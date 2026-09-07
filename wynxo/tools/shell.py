"""Running commands, on whichever OS the user is actually on."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import subprocess
import time
from collections import deque

from ..platforms import default_shell
from ..schema import Field, Schema
from .base import Tool, ToolResult

MAX_OUTPUT = 30_000

TAIL_LINES = 200
"""How much of the end to keep when a command is very chatty. The tail is
what matters -- a failing build says why on its last lines."""

MAX_LINE_BYTES = 16_384
"""A "line" longer than this is a progress bar redrawing with \r, not a
line. Flushed rather than buffered until the process exits."""


def _new_process_group() -> dict:
    """Keyword arguments that give the command its own process group."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags} if flags else {}
    return {"start_new_session": True}


def _signal_group(process, terminate: bool) -> None:
    """Signal the command's whole process group, falling back to one process."""
    if process.pid is None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=10)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        sig = signal.SIGTERM if terminate else signal.SIGKILL
        try:
            os.killpg(os.getpgid(process.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.terminate() if terminate else process.kill()
    except (ProcessLookupError, OSError, ValueError):
        pass


def _close_transports(process, force: bool = False) -> None:
    """Retire a finished process's pipe transports."""
    if not force and getattr(process, "returncode", None) is None:
        return
    transport = getattr(process, "_transport", None)
    if transport is None:
        return
    try:
        transport.close()
    except Exception:
        pass


def _clean(raw: bytes) -> str:
    """Output as text, collapsing in-place progress redraws."""
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"\r+(\n|$)", r"\1", text)
    if "\r" in text:
        text = text.split("\r")[-1]
    return text.rstrip("\n")


_EVERYTHING = {"/", "/*", "/.", "~", "~/", "~/*", "*", "/usr", "/etc",
               "/home", "/var", "/bin", "/lib", "/boot", "/sys", "/proc"}
_FORMATTERS = ("mkfs", "mke2fs", "mkdosfs", "newfs", "diskpart")
_TURNS_IT_OFF = {"shutdown", "reboot", "halt", "poweroff"}
_FORK_BOMB = ":(){:|:&};:"
_RAW_DISK = re.compile(r">\s*/dev/(sd|nvme|hd|disk|vd)", re.IGNORECASE)
_WINDOWS_ROOT = re.compile(r"^[a-z]:[\\/]?$", re.IGNORECASE)
_SEPARATORS = re.compile(r"&&|\|\||[;|&\n\r]")


def _split_segments(line: str) -> list[str]:
    """The line's separate commands, without splitting inside quotes."""
    segments, current, quote, index = [], [], "", 0
    while index < len(line):
        char = line[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            elif char == "\\" and index + 1 < len(line) and quote == '"':
                index += 1
                current.append(line[index])
        elif char in "'\"":
            quote = char
            current.append(char)
        elif match := _SEPARATORS.match(line, index):
            segments.append("".join(current))
            current = []
            index = match.end()
            continue
        else:
            current.append(char)
        index += 1
    segments.append("".join(current))
    return [s for s in segments if s.strip()]


def _commands_in(line: str, depth: int = 0) -> list[list[str]]:
    """Return separate commands, recursively including inline shell scripts."""
    out = []
    for segment in _split_segments(line):
        try:
            tokens = shlex.split(segment, posix=os.name != "nt")
        except ValueError:
            tokens = segment.split()
        if not tokens:
            continue
        out.append(tokens)
        script = _script_of(_unwrap(tokens))
        if script and depth < 4:
            out.extend(_commands_in(script, depth + 1))
    return out


_WRAPPERS = {"sudo", "doas", "env", "nice", "ionice", "nohup", "time",
             "command", "exec", "stdbuf", "xargs"}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "ash", "busybox"}


def _unwrap(tokens: list[str]) -> list[str]:
    """Strip wrappers off the front to find the command being run."""
    while tokens and tokens[0].lower().rsplit("/", 1)[-1] in _WRAPPERS:
        tokens = tokens[1:]
        while tokens and (tokens[0].startswith("-") or "=" in tokens[0]):
            tokens = tokens[1:]
    return tokens


def _dequote_script(script: str) -> str:
    """Undo quote retention from ``shlex(..., posix=False)`` on Windows.

    The safety parser intentionally follows the host tokenizer for ordinary
    commands. On Windows that tokenizer retains the surrounding quotes of
    ``bash -c 'rm -rf /'``. Feeding that quoted token back into the recursive
    parser made the whole script look like one harmless command and allowed a
    destructive command to hide behind ``sh -c``/``bash -lc``. Only remove one
    matching outer quote pair; inner quoting still belongs to the script.
    """
    script = script.strip()
    if len(script) >= 2 and script[0] == script[-1] and script[0] in "'\"":
        return script[1:-1]
    return script


def _script_of(tokens: list[str]) -> str | None:
    """The inline script a shell was asked to run, if any."""
    if not tokens or tokens[0].lower().rsplit("/", 1)[-1] not in _SHELLS:
        return None
    rest = tokens[1:]
    while rest:
        flag, rest = rest[0], rest[1:]
        if not flag.startswith("-"):
            return None
        letters = flag[1:]
        if (not flag.startswith("--") and "c" in letters) or flag == "--command":
            return _dequote_script(rest[0]) if rest else None
    return None


def hard_refusal(line: str) -> str:
    """Why this must not run at all, or "" if it may be asked about."""
    if _FORK_BOMB in "".join(line.split()):
        return "a fork bomb"
    if _RAW_DISK.search(line):
        return "a write straight to a raw disk device"

    for tokens in _commands_in(line):
        tokens = _unwrap(tokens)
        if not tokens:
            continue
        head = tokens[0].lower().rsplit("/", 1)[-1]
        arguments = [t for t in tokens[1:] if not t.startswith("-")]
        if head == "rm" and any(a.rstrip("/") in
                                {e.rstrip("/") for e in _EVERYTHING} or
                                a in _EVERYTHING for a in arguments):
            return "a recursive delete of the whole filesystem"
        if head.startswith(_FORMATTERS):
            return "formatting a filesystem"
        if head == "format" and any(_WINDOWS_ROOT.match(a) for a in arguments):
            return "formatting a drive"
        if head in ("del", "rd", "rmdir") and any(_WINDOWS_ROOT.match(a)
                                                  for a in arguments):
            return "deleting a whole drive"
        if head in _TURNS_IT_OFF:
            return "shutting the machine down"
        if head == "dd" and any(a.startswith("of=/dev/") for a in tokens[1:]):
            return "writing straight to a device"
    return ""


class ShellInput(Schema):
    command = Field(str, "The command line to run.")
    timeout = Field(int, "Seconds before the command is killed.", default=120, ge=1, le=900)
    cwd = Field(str, "Working directory, relative to the project root.", default="")
    background = Field(bool, "Start the command and return immediately with a "
                             "job id, instead of waiting for it to finish. "
                             "Poll it later with background_poll.",
                       default=False)


class Shell(Tool):
    name = "shell"
    description = (
        "Run a shell command in the project directory and return its output. "
        "Use this for git, tests, linters, package managers and build tools. "
        "It is PowerShell on Windows and your login shell elsewhere (Termux "
        "included), so write the command for the platform you were told you "
        "are on."
    )
    Input = ShellInput
    mutating = True
    concurrency_safe = False

    def __init__(self, workspace, boundary=None, shield=None,
                 max_output: int = MAX_OUTPUT):
        super().__init__(workspace, boundary, shield)
        self.max_output = max(int(max_output or 0), 1000)

    async def run(self, args: ShellInput) -> ToolResult:
        command = args.command.strip()
        if not command:
            return ToolResult.failure("Empty command.")

        if reason := hard_refusal(command):
            return ToolResult.failure(
                f"Refusing to run this: it is {reason}, which is destructive "
                "and not reversible. If you genuinely need it, run it "
                "yourself outside the agent."
            )

        cwd = self.resolve_path(args.cwd) if args.cwd else self.workspace
        if not cwd.is_dir():
            return ToolResult.failure(f"{self.relative(cwd)} is not a directory.")

        shell, flags = default_shell()
        env = dict(os.environ)
        env.update({
            "GIT_PAGER": "cat", "PAGER": "cat", "GIT_TERMINAL_PROMPT": "0",
            "DEBIAN_FRONTEND": "noninteractive", "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1", "CI": "1",
        })

        try:
            process = await asyncio.create_subprocess_exec(
                shell, *flags, command,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **_new_process_group(),
            )
        except OSError as exc:
            return ToolResult.failure(f"Could not start {shell}: {exc}")

        if args.background:
            return await _launch_background(process, command)

        try:
            output, timed_out = await self._stream(process, args.timeout)
        except (asyncio.CancelledError, KeyboardInterrupt):
            await self._terminate(process)
            await self._close_streams(process)
            raise
        if timed_out:
            await self._terminate(process)
            await self._close_streams(process)
            return ToolResult.failure(
                f"Command timed out after {args.timeout}s and was killed: "
                f"{command}\n\nOutput before it was killed:\n"
                f"{output or '(none)'}",
                command=command, timed_out=True, cancelled=False,
                stdout=output, stderr="", exit_code=None,
            )
        await process.wait()
        await self._close_streams(process)

        code = process.returncode or 0
        if code == 1 and output:
            import re as _re
            match = _re.search(r"exit(?:ed)?\s+code\s+(-?\d+)", output, _re.IGNORECASE)
            if match:
                code = int(match.group(1))
        if code == 0:
            return ToolResult.success(
                output or "(no output)",
                display=f"$ {command}",
                command=command, stdout=output, stderr="", exit_code=0,
                timed_out=False, cancelled=False,
            )
        return ToolResult(
            ok=False,
            output=f"exit code {code}\n{output or '(no output)'}",
            display=f"$ {command}",
            error=f"exit code {code}",
            metadata={"exit_code": code, "command": command,
                      "stdout": output, "stderr": "", "timed_out": False,
                      "cancelled": False},
        )

    async def _stream(self, process, timeout: int) -> tuple[str, bool]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        head: list[str] = []
        tail: deque[str] = deque(maxlen=TAIL_LINES)
        head_chars = 0
        dropped = 0
        pending = b""

        async def emit(line: str) -> None:
            nonlocal head_chars, dropped
            if head_chars < self.max_output // 2:
                head.append(line)
                head_chars += len(line) + 1
            else:
                if len(tail) == tail.maxlen:
                    dropped += 1
                tail.append(line)
            if self.on_output is not None:
                try:
                    await self.on_output(line)
                except Exception:
                    pass

        timed_out = False
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                break
            try:
                chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=remaining)
            except asyncio.TimeoutError:
                timed_out = True
                break
            if not chunk:
                break
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for raw_line in lines:
                await emit(_clean(raw_line))
            if len(pending) > MAX_LINE_BYTES:
                await emit(_clean(pending))
                pending = b""

        if pending:
            await emit(_clean(pending))

        body = "\n".join(head)
        if tail:
            gap = (f"\n\n... [{dropped} lines omitted] ...\n\n"
                   if dropped else "\n")
            body += gap + "\n".join(tail)
        return body.strip(), timed_out

    @staticmethod
    async def _close_streams(process) -> None:
        _close_transports(process)

    @staticmethod
    async def _terminate(process) -> None:
        if process.returncode is not None:
            return
        _signal_group(process, terminate=True)
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5.0)
            return
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
        except asyncio.CancelledError:
            _signal_group(process, terminate=False)
            raise
        _signal_group(process, terminate=False)
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError, asyncio.CancelledError):
            pass


_BACKGROUND: dict[str, dict] = {}
_ATEXIT_REGISTERED = False


def shutdown_background(timeout: float = 3.0) -> int:
    """Stop every background job still running. Returns how many were killed."""
    doomed = [job["process"] for job in _BACKGROUND.values()
              if job["process"].returncode is None and not job.get("stopped")]
    if not doomed:
        return 0
    for job in _BACKGROUND.values():
        job["stopped"] = True
    import logging

    asyncio_log = logging.getLogger("asyncio")
    previous = asyncio_log.level
    asyncio_log.setLevel(logging.CRITICAL)
    try:
        for process in doomed:
            _signal_group(process, terminate=True)
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if all(_gone(process) for process in doomed):
                break
            time.sleep(0.05)
        for process in doomed:
            if not _gone(process):
                _signal_group(process, terminate=False)
            _reap_after_loop_close(process, timeout=1.0)
            _close_transports(process, force=True)
    finally:
        asyncio_log.setLevel(previous)
    return len(doomed)


def _reap_after_loop_close(process, timeout: float = 0.0) -> bool:
    """Reap an exited child when its event loop can no longer do so."""
    loop = getattr(process, "_loop", None)
    transport = getattr(process, "_transport", None)
    child = getattr(transport, "_proc", None)
    if loop is None or not loop.is_closed() or child is None:
        return False
    try:
        if timeout > 0:
            return child.wait(timeout=timeout) is not None
        return child.poll() is not None
    except (OSError, subprocess.TimeoutExpired):
        return False


def _gone(process) -> bool:
    """True when the process is no longer running."""
    if process.returncode is not None or _reap_after_loop_close(process):
        return True
    if process.pid is None or os.name == "nt":
        return process.returncode is not None
    try:
        os.kill(process.pid, 0)
        return False
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False


async def _launch_background(process, command: str) -> ToolResult:
    """Register a running process as a pollable job and return immediately."""
    import atexit
    import uuid

    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(shutdown_background)
        _ATEXIT_REGISTERED = True

    job_id = uuid.uuid4().hex[:8]
    job = {
        "process": process,
        "command": command,
        "output": bytearray(),
        "done": asyncio.Event(),
        "exit_code": None,
        "timed_out": False,
    }
    _BACKGROUND[job_id] = job
    asyncio.get_running_loop().create_task(_background_reader(job))
    return ToolResult.success(
        f"Started in the background as job {job_id}: {command}\n"
        "Poll with background_poll(job_id).",
        display=f"$ {command} &  (job {job_id})",
        job_id=job_id, command=command,
    )


async def _background_reader(job: dict) -> None:
    """Drain the process's output into the job buffer until it exits."""
    process = job["process"]
    try:
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                break
            if len(job["output"]) < MAX_OUTPUT:
                job["output"].extend(chunk[:max(0, MAX_OUTPUT - len(job["output"]))])
        await process.wait()
    except (asyncio.CancelledError, ProcessLookupError, OSError):
        pass
    finally:
        job["exit_code"] = process.returncode
        job["done"].set()
        _close_transports(process)


def _job_output(job: dict) -> str:
    return _clean(bytes(job["output"])).strip()


class BackgroundPollInput(Schema):
    job_id = Field(str, "The job id returned by shell(background=true).")
    kill = Field(bool, "Set true to stop the job and its whole process group.", default=False)


class BackgroundPoll(Tool):
    name = "background_poll"
    description = (
        "Check on a command started in the background with "
        "shell(background=true): whether it has finished, and what it has "
        "printed so far. Set kill=true to stop it. Use it to run long "
        "builds or installs while continuing to work."
    )
    Input = BackgroundPollInput
    mutating = True
    concurrency_safe = False

    async def run(self, args: BackgroundPollInput) -> ToolResult:
        job = _BACKGROUND.get(args.job_id)
        if job is None:
            return ToolResult.failure(
                f"No background job {args.job_id!r}. Jobs live only for the "
                "lifetime of this session.")
        process = job["process"]
        if args.kill and process.returncode is None:
            _signal_group(process, terminate=True)
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError, asyncio.CancelledError):
                _signal_group(process, terminate=False)
            await job["done"].wait()
            return ToolResult.success(
                f"Killed job {args.job_id}: {job['command']}",
                job_id=args.job_id, exit_code=job["exit_code"], finished=True)
        if not job["done"].is_set():
            tail = _job_output(job)
            return ToolResult.success(
                f"Job {args.job_id} is still running: {job['command']}"
                + (f"\n\n{tail[-2000:]}" if tail else ""),
                job_id=args.job_id, finished=False,
                exit_code=None, stdout=tail)
        output = _job_output(job)
        code = job["exit_code"] or 0
        if code == 0:
            return ToolResult.success(
                output or "(no output)",
                job_id=args.job_id, finished=True, exit_code=0,
                stdout=output)
        return ToolResult(
            ok=False,
            output=f"exit code {code}\n{output or '(no output)'}",
            error=f"exit code {code}",
            metadata={"job_id": args.job_id, "finished": True,
                      "exit_code": code, "output": output},
        )
