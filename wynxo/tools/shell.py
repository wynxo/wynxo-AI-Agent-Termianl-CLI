"""Running commands, on whichever OS the user is actually on."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import subprocess
from collections import deque

from ..platforms import default_shell
from ..schema import Field, Schema
from .base import Tool, ToolResult

MAX_OUTPUT = 30_000

TAIL_LINES = 200
"""How much of the end to keep when a command is very chatty. The tail is
what matters -- a failing build says why on its last lines."""

MAX_LINE_BYTES = 16_384
"""A "line" longer than this is a progress bar redrawing with \\r, not a
line. Flushed rather than buffered until the process exits."""


def _new_process_group() -> dict:
    """Keyword arguments that give the command its own process group."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags} if flags else {}
    # POSIX: a new session, which is also a new process group.
    return {"start_new_session": True}


def _signal_group(process, terminate: bool) -> None:
    """Signal the command's whole process group, falling back to the one
    process where the platform will not do groups."""
    if process.pid is None:
        return
    if os.name == "nt":
        # Windows has no process groups in the POSIX sense; taskkill /T is
        # the equivalent, and /F is the only reliable form of it.
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=10)
            return
        except (OSError, subprocess.SubprocessError):
            pass          # fall through to the single-process attempt
    else:
        sig = signal.SIGTERM if terminate else signal.SIGKILL
        try:
            os.killpg(os.getpgid(process.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass          # already gone, or never got its own group
    try:
        process.terminate() if terminate else process.kill()
    except (ProcessLookupError, OSError, ValueError):
        pass


def _clean(raw: bytes) -> str:
    """Output as text. Carriage returns are collapsed so a progress bar
    reads as its final state rather than every frame at once.

    Windows child processes commonly end lines with CRLF, and a pipe can
    add extra carriage returns (PowerShell turning ``\n`` into ``\r\n``);
    those are line endings, not progress-bar frames, so they are stripped
    *before* the lone-``\r`` collapse. Without that a ``\r\r\n`` line
    collapsed to an empty string on Windows.
    """
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"\r+(\n|$)", r"\1", text)   # CRLF / trailing CR -> LF / end
    if "\r" in text:
        # A genuine in-place redraw: keep only the final frame.
        text = text.split("\r")[-1]
    return text.rstrip("\n")

# Commands that are almost never what a confused model meant, and are
# unrecoverable when they are wrong. These are refused outright rather than
# merely prompted for, because a yes/no prompt is exactly the thing a user
# clicks through on autopilot.
#
# Matched against the command being run rather than as a substring of the
# line. Substrings refused far too much: "rm -rf /tmp/build" starts with
# "rm -rf /", and "git commit -m 'handle shutdown cleanly'" contains the
# word shutdown, as does "grep -rn reboot src". All three were refused
# outright, with no way past it.

_EVERYTHING = {"/", "/*", "/.", "~", "~/", "~/*", "*", "/usr", "/etc",
               "/home", "/var", "/bin", "/lib", "/boot", "/sys", "/proc"}
"""Targets that mean "the machine" rather than "this project"."""

_FORMATTERS = ("mkfs", "mke2fs", "mkdosfs", "newfs", "diskpart")
_TURNS_IT_OFF = {"shutdown", "reboot", "halt", "poweroff"}
_FORK_BOMB = ":(){:|:&};:"
_RAW_DISK = re.compile(r">\s*/dev/(sd|nvme|hd|disk|vd)", re.IGNORECASE)
_WINDOWS_ROOT = re.compile(r"^[a-z]:[\\/]?$", re.IGNORECASE)

_SEPARATORS = re.compile(r"&&|\|\||[;|&\n\r]")


def _commands_in(line: str) -> list[list[str]]:
    """The separate commands a line would run, each as its tokens."""
    out = []
    for segment in _SEPARATORS.split(line):
        if not segment.strip():
            continue
        try:
            tokens = shlex.split(segment, posix=os.name != "nt")
        except ValueError:
            tokens = segment.split()      # unbalanced quotes; do the crude thing
        if tokens:
            out.append(tokens)
    return out


_WRAPPERS = {"sudo", "doas", "env", "nice", "ionice", "nohup", "time",
             "command", "exec", "stdbuf"}
"""Things that run something else. "sudo rm -rf /" is still "rm -rf /"."""


def _unwrap(tokens: list[str]) -> list[str]:
    """Strip the wrappers off the front to find the command being run."""
    while tokens and tokens[0].lower().rsplit("/", 1)[-1] in _WRAPPERS:
        tokens = tokens[1:]
        while tokens and (tokens[0].startswith("-") or "=" in tokens[0]):
            tokens = tokens[1:]
    return tokens


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
        # The agent threads config.max_command_output_chars through here; the
        # hardcoded default keeps every other construction site working.
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
        # Stop interactive pagers and prompts from hanging the agent forever.
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
                # Its own process group, so the whole command can be killed
                # rather than just the shell that launched it. `make -j8` and
                # `npm install` spawn workers, and killing their parent leaves
                # those workers running -- eating the machine long after the
                # user thinks they stopped it.
                **_new_process_group(),
            )
        except OSError as exc:
            return ToolResult.failure(f"Could not start {shell}: {exc}")

        if args.background:
            return await _launch_background(process, command)

        try:
            output, timed_out = await self._stream(process, args.timeout)
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Ctrl-C. Without this the await is abandoned and the command
            # carries on in the background, still writing to the project,
            # while the user believes they stopped it.
            await self._terminate(process)
            await self._close_streams(process)
            raise
        if timed_out:
            await self._terminate(process)
            await self._close_streams(process)
            # The output is handed back rather than discarded. A command that
            # hangs is exactly when its last few lines matter most -- they say
            # which test wedged or which download stalled -- and throwing them
            # away leaves the model with nothing to act on but the word
            # "timeout".
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
        # A shell may encode a child failure in its own status; on Windows
        # PowerShell commonly emits the real code in the output while exiting
        # with 1. Preserve the useful status for structured consumers.
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
        """Read the command's output as it arrives, not once it is over.

        `communicate()` waits for the process to exit, so a five-minute test
        run or an `npm install` showed absolutely nothing until it finished
        -- and if it hit the timeout, the output that would have explained
        why was thrown away with it. Both of those are worst exactly when
        something is going wrong.

        Read in chunks rather than by line: asyncio's readline() raises once
        a line exceeds its buffer limit, and a progress bar that redraws with
        \r is one enormous line.
        """
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
                # A tool that crashes the turn because the UI hiccuped would
                # be a poor trade for a progress display.
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
                chunk = await asyncio.wait_for(process.stdout.read(4096),
                                               timeout=remaining)
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
                # A progress bar rewriting one line forever. Flush what we
                # have so it is not held in memory until the process exits.
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
        """Retire the pipe transports before the loop does.

        On Windows the ProactorEventLoop hands a subprocess a pipe transport
        for stdout. If the loop closes while that transport is still alive
        -- the exact shape of a cancelled or timed-out command -- the
        deallocator later runs against an already-closed socket and the
        interpreter prints "Exception ignored while calling deallocator"
        (asyncio: I/O operation on closed pipe). Explicitly closing the
        stream retires the transport cleanly, which any asyncio program
        should do once the process has been reaped.
        """
        try:
            stream = getattr(process, "stdout", None)
            if stream is not None:
                stream.close()
        except Exception:
            pass

    @staticmethod
    async def _terminate(process) -> None:
        """Stop the command and everything it started.

        Politely first: SIGTERM to the group gives a test runner the chance
        to tear down its own children and remove its temp files. SIGKILL is
        the follow-up for anything that ignores it.
        """
        if process.returncode is not None:
            return
        _signal_group(process, terminate=True)
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5.0)
            return
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
        except asyncio.CancelledError:
            # Interrupted again while cleaning up. Finish the job -- leaving
            # a half-killed process group is the thing we are here to avoid.
            _signal_group(process, terminate=False)
            raise
        _signal_group(process, terminate=False)
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError, asyncio.CancelledError):
            pass


# -- background jobs ---------------------------------------------------------
#
# shell(background=true) returns a job id and keeps the process alive; the
# agent polls with background_poll and keeps chatting in between. The store
# is process-local and per-interpreter: a job cannot outlive its session,
# which is exactly the boundary a background job should respect.

_BACKGROUND: dict[str, dict] = {}
"""job id -> job state, for the lifetime of this interpreter."""


async def _launch_background(process, command: str) -> ToolResult:
    """Register a running process as a pollable job and return immediately."""
    import uuid

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
            # Bounded like the foreground path: a chatty job must not grow
            # without limit in memory while nobody is looking at it.
            if len(job["output"]) < MAX_OUTPUT:
                job["output"].extend(chunk[:max(0, MAX_OUTPUT - len(job["output"]))])
        await process.wait()
    except (asyncio.CancelledError, ProcessLookupError, OSError):
        pass
    finally:
        job["exit_code"] = process.returncode
        job["done"].set()
        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            pass


def _job_output(job: dict) -> str:
    return _clean(bytes(job["output"])).strip()


class BackgroundPollInput(Schema):
    job_id = Field(str, "The job id returned by shell(background=true).")
    kill = Field(bool, "Set true to stop the job and its whole process group.",
                 default=False)


class BackgroundPoll(Tool):
    name = "background_poll"
    description = (
        "Check on a command started in the background with "
        "shell(background=true): whether it has finished, and what it has "
        "printed so far. Set kill=true to stop it. Use it to run long "
        "builds or installs while continuing to work."
    )
    Input = BackgroundPollInput
    mutating = True      # kill=true changes the world
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
                await asyncio.wait_for(asyncio.shield(process.wait()),
                                       timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError,
                    asyncio.CancelledError):
                _signal_group(process, terminate=False)
            await job["done"].wait()
            return ToolResult.success(
                f"Killed job {args.job_id}: {job['command']}",
                job_id=args.job_id, exit_code=job["exit_code"],
                finished=True)
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
