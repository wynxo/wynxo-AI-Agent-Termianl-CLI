"""Running commands, on whichever OS the user is actually on."""

from __future__ import annotations

import asyncio
import os
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


def _clean(raw: bytes) -> str:
    """One line of output, as text. Carriage returns are collapsed so a
    progress bar reads as its final state rather than every frame at once."""
    text = raw.decode("utf-8", "replace")
    if "\r" in text:
        text = text.split("\r")[-1] or text.rstrip("\r")
    return text.rstrip("\n")

# Commands that are almost never what a confused model meant, and are
# unrecoverable when they are wrong. These are refused outright rather than
# merely prompted for, because a yes/no prompt is exactly the thing a user
# clicks through on autopilot.
HARD_BLOCKED = (
    "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -rf ~/", ":(){:|:&};:",
    "mkfs", "dd if=/dev/zero of=/dev", "> /dev/sda",
    "format c:", "del /f /s /q c:\\", "rd /s /q c:\\",
    "shutdown", "reboot", "halt",
)


class ShellInput(Schema):
    command = Field(str, "The command line to run.")
    timeout = Field(int, "Seconds before the command is killed.", default=120, ge=1, le=900)
    cwd = Field(str, "Working directory, relative to the project root.", default="")


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

    async def run(self, args: ShellInput) -> ToolResult:
        command = args.command.strip()
        if not command:
            return ToolResult.failure("Empty command.")

        lowered = " ".join(command.lower().split())
        for blocked in HARD_BLOCKED:
            if blocked in lowered:
                return ToolResult.failure(
                    f"Refusing to run this: it contains {blocked!r}, which is "
                    "destructive and not reversible. If you genuinely need it, "
                    "run it yourself outside the agent."
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
            )
        except OSError as exc:
            return ToolResult.failure(f"Could not start {shell}: {exc}")

        output, timed_out = await self._stream(process, args.timeout)
        if timed_out:
            await self._terminate(process)
            # The output is handed back rather than discarded. A command that
            # hangs is exactly when its last few lines matter most -- they say
            # which test wedged or which download stalled -- and throwing them
            # away leaves the model with nothing to act on but the word
            # "timeout".
            return ToolResult.failure(
                f"Command timed out after {args.timeout}s and was killed: "
                f"{command}\n\nOutput before it was killed:\n"
                f"{output or '(none)'}"
            )
        await process.wait()

        code = process.returncode or 0
        if code == 0:
            return ToolResult.success(
                output or "(no output)",
                display=f"$ {command}",
                exit_code=0,
            )
        return ToolResult(
            ok=False,
            output=f"exit code {code}\n{output or '(no output)'}",
            display=f"$ {command}",
            error=f"exit code {code}",
            metadata={"exit_code": code},
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
            if head_chars < MAX_OUTPUT // 2:
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
    async def _terminate(process) -> None:
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
