"""Running commands, on whichever OS the user is actually on."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

MAX_OUTPUT = 30_000

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


def default_shell() -> tuple[str, list[str]]:
    """Pick a shell and the flag that makes it run a command string."""
    if sys.platform == "win32":
        # PowerShell where available -- cmd.exe quoting is a source of
        # endless subtle breakage, and pwsh/powershell is on every
        # supported Windows.
        for exe in ("pwsh", "powershell"):
            if shutil.which(exe):
                return exe, ["-NoProfile", "-NonInteractive", "-Command"]
        return os.environ.get("COMSPEC", "cmd.exe"), ["/c"]
    shell = os.environ.get("SHELL")
    if shell and shutil.which(shell):
        return shell, ["-c"]
    for exe in ("bash", "sh"):
        if path := shutil.which(exe):
            return path, ["-c"]
    return "/bin/sh", ["-c"]


class ShellInput(BaseModel):
    command: str = Field(description="The command line to run.")
    timeout: int = Field(120, ge=1, le=900, description="Seconds before the command is killed.")
    cwd: str = Field("", description="Working directory, relative to the project root.")


class Shell(Tool):
    name = "shell"
    description = (
        "Run a shell command in the project directory and return its output. "
        "Use this for git, tests, linters, package managers and build tools. "
        "It is PowerShell on Windows and your login shell elsewhere, so write "
        "the command for the platform you were told you are on."
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

        try:
            raw, _ = await asyncio.wait_for(process.communicate(), timeout=args.timeout)
        except asyncio.TimeoutError:
            await self._terminate(process)
            return ToolResult.failure(
                f"Command timed out after {args.timeout}s and was killed: {command}"
            )

        output = raw.decode("utf-8", "replace").strip()
        if len(output) > MAX_OUTPUT:
            head = output[: MAX_OUTPUT // 2]
            tail = output[-MAX_OUTPUT // 2 :]
            omitted = len(output) - MAX_OUTPUT
            output = f"{head}\n\n... [{omitted} characters omitted] ...\n\n{tail}"

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
