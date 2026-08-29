"""Developer workflow tools: git, and tests."""

from __future__ import annotations

import os
import time

from ..schema import Field, Schema
from .. import testing
from .base import Tool, ToolResult
from .shell import Shell


class GitInput(Schema):
    action = Field(str, "What to look at: status, diff, or log.", default="status",
                   choices=("status", "diff", "log"))
    path = Field(str, "Repository path, relative to the workspace.", default=".")


_COMMANDS = {
    "status": "status --short --branch",
    "diff": "diff --no-ext-diff",
    "log": "log -8 --oneline --decorate",
}


class Git(Tool):
    name = "git"
    description = "Inspect the repository: status (branch and working-tree changes), " \
                  "diff (uncommitted changes), or log (recent commits). Read-only."
    Input = GitInput

    async def run(self, args: GitInput) -> ToolResult:
        command = _COMMANDS.get(args.action, _COMMANDS["status"])
        shell = Shell(self.workspace, self.boundary, self.shield)
        return await shell.invoke({
            "command": f"git -C {self._quote(args.path)} {command}",
            "timeout": 30,
        })

    @staticmethod
    def _quote(path: str) -> str:
        import shlex
        return shlex.quote(path) if os.name != "nt" else '"' + path.replace('"', '\\"') + '"'


class TestsInput(Schema):
    command = Field(str, "Test command; empty means auto-detect the project command.", default="")
    timeout = Field(int, "Maximum test duration in seconds.", default=120, ge=1, le=900)


class RunTests(Tool):
    name = "run_tests"
    description = "Run the project's tests and return structured output, exit status, and duration."
    Input = TestsInput
    mutating = True
    concurrency_safe = False

    async def run(self, args: TestsInput) -> ToolResult:
        command = args.command.strip()
        runner = testing.detect(self.workspace) if not command else None
        if not command and runner is None:
            return ToolResult.failure("Could not detect a test command; provide command explicitly.")
        command = command or runner.command
        started = time.monotonic()
        shell = Shell(self.workspace, self.boundary, self.shield)
        result = await shell.invoke({"command": command, "timeout": args.timeout})
        duration = time.monotonic() - started
        result.metadata.update({"command": command, "duration": duration,
                                "exit_code": result.metadata.get("exit_code", 0 if result.ok else 1),
                                "timed_out": "timed out" in result.output.lower()})
        if not result.ok:
            import re
            match = re.search(r"exit code ([-]?\d+)", result.output, re.IGNORECASE)
            if match:
                result.metadata["exit_code"] = int(match.group(1))
        result.display = f"{'passed' if result.ok else 'failed'}: {command} ({duration:.1f}s)"
        return result
