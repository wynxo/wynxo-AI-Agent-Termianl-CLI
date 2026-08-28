"""Developer workflow tools: repository inspection, git, and tests."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

from ..schema import Field, Schema
from .. import testing
from .base import Tool, ToolResult
from .shell import Shell


class RepoInput(Schema):
    path = Field(str, "Repository path, relative to the workspace.", default=".")


class GitStatus(Tool):
    name = "git_status"
    description = "Show the current branch and working-tree changes without modifying anything."
    Input = RepoInput

    async def run(self, args: RepoInput) -> ToolResult:
        return await self._git("status --short --branch", args.path)

    async def _git(self, command: str, path: str) -> ToolResult:
        shell = Shell(self.workspace, self.boundary, self.shield)
        return await shell.invoke({"command": f"git -C {self._quote(path)} {command}", "timeout": 30})

    @staticmethod
    def _quote(path: str) -> str:
        import shlex
        return shlex.quote(path) if os.name != "nt" else '"' + path.replace('"', '\\"') + '"'


class GitDiff(GitStatus):
    name = "git_diff"
    description = "Inspect the uncommitted diff, optionally including staged changes."

    async def run(self, args: RepoInput) -> ToolResult:
        return await self._git("diff --no-ext-diff", args.path)


class GitLog(GitStatus):
    name = "git_log"
    description = "Show recent commits for understanding project history."

    async def run(self, args: RepoInput) -> ToolResult:
        return await self._git("log -8 --oneline --decorate", args.path)


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
