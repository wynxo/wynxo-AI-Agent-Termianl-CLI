from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.tools import build_registry
from wynxo.tools.base import ToolResult
from wynxo.tools.dev import GitDiff, GitLog, GitStatus, RunTests
from wynxo.tools.files import EditFile


def test_registry_exposes_coding_agent_tools(tmp_path: Path):
    names = set(build_registry(tmp_path).names())
    assert {
        "read_file", "write_file", "edit_file", "list_directory",
        "find_files", "search_text", "shell", "git_status", "git_diff",
        "git_log", "run_tests",
    } <= names


def test_edit_reports_stale_or_ambiguous_content(tmp_path: Path):
    path = tmp_path / "x.py"
    path.write_text("value = 1\nvalue = 1\n")
    result = asyncio.run(EditFile(tmp_path).invoke({
        "path": "x.py", "old_text": "value = 1", "new_text": "value = 2"
    }))
    assert not result.ok
    assert "appears 2 times" in result.output
    assert path.read_text() == "value = 1\nvalue = 1\n"


def test_run_tests_returns_structured_failure(tmp_path: Path):
    result = asyncio.run(RunTests(tmp_path).invoke({
        "command": "python -c \"import sys; print('bad'); sys.exit(3)\""
    }))
    assert not result.ok
    assert result.metadata["exit_code"] in (1, 3)
    assert result.metadata["command"]
    assert result.metadata["duration"] >= 0


def test_git_tools_are_read_only_and_structured(tmp_path: Path):
    asyncio.run(asyncio.create_subprocess_exec("git", "init", str(tmp_path),
                                                stdout=asyncio.subprocess.DEVNULL,
                                                stderr=asyncio.subprocess.DEVNULL))
    status = asyncio.run(GitStatus(tmp_path).invoke({}))
    diff = asyncio.run(GitDiff(tmp_path).invoke({}))
    log = asyncio.run(GitLog(tmp_path).invoke({}))
    assert status.metadata["exit_code"] == 0
    assert diff.metadata["exit_code"] == 0
    assert log.metadata["exit_code"] != 0 or "fatal" in log.output.lower()
