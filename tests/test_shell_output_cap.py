"""config.max_command_output_chars must actually reach the shell tool.

The setting used to exist in config while the shell kept a hardcoded
constant -- a config option that was a lie. This runs on Windows too,
unlike the POSIX-only streaming tests.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from wynxo.tools.shell import Shell, ShellInput


def run(tool: Shell, command: str, timeout: int = 60):
    return asyncio.run(tool.run(ShellInput(command=command, timeout=timeout)))


def chatty(lines: int = 600) -> str:
    """A command that prints ``lines`` wide lines, spelled for this shell.

    The shell tool runs PowerShell on Windows and the login shell elsewhere,
    and the two disagree about how to run a quoted program path: PowerShell
    needs its `&` call operator, bash treats `&` as "background the empty
    command" and fails with a syntax error. This file claims to run on both,
    so it has to spell the command for whichever one it got -- hardcoding
    the PowerShell form meant it could not pass on Linux at all.
    """
    script = "import sys; [print('x'*80) for _ in range(%d)]" % lines
    if os.name == "nt":
        return f"& '{sys.executable}' -c \"{script}\""
    return f"'{sys.executable}' -c \"{script}\""


def test_max_output_config_changes_what_is_kept(tmp_path: Path):
    big = chatty()
    full = run(Shell(workspace=tmp_path), big)
    capped = run(Shell(workspace=tmp_path, max_output=2000), big)
    assert full.ok, full.error
    assert capped.ok, capped.error
    assert len(capped.output) < len(full.output), (
        "a smaller max_output must retain less output")
    # The tail survives even a small cap: the last lines are what a failing
    # build explains itself with.
    assert capped.output.strip().endswith("x" * 80)


def test_default_max_output_matches_the_old_constant(tmp_path: Path):
    tool = Shell(workspace=tmp_path)
    assert tool.max_output == 30_000
