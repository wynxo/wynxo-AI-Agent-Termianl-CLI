"""config.max_command_output_chars must actually reach the shell tool.

The setting used to exist in config while the shell kept a hardcoded
constant -- a config option that was a lie. This runs on Windows too,
unlike the POSIX-only streaming tests.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from wynxo.tools.shell import Shell, ShellInput


def run(tool: Shell, command: str, timeout: int = 60):
    return asyncio.run(tool.run(ShellInput(command=command, timeout=timeout)))


def test_max_output_config_changes_what_is_kept(tmp_path: Path):
    py = sys.executable
    big = f"& '{py}' -c \"import sys; [print('x'*80) for _ in range(600)]\""
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
