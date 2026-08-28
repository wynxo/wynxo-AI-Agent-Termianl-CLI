from __future__ import annotations

from pathlib import Path

import pytest

from wynxo.hardening import (
    _atomic_write_back,
    windows_hard_refusal,
    windows_is_read_only_command,
)
from wynxo.tools import Glob, ListDir


def test_atomic_write_preserves_existing_mode(tmp_path: Path):
    target = tmp_path / "script.py"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o750)
    old_mode = target.stat().st_mode & 0o777

    _atomic_write_back(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert target.stat().st_mode & 0o777 == old_mode
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_windows_drive_destructive_commands_are_rejected():
    assert windows_hard_refusal("Remove-Item -Recurse -Force C:\\")
    assert windows_hard_refusal("Format-Volume -DriveLetter C")
    assert windows_hard_refusal("Stop-Computer -Force")


def test_windows_project_mutations_are_not_misclassified_as_read_only():
    assert not windows_is_read_only_command("Set-Content .env 'SECRET=1'")
    assert not windows_is_read_only_command("Remove-Item build")


def test_windows_compound_commands_are_not_read_only():
    assert not windows_is_read_only_command("Get-ChildItem; Remove-Item x")
    assert not windows_is_read_only_command("Get-Content x | Set-Content y")


def test_windows_safe_queries_are_read_only():
    assert windows_is_read_only_command("Get-ChildItem src")
    assert windows_is_read_only_command("Get-Content README.md")
    assert windows_is_read_only_command("git status")
    assert not windows_is_read_only_command("npm install")


@pytest.mark.asyncio
async def test_hidden_project_config_is_discoverable(tmp_path: Path):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "workflows.yml").write_text("name: ci\n", encoding="utf-8")
    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".vscode" / "settings.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("internal\n", encoding="utf-8")

    glob = await Glob(tmp_path).invoke({"pattern": "**/*"})
    listing = await ListDir(tmp_path).invoke({"path": ".", "depth": 2})

    assert glob.ok
    assert ".github/workflows.yml" in glob.output.replace("\\", "/")
    assert ".vscode/settings.json" in glob.output.replace("\\", "/")
    assert ".git/config" not in glob.output.replace("\\", "/")
    assert ".github/" in listing.output.replace("\\", "/")
    assert ".vscode/" in listing.output.replace("\\", "/")
    assert ".git/" not in listing.output.replace("\\", "/")
