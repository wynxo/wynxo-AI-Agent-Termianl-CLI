from __future__ import annotations

from pathlib import Path

import pytest

from wynxo.tools.files import EditFile, ListDir, WriteFile


@pytest.mark.asyncio
async def test_write_file_refuses_binary_overwrite(tmp_path: Path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"\x00\x01\x02\x00binary")

    result = await WriteFile(tmp_path).invoke({"path": "data.bin", "content": "oops"})

    assert not result.ok
    assert target.read_bytes() == b"\x00\x01\x02\x00binary"


@pytest.mark.asyncio
async def test_write_file_refuses_secret_target(tmp_path: Path):
    target = tmp_path / ".env"
    target.write_text("SECRET=old\n", encoding="utf-8")

    result = await WriteFile(tmp_path).invoke({"path": ".env", "content": "SECRET=new\n"})

    assert not result.ok
    assert target.read_text(encoding="utf-8") == "SECRET=old\n"


@pytest.mark.asyncio
async def test_edit_file_refuses_secret_target(tmp_path: Path):
    target = tmp_path / "id_rsa"
    target.write_text("private-key\n", encoding="utf-8")

    result = await EditFile(tmp_path).invoke({
        "path": "id_rsa", "old_text": "private-key", "new_text": "changed"
    })

    assert not result.ok
    assert target.read_text(encoding="utf-8") == "private-key\n"


@pytest.mark.asyncio
async def test_list_dir_shows_project_dot_directories_but_not_git(tmp_path: Path):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "workflows.yml").write_text("name: ci\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("internal\n", encoding="utf-8")

    result = await ListDir(tmp_path).invoke({"path": ".", "depth": 2})

    assert result.ok
    normalized = result.output.replace("\\", "/")
    assert ".github/" in normalized
    assert ".github/workflows.yml" in normalized
    assert ".git/" not in normalized


@pytest.mark.asyncio
async def test_write_file_preserves_existing_mode(tmp_path: Path):
    target = tmp_path / "script.sh"
    target.write_text("echo old\n", encoding="utf-8")
    target.chmod(0o750)

    result = await WriteFile(tmp_path).invoke({"path": "script.sh", "content": "echo new\n"})

    assert result.ok
    assert target.read_text(encoding="utf-8") == "echo new\n"
    assert target.stat().st_mode & 0o777 == 0o750
