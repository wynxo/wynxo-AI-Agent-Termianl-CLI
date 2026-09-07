"""Regression tests for developer-data and large-project hardening."""

from __future__ import annotations

import hashlib
import os
import stat

import pytest

from wynxo.tools.files import MAX_READ_BYTES, ReadFile, WriteFile
from wynxo.tools.search import _project_files
from wynxo.tools import shell as shell_module


@pytest.mark.asyncio
async def test_write_file_refuses_to_replace_binary_data(tmp_path):
    path = tmp_path / "asset.bin"
    original = bytes(range(256)) * 40
    path.write_bytes(original)

    result = await WriteFile(tmp_path).invoke(
        {"path": "asset.bin", "content": "this must not replace the binary"}
    )

    assert result.ok is False
    assert "binary" in result.output.lower()
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_write_file_refuses_whole_replacement_of_large_file(tmp_path):
    path = tmp_path / "generated.txt"
    original = b"x" * (MAX_READ_BYTES + 1)
    path.write_bytes(original)

    result = await WriteFile(tmp_path).invoke(
        {"path": "generated.txt", "content": "tiny replacement\n"}
    )

    assert result.ok is False
    assert result.metadata.get("protected_large_file") is True
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_write_file_rejects_stale_read_hash(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n", encoding="utf-8")
    stale = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("value = 2\n", encoding="utf-8")

    result = await WriteFile(tmp_path).invoke(
        {"path": "module.py", "content": "value = 3\n", "expected_hash": stale}
    )

    assert result.ok is False
    assert result.metadata.get("stale") is True
    assert path.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_read_file_returns_hash_for_safe_follow_up_write(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n", encoding="utf-8")

    result = await ReadFile(tmp_path).invoke({"path": "module.py"})

    assert result.ok
    assert result.metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(os.name == "nt", reason="Unix executable mode bits")
@pytest.mark.asyncio
async def test_atomic_replacement_preserves_executable_bit(tmp_path):
    path = tmp_path / "run.sh"
    path.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)

    result = await WriteFile(tmp_path).invoke(
        {"path": "run.sh", "content": "#!/bin/sh\necho new\n"}
    )

    assert result.ok
    assert path.stat().st_mode & stat.S_IXUSR


def test_project_walk_is_bounded_and_prunes_noise(tmp_path):
    (tmp_path / "src").mkdir()
    for index in range(10):
        (tmp_path / "src" / f"f{index}.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "noise.py").write_text("noise\n", encoding="utf-8")

    found = list(_project_files(tmp_path, limit=3))

    assert len(found) == 3
    assert all("node_modules" not in path.parts for path in found)


def test_nested_posix_shell_refusal_survives_windows_quote_rules(monkeypatch):
    # shlex(posix=False) retains the quote pair around the -c payload. That
    # used to hide the destructive command from the recursive safety parser.
    monkeypatch.setattr(shell_module.os, "name", "nt")

    assert shell_module.hard_refusal("bash -c 'rm -rf /'")
    assert shell_module.hard_refusal("zsh -lc 'sh -c \"rm -rf /\"'")
