from pathlib import Path

import pytest

from wynxo.tools.files import ReadFile
from wynxo.tools.search import Grep


@pytest.mark.asyncio
async def test_read_file_supports_one_based_range_and_truncation(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\nthree\nfour\n")
    result = await ReadFile(tmp_path).invoke({"path": "sample.txt", "start_line": 2, "end_line": 3})
    assert result.ok
    assert "two" in result.output and "three" in result.output
    assert "one" not in result.output and "four" not in result.output
    assert result.metadata["truncated"] is True


@pytest.mark.asyncio
async def test_read_file_max_bytes_is_reported(tmp_path: Path):
    (tmp_path / "sample.txt").write_text("abcdefghij\n")
    result = await ReadFile(tmp_path).invoke({"path": "sample.txt", "max_bytes": 5})
    assert result.ok
    assert result.metadata["truncated"] is True
    assert "output truncated" in result.output


@pytest.mark.asyncio
async def test_search_supports_literal_matching(tmp_path: Path):
    (tmp_path / "sample.py").write_text("a+b\naXXb\n")
    result = await Grep(tmp_path).invoke({"path": ".", "pattern": "a+b", "literal": True})
    assert result.ok
    assert result.metadata["matches"] == 1
    assert "sample.py:1" in result.output
