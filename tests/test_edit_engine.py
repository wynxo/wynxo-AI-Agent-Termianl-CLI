from pathlib import Path

from wynxo.tools.files import EditFile, ReadFile, WriteFile


async def test_range_edit_returns_diff_metadata(tmp_path: Path):
    path = tmp_path / "sample.py"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = EditFile(tmp_path)
    result = await tool.invoke({"path": "sample.py", "start_line": 2, "end_line": 2, "new_text": "TWO\n"})
    assert result.ok
    assert result.metadata["changed"] is True
    assert result.metadata["additions"] == 1
    assert result.metadata["deletions"] == 1
    assert "-two" in result.display and "+TWO" in result.display
    assert path.read_text(encoding="utf-8") == "one\nTWO\nthree\n"


async def test_write_file_reports_creation_diff_and_hash(tmp_path: Path):
    result = await WriteFile(tmp_path).invoke({"path": "new.py", "content": "print('ok')\n"})
    assert result.ok
    assert result.metadata["created"] is True
    assert result.metadata["additions"] == 1
    assert result.metadata["sha256"]


async def test_range_edit_rejects_mixed_exact_arguments(tmp_path: Path):
    (tmp_path / "x.txt").write_text("a\nb\n", encoding="utf-8")
    result = await EditFile(tmp_path).invoke({
        "path": "x.txt", "start_line": 1, "end_line": 1,
        "old_text": "a", "new_text": "z",
    })
    assert not result.ok
    assert "either exact" in result.error
