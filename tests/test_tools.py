import sys

import pytest

from wynxo.tools import build_registry
from wynxo.tools.files import EditFile, ListDir, ReadFile, WriteFile
from wynxo.tools.search import Glob, Grep
from wynxo.tools.shell import Shell
from wynxo.tools.todo import TodoWrite


class TestReadFile:
    async def test_reads_with_line_numbers(self, tmp_path):
        (tmp_path / "a.py").write_text("one\ntwo\n")
        result = await ReadFile(tmp_path).invoke({"path": "a.py"})
        assert result.ok
        assert "1\tone" in result.output and "2\ttwo" in result.output

    async def test_missing_file_suggests_a_near_match(self, tmp_path):
        (tmp_path / "config.py").write_text("x\n")
        result = await ReadFile(tmp_path).invoke({"path": "confg.py"})
        assert not result.ok
        assert "config.py" in result.output

    async def test_binary_is_refused(self, tmp_path):
        (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
        result = await ReadFile(tmp_path).invoke({"path": "b.bin"})
        assert not result.ok and "binary" in result.output

    async def test_offset_and_limit(self, tmp_path):
        (tmp_path / "n.txt").write_text("\n".join(str(i) for i in range(100)))
        result = await ReadFile(tmp_path).invoke({"path": "n.txt", "offset": 50, "limit": 5})
        assert "51\t50" in result.output
        assert "of 100" in result.output

    async def test_cp1252_file_still_reads(self, tmp_path):
        # A file saved by a Windows editor must not read as "does not exist".
        (tmp_path / "w.txt").write_bytes("caf\xe9\n".encode("cp1252"))
        result = await ReadFile(tmp_path).invoke({"path": "w.txt"})
        assert result.ok and "caf" in result.output

    async def test_invalid_arguments_explain_the_schema(self, tmp_path):
        result = await ReadFile(tmp_path).invoke({"wrong": "key"})
        assert not result.ok
        assert "read_file(path" in result.output


class TestWriteAndEdit:
    async def test_write_creates_parents(self, tmp_path):
        result = await WriteFile(tmp_path).invoke(
            {"path": "deep/nested/f.py", "content": "x=1\n"})
        assert result.ok
        assert (tmp_path / "deep/nested/f.py").read_text() == "x=1\n"

    async def test_write_preserves_exact_bytes(self, tmp_path):
        # newline="" -- a CRLF rewrite would corrupt files inside a git repo.
        await WriteFile(tmp_path).invoke({"path": "f.txt", "content": "a\nb\n"})
        assert (tmp_path / "f.txt").read_bytes() == b"a\nb\n"

    async def test_edit_replaces_once(self, tmp_path):
        (tmp_path / "e.py").write_text("a = 1\n")
        result = await EditFile(tmp_path).invoke(
            {"path": "e.py", "old_text": "a = 1", "new_text": "a = 2"})
        assert result.ok
        assert (tmp_path / "e.py").read_text() == "a = 2\n"

    async def test_ambiguous_edit_is_refused(self, tmp_path):
        (tmp_path / "e.py").write_text("x\nx\n")
        result = await EditFile(tmp_path).invoke(
            {"path": "e.py", "old_text": "x", "new_text": "y"})
        assert not result.ok and "2 times" in result.output
        assert (tmp_path / "e.py").read_text() == "x\nx\n"

    async def test_replace_all_is_allowed_explicitly(self, tmp_path):
        (tmp_path / "e.py").write_text("x\nx\n")
        result = await EditFile(tmp_path).invoke(
            {"path": "e.py", "old_text": "x", "new_text": "y", "replace_all": True})
        assert result.ok
        assert (tmp_path / "e.py").read_text() == "y\ny\n"

    async def test_whitespace_mismatch_says_so(self, tmp_path):
        # Tab-indented file, space-indented old_text: the single most common
        # reason an edit fails, so the error has to name the cause.
        (tmp_path / "e.py").write_text("\tindented = 1\n")
        result = await EditFile(tmp_path).invoke(
            {"path": "e.py", "old_text": "    indented = 1", "new_text": "z"})
        assert not result.ok
        assert "whitespace" in result.output.lower()
        assert (tmp_path / "e.py").read_text() == "\tindented = 1\n"

    async def test_edit_produces_a_diff(self, tmp_path):
        (tmp_path / "e.py").write_text("a\n")
        result = await EditFile(tmp_path).invoke(
            {"path": "e.py", "old_text": "a", "new_text": "b"})
        assert "-a" in result.display and "+b" in result.display

    async def test_escape_outside_workspace_is_blocked(self, tmp_path):
        result = await WriteFile(tmp_path).invoke(
            {"path": "../evil.txt", "content": "x"})
        assert not result.ok and "outside the workspace" in result.output


class TestSearch:
    async def test_glob_finds_by_pattern(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src/a.py").write_text("")
        (tmp_path / "src/b.js").write_text("")
        result = await Glob(tmp_path).invoke({"pattern": "*.py"})
        assert "a.py" in result.output and "b.js" not in result.output

    async def test_glob_skips_ignored_directories(self, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules/x.py").write_text("")
        (tmp_path / "real.py").write_text("")
        result = await Glob(tmp_path).invoke({"pattern": "*.py"})
        assert "real.py" in result.output and "node_modules" not in result.output

    async def test_grep_reports_file_and_line(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\ndef go():\n    pass\n")
        result = await Grep(tmp_path).invoke({"pattern": r"def \w+"})
        assert "a.py:2:" in result.output

    async def test_grep_glob_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.txt").write_text("target\n")
        result = await Grep(tmp_path).invoke({"pattern": "target", "glob": "*.py"})
        assert "a.py" in result.output and "b.txt" not in result.output

    async def test_invalid_regex_is_explained(self, tmp_path):
        result = await Grep(tmp_path).invoke({"pattern": "([unclosed"})
        assert not result.ok and "Invalid regex" in result.output

    async def test_no_match_is_success_not_failure(self, tmp_path):
        # A model must not treat "not found" as an error to retry.
        (tmp_path / "a.py").write_text("nothing\n")
        result = await Grep(tmp_path).invoke({"pattern": "zzz"})
        assert result.ok and "No matches" in result.output


class TestListDir:
    async def test_renders_a_tree(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src/a.py").write_text("")
        result = await ListDir(tmp_path).invoke({"path": "."})
        assert "src/" in result.output and "a.py" in result.output

    async def test_hides_vcs_and_build_noise(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "keep.py").write_text("")
        result = await ListDir(tmp_path).invoke({"path": "."})
        assert "keep.py" in result.output
        assert ".git" not in result.output and "__pycache__" not in result.output


class TestShell:
    async def test_runs_and_captures_output(self, tmp_path):
        command = "echo hello" if sys.platform != "win32" else "Write-Output hello"
        result = await Shell(tmp_path).invoke({"command": command})
        assert result.ok and "hello" in result.output

    async def test_nonzero_exit_is_a_failure_with_the_output(self, tmp_path):
        command = "echo oops; exit 3" if sys.platform != "win32" else "Write-Output oops; exit 3"
        result = await Shell(tmp_path).invoke({"command": command})
        assert not result.ok
        assert "exit code 3" in result.output and "oops" in result.output

    async def test_destructive_command_is_refused(self, tmp_path):
        result = await Shell(tmp_path).invoke({"command": "rm -rf /"})
        assert not result.ok and "Refusing" in result.output

    @pytest.mark.skipif(sys.platform == "win32", reason="posix timeout semantics")
    async def test_timeout_kills_the_process(self, tmp_path):
        result = await Shell(tmp_path).invoke({"command": "sleep 5", "timeout": 1})
        assert not result.ok and "timed out" in result.output

    async def test_runs_in_the_workspace(self, tmp_path):
        (tmp_path / "marker.txt").write_text("")
        command = "ls" if sys.platform != "win32" else "Get-ChildItem -Name"
        result = await Shell(tmp_path).invoke({"command": command})
        assert "marker.txt" in result.output


class TestTodo:
    async def test_renders_and_counts(self, tmp_path):
        tool = TodoWrite(tmp_path)
        result = await tool.invoke({"items": [
            {"task": "read", "status": "done"},
            {"task": "patch", "status": "in_progress"},
            {"task": "test", "status": "pending"},
        ]})
        assert result.ok and "1/3 done" in result.output
        assert "[x] read" in result.display and "[>] patch" in result.display
        assert tool.outstanding() == ["patch", "test"]

    async def test_warns_on_two_in_progress(self, tmp_path):
        result = await TodoWrite(tmp_path).invoke({"items": [
            {"task": "a", "status": "in_progress"},
            {"task": "b", "status": "in_progress"},
        ]})
        assert "one at a time" in result.output

    async def test_rejects_a_bad_status(self, tmp_path):
        result = await TodoWrite(tmp_path).invoke({"items": [{"task": "a", "status": "nope"}]})
        assert not result.ok


class TestRegistry:
    def test_schemas_are_well_formed(self, tmp_path):
        for schema in build_registry(tmp_path).ollama_schemas():
            function = schema["function"]
            assert function["name"] and function["description"]
            assert schema["type"] == "function"
            assert "properties" in function["parameters"]

    def test_shell_can_be_disabled(self, tmp_path):
        assert "shell" not in build_registry(tmp_path, allow_shell=False).names()
        assert "shell" in build_registry(tmp_path, allow_shell=True).names()

    def test_mutating_flags_are_right(self, tmp_path):
        registry = build_registry(tmp_path)
        assert registry.get("write_file").mutating
        assert registry.get("edit_file").mutating
        assert registry.get("shell").mutating
        assert not registry.get("read_file").mutating
        assert not registry.get("grep").mutating
