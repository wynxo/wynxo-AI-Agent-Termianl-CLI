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
        assert not result.ok and "outside the project directory" in result.output


class TestATimeoutTheToolAskedFor:
    """The shell accepts up to nine hundred seconds and invoke() capped
    every call at a hundred and twenty. So "run the test suite" on a suite
    that takes five minutes was killed at two, and what came back was
    "shell timed out after 120s" -- without the output that would have said
    which test it had reached."""

    async def test_the_tools_own_timeout_wins(self, tmp_path):
        tool = Shell(tmp_path)
        assert tool.timeout_for(tool.validate({"command": "x", "timeout": 600})) \
            > 120

    async def test_invoke_actually_asks_the_tool(self, tmp_path):
        """The bug was the call site, not the answer: invoke() knew how to
        ask and used a constant instead."""
        tool = Shell(tmp_path)
        asked = []
        answer = tool.timeout_for

        def spy(args):
            asked.append(answer(args))
            return asked[-1]

        tool.timeout_for = spy
        await tool.invoke({"command": "echo hi", "timeout": 600})
        assert asked and asked[0] > 120

    async def test_a_tool_with_nothing_to_say_keeps_the_default(self, tmp_path):
        tool = ReadFile(tmp_path)
        assert tool.timeout_for(tool.validate({"path": "x"})) == \
            tool.DEFAULT_TIMEOUT

    async def test_the_grace_lets_the_tool_report_first(self, tmp_path):
        """A tool that knows it timed out says what it saw; an outer cap
        firing at the same moment replaces that with one bare line."""
        tool = Shell(tmp_path)
        result = await tool.invoke(
            {"command": "echo starting; sleep 8", "timeout": 1})
        assert result.ok is False
        assert "Output before it was killed" in result.output
        assert "starting" in result.output

    async def test_an_explicit_timeout_still_overrides(self, tmp_path):
        result = await Shell(tmp_path).invoke(
            {"command": "sleep 5", "timeout": 30}, timeout=0.5)
        assert result.ok is False
        assert "timed out after 0s" in result.output

    async def test_a_true_in_the_field_is_not_a_timeout(self, tmp_path):
        """bool is an int in Python, and True would mean one second."""
        tool = ReadFile(tmp_path)
        args = tool.validate({"path": "x"})
        args.timeout = True
        assert tool.timeout_for(args) == tool.DEFAULT_TIMEOUT


class TestWhatTheModelIsToldWhenAPathIsRefused:
    """The boundary's message is written to be acted on -- it names the flag
    that would widen the scope. Announcing it as "read_file raised
    PermissionError" reads like wynxo broke and buries that sentence."""

    async def test_the_refusal_speaks_for_itself(self, tmp_path):
        work = tmp_path / "project"
        work.mkdir()
        (tmp_path / "secret.txt").write_text("no\n")
        result = await ReadFile(work).invoke({"path": "../secret.txt"})
        assert result.ok is False
        assert "raised" not in result.output
        assert "PermissionError" not in result.output
        assert "outside the project directory" in result.output
        assert "--scope" in result.output

    async def test_a_tool_that_genuinely_crashes_still_says_so(self, tmp_path):
        """The catch-all is still there for what it is for."""
        tool = ReadFile(tmp_path)

        async def boom(_args):
            raise ZeroDivisionError("nope")

        tool.run = boom
        result = await tool.invoke({"path": "x"})
        assert "raised ZeroDivisionError" in result.output


class TestAFileIsSavedTheWayItWasStored:
    """An edit changes what was asked for and nothing else.

    Reading was forgiving -- UTF-16, cp1252, a byte-order mark -- and
    writing was not: everything went back as plain UTF-8. So a one-line
    change rewrote every other byte in the file. A UTF-16 PowerShell script
    came back as UTF-8 with no BOM, a cp1252 file's accented characters were
    re-encoded end to end, and a BOM that mattered simply disappeared.
    """

    CASES = {
        "utf16.ps1": ("Write-Host 'hello'\nWrite-Host 'second'\n", "utf-16"),
        "cp1252.txt": ("café résumé\nsecond line\n", "cp1252"),
        "bom.py": ("# héllo\nsecond = 1\n", "utf-8-sig"),
        "plain.py": ("x = 1\nsecond = 2\n", "utf-8"),
        "crlf.txt": ("one\r\nsecond\r\n", "utf-8"),
    }

    def _write(self, tmp_path, name):
        text, encoding = self.CASES[name]
        (tmp_path / name).write_text(text, encoding=encoding, newline="")
        return text, encoding

    @pytest.mark.parametrize("name", list(CASES))
    async def test_only_the_edited_span_changes(self, tmp_path, name):
        text, encoding = self._write(tmp_path, name)
        result = await EditFile(tmp_path).invoke(
            {"path": name, "old_text": "second", "new_text": "SECOND"})
        assert result.ok, result.output
        assert (tmp_path / name).read_bytes() == \
            text.replace("second", "SECOND", 1).encode(encoding)

    @pytest.mark.parametrize("name", list(CASES))
    async def test_reading_it_back_works_too(self, tmp_path, name):
        """UTF-16 is half NUL bytes, so the binary sniff called every
        PowerShell script binary and refused to open it."""
        self._write(tmp_path, name)
        result = await ReadFile(tmp_path).invoke({"path": name})
        assert result.ok, result.output
        assert "second" in result.output

    async def test_replacing_the_whole_file_keeps_its_encoding(self, tmp_path):
        self._write(tmp_path, "utf16.ps1")
        await WriteFile(tmp_path).invoke(
            {"path": "utf16.ps1", "content": "Write-Host 'new'\n"})
        assert (tmp_path / "utf16.ps1").read_bytes() == \
            "Write-Host 'new'\n".encode("utf-16")

    async def test_a_new_file_is_plain_utf8(self, tmp_path):
        await WriteFile(tmp_path).invoke({"path": "new.py", "content": "x = 1\n"})
        assert (tmp_path / "new.py").read_bytes() == b"x = 1\n"

    async def test_an_encoding_that_cannot_hold_the_new_text_says_so(
            self, tmp_path):
        """Better than a file full of question marks nobody was warned about."""
        (tmp_path / "old.txt").write_text("café\nsecond\n", encoding="cp1252",
                                          newline="")
        result = await EditFile(tmp_path).invoke(
            {"path": "old.txt", "old_text": "second", "new_text": "→ arrow"})
        assert result.ok
        assert "UTF-8" in result.output
        assert (tmp_path / "old.txt").read_bytes().decode("utf-8") == \
            "café\n→ arrow\n"

    async def test_a_real_binary_file_is_still_refused(self, tmp_path):
        (tmp_path / "blob.bin").write_bytes(bytes(range(256)) * 40)
        for tool, args in (
            (EditFile, {"path": "blob.bin", "old_text": "a", "new_text": "b"}),
            (ReadFile, {"path": "blob.bin"}),
        ):
            result = await tool(tmp_path).invoke(args)
            assert result.ok is False
            assert "binary" in result.output


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


class TestALinkIsNotADirectory:
    """Two directories pointing at each other made the listing a tree of
    itself repeating -- a/to_b/to_a/to_b for as many levels as were asked
    for -- and a link back to the project root duplicated the whole listing
    under a name nothing actually lives at."""

    def _tangle(self, tmp_path):
        import os

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1\n")
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        os.symlink(tmp_path / "b", tmp_path / "a" / "to_b",
                   target_is_directory=True)
        os.symlink(tmp_path / "a", tmp_path / "b" / "to_a",
                   target_is_directory=True)
        os.symlink(tmp_path, tmp_path / "whole", target_is_directory=True)
        return tmp_path

    async def test_the_loop_is_not_walked(self, tmp_path):
        pytest.skip("Symlinks require admin on Windows", allow_module_level=False) if sys.platform == "win32" else None
        self._tangle(tmp_path)
        result = await ListDir(tmp_path).invoke({"path": ".", "depth": 5})
        assert result.ok
        assert "to_b" in result.output
        assert "to_a/" not in result.output.replace("to_a/ ->", "")

    async def test_a_link_says_where_it_goes(self, tmp_path):
        pytest.skip("Symlinks require admin on Windows", allow_module_level=False) if sys.platform == "win32" else None
        self._tangle(tmp_path)
        result = await ListDir(tmp_path).invoke({"path": ".", "depth": 5})
        assert "->" in result.output

    async def test_the_real_tree_is_still_listed(self, tmp_path):
        pytest.skip("Symlinks require admin on Windows", allow_module_level=False) if sys.platform == "win32" else None
        self._tangle(tmp_path)
        result = await ListDir(tmp_path).invoke({"path": ".", "depth": 5})
        assert "app.py" in result.output
        # ... and only once, rather than again under the link to the root.
        assert result.output.count("app.py") == 1


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


class TestWhatIsRefusedOutright:
    """A handful of commands are refused rather than prompted for, because a
    yes/no question is the thing a person clicks through on autopilot.

    They were matched as substrings of the whole line, which refused far too
    much: "rm -rf /tmp/build" starts with "rm -rf /", and a commit message
    with the word "shutdown" in it contains "shutdown". All refused
    outright, with nothing the model could do about it.
    """

    @pytest.mark.parametrize("command", [
        "rm -rf /tmp/build",
        "rm -rf ./build",
        "rm -rf build",
        "rm -rf node_modules",
        "git commit -m 'handle shutdown cleanly'",
        "grep -rn reboot src",
        "echo 'format c: was a joke'",
        "find . -name '*.pyc' -delete",
        "time npm test",
        "env FOO=1 pytest",
        # A shell running an inline script is ordinary work far more often
        # than not; unwrapping -c must not make every one of these a refusal.
        "bash -c 'pytest -q'",
        "sh -c 'echo hi; ls'",
        "bash -lc 'make -j8'",
        "bash script.sh",
        "echo 'rm -rf /' > notes.txt",
    ])
    async def test_ordinary_work_is_not_refused(self, tmp_path, command):
        from wynxo.tools.shell import hard_refusal

        assert hard_refusal(command) == "", command

    @pytest.mark.parametrize("command", [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf ~/",
        "rm -rf /usr",
        "sudo rm -rf /",
        "env rm -rf /",
        "nohup rm -rf ~",
        "shutdown -h now",
        "sudo shutdown now",
        "reboot",
        "halt",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "echo hi > /dev/sda",
        ":(){:|:&};:",
        # A separator does not launder it.
        "ls && rm -rf /",
        "ls\nrm -rf /",
        "format c:",
        "rd /s /q c:\\",
        # A shell does not launder it either. `sh -c "rm -rf /"` is one
        # token away from `sudo rm -rf /`, and used to sail straight past.
        "bash -c 'rm -rf /'",
        'sh -c "rm -rf /"',
        "/bin/sh -c 'rm -rf ~'",
        # Short options cluster, so -c is a letter in the middle of the flag.
        "bash -lc 'rm -rf /'",
        "bash -i -c 'rm -rf /'",
        # The separator lives inside the quotes, so splitting on it first
        # tore the dangerous half into an unparseable fragment.
        "sh -c 'echo building; rm -rf /'",
        # And a nested shell is still a shell.
        'sh -c \'sh -c "rm -rf /"\'',
        "zsh -c 'shutdown now'",
    ])
    async def test_these_never_run(self, command):
        from wynxo.tools.shell import hard_refusal

        assert hard_refusal(command), command

    async def test_the_refusal_reaches_the_model(self, tmp_path):
        result = await Shell(tmp_path).invoke({"command": "rm -rf /"})
        assert result.ok is False
        assert "Refusing to run" in result.output
        assert "run it yourself" in result.output


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


class TestAnEmptyOldText:
    """Empty matches between every character, so the count that would
    otherwise come back ("appears 7 times") describes nothing and tells the
    model nothing about what to do differently."""

    def test_edit_file_says_it_is_empty(self, tmp_path):
        import asyncio

        from wynxo.tools.files import EditFile, EditInput

        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        result = asyncio.run(EditFile(workspace=tmp_path).run(
            EditInput(path="a.py", old_text="", new_text="X")))
        assert result.ok is False
        assert "empty" in result.output and "times" not in result.output

    def test_it_points_at_write_file_instead(self, tmp_path):
        import asyncio

        from wynxo.tools.files import EditFile, EditInput

        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        result = asyncio.run(EditFile(workspace=tmp_path).run(
            EditInput(path="a.py", old_text="", new_text="X")))
        assert "write_file" in result.output

    def test_multi_edit_names_which_edit(self, tmp_path):
        import asyncio

        from wynxo.tools.files import MultiEdit, MultiEditInput

        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        result = asyncio.run(MultiEdit(workspace=tmp_path).run(
            MultiEditInput(path="a.py", edits=[
                {"old_text": "a = 1", "new_text": "a = 2"},
                {"old_text": "", "new_text": "X"}])))
        assert result.ok is False and "edit 2" in result.output

    def test_the_file_is_untouched(self, tmp_path):
        """A refused batch must not half-apply."""
        import asyncio

        from wynxo.tools.files import MultiEdit, MultiEditInput

        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        asyncio.run(MultiEdit(workspace=tmp_path).run(
            MultiEditInput(path="a.py", edits=[
                {"old_text": "a = 1", "new_text": "a = 2"},
                {"old_text": "", "new_text": "X"}])))
        assert (tmp_path / "a.py").read_text() == "a = 1\n"


class TestShellStreamCleanup:
    """Cancelled and timed-out commands must retire their pipe transports.

    On Windows the ProactorEventLoop hands a subprocess a pipe transport for
    stdout; when the loop closes while that transport is still alive, its
    deallocator later runs against an already-closed socket and the
    interpreter prints "Exception ignored while calling deallocator
    (asyncio: I/O operation on closed pipe)". Explicitly closing the stream
    is what makes the teardown clean.
    """

    async def test_a_live_stream_is_closed(self):
        from unittest.mock import MagicMock

        stream = MagicMock()
        process = MagicMock()
        process.stdout = stream
        await Shell._close_streams(process)
        stream.close.assert_called_once()

    async def test_no_stream_is_not_an_error(self):
        from unittest.mock import MagicMock

        process = MagicMock()
        process.stdout = None
        await Shell._close_streams(process)      # must not raise

    async def test_an_already_closed_stream_is_tolerated(self):
        from unittest.mock import MagicMock

        stream = MagicMock()
        stream.close.side_effect = ValueError("I/O operation on closed pipe")
        process = MagicMock()
        process.stdout = stream
        await Shell._close_streams(process)      # must not raise
