"""Undo for file changes."""

import pytest

from wynxo.checkpoints import Checkpoints


class TestUndo:
    def test_restores_previous_content(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("original\n")
        checkpoints = Checkpoints()
        checkpoints.capture(target, "edit_file")
        target.write_text("changed\n")

        done, message = checkpoints.undo()
        assert done and target.read_text() == "original\n"
        assert "edit_file" in message

    def test_deletes_a_file_that_did_not_exist(self, tmp_path):
        target = tmp_path / "new.py"
        checkpoints = Checkpoints()
        checkpoints.capture(target, "write_file")
        target.write_text("created\n")

        done, message = checkpoints.undo()
        assert done and not target.exists()
        assert "Deleted" in message

    def test_unwinds_in_reverse_order(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("v1\n")
        checkpoints = Checkpoints()
        checkpoints.capture(target, "edit_file"); target.write_text("v2\n")
        checkpoints.capture(target, "edit_file"); target.write_text("v3\n")

        checkpoints.undo()
        assert target.read_text() == "v2\n"
        checkpoints.undo()
        assert target.read_text() == "v1\n"

    def test_nothing_to_undo(self, tmp_path):
        done, message = Checkpoints().undo()
        assert not done and "Nothing to undo" in message

    def test_undoing_an_already_deleted_file_is_not_an_error(self, tmp_path):
        target = tmp_path / "gone.py"
        target.write_text("x\n")
        checkpoints = Checkpoints()
        checkpoints.capture(target, "write_file")
        target.unlink()
        done, _ = checkpoints.undo()
        assert done and target.read_text() == "x\n"

    def test_history_is_most_recent_first(self, tmp_path):
        checkpoints = Checkpoints()
        for name in ("a", "b", "c"):
            path = tmp_path / f"{name}.py"
            path.write_text("x")
            checkpoints.capture(path, "edit_file", label=f"{name}.py")
        assert [s.label for s in checkpoints.history()] == ["c.py", "b.py", "a.py"]

    def test_stack_is_bounded(self, tmp_path):
        from wynxo.checkpoints import MAX_SNAPSHOTS
        checkpoints = Checkpoints()
        target = tmp_path / "a.py"
        target.write_text("x")
        for _ in range(MAX_SNAPSHOTS + 40):
            checkpoints.capture(target, "edit_file")
        assert len(checkpoints) == MAX_SNAPSHOTS

    def test_very_large_files_are_skipped(self, tmp_path):
        from wynxo.checkpoints import MAX_FILE_BYTES
        target = tmp_path / "big.bin"
        target.write_text("x" * (MAX_FILE_BYTES + 10))
        checkpoints = Checkpoints()
        checkpoints.capture(target, "write_file")
        assert len(checkpoints) == 0, "must not hold a huge file in memory"

    def test_capture_of_an_unreadable_path_is_ignored(self, tmp_path):
        checkpoints = Checkpoints()
        checkpoints.capture(tmp_path / "no" / "such" / "dir" / "f.py", "write_file")
        assert len(checkpoints) == 1   # recorded as "did not exist"
        done, _ = checkpoints.undo()
        assert done


class TestTurnScopedChanges:
    """Review mode needs "what changed during this turn", which is not the
    same question undo answers."""

    def test_the_earliest_snapshot_per_file_wins(self, tmp_path):
        """Three edits to one file during a turn are one change from how it
        started, not three overlapping ones."""
        from wynxo.checkpoints import Checkpoints

        # newline="" because a snapshot is now what is on disk, byte for
        # byte. write_text() without it turns "\n" into "\r\n" on Windows,
        # and the assertion below would be about Python's newline
        # translation rather than about checkpoints.
        target = tmp_path / "a.py"
        target.write_text("one\n", newline="")
        points = Checkpoints()
        mark = points.mark()
        points.capture(target, "write_file")
        target.write_text("two\n", newline="")
        points.capture(target, "edit_file")
        target.write_text("three\n", newline="")

        changes = points.changes_since(mark)
        assert len(changes) == 1
        assert changes[0].content == "one\n"

    def test_several_files_all_appear(self, tmp_path):
        from wynxo.checkpoints import Checkpoints

        points = Checkpoints()
        mark = points.mark()
        for name in ("a.py", "b.py", "c.py"):
            path = tmp_path / name
            points.capture(path, "write_file")
            path.write_text("x\n")
        assert len(points.changes_since(mark)) == 3

    def test_changes_before_the_mark_are_not_included(self, tmp_path):
        """An earlier turn's edits are not this turn's business."""
        from wynxo.checkpoints import Checkpoints

        points = Checkpoints()
        old = tmp_path / "old.py"
        points.capture(old, "write_file"); old.write_text("x\n")

        mark = points.mark()
        new = tmp_path / "new.py"
        points.capture(new, "write_file"); new.write_text("y\n")

        changes = points.changes_since(mark)
        assert [c.path.name for c in changes] == ["new.py"]

    def test_reverting_a_turn_restores_the_starting_state(self, tmp_path):
        from wynxo.checkpoints import Checkpoints

        target = tmp_path / "a.py"
        target.write_text("original\n")
        points = Checkpoints()
        mark = points.mark()
        points.capture(target, "write_file"); target.write_text("changed\n")
        points.capture(target, "edit_file"); target.write_text("changed twice\n")

        reverted, problems = points.revert_since(mark)
        assert reverted == 2
        assert problems == []
        assert target.read_text() == "original\n"

    def test_reverting_removes_files_the_turn_created(self, tmp_path):
        from wynxo.checkpoints import Checkpoints

        created = tmp_path / "new.py"
        points = Checkpoints()
        mark = points.mark()
        points.capture(created, "write_file")     # did not exist yet
        created.write_text("x\n")

        points.revert_since(mark)
        assert not created.exists()

    def test_reverting_leaves_earlier_turns_alone(self, tmp_path):
        from wynxo.checkpoints import Checkpoints

        keep = tmp_path / "keep.py"
        keep.write_text("first\n")
        points = Checkpoints()
        points.capture(keep, "write_file"); keep.write_text("second\n")

        mark = points.mark()
        other = tmp_path / "other.py"
        points.capture(other, "write_file"); other.write_text("x\n")

        points.revert_since(mark)
        assert keep.read_text() == "second\n", "an earlier turn was rolled back"
        assert not other.exists()

    def test_reverting_nothing_is_not_an_error(self):
        from wynxo.checkpoints import Checkpoints

        points = Checkpoints()
        assert points.revert_since(points.mark()) == (0, [])


class TestUndoPutsBackExactlyWhatWasThere:
    """An undo that changes anything except what was edited is not an undo.

    The snapshot was read with the default newline handling, which
    translates CRLF to LF, and written back untranslated -- so undoing one
    edit to a CRLF file converted the whole file to LF. Inside a git repo
    that is every line showing as changed.
    """

    CASES = {
        "utf16.ps1": ("Write-Host 'hi'\n", "utf-16"),
        "cp1252.txt": ("café résumé\n", "cp1252"),
        "bom.py": ("# héllo\n", "utf-8-sig"),
        "crlf.txt": ("one\r\ntwo\r\n", "utf-8"),
        "mixed.txt": ("a\rb\r\nc\n", "utf-8"),
        "lf.py": ("x = 1\n", "utf-8"),
    }

    @pytest.mark.parametrize("name", list(CASES))
    def test_the_bytes_come_back(self, tmp_path, name):
        text, encoding = self.CASES[name]
        path = tmp_path / name
        path.write_text(text, encoding=encoding, newline="")
        before = path.read_bytes()

        checkpoints = Checkpoints()
        checkpoints.capture(path, "edit_file")
        path.write_bytes(b"CLOBBERED")
        did, _ = checkpoints.undo()

        assert did is True
        assert path.read_bytes() == before

    def test_even_a_file_that_is_not_text(self, tmp_path):
        """A tool may refuse to edit one, but a shell command can write
        anything, and undo has to put back what it found."""
        path = tmp_path / "blob.bin"
        path.write_bytes(bytes(range(256)) * 4)
        before = path.read_bytes()

        checkpoints = Checkpoints()
        checkpoints.capture(path, "shell")
        path.write_bytes(b"gone")
        checkpoints.undo()

        assert path.read_bytes() == before
