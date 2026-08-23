"""Undo for file changes."""

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
