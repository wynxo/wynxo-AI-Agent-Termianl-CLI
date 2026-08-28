    same question undo answers."""

    def test_the_earliest_snapshot_per_file_wins(self, tmp_path):
        """Three edits to one file during a turn are one change from how it
        started, not three overlapping ones."""
        from wynxo.checkpoints import Checkpoints

        # Snapshots preserve the bytes that are actually on disk so undo can
        # restore CRLF, BOMs and non-UTF-8 content exactly.
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
        assert changes[0].content == b"one\n"

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