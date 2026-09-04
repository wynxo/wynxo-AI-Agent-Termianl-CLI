"""Reverting one file from the review flow.

The end-of-turn review offers "step through", which reverts files one at a
time. That path had no drift check: it wrote the snapshot straight back, so
answering "revert" to a file the user had saved in their editor since the
agent touched it destroyed that save and reported success. ``/undo`` had the
check; these two paths were the same operation written twice, and only one of
them was safe.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from wynxo import cli
from wynxo.checkpoints import Checkpoints


class _UI:
    def __init__(self):
        self.warnings: list[str] = []
        self.successes: list[str] = []

    def shorten_path(self, path):
        return path

    def warn(self, message):
        self.warnings.append(message)

    def success(self, message):
        self.successes.append(message)


def _repl(checkpoints, answers):
    """A Repl stub answering the step-through questions from ``answers``."""
    ui = _UI()
    replies = list(answers)

    async def question(_prompt, _answers, default=""):
        return replies.pop(0) if replies else default

    return SimpleNamespace(ui=ui, _question=question,
                           agent=SimpleNamespace(checkpoints=checkpoints))


def _step(repl, checkpoints, mark):
    asyncio.run(cli.Repl._step_through(
        repl, mark, checkpoints.changes_since(mark)))


class TestSteppingThroughAReview:
    def test_reverting_one_file_puts_it_back(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("original\n")
        checkpoints = Checkpoints()
        mark = checkpoints.mark()
        checkpoints.capture(target, "edit_file")
        target.write_text("the agent's edit\n")
        checkpoints.mark_expected(target)

        repl = _repl(checkpoints, ["r"])
        _step(repl, checkpoints, mark)

        assert target.read_text() == "original\n"
        assert repl.ui.successes == ["kept 0, reverted 1"]

    def test_it_refuses_over_the_users_own_save(self, tmp_path):
        """The whole reason the check exists. Between the agent's edit and
        the answer to this question, the user saved the file in their
        editor; putting the old copy back would destroy that."""
        target = tmp_path / "a.py"
        target.write_text("original\n")
        checkpoints = Checkpoints()
        mark = checkpoints.mark()
        checkpoints.capture(target, "edit_file")
        target.write_text("the agent's edit\n")
        checkpoints.mark_expected(target)

        target.write_text("MY OWN WORK\n")          # the user, in their editor

        repl = _repl(checkpoints, ["r"])
        _step(repl, checkpoints, mark)

        assert target.read_text() == "MY OWN WORK\n"
        assert repl.ui.warnings, "a refusal has to be said out loud"
        assert "changed after it was edited" in repl.ui.warnings[0]
        # And it must not be counted as done, either.
        assert repl.ui.successes == ["kept 0, reverted 0"]

    def test_keeping_a_file_leaves_it_alone(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("original\n")
        checkpoints = Checkpoints()
        mark = checkpoints.mark()
        checkpoints.capture(target, "edit_file")
        target.write_text("the agent's edit\n")
        checkpoints.mark_expected(target)

        repl = _repl(checkpoints, ["k"])
        _step(repl, checkpoints, mark)

        assert target.read_text() == "the agent's edit\n"
        assert repl.ui.successes == ["kept 1, reverted 0"]

    def test_one_blocked_file_does_not_stop_the_others(self, tmp_path):
        """Step-through is per file. A file the user has since saved is
        skipped with a reason; the rest of the answers still apply."""
        safe = tmp_path / "safe.py"
        drifted = tmp_path / "drifted.py"
        for path in (safe, drifted):
            path.write_text("original\n")

        checkpoints = Checkpoints()
        mark = checkpoints.mark()
        for path in (safe, drifted):
            checkpoints.capture(path, "edit_file")
            path.write_text("the agent's edit\n")
            checkpoints.mark_expected(path)
        drifted.write_text("MY OWN WORK\n")

        repl = _repl(checkpoints, ["r", "r"])
        _step(repl, checkpoints, mark)

        assert safe.read_text() == "original\n"
        assert drifted.read_text() == "MY OWN WORK\n"
        assert repl.ui.successes == ["kept 0, reverted 1"]

    def test_a_created_file_is_deleted(self, tmp_path):
        target = tmp_path / "new.py"
        checkpoints = Checkpoints()
        mark = checkpoints.mark()
        checkpoints.capture(target, "write_file")
        target.write_text("created\n")
        checkpoints.mark_expected(target)

        repl = _repl(checkpoints, ["r"])
        _step(repl, checkpoints, mark)

        assert not target.exists()

    def test_a_created_file_the_user_then_wrote_to_survives(self, tmp_path):
        """Deleting is the destructive direction, so it gets the same check
        as overwriting does."""
        target = tmp_path / "new.py"
        checkpoints = Checkpoints()
        mark = checkpoints.mark()
        checkpoints.capture(target, "write_file")
        target.write_text("created\n")
        checkpoints.mark_expected(target)

        target.write_text("MY OWN WORK\n")

        repl = _repl(checkpoints, ["r"])
        _step(repl, checkpoints, mark)

        assert target.read_text() == "MY OWN WORK\n"
        assert repl.ui.warnings


class TestTheTwoPathsAgree:
    """``/undo`` and the review flow are the same operation. They are only
    safe together if the file handling has one implementation."""

    def _prepared(self, tmp_path, name):
        target = tmp_path / name
        target.write_text("original\n")
        checkpoints = Checkpoints()
        checkpoints.capture(target, "edit_file")
        target.write_text("the agent's edit\n")
        checkpoints.mark_expected(target)
        return target, checkpoints

    def test_both_refuse_a_drifted_file(self, tmp_path):
        for name in ("undo.py", "step.py"):
            target, checkpoints = self._prepared(tmp_path, name)
            target.write_text("MY OWN WORK\n")

            if name == "undo.py":
                ok, message = checkpoints.undo()
            else:
                ok, message = checkpoints.restore(checkpoints.peek())

            assert not ok, name
            assert "changed after it was edited" in message, name
            assert target.read_text() == "MY OWN WORK\n", name

    def test_a_refusal_keeps_the_snapshot(self, tmp_path):
        """Refusing is not the same as having undone it. The record has to
        outlive the refusal or the undo is gone once the conflict is."""
        target, checkpoints = self._prepared(tmp_path, "a.py")
        target.write_text("MY OWN WORK\n")

        assert checkpoints.undo()[0] is False
        assert len(checkpoints) == 1

        target.write_text("the agent's edit\n")     # conflict resolved
        assert checkpoints.undo()[0] is True
        assert target.read_text() == "original\n"

    def test_restore_does_not_touch_the_stack(self, tmp_path):
        """Step-through reverts out of order, so it cannot pop: taking one
        snapshot out of the middle would put the later ones back too."""
        target, checkpoints = self._prepared(tmp_path, "a.py")
        before = len(checkpoints)

        assert checkpoints.restore(checkpoints.peek())[0] is True
        assert len(checkpoints) == before
