from wynxo.task_state import TaskStateMachine


def test_task_state_records_distinct_evidence():
    state = TaskStateMachine()
    state.begin("repair tests")
    state.record_success("read_file: loaded source")
    state.record_failure("pytest: assertion failed")
    state.set_root_cause("incorrect boundary condition")
    assert state.successes == ["read_file: loaded source"]
    assert state.failures == ["pytest: assertion failed"]
    assert state.root_cause == "incorrect boundary condition"


def test_task_state_deduplicates_evidence():
    state = TaskStateMachine()
    state.record_failure("same failure")
    state.record_failure("same failure")
    state.record_success("same success")
    state.record_success("same success")
    assert state.failures == ["same failure"]
    assert state.successes == ["same success"]


class TestTheReadmeListsEveryCommand:
    """The README claimed the full-screen layout was gone while it was still
    what the app started by default. Documentation that has drifted from the
    code is not a smaller problem than a bug -- it is the same problem, read
    by somebody who then trusts it.

    A command nobody can find is a command nobody uses, so the listing has
    to name all of them.
    """

    def _readme(self) -> str:
        import pathlib

        import wynxo

        root = pathlib.Path(wynxo.__file__).resolve().parent.parent
        return (root / "README.md").read_text(encoding="utf-8")

    def test_every_command_is_mentioned(self):
        from wynxo.cli import COMMANDS

        readme = self._readme()
        missing = sorted(name for name in COMMANDS if name not in readme)
        assert not missing, f"undocumented: {missing}"

    def test_the_readme_does_not_describe_a_ui_that_is_gone(self):
        """F2, mouse capture and the alternate screen are all removed. A
        troubleshooting entry may say they *used* to exist -- that is what
        somebody arriving with the old behaviour in mind will search for --
        but nothing may present them as current."""
        import re

        readme = self._readme()
        for line in readme.splitlines():
            if re.search(r"\bF2\b", line):
                assert "used to" in line or "gone" in line, line
