from pathlib import Path

from wynxo.navigation import affected_tests, symbols
from wynxo.task_state import TaskState, TaskStateMachine


def test_task_state_tracks_objective_and_progress(tmp_path: Path):
    state = TaskStateMachine()
    state.begin("Fix the Windows subprocess issue")
    assert state.state is TaskState.THINKING
    state.add_file("wynxo/shell.py")
    state.add_file("wynxo/shell.py", changed=True)
    assert state.objective.startswith("Fix")
    assert state.relevant_files == ["wynxo/shell.py"]
    assert state.changed_files == ["wynxo/shell.py"]
    # record_action returns the repeat count since the last progress event.
    assert state.record_action("search:subprocess") == 0
    assert state.record_action("search:subprocess") == 1
    assert state.record_action("search:subprocess") == 2
    assert "Repeated action" in state.blockers[-1]
    # A different action is fresh again.
    assert state.record_action("read:shell.py") == 0


def test_progress_resets_repeat_counts(tmp_path: Path):
    state = TaskStateMachine()
    state.begin("Fix the launcher")
    assert state.record_action("grep:spawn") == 0
    assert state.record_action("grep:spawn") == 1
    state.mark_progress()          # an edit landed in between
    assert state.record_action("grep:spawn") == 0


def test_recovery_block_is_built_from_recorded_state(tmp_path: Path):
    state = TaskStateMachine()
    state.begin("Fix the failing test")
    state.add_file("calc.py", changed=True)
    state.record_failure("tests failed: pytest")
    state.set_root_cause("off-by-one in add()")
    state.record_action("read:calc.py")
    state.record_action("read:calc.py")
    block = state.recovery_block()
    assert block.startswith("RECOVERY")
    assert "Fix the failing test" in block
    assert "Repeated action" in block
    assert "off-by-one" in block
    assert "changed files: calc.py" in block


def test_completion_report_only_for_coding_evidence(tmp_path: Path):
    state = TaskStateMachine()
    state.begin("hello")           # small talk: begin() records an objective
    assert state.completion_report() is None

    state = TaskStateMachine()
    state.begin("Fix the bug")
    state.add_file("calc.py", changed=True)
    state.record_verification("pytest")
    report = state.completion_report()
    assert report is not None
    assert "✓ completed" in report
    assert "calc.py" in report
    assert "pytest" in report

    state = TaskStateMachine()
    state.begin("Fix the bug")
    state.add_file("calc.py", changed=True)
    state.record_failure("tests failed: pytest")
    report = state.completion_report()
    assert "⚠ partially completed" in report
    assert "tests failed" in report

    # A failed grep mid-investigation is exploration, not a blocker: the
    # report still reads completed when the edit landed and tests passed.
    state = TaskStateMachine()
    state.begin("Fix the bug")
    state.add_file("calc.py", changed=True)
    state.record_failure("grep: no matches")
    state.record_verification("pytest")
    report = state.completion_report()
    assert "✓ completed" in report
    assert "partially" not in report
    assert "no matches" not in report


def test_python_symbol_navigation_and_affected_tests(tmp_path: Path):
    source = tmp_path / "agent.py"
    source.write_text("class Agent:\n    def run(self):\n        pass\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    test_file = tests / "test_agent.py"
    test_file.write_text("def test_run(): pass\n")
    found = symbols(source)
    assert [item["name"] for item in found] == ["Agent", "run"]
    assert affected_tests([source], tests) == [test_file]
