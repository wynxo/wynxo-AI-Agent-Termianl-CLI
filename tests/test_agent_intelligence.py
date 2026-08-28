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
    assert state.record_action("search:subprocess")
    assert not state.record_action("search:subprocess")


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
