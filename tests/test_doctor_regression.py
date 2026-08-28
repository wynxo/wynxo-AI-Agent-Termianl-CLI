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
