from wynxo.task_state import TaskState, TaskStateMachine


def test_task_state_machine_accepts_normal_workflow():
    machine = TaskStateMachine()
    assert machine.transition(TaskState.THINKING)
    assert machine.transition(TaskState.PLANNING)
    assert machine.transition(TaskState.EXECUTING)
    assert machine.transition(TaskState.TESTING)
    assert machine.transition(TaskState.COMPLETED)
    assert machine.state is TaskState.COMPLETED


def test_task_state_machine_rejects_invalid_transition_without_mutation():
    machine = TaskStateMachine()
    assert not machine.transition(TaskState.COMPLETED)
    assert machine.state is TaskState.IDLE
    assert machine.transition(TaskState.THINKING)
    assert not machine.transition(TaskState.IDLE)
    assert machine.state is TaskState.THINKING


def test_task_state_machine_can_reset_after_cancellation():
    machine = TaskStateMachine()
    machine.transition(TaskState.THINKING)
    assert machine.transition(TaskState.CANCELLED)
    assert machine.transition(TaskState.IDLE)
    assert machine.state is TaskState.IDLE
