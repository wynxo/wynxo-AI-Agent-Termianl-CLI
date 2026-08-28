"""Validated high-level task state shared by the agent and presentation."""

from __future__ import annotations

from enum import Enum


class TaskState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    TESTING = "testing"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED: dict[TaskState, frozenset[TaskState]] = {
    TaskState.IDLE: frozenset({TaskState.THINKING, TaskState.PLANNING, TaskState.EXECUTING}),
    TaskState.THINKING: frozenset({TaskState.PLANNING, TaskState.EXECUTING, TaskState.TESTING, TaskState.FAILED, TaskState.CANCELLED, TaskState.COMPLETED}),
    TaskState.PLANNING: frozenset({TaskState.EXECUTING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.EXECUTING: frozenset({TaskState.THINKING, TaskState.TESTING, TaskState.RECOVERING, TaskState.FAILED, TaskState.CANCELLED, TaskState.COMPLETED}),
    TaskState.TESTING: frozenset({TaskState.RECOVERING, TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.RECOVERING: frozenset({TaskState.THINKING, TaskState.EXECUTING, TaskState.TESTING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.COMPLETED: frozenset({TaskState.IDLE}),
    TaskState.FAILED: frozenset({TaskState.IDLE, TaskState.THINKING, TaskState.EXECUTING}),
    TaskState.CANCELLED: frozenset({TaskState.IDLE, TaskState.THINKING, TaskState.EXECUTING}),
}


class TaskStateMachine:
    def __init__(self) -> None:
        self.state = TaskState.IDLE

    def transition(self, target: TaskState) -> bool:
        if target is self.state:
            return True
        if target not in _ALLOWED[self.state]:
            return False
        self.state = target
        return True

    def reset(self) -> None:
        self.state = TaskState.IDLE
