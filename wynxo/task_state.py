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
        self.objective = ""
        self.root_cause = ""
        self.relevant_files: list[str] = []
        self.changed_files: list[str] = []
        self.failures: list[str] = []
        self.successes: list[str] = []
        self.blockers: list[str] = []
        self.action_fingerprints: list[str] = []
        self.inspected_files: list[str] = []
        self.verification: list[str] = []

    def begin(self, objective: str) -> None:
        self.reset()
        self.objective = objective.strip()
        self.transition(TaskState.THINKING)

    def record_action(self, fingerprint: str) -> bool:
        """Record an action and reject recent repeats without new evidence."""
        if fingerprint in self.action_fingerprints[-3:]:
            self.blockers.append(f"Repeated action: {fingerprint}")
            return False
        self.action_fingerprints.append(fingerprint)
        return True

    def record_failure(self, failure: str) -> None:
        if failure and failure not in self.failures:
            self.failures.append(failure)

    def record_success(self, success: str) -> None:
        if success and success not in self.successes:
            self.successes.append(success)

    def set_root_cause(self, cause: str) -> None:
        self.root_cause = cause.strip()

    def add_file(self, path: str, changed: bool = False) -> None:
        target = self.changed_files if changed else self.relevant_files
        if path and path not in target:
            target.append(path)
        if path and not changed and path not in self.inspected_files:
            self.inspected_files.append(path)

    def record_verification(self, check: str) -> None:
        if check and check not in self.verification:
            self.verification.append(check)

    def transition(self, target: TaskState) -> bool:
        if target is self.state:
            return True
        if target not in _ALLOWED[self.state]:
            return False
        self.state = target
        return True

    def reset(self) -> None:
        self.state = TaskState.IDLE
        self.objective = ""
        self.root_cause = ""
        self.relevant_files = []
        self.changed_files = []
        self.failures = []
        self.successes = []
        self.blockers = []
        self.action_fingerprints = []
        self.inspected_files = []
        self.verification = []
