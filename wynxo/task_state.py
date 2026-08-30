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
        self.changed_files: list[str] = []
        self.failures: list[str] = []
        self.successes: list[str] = []
        self.blockers: list[str] = []
        self.action_fingerprints: list[str] = []
        self.inspected_files: list[str] = []
        self.verification: list[str] = []
        self._repeat_counts: dict[str, int] = {}
        """Per-action repeat counts since the last progress event."""

    def begin(self, objective: str) -> None:
        self.reset()
        self.objective = objective.strip()
        self.transition(TaskState.THINKING)

    def record_action(self, fingerprint: str) -> int:
        """Record an action; return how many times it has been repeated since
        the last progress event (0 for a fresh action, 1 for its first
        repeat, and so on).

        mark_progress() resets the counters, so a legitimate re-read or
        re-run after an edit or a passing check never counts against the
        task -- only an action that keeps coming back with nothing changing
        in between does. Repeats land in ``blockers`` for the recovery
        prompt the agent can surface to the model.
        """
        self.action_fingerprints.append(fingerprint)
        count = self._repeat_counts.get(fingerprint, 0) + 1
        self._repeat_counts[fingerprint] = count
        if count >= 2:
            self.blockers.append(f"Repeated action: {fingerprint}")
        return count - 1

    def mark_progress(self) -> None:
        """A step that measurably moved the task forward resets repeat
        counts: an edit landed, a check passed, the plan changed."""
        self._repeat_counts = {}

    def record_failure(self, failure: str) -> None:
        if failure and failure not in self.failures:
            self.failures.append(failure)

    def record_success(self, success: str) -> None:
        if success and success not in self.successes:
            self.successes.append(success)

    def clear_blocking_failures(self) -> None:
        """A check that previously failed now passes. The old failure is
        history, not current state: the completion report must not keep
        saying "partially completed" for a failure that was fixed and
        re-verified."""
        self.failures = [f for f in self.failures
                         if not f.startswith(("tests failed",
                                              "syntax check failed"))]

    def set_root_cause(self, cause: str) -> None:
        self.root_cause = cause.strip()

    def add_file(self, path: str, changed: bool = False) -> None:
        """Record a file this turn touched.

        One list per kind. A file that was only looked at used to be
        appended to two lists that were kept in lockstep and never read
        apart -- and neither of them reached the model or the screen, so the
        whole of it was bookkeeping for nobody. What was inspected is worth
        keeping, because a stuck turn benefits from being told where it has
        already been; the second copy was not.
        """
        if not path:
            return
        target = self.changed_files if changed else self.inspected_files
        if path not in target:
            target.append(path)

    @property
    def relevant_files(self) -> list[str]:
        """What the turn has looked at. Kept as a name because it reads
        better at the call sites that ask the question that way."""
        return self.inspected_files

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

    def recovery_block(self) -> str:
        """A structured, model-visible description of why the task is stuck.

        Built entirely from recorded state -- what was repeated, what has
        failed, what the objective was, what has changed -- so the model
        gets the same evidence the UI has, not a prose guess at it.
        """
        lines = [
            "RECOVERY",
            "",
            "The task is not making progress. Recorded state:",
        ]
        if self.objective:
            lines.append(f"  objective: {self.objective[:300]}")
        blockers = self.blockers[-4:]
        if blockers:
            lines.append("  repeated actions:")
            for blocker in blockers:
                lines.append(f"    - {blocker}")
        if self.failures:
            lines.append("  failures:")
            for failure in self.failures[-4:]:
                lines.append(f"    - {failure[:200]}")
        if self.changed_files:
            lines.append("  changed files: " + ", ".join(self.changed_files[-6:]))
        if self.inspected_files:
            # Recorded all along and never shown to anybody. A turn that is
            # repeating itself is exactly the one that benefits from being
            # told where it has already looked.
            lines.append("  already inspected: "
                         + ", ".join(self.inspected_files[-8:]))
        if self.root_cause:
            lines.append(f"  root cause so far: {self.root_cause[:200]}")
        lines.append("")
        lines.append(
            "Try a different strategy now: another way to inspect, another "
            "hypothesis, or a narrower question. Do not repeat the actions "
            "above."
        )
        return "\n".join(lines)

    def completion_report(self) -> str | None:
        """A compact evidence summary of a finished coding turn, or None
        when there is nothing to report.

        Pure conversation (small talk, a question) has no changed files, no
        failures and no verification, so it gets no report. Coding turns
        get exactly what the machine recorded -- never model prose -- so a
        "fixed" claim cannot outrun the checks that ran.
        """
        changed = self.changed_files
        verified = self.verification
        # Blocking failures are the ones the completion gate cares about: a
        # test or syntax run that failed at the end. A failed grep mid-
        # investigation is normal exploration and must not flip the report
        # to "partially completed".
        blocking = [f for f in self.failures
                    if f.startswith(("tests failed", "syntax check failed"))]
        if not (changed or blocking or verified):
            return None

        out = []
        if blocking:
            out.append(f"⚠ partially completed · {len(blocking)} blocking failure(s)")
        else:
            out.append("✓ completed")
        if changed:
            out.append("  changed: " + ", ".join(changed[:8]))
            if len(changed) > 8:
                out[-1] += f" (+{len(changed) - 8} more)"
        if verified:
            out.append("  verification: " + "; ".join(verified[:4]))
        if blocking:
            out.append("  issues:")
            for failure in blocking[:4]:
                out.append(f"    ✕ {failure[:160]}")
        return "\n".join(out)

    def reset(self) -> None:
        self.state = TaskState.IDLE
        self.objective = ""
        self.root_cause = ""
        self.changed_files = []
        self.failures = []
        self.successes = []
        self.blockers = []
        self.action_fingerprints = []
        self.inspected_files = []
        self.verification = []
        self._repeat_counts = {}
