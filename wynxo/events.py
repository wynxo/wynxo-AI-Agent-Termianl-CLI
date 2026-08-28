"""Structured execution events shared by orchestration and presentation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class ExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class ToolEvent:
    tool: str
    summary: str = ""
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: ExecutionState = ExecutionState.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    output: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        end = self.finished_at or time.monotonic()
        return max(0.0, end - (self.started_at or end))

    def start(self) -> None:
        self.state = ExecutionState.RUNNING
        self.started_at = time.monotonic()

    def finish(self, success: bool, output: str = "", error: str = "", **metadata) -> None:
        # Lifecycle methods are idempotent: late callbacks after cancellation
        # must not turn a cancelled execution back into a success/failure.
        if self.state in (ExecutionState.SUCCESS, ExecutionState.FAILURE, ExecutionState.CANCELLED):
            return
        self.state = ExecutionState.SUCCESS if success else ExecutionState.FAILURE
        self.finished_at = time.monotonic()
        self.output = output
        self.error = error
        self.metadata.update(metadata)

    def cancel(self, reason: str = "cancelled") -> None:
        if self.state in (ExecutionState.SUCCESS, ExecutionState.FAILURE, ExecutionState.CANCELLED):
            return
        self.state = ExecutionState.CANCELLED
        self.finished_at = time.monotonic()
        self.error = reason

    def compact(self) -> str:
        if self.state is ExecutionState.RUNNING:
            return f"→ {self.tool} {self.summary}".rstrip()
        mark = "✓" if self.state is ExecutionState.SUCCESS else "✕"
        if self.state is ExecutionState.CANCELLED:
            mark = "⏹"
        detail = self.metadata.get("display") or self.error or self.summary
        return f"{mark} {detail}  {self.duration:.2f}s".rstrip()
