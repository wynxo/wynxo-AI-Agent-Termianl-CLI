"""Small, bounded inference instrumentation.

Metrics are deliberately local and cheap.  They make the cost of hidden
requests visible without turning the normal terminal into a dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class RequestMetric:
    model: str
    mode: str
    tools_supplied: bool
    prompt_tokens_estimate: int
    started_at: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    finished_at: float | None = None
    output_tokens: int = 0
    eval_seconds: float = 0.0
    request_kind: str = "generation"

    @property
    def time_to_first_token(self) -> float | None:
        if self.first_token_at is None:
            return None
        return max(0.0, self.first_token_at - self.started_at)

    @property
    def tokens_per_second(self) -> float:
        if self.eval_seconds <= 0:
            return 0.0
        return self.output_tokens / self.eval_seconds


@dataclass
class InferenceMetrics:
    """Bounded history for /context and tests; never grows with the session."""

    max_samples: int = 32
    requests_per_turn: int = 0
    _turn_requests: int = 0
    samples: list[RequestMetric] = field(default_factory=list)

    def begin_turn(self) -> None:
        self._turn_requests = 0

    def begin(self, *, model: str, mode: str, tools_supplied: bool,
              prompt_tokens_estimate: int, request_kind: str = "generation") -> RequestMetric:
        self._turn_requests += 1
        self.requests_per_turn = self._turn_requests
        return RequestMetric(
            model=model, mode=mode, tools_supplied=tools_supplied,
            prompt_tokens_estimate=max(0, int(prompt_tokens_estimate)),
            request_kind=request_kind,
        )

    def finish(self, sample: RequestMetric) -> RequestMetric:
        sample.finished_at = time.monotonic()
        self.samples.append(sample)
        del self.samples[:-self.max_samples]
        return sample

    @property
    def last(self) -> RequestMetric | None:
        return self.samples[-1] if self.samples else None
