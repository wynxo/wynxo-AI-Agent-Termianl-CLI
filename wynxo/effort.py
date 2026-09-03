"""Effort levels.

Effort is a *scheduler policy*, not a single model knob. Local models mostly do
not expose a native reasoning budget, so a level that only forwarded a
``reasoning_effort`` field would collapse into two or three real settings.

Instead each level controls how many chances the model gets to be right:
how much it plans, how many tool iterations it may spend, how many times it
re-checks its own work, how wide it fans out, and how much context it is
allowed to keep. A 30B model at ``max`` genuinely beats itself at ``low``,
not because it thinks harder per token but because it catches its own mistakes.

Where a model *does* have a native dial (Qwen3's thinking mode, gpt-oss's
``reasoning_effort``) the policy drives that too, via `native_effort` and
`thinking`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

EffortName = Literal["low", "medium", "high", "xhigh", "max", "ultra"]

ORDER: tuple[EffortName, ...] = ("low", "medium", "high", "xhigh", "max", "ultra")

PlanMode = Literal["none", "inline", "explicit", "critique"]


@dataclass(frozen=True)
class EffortPolicy:
    """How hard the agent is allowed to work on one request."""

    name: EffortName

    headline: str
    """One line, in plain terms: how much it thinks and how smart that makes
    it. This is what people actually choose on."""

    speed: str
    """Relative wall-clock cost, shown beside the headline.

    Kept out of the headline rather than repeated in it. Every headline used
    to name the speed too -- "a little thinking, quick, reasonably smart" --
    so the picker either said it twice or, as it did for the whole of this
    field's life, never said it at all."""

    # --- loop shape -------------------------------------------------------
    plan: PlanMode
    """``none`` acts immediately. ``inline`` asks for a one-line plan in the
    same turn. ``explicit`` runs a separate read-only planning pass first.
    ``critique`` additionally has the model attack its own plan before acting."""

    max_iterations: int
    """Hard ceiling on tool-call round trips for a single user request."""

    verify_rounds: int
    """Self-review passes after the model believes it is done. -1 = until the
    review comes back clean (bounded by ``max_verify_rounds``)."""

    max_verify_rounds: int
    """Absolute ceiling when ``verify_rounds`` is -1, so 'until clean' cannot
    spin forever on a model that never admits it is done."""

    parallel_samples: int
    """Independent attempts at the *plan*, reconciled by the model before it
    acts. >1 costs real time on a local box; only the top levels use it."""

    # --- context ----------------------------------------------------------
    context_budget: int
    """Tokens of conversation to keep before compaction kicks in. ``0`` means
    use the whole served context window."""

    max_tool_output: int
    """Characters of a single tool result kept verbatim before truncation.
    Low effort keeps less so the window lasts longer."""

    # --- model knobs ------------------------------------------------------
    thinking: bool
    """Qwen3 / DeepSeek-style thinking mode."""

    think_level: str | None
    """Ollama's native reasoning dial, sent as ``think``. It accepts
    ``"low" | "medium" | "high" | "max"`` as well as a plain boolean
    (api/types.go, ThinkValue). Older servers only understand the boolean, so
    the client downgrades automatically if a level is rejected.

    The mapping is deliberately graduated rather than name-matched: by the
    time you are at wynxo's ``high`` you are already getting a planning pass
    and a verification round, which is more added rigour than the raw
    thinking dial contributes."""

    temperature: float
    num_predict: int
    """Max tokens per single model response. -1 = model default."""

    # --- behaviour --------------------------------------------------------
    repair_attempts: int
    """How many times a malformed tool call is handed back for repair before
    the turn is failed. Local models emit bad JSON often enough that this
    matters more than anything else on this list."""

    def bump(self, delta: int) -> "EffortPolicy":
        """Return the policy ``delta`` steps up or down the ladder."""
        i = ORDER.index(self.name)
        j = max(0, min(len(ORDER) - 1, i + delta))
        return POLICIES[ORDER[j]]

    def describe(self) -> str:
        """The plain-language summary plus what it mechanically does."""
        return f"{self.headline} -- {self.mechanics()}"

    def mechanics(self) -> str:
        plan = {
            "none": "no planning",
            "inline": "inline plan",
            "explicit": "planning pass",
            "critique": "plan + self-critique",
        }[self.plan]
        if self.verify_rounds < 0:
            verify = f"verify until clean (cap {self.max_verify_rounds})"
        elif self.verify_rounds == 0:
            verify = "no verification"
        else:
            verify = f"{self.verify_rounds} verify round(s)"
        bits = [plan, f"{self.max_iterations} tool iters", verify]
        if self.parallel_samples > 1:
            bits.append(f"{self.parallel_samples}x plan consensus")
        if not self.thinking:
            bits.append("no thinking")
        elif self.think_level:
            bits.append(f'think "{self.think_level}"')
        else:
            bits.append("thinking on")
        return ", ".join(bits)


POLICIES: dict[EffortName, EffortPolicy] = {
    "low": EffortPolicy(
        name="low",
        headline="least thinking, least smart",
        speed="fastest",
        plan="none",
        max_iterations=6,
        verify_rounds=0,
        max_verify_rounds=0,
        parallel_samples=1,
        context_budget=8_000,
        max_tool_output=4_000,
        thinking=False,
        think_level=None,
        temperature=0.3,
        num_predict=1_024,
        repair_attempts=1,
    ),
    "medium": EffortPolicy(
        name="medium",
        headline="a little thinking, reasonably smart",
        speed="quick",
        plan="inline",
        max_iterations=16,
        verify_rounds=0,
        max_verify_rounds=0,
        parallel_samples=1,
        context_budget=16_000,
        max_tool_output=8_000,
        thinking=False,
        think_level=None,
        temperature=0.4,
        num_predict=2_048,
        repair_attempts=2,
    ),
    "high": EffortPolicy(
        name="high",
        headline="real thinking, noticeably smarter",
        speed="slower",
        plan="explicit",
        max_iterations=40,
        verify_rounds=1,
        max_verify_rounds=1,
        parallel_samples=1,
        context_budget=32_000,
        max_tool_output=12_000,
        thinking=True,
        think_level="medium",
        temperature=0.5,
        num_predict=4_096,
        repair_attempts=3,
    ),
    "xhigh": EffortPolicy(
        name="xhigh",
        headline="hard thinking, very thorough",
        speed="slow",
        plan="explicit",
        max_iterations=80,
        verify_rounds=2,
        max_verify_rounds=2,
        parallel_samples=1,
        context_budget=64_000,
        max_tool_output=16_000,
        thinking=True,
        think_level="high",
        temperature=0.5,
        num_predict=8_192,
        repair_attempts=3,
    ),
    "max": EffortPolicy(
        name="max",
        headline="maximum thinking, near its best",
        speed="very slow",
        plan="critique",
        max_iterations=150,
        verify_rounds=-1,
        max_verify_rounds=4,
        parallel_samples=2,
        context_budget=0,
        max_tool_output=24_000,
        thinking=True,
        think_level="max",
        temperature=0.6,
        num_predict=-1,
        repair_attempts=4,
    ),
    "ultra": EffortPolicy(
        name="ultra",
        headline="most thinking possible, smartest",
        speed="slowest",
        plan="critique",
        max_iterations=400,
        verify_rounds=-1,
        max_verify_rounds=8,
        parallel_samples=3,
        context_budget=0,
        max_tool_output=32_000,
        thinking=True,
        think_level="max",
        temperature=0.7,
        num_predict=-1,
        repair_attempts=5,
    ),
}


def resolve(name: str) -> EffortPolicy:
    """Look up a policy by its own name.

    No aliases. There are six levels with short, plain names, and /effort
    lists them -- a second vocabulary of "x", "u", "insane" and "maximum" on
    top of that is more to remember rather than less, and made it easy to
    pick a level you did not mean.
    """
    key = name.strip().lower()
    if key not in POLICIES:
        raise KeyError(
            f"unknown effort level {name!r}; choose one of {', '.join(ORDER)}"
        )
    return POLICIES[key]


def override(policy: EffortPolicy, **kwargs) -> EffortPolicy:
    """Per-session tweaks on top of a named level (e.g. thinking off at high)."""
    return replace(policy, **kwargs)
