"""Shared edit-analysis helpers used by tools and verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffStats:
    additions: int
    deletions: int
    changed: bool


def diff_stats(diff: str) -> DiffStats:
    additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return DiffStats(additions, deletions, additions > 0 or deletions > 0)


def replacement_ratio(before: str, after: str) -> float:
    """Approximate how much of a file an edit replaces."""
    total = max(1, len(before))
    common = 0
    for left, right in zip(before.splitlines(), after.splitlines()):
        if left == right:
            common += len(left)
    return max(0.0, min(1.0, 1.0 - common / total))
