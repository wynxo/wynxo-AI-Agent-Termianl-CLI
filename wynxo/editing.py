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
    """Approximate how much of a file an edit replaces.

    Measured in content characters (newlines excluded): an unchanged line
    contributes its length to ``common``, everything else counts as
    replaced. Using the raw byte length of ``before`` as the denominator
    would let the newline characters dominate on small files -- an
    unchanged ``a\\nb\\nc\\n`` would score 0.5 and trip a ``large_rewrite``
    flag on a no-op edit."""
    total = max(1, sum(len(line) for line in before.splitlines()))
    common = 0
    # Deliberately not strict. The two files are different lengths --
    # that is the whole reason for measuring how much of one survives in
    # the other -- so stopping at the shorter is the intended behaviour
    # here, unlike everywhere else zip() is used in this project.
    for left, right in zip(before.splitlines(),
                          after.splitlines(), strict=False):
        if left == right:
            common += len(left)
    return max(0.0, min(1.0, 1.0 - common / total))
