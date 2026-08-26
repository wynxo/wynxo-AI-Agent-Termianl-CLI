"""Every pattern in wynxo must cost time in proportion to its input.

A regex with an open quantifier in front of a required literal is the
classic way to turn a fast program into a hung one: the engine eats to the
end of the text at every position, fails, and hands the characters back one
at a time. Twice the text, four times the work.

Two of those shipped. The credential mask ran on every read and every tool
result, so a minified bundle -- one enormous line -- stopped wynxo dead for
minutes with no error and nothing on screen to say why. The speech filter
did the same to any long answer.

Neither was found by reading the patterns; both were found by timing them.
So this is the timing, kept, and pointed at every pattern in the package
rather than at the two that were already wrong. It is a class of bug, and
this is the shape of test that catches the class.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import time

import pytest

import wynxo

SMALL, LARGE = 10_000, 40_000
"""Four times the text. Linear costs ~4x more; quadratic ~16x.

Sized so a quadratic pattern is unmistakable rather than marginal. At
20,000 characters the markdown-link rule came in at 0.226s against a
0.25s floor: it passed here and failed on CI. The bug was real either
way, and all the smaller sample decided was which machine found out."""

FLOOR = 0.25
"""Seconds. Below this a pattern is fast enough that the ratio is only
measuring scheduler noise, so no ratio can fail it."""

GROWTH = 8.0
"""How much worse than linear before it counts. Well clear of 4x for the
honest patterns, well under the 16x a quadratic one shows."""


def _probes(size: int) -> dict[str, str]:
    """Text with no structure in it, in the shapes patterns get greedy about.

    None of these should match anything. That is the point: a healthy
    pattern fails fast on all of them, and an unhealthy one spends the
    whole input proving it cannot match.
    """
    return {
        "plain": "x" * size,
        "urlish": "a+b-c.d" * (size // 7),
        "spaces": "a " * (size // 2),
        "punctuation": "a:b=c," * (size // 6),
        "slashes": "a/b\\c" * (size // 5),
        "quotes": "a\"b'c" * (size // 5),
        "brackets": "a[b](c)" * (size // 7),
        "markup": "a`b*c_" * (size // 6),
        "escapes": "\x1b[31ma" * (size // 6),
        "lines": "ab\n" * (size // 3),
        "digits": "1234567890" * (size // 10),
        "identifiers": "Ab1_-." * (size // 6),
        "unclosed": "[<({" * (size // 4),
        "tagish": "<a b" * (size // 4),
        "rules": "-=_ " * (size // 4),
        "emphasis": "**a" * (size // 3),
    }


def _patterns() -> list[tuple[str, re.Pattern]]:
    """Every compiled pattern reachable from the package, once each."""
    found: dict[tuple[str, int], tuple[str, re.Pattern]] = {}
    for module in pkgutil.walk_packages(wynxo.__path__, "wynxo."):
        try:
            loaded = importlib.import_module(module.name)
        except Exception:
            # A module that will not import on this platform is that
            # module's problem, not this test's.
            continue
        for name in dir(loaded):
            value = getattr(loaded, name, None)
            if not isinstance(value, re.Pattern):
                continue
            if not isinstance(value.pattern, str):
                continue
            found.setdefault((value.pattern, value.flags),
                             (f"{module.name}.{name}", value))
    return sorted(found.values())


def _cost(pattern: re.Pattern, text: str) -> float:
    start = time.perf_counter()
    list(pattern.finditer(text))
    return time.perf_counter() - start


ALL = _patterns()


def test_the_sweep_actually_found_the_patterns():
    """Guards the test itself: a walk that silently found nothing would
    pass every case below without ever running one."""
    names = {name for name, _ in ALL}
    assert len(ALL) > 20
    assert "wynxo.secrets._URL_CRED" in names
    assert "wynxo.speech._PATHY" in names


@pytest.mark.parametrize("name,pattern", ALL, ids=[n for n, _ in ALL])
def test_a_pattern_costs_no_more_than_its_text(name, pattern):
    for shape, large in _probes(LARGE).items():
        small = _probes(SMALL)[shape]
        try:
            quick, slow = _cost(pattern, small), _cost(pattern, large)
        except Exception:
            continue        # a pattern this text cannot be fed is not the point
        if slow <= FLOOR or slow <= quick * GROWTH:
            continue
        # Measure again before failing. A single slow reading can be the
        # machine rather than the pattern, and a test that cries wolf on a
        # busy runner gets deleted rather than believed.
        quick, slow = _cost(pattern, small), _cost(pattern, large)
        if slow <= FLOOR or slow <= quick * GROWTH:
            continue
        pytest.fail(
            f"{name} grows faster than its input on {shape!r} text: "
            f"{SMALL} chars took {quick:.3f}s but {LARGE} chars took "
            f"{slow:.3f}s -- {slow / max(quick, 1e-9):.0f}x for 4x the "
            f"text. Something in {pattern.pattern[:120]!r} backtracks. "
            f"Bound the open quantifier, or anchor what can start a match."
        )
