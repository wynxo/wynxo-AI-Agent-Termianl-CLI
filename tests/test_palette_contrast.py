"""Every colour that carries words has to be readable as words.

"Quiet" and "invisible" are one nudge apart, and the first draft of the
faint role landed on the wrong side of it: #6b6280 is 3.68:1 against a
black terminal, under the 4.5:1 at which body text stops being reliably
legible -- and faint was carrying the completion report's evidence, the
plan's progress count and the marker saying a message of yours was queued.

Two things came out of that. The roles were sorted: what you read is muted,
what you may ignore is faint. And faint was mixed up to clear the threshold
anyway, because a role named "ignorable" still has to be legible when you
go looking at it.

The ratios here are the WCAG relative-luminance formula. It is a rule about
text on backgrounds, so it is applied to the roles that draw text and not
to accent_dim, which exists only as a shadow colour inside the companion
sprite -- pixels next to other pixels, where what matters is telling them
apart from their neighbours rather than reading them.
"""
from __future__ import annotations

import pytest

from wynxo import theme

TEXT_ROLES = ("accent", "text", "muted", "faint", "good", "warn", "bad",
              "bar_text", "bar_dim", "bar_accent")

READABLE = 4.5
"""WCAG AA for body text. Not decoration: every one of these draws words."""


def _luminance(colour: str) -> float:
    colour = colour.lstrip("#")
    channels = (int(colour[i:i + 2], 16) / 255 for i in (0, 2, 4))
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _hex_palettes():
    for name in theme.names():
        palette = theme.resolve(name)
        # Two themes are built from the terminal's own sixteen colours,
        # whose actual values belong to the user's terminal profile. There
        # is nothing here to measure and nothing we could fix if there were.
        if palette.accent.startswith("#"):
            yield name, palette


@pytest.mark.parametrize("name,palette", list(_hex_palettes()),
                         ids=lambda v: v if isinstance(v, str) else "")
class TestEveryThemeIsLegible:
    def test_text_roles_are_readable_on_a_black_terminal(self, name, palette):
        for role in TEXT_ROLES:
            colour = getattr(palette, role)
            ratio = contrast(colour, "#000000")
            assert ratio >= READABLE, \
                f"{name}.{role} {colour} is {ratio:.2f}:1 on black"

    def test_the_strip_is_readable_on_its_own_background(self, name, palette):
        """The status strip paints its own background, so its colours are
        measured against that rather than against the terminal."""
        for role in ("bar_text", "bar_dim", "bar_accent"):
            colour = getattr(palette, role)
            ratio = contrast(colour, palette.bar_bg)
            assert ratio >= READABLE, \
                f"{name}.{role} {colour} is {ratio:.2f}:1 on the strip"

    def test_faint_is_quieter_than_muted_and_muted_than_text(self, name, palette):
        """Readable is not the same as loud. The hierarchy is the point of
        having three of these, and lifting faint must not flatten it."""
        black = "#000000"
        assert contrast(palette.faint, black) < contrast(palette.muted, black)
        assert contrast(palette.muted, black) < contrast(palette.text, black)

    def test_success_warning_and_failure_are_told_apart(self, name, palette):
        """Not by name -- by hue. Three status colours that read alike are
        three colours doing one colour's work."""
        for a, b in (("good", "bad"), ("good", "warn"), ("warn", "bad")):
            first, second = getattr(palette, a), getattr(palette, b)
            assert first != second, f"{name}: {a} and {b} are the same colour"
