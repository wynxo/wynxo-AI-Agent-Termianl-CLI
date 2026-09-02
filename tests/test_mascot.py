"""One character, one size, and pixels a terminal can be trusted with.

The companion is a half-block sprite: two pixels to a cell, ``▀`` drawn in
the top pixel's colour over the bottom pixel's. That buys a silhouette --
ears, a head that tapers, shoulders, a laptop -- at a size where the face
before it could only be punctuation.

What these pin is the part that goes wrong silently. A frame a pixel wider
than the rest shifts the text set beside it; a frame a row shorter makes
the live region change height between states, which reads as the whole
screen jumping. Neither shows up in a screenshot of one frame, and both are
obvious in motion, which is the worst combination to leave untested.

There is no second drawing to keep in step any more. ``pet.py`` had a face,
``companion.py`` had a full set of staged ASCII scenes nothing ever drew,
and ``motion.py`` wrapped those a third time for previews. The face and the
scenes are gone; the previews draw this.
"""
from __future__ import annotations

import unicodedata

import pytest
from rich.cells import cell_len

from wynxo import sprite
from wynxo.companion import State
from wynxo.theme import PURPLE, names, resolve


def _frames():
    for state, frames in sprite.FRAMES.items():
        for index, pixels in enumerate(frames):
            yield state, index, pixels


class TestTheBlockNeverChangesSize:
    @pytest.mark.parametrize("state,index,pixels", list(_frames()))
    def test_every_frame_is_the_same_pixel_grid(self, state, index, pixels):
        assert len(pixels) == sprite.HEIGHT * 2, f"{state.value}[{index}]"
        for row, line in enumerate(pixels):
            assert len(line) == sprite.WIDTH, \
                f"{state.value}[{index}] row {row}: {len(line)}"

    @pytest.mark.parametrize("state", list(State))
    def test_every_state_draws_the_same_block(self, state):
        for frame in range(len(sprite.FRAMES[state])):
            rows = sprite.rows(state, frame, PURPLE)
            assert len(rows) == sprite.HEIGHT
            for row in rows:
                assert row.cell_len == sprite.WIDTH, \
                    f"{state.value}[{frame}]: {row.plain!r}"

    def test_the_text_beside_the_companion_never_moves(self):
        """The failure this is really about: the status lines set to the
        right of the character sitting still while the state changes."""
        starts = set()
        for state in State:
            for row in sprite.rows(state, 0, PURPLE):
                starts.add(row.cell_len + len("  status"))
        assert len(starts) == 1, starts


class TestEveryCellIsSafeInALineOfText:
    @pytest.mark.parametrize("state", list(State))
    def test_only_the_four_glyphs_are_drawn(self, state):
        for frame in range(len(sprite.FRAMES[state])):
            for row in sprite.rows(state, frame, PURPLE):
                assert set(row.plain) <= set(sprite.GLYPHS), row.plain

    def test_the_glyphs_are_one_cell_and_ambiguous(self):
        """Documenting the constraint the technique comes with.

        ▀ and ▄ are the only characters that split a cell horizontally, and
        they are East Asian Width Ambiguous, as is █. There is no Neutral
        alternative to pick instead -- so half-block art cannot be made safe
        in a locale that draws Ambiguous wide, and the honest answer is to
        decline to draw it there rather than to pretend otherwise. That is
        what the locale gate below is for; this pins why it has to exist.
        """
        for char in sprite.GLYPHS:
            assert cell_len(char) == 1, char
        assert {unicodedata.east_asian_width(c) for c in sprite.GLYPHS.strip()} \
            == {"A"}

    def test_transparent_pixels_paint_no_background(self):
        """A shape on the conversation, not a coloured rectangle sitting on
        it. Only cells where two different opaque colours stack may set a
        background at all."""
        for state in State:
            rows = sprite.rows(state, 0, PURPLE)
            for row in rows:
                for span in row.spans:
                    text = row.plain[span.start:span.end]
                    if "on " in str(span.style):
                        assert " " not in text, (state.value, text)


class TestItIsOneCharacter:
    def test_every_frame_keeps_the_ears_and_the_laptop(self):
        """The silhouette is the character. Written as one base with rows
        edited, so a state cannot quietly become a different animal."""
        for state, index, pixels in _frames():
            assert set(pixels[9]) <= set(".LC"), f"{state.value}[{index}]"
            assert "F" in pixels[3], f"{state.value}[{index}]: no head"

    def test_success_and_error_do_not_share_a_silhouette(self):
        """The one pair that must never be confused at a glance, since one
        of them means stop reading and look. Compared as glyphs, not as
        colour: a red closed eye and a happy closed eye are the same shape,
        and that is exactly how they used to be told apart."""
        good = [r.plain for r in sprite.rows(State.SUCCESS, 0, PURPLE)]
        bad = [r.plain for r in sprite.rows(State.ERROR, 0, PURPLE)]
        assert good != bad

    def test_a_blink_survives_without_colour(self):
        """Dimming the eye pixel left the cell's glyph identical, so the
        character never blinked on a terminal without truecolour."""
        frames = {tuple(r.plain for r in sprite.rows(State.IDLE, f, PURPLE))
                  for f in range(len(sprite.FRAMES[State.IDLE]))}
        assert len(frames) > 1


class TestTheColourComesFromTheTheme:
    def test_every_ink_is_a_role_not_a_colour(self):
        for char, role in sprite.INK.items():
            if not role:
                continue
            assert not role.startswith("#"), char
            assert getattr(PURPLE, role, None), (char, role)

    @pytest.mark.parametrize("theme", names())
    def test_switching_theme_switches_the_companion(self, theme):
        palette = resolve(theme)
        drawn = sprite.rows(State.CODING, 0, palette)
        styles = {str(span.style) for row in drawn for span in row.spans}
        assert any(palette.accent in style for style in styles), theme


class TestItGivesWayOnASmallTerminal:
    def test_no_sprite_where_half_blocks_will_not_render(self):
        assert sprite.fits(120, unicode_ok=False) is False

    def test_no_sprite_on_a_narrow_terminal(self):
        assert sprite.fits(40, unicode_ok=True) is False
        assert sprite.fits(sprite.MIN_COLUMNS, unicode_ok=True) is True

    def test_no_sprite_where_ambiguous_width_draws_wide(self, monkeypatch):
        """A CJK locale may draw ▀ as two cells, which would double the
        sprite and shift the text beside it by fourteen columns a frame."""
        monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
        assert sprite.fits(120, unicode_ok=True) is False
        monkeypatch.setenv("LC_ALL", "en_GB.UTF-8")
        assert sprite.fits(120, unicode_ok=True) is True

    def test_it_leaves_room_for_the_words_beside_it(self):
        """The companion is seventh in the hierarchy and the status lines
        are third and fourth, so the threshold has to leave the words a
        usable column rather than just fitting the picture."""
        assert sprite.MIN_COLUMNS >= sprite.WIDTH * 2
