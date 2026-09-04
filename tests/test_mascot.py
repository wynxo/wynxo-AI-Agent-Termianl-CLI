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
        edited, so a state cannot quietly become a different animal.

        Searched, not indexed. These used to name the row the laptop was
        on, and when the drawing grew a row every state failed at once --
        which said the sprite had changed size, not that it had stopped
        being the same character, and that is the only thing worth failing
        over here.
        """
        for state, index, pixels in _frames():
            where = f"{state.value}[{index}]"
            body = "".join(pixels)
            assert "F" in pixels[0] + pixels[1], f"{where}: no ears"
            assert "F" in pixels[3], f"{where}: no head"
            assert "L" in body, f"{where}: no laptop"
            assert "P" in body, f"{where}: nothing of him showing"

    def test_the_screen_is_always_inside_the_laptop(self):
        """A lit pixel loose on the desk is a bug you only see in motion:
        the screen contents animate per state, and a row edited a column
        wide leaves them glowing beside the machine rather than in it."""
        for state, index, pixels in _frames():
            for row, line in enumerate(pixels):
                if set(line) & set("SGR"):
                    assert set(line) <= set(".fLKSGRC"), \
                        f"{state.value}[{index}] row {row}: {line}"

    def test_every_frame_has_both_arms(self):
        """The arms are what the sprite grew four columns and a row for. A
        state that drops one has the character reaching for something,
        which is a thing none of them mean.

        Looked for anywhere below the head rather than on one named row.
        The arms move between states -- a hand goes up to the face while
        he thinks -- so the row they cross is not a constant and never was.
        """
        half = sprite.WIDTH // 2
        for state, index, pixels in _frames():
            below = pixels[8:]
            assert any("f" in row[:half] for row in below), \
                f"{state.value}[{index}]: no left arm"
            assert any("f" in row[half:] for row in below), \
                f"{state.value}[{index}]: no right arm"

    def test_the_paws_are_somewhere_in_every_frame(self):
        """On the near edge, up on the deck, or at the face while thinking
        -- but never gone. A character whose hands vanish for a frame reads
        as a glitch, and that is what the paws-off states used to be."""
        for state, index, pixels in _frames():
            below = "".join(pixels[5:])
            assert "f" in below, f"{state.value}[{index}]: no paws anywhere"

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


class TestTheGalleryFitsTheTerminal:
    """/animate list and /pet draw every state side by side. How many fit
    across is a property of the terminal, and it used to be the number 4 --
    which needed seventy columns and was drawn on any terminal at all, so a
    narrow one wrapped every row and the cats came apart into bands of
    pixels with their labels adrift underneath."""

    WIDTHS = (20, 40, 60, 62, 80, 100, 120, 200)

    def test_a_row_of_states_fits_the_width_it_was_measured_for(self):
        from wynxo.cli import STATE_GAP, STATE_INDENT, gallery_columns

        for width in self.WIDTHS:
            drawn = (STATE_INDENT
                     + gallery_columns(width) * (sprite.WIDTH + len(STATE_GAP)))
            assert drawn <= width or gallery_columns(width) == 1, width

    def test_a_terminal_narrower_than_one_state_still_gets_one(self):
        """Rather than zero columns, an empty gallery, or a division by
        zero. One drawing that overflows says more than none."""
        from wynxo.cli import gallery_columns

        assert gallery_columns(10) == 1
        assert gallery_columns(0) == 1

    def test_a_wider_terminal_never_shows_fewer(self):
        from wynxo.cli import gallery_columns

        counts = [gallery_columns(w) for w in self.WIDTHS]
        assert counts == sorted(counts)

    def _lines(self, width: int) -> list[str]:
        import io

        from wynxo.cli import Repl
        from wynxo.ui import UI

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = width
        repl = Repl.__new__(Repl)
        repl.ui = ui
        repl._show_states()
        return ui.console.file.getvalue().splitlines()

    def test_nothing_the_gallery_draws_is_wrapped(self):
        """The real symptom. A wrapped sprite row is not a narrower sprite,
        it is half a cat on the next line -- so the check is that the
        console never had to wrap anything, counted in lines."""
        from math import ceil

        from wynxo.cli import gallery_columns

        for width in (60, 62, 80, 100, 120):
            groups = ceil(len(State) / gallery_columns(width))
            # Per group: the sprite's rows, the labels, and a blank line.
            assert len(self._lines(width)) == groups * (sprite.HEIGHT + 2), \
                width

    def test_every_state_is_still_shown(self):
        body = "\n".join(self._lines(62))
        for state in State:
            assert state.value in body, state.value
