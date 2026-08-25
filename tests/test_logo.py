"""The start-up logo.

Three things decide whether a splash screen is a pleasure or an obstacle:
it has to fit, it has to be brief, and it has to know when not to run.
"""

from __future__ import annotations

import asyncio

import pytest

from wynxo import logo
from wynxo.ui import UI


class TestFittingTheTerminal:
    def test_art_wider_than_the_terminal_is_brought_down(self):
        """The bundled art is 160 columns. A logo that wraps is worse than
        no logo."""
        lines = logo.fit(logo.read("wyn"), width=70, max_height=40)
        assert lines and max(len(l) for l in lines) <= 70

    def test_a_short_terminal_binds_the_height(self):
        lines = logo.fit(logo.read("wyn"), width=100, max_height=8)
        assert len(lines) <= 8

    def test_the_picture_survives_the_shrink(self):
        """Resampled rather than cropped: cropping loses half of it, and
        taking every other line loses the shading."""
        lines = logo.fit(logo.read("wyn"), width=60, max_height=30)
        ink = sum(1 for line in lines for ch in line if ch.strip())
        assert ink > len(lines) * 8, "the picture came out mostly empty"

    def test_proportions_are_kept(self):
        """Source art is already in character cells, so its shape is right
        and must simply be preserved."""
        art = logo.read("wyn")
        rows = [r for r in art.split("\n") if r.strip()]
        ratio = len(rows) / max(len(r) for r in rows)
        lines = logo.fit(art, width=80, max_height=200)
        assert len(lines) / max(len(l) for l in lines) == pytest.approx(
            ratio, abs=0.25)

    def test_a_silly_width_does_not_divide_by_zero(self):
        assert logo.fit(logo.read("wyn"), width=1, max_height=1)

    def test_empty_art_gives_nothing(self):
        assert logo.fit("", 40, 20) == []
        assert logo.fit("   \n  \n", 40, 20) == []


class TestTheLogosOnDisk:
    def test_they_are_found(self):
        assert "wyn" in logo.available()

    def test_every_one_of_them_renders(self):
        for name in logo.available():
            assert logo.fit(logo.read(name), 60, 20), f"{name} came out empty"

    def test_an_unknown_name_is_empty_rather_than_an_error(self):
        assert logo.read("no-such-logo") == ""


class TestTheColourSweep:
    def test_the_sweep_ends_where_it_began(self):
        """A hue that wrapped the long way would put a lime stripe across
        the picture."""
        first, last = logo.SWEEP[0], logo.SWEEP[-1]
        assert all(abs(a - b) < 60 for a, b in zip(first, last))

    def test_it_is_pink_through_purple_with_no_green(self):
        for r, g, b in logo.SWEEP:
            assert g < r and g < b, f"({r},{g},{b}) is greenish"

    def test_the_phase_moves_the_colour(self):
        assert logo.colour_at(0, 0, 0) != logo.colour_at(0, 0, 1)

    def test_it_travels_diagonally(self):
        """Folding the row into the offset makes the band move across and
        down together instead of as a flat wipe."""
        assert logo.colour_at(0, 0, 0) != logo.colour_at(1, 0, 0)

    def test_a_frame_covers_every_line(self):
        lines = ["##", "##", "##"]
        assert str(logo.frame(lines, 0)).count("\n") == 3

    def test_nothing_paints_a_background(self):
        """Only foreground colour is set, so the gaps in the art stay
        transparent -- a background would put a solid rectangle behind the
        picture and over whatever the terminal already had."""
        frame = logo.frame(["  ##  ", "######"], 0)
        assert frame.spans
        for span in frame.spans:
            assert "on " not in str(span.style)


class TestWhenNotToRun:
    def test_not_when_animations_are_off(self):
        assert logo.should_play(UI(), animations=False) is False

    def test_not_without_a_terminal(self):
        assert logo.should_play(UI(), animations=True) is False

    def test_not_on_a_phone_width_terminal(self):
        """is_terminal is read-only on a rich Console, so the console is
        replaced rather than patched."""
        ui = UI()
        ui.console = type("Terminal", (), {"is_terminal": True})()
        ui.narrow = False
        assert logo.should_play(ui, animations=True) is True
        ui.narrow = True
        assert logo.should_play(ui, animations=True) is False

    def test_a_missing_logo_draws_nothing(self):
        assert asyncio.run(logo.play(UI(), "no-such-logo")) is False

    def test_it_falls_back_to_a_still_frame(self):
        """Everywhere a repainting widget cannot go -- which includes the
        chat layout's transcript -- it is still coloured, just not moving."""
        from wynxo.tui import Transcript

        page = Transcript(width=80)
        ui = UI()
        ui.console = page.console
        ui.live_ok = False
        assert asyncio.run(logo.play(ui, "wyn", animations=True)) is True
        page.drain()
        body = "\n".join(page.lines)
        assert "\x1b[" in body, "the still frame lost its colour"
        assert "?25" not in body and "\r" not in body

    def test_it_is_brief(self):
        """Anything longer is a thing you sit through, every start-up."""
        assert logo.FRAMES * logo.FRAME_TIME < 1.2


class TestTheHeightCap:
    def test_it_measures_the_screen_not_the_console(self, monkeypatch):
        """Under the chat layout the console is a buffer with a nominal
        height of ten thousand. Asking it reported a screen big enough for
        any logo, so the cap never applied and the picture filled the
        window."""
        from wynxo.tui import Transcript

        ui = UI()
        ui.console = Transcript(width=80).console
        assert logo._rows(ui) < 1_000

    def test_the_logo_takes_at_most_half_the_screen(self, monkeypatch):
        import shutil

        monkeypatch.setattr(shutil, "get_terminal_size",
                            lambda _d=None: type("S", (), {"columns": 100,
                                                           "lines": 40})())
        ui = UI()
        assert len(logo.fit(logo.read("wyn"), 90,
                            max_height=logo._rows(ui) // 2)) <= 20


class TestTheSetting:
    def test_the_default_is_the_bundled_logo(self):
        from wynxo.config import Config

        assert Config().logo in logo.available()

    def test_the_command_exists(self):
        from wynxo.cli import COMMANDS

        assert "/logo" in COMMANDS


class TestArtThatAlreadyFits:
    """Line art must not be round-tripped through the ink ramp.

    Resampling substitutes a character of similar weight for every one,
    which is right for a photograph and ruinous for hand-drawn art -- every
    / and \\ comes back as a +.
    """

    ART = "  /\\_/\\\n ( o.o )\n  > ^ <"

    def test_it_is_returned_character_for_character(self):
        assert logo.fit(self.ART, width=40, max_height=20) == [
            "  /\\_/\\", " ( o.o )", "  > ^ <"]

    def test_the_bundled_line_art_survives(self):
        drawn = logo.fit(logo.read("cat"), width=100, max_height=40)
        assert any("/\\" in line for line in drawn)
        assert any("\\_/" in line for line in drawn)

    def test_art_too_wide_is_still_resampled(self):
        """The photograph has to shrink; only art that already fits is
        passed through."""
        lines = logo.fit(logo.read("wyn"), width=50, max_height=40)
        assert max(len(l) for l in lines) <= 50

    def test_squeezing_line_art_does_not_crash(self):
        assert logo.fit(logo.read("cat"), width=6, max_height=40)
