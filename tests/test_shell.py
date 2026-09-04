"""The composed screen: header, rail, illustration, cards, status bar.

The shell is the one part of the interface that is a *layout* rather than a
line of output, and layouts fail in ways single lines do not: a column one
cell too wide wraps every row of the picture beside it, a picture measured
from the width overflows a short terminal, and a piece that gives way at
one size but not another leaves a hole. These pin the arithmetic.

Nothing here asserts on colour. Which violet the hair is drawn in is a fact
about the palette and it is tested where palettes are; whether there is a
head with two eyes under a pair of ears is a fact about the drawing.
"""

from __future__ import annotations

import io
import re
import unicodedata

import pytest
from rich.cells import cell_len

from wynxo import pixelart, portrait, shell
from wynxo.theme import PALETTES, names, resolve
from wynxo.ui import UI, Glyphs


def drawn(ui: UI, renderable, width: int | None = None) -> list[str]:
    """A renderable as the lines it puts on screen, without the escapes."""
    width = width or ui.width
    buffer = io.StringIO()
    ui.console.file = buffer
    ui.console._width = width
    ui.console.print(renderable)
    return [re.sub(r"\x1b\[[0-9;]*m", "", line)
            for line in buffer.getvalue().splitlines()]


def screen(width: int, height: int = 48, theme: str = "catboy") -> UI:
    ui = UI(theme=theme)
    ui.g = Glyphs(True)
    ui.width = ui.console.width = width
    ui.narrow = width < 60
    shell.terminal_height = lambda default=24: height
    return ui


@pytest.fixture(autouse=True)
def _restore_height():
    from wynxo.platforms import terminal_height

    yield
    shell.terminal_height = terminal_height


# -- the pixel engine --------------------------------------------------------

class TestTheDitherMatrix:
    def test_it_is_a_permutation_of_every_level(self):
        """The failure this is really for. A hand-written 8x8 that merely
        looked plausible clustered into 4x4 blocks, and every dithered
        surface in the artwork came out in visible squares -- which reads
        as a rendering bug rather than as texture."""
        values = sorted(v for row in pixelart._BAYER for v in row)
        assert values == list(range(len(pixelart._BAYER) ** 2))

    def test_a_density_produces_about_that_many_pixels(self):
        for wanted in (0.1, 0.25, 0.5, 0.75):
            art = pixelart.Canvas(64, 64)
            art.dither(0, 0, 1, 1, "x", wanted)
            lit = sum(row.count("x") for row in art.px) / (64 * 64)
            assert abs(lit - wanted) < 0.03, wanted


class TestColourArithmetic:
    def test_a_palette_without_hex_is_left_alone(self):
        """The plain and minimal palettes name ANSI colours, which have no
        numeric value to blend. Mixing has to decline rather than guess."""
        assert not pixelart.is_hex("bright_magenta")
        assert pixelart.mix("bright_magenta", "#ffffff", 0.5) == "bright_magenta"
        assert pixelart.darken("default", 0.5) == "default"

    def test_mixing_moves_towards_the_second_colour(self):
        assert pixelart.mix("#000000", "#ffffff", 0.0) == "#000000"
        assert pixelart.mix("#000000", "#ffffff", 1.0) == "#ffffff"
        assert pixelart.mix("#000000", "#ffffff", 0.5) == "#808080"


class TestTheCanvasIsSafeInALineOfText:
    def test_a_canvas_is_exactly_half_its_pixels_tall(self):
        for cells in (32, 33, 40, 46, 60):
            art = portrait.draw(cells)
            rows = art.rows(portrait.Ink.of(PALETTES["catboy"]).style)
            assert len(rows) == art.height // 2
            for row in rows:
                assert row.cell_len == cells, (cells, row.plain)

    def test_a_width_below_the_minimum_is_raised_rather_than_honoured(self):
        """``draw`` clamps rather than producing a picture nobody should
        see. The decision not to draw at all belongs to the layout, which
        asks ``fits`` first."""
        assert portrait.draw(4).width == portrait.MIN_CELLS

    def test_only_the_four_glyphs_are_drawn(self):
        for row in portrait.rows(40, PALETTES["catboy"]):
            assert set(row.plain) <= set(pixelart.GLYPHS), row.plain

    def test_the_glyphs_are_one_cell_each(self):
        for char in pixelart.GLYPHS:
            assert cell_len(char) == 1, char
        assert {unicodedata.east_asian_width(c)
                for c in pixelart.GLYPHS.strip()} == {"A"}

    def test_transparent_pixels_paint_no_background(self):
        """A picture on the conversation, not a coloured rectangle sitting
        on it: only a cell where two different opaque colours stack may set
        a background at all."""
        for row in portrait.rows(40, PALETTES["catboy"]):
            for span in row.spans:
                if "on " in str(span.style):
                    assert " " not in row.plain[span.start:span.end]


# -- the illustration --------------------------------------------------------

def ink_map(cells: int) -> list[str]:
    return ["".join(row) for row in portrait.draw(cells).px]


class TestItIsTheSameCharacterAtEverySize:
    SIZES = (32, 36, 40, 43, 46, 52, 60)

    @pytest.mark.parametrize("cells", SIZES)
    def test_he_has_ears_over_his_head(self, cells):
        """Above the hair, not beside it. The ears were drawn before the
        hair mass once and all that survived was the tip -- two spikes over
        the crown, which read as antennae rather than as ears."""
        rows = ink_map(cells)
        top = rows[: len(rows) // 6]
        assert any("e" in row for row in top), cells
        assert any("k" in row for row in top), cells

    @pytest.mark.parametrize("cells", SIZES)
    def test_he_has_two_eyes_side_by_side(self, cells):
        """Two of them, one either side of the middle of his face.

        Asserted as "iris on both sides and none in the bridge between"
        rather than by counting runs: an iris has a pupil in the middle of
        it, so a single eye is already several runs, and how many depends
        on where the rounding lands at each size."""
        rows = ink_map(cells)
        columns = {x for row in rows
                   for x, ink in enumerate(row) if ink == "i"}
        middle = cells / 2
        assert any(x < middle - 1 for x in columns), (cells, columns)
        assert any(x > middle + 1 for x in columns), (cells, columns)
        assert not any(abs(x - middle) < 1 for x in columns), (cells, columns)

    @pytest.mark.parametrize("cells", SIZES)
    def test_the_eyes_sit_below_a_forehead(self, cells):
        """The single thing that keeps this a person rather than an animal
        at terminal resolution. Eyes in the middle of a circle read as a
        cat's face however human the rest of the drawing is."""
        rows = ink_map(cells)
        eyes = min(y for y, row in enumerate(rows) if "i" in row)
        head = min(y for y, row in enumerate(rows) if "k" in row)
        assert eyes - head > len(rows) * 0.12, (cells, head, eyes)

    @pytest.mark.parametrize("cells", SIZES)
    def test_he_is_at_a_laptop_on_a_desk(self, cells):
        rows = ink_map(cells)
        body = "".join(rows)
        for ink, what in (("l", "laptop"), ("L", "a lit screen edge"),
                          ("t", "a desk"), ("h", "a hoodie"),
                          ("n", "skin")):
            assert ink in body, f"{cells}: no {what}"

    @pytest.mark.parametrize("cells", SIZES)
    def test_the_laptop_is_in_front_of_him(self, cells):
        """Between him and the viewer. Behind him it is a prop standing on
        a shelf, and the pose stops being someone sitting at a desk."""
        rows = ink_map(cells)
        lowest_face = max(y for y, row in enumerate(rows) if "i" in row)
        lid = min(y for y, row in enumerate(rows) if "l" in row)
        assert lid > lowest_face, cells

    @pytest.mark.parametrize("cells", SIZES)
    def test_both_hands_are_somewhere(self, cells):
        """One at the keyboard and one holding his face up. A frame with a
        single hand is a different pose from the one this is."""
        rows = ink_map(cells)
        half = cells // 2
        skin = [(y, x) for y, row in enumerate(rows)
                for x, ink in enumerate(row) if ink in "nd"]
        below = [(y, x) for y, x in skin if y > len(rows) * 0.5]
        assert any(x < half for _, x in below), cells
        assert any(x > half for _, x in below), cells

    def test_the_drawing_is_deterministic(self):
        """It is redrawn on every resize. A bookshelf that reshuffles when
        the window changes width is a distraction rather than a room."""
        assert ink_map(40) == ink_map(40)


class TestItKeepsItsProportions:
    def test_a_width_chosen_for_a_height_actually_fits_it(self):
        """The property the layout leans on. ``cells_for`` is asked how
        wide a picture may be given the rows available, and the answer has
        to be one that ``rows_for`` will not then exceed."""
        for rows in range(8, 60):
            cells = portrait.cells_for(rows)
            assert portrait.rows_for(cells) <= rows, (rows, cells)

    def test_the_aspect_never_drifts(self):
        """Normalised coordinates are only isotropic on the shape they were
        composed at. Left free, a wider box gave him an oval head and horns
        for ears."""
        for cells in (32, 40, 46, 60, 80):
            art = portrait.draw(cells)
            assert abs(art.height / art.width - portrait.ASPECT) < 0.05, cells


class TestTheColourComesFromTheTheme:
    def test_every_ink_code_names_a_shade(self):
        ink = portrait.Ink.of(PALETTES["catboy"])
        for code, field in portrait._CODES.items():
            assert getattr(ink, field), code
            assert ink.style(code)
        assert ink.style(" ") == ""

    @pytest.mark.parametrize("theme", names())
    def test_every_palette_produces_a_drawable_scene(self, theme):
        rows = portrait.rows(40, resolve(theme))
        assert len(rows) == portrait.rows_for(40)
        assert any(str(span.style) for row in rows for span in row.spans), theme

    def test_a_sixteen_colour_palette_degrades_to_roles(self):
        """The plain palette has nothing to blend, so the artwork has to
        fall back to flat roles rather than raise."""
        ink = portrait.Ink.of(PALETTES["plain"])
        assert not any(str(v).startswith("#") for v in ink.__dict__.values())

    def test_two_themes_do_not_draw_the_same_picture(self):
        catboy = {str(s.style) for r in portrait.rows(40, PALETTES["catboy"])
                  for s in r.spans}
        ember = {str(s.style) for r in portrait.rows(40, PALETTES["ember"])
                 for s in r.spans}
        assert catboy != ember


class TestItGivesWayRatherThanDegrading:
    def test_no_picture_where_half_blocks_will_not_render(self):
        assert portrait.fits(120, unicode_ok=False) is False

    def test_no_picture_below_the_size_a_face_survives(self):
        assert portrait.fits(portrait.MIN_CELLS - 1, unicode_ok=True) is False
        assert portrait.fits(portrait.MIN_CELLS, unicode_ok=True) is True

    def test_no_picture_where_ambiguous_width_draws_wide(self, monkeypatch):
        """A CJK locale may draw ▀ as two cells, which would make every row
        of the picture twice the width the layout measured for it."""
        monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
        assert portrait.fits(120, unicode_ok=True) is False
        monkeypatch.setenv("LC_ALL", "en_GB.UTF-8")
        assert portrait.fits(120, unicode_ok=True) is True


# -- the composed screen -----------------------------------------------------

class TestTheScreenFitsTheTerminal:
    WIDTHS = (60, 66, 72, 76, 80, 90, 100, 110, 120, 160, 200)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_nothing_is_ever_wider_than_the_terminal(self, width):
        """One cell over and every row of the illustration wraps, which
        does not look like a narrower picture -- it looks like half a cat
        on the next line."""
        ui = screen(width)
        for line in drawn(ui, shell.home(ui, model="m", version="v0.1.0")):
            assert cell_len(line) <= width, (width, line)

    @pytest.mark.parametrize("height", (20, 24, 30, 40, 60))
    def test_a_short_terminal_gets_a_smaller_picture_or_none(self, height):
        """Sized from the rows, not the columns. Measured from the width
        instead, a hero image pushes the input box it is introducing off
        the bottom of the screen."""
        ui = screen(140, height=height)
        art = shell._illustration(ui, True, shell._column_rows(
            shell.GREETING, shell.DEFAULT_SUGGESTIONS))
        if art is not None:
            assert art[2] <= max(0, height - shell.HEIGHT_OVERHEAD), height

    def test_the_picture_is_dropped_rather_than_shrunk(self):
        """Below the size a face survives at, there is no honest small
        version -- so the conversation gets the whole width instead."""
        ui = screen(64)
        assert shell._illustration(ui, False, 18) is None

    def test_the_rail_gives_way_before_the_character_does(self):
        """The rail says what the application is made of; the character is
        what it is. Fifteen columns is the difference between drawing him
        and dropping him, and he outranks it."""
        ui = screen(80)
        assert 80 < shell.RAIL_FROM
        assert shell._illustration(ui, False, 18) is not None


class TestTheHeader:
    def test_the_wordmark_is_three_rows_of_pixels(self):
        ui = screen(100)
        mark = shell.wordmark(ui.palette)
        lines = mark.plain.splitlines()
        assert len(lines) == shell.WORDMARK_ROWS
        assert {cell_len(line) for line in lines} == {shell.WORDMARK_CELLS}

    def test_it_carries_the_name_the_tagline_and_the_version(self):
        ui = screen(110)
        body = "\n".join(drawn(ui, shell.header(ui, "v9.9.9")))
        for line in shell.TAGLINE:
            assert line in body
        assert "v9.9.9" in body

    def test_a_terminal_without_pixels_still_gets_a_name(self):
        ui = screen(100)
        ui.g = Glyphs(False)
        body = "\n".join(drawn(ui, shell.header(ui, "v1")))
        assert "wynxo" in body
        assert not set(body) & set("▀▄█")


class TestTheRail:
    def test_exactly_one_item_is_marked(self):
        ui = screen(100)
        lines = drawn(ui, shell.rail(ui, "tools"), width=shell.RAIL_CELLS)
        marked = [line for line in lines if line.startswith("│")]
        assert len(marked) == 1
        assert "tools" in marked[0]

    def test_every_section_is_listed(self):
        ui = screen(100)
        body = "\n".join(drawn(ui, shell.rail(ui), width=shell.RAIL_CELLS))
        for name, _, _ in shell.NAV:
            assert name in body

    def test_it_fits_the_columns_it_claims(self):
        ui = screen(100)
        for line in drawn(ui, shell.rail(ui), width=shell.RAIL_CELLS):
            assert cell_len(line) <= shell.RAIL_CELLS, line

    def test_an_ascii_terminal_gets_ascii_icons(self):
        ui = screen(100)
        ui.g = Glyphs(False)
        body = "\n".join(drawn(ui, shell.rail(ui), width=shell.RAIL_CELLS))
        assert body.isascii(), body


class TestTheConversationPieces:
    def test_the_state_chip_hugs_its_label(self):
        """A bubble as wide as the column is not a bubble, it is a banner."""
        ui = screen(120)
        lines = drawn(ui, shell.chip(ui, "thinking..."))
        assert len(lines) == 2
        assert cell_len(lines[0]) < 30
        assert "thinking..." in lines[0]

    def test_a_resting_state_takes_no_ellipsis(self):
        """"thinking..." says something is happening. "ready..." says
        nothing is, at length."""
        assert shell.state_label("thinking") == "thinking..."
        assert shell.state_label("ready") == "ready"
        ui = screen(110)
        body = "\n".join(drawn(ui, shell.home(ui, model="m", version="v1")))
        assert "ready" in body and "ready..." not in body

    def test_a_user_message_is_sized_to_what_was_said(self):
        ui = screen(120)
        lines = drawn(ui, shell.user_message(ui, "hi"))
        assert cell_len(lines[0]) < 20, lines
        assert ui.g.caret in lines[1] and "hi" in lines[1]

    def test_the_suggestions_are_a_list_not_a_panel(self):
        """It sits between two outlined things, and a third border between
        them turns the column into a stack of boxes."""
        ui = screen(80)
        body = drawn(ui, shell.suggestions(ui, shell.DEFAULT_SUGGESTIONS))
        assert body[0].strip() == "suggestions:"
        assert not any(set(line) & set("╭╮╰╯│") for line in body)
        for command, what in shell.DEFAULT_SUGGESTIONS:
            assert any(command in line and what in line for line in body)

    def test_the_input_box_is_the_last_thing_on_the_screen(self):
        """It is where the real composer opens, so nothing may sit under
        it -- the caret prompt_toolkit draws lands on the next row."""
        ui = screen(110)
        lines = drawn(ui, shell.home(ui, model="m", version="v1"))
        body = [line for line in lines if line.strip()]
        placeholder = max(i for i, line in enumerate(body)
                          if "Type a message" in line)
        after = body[placeholder + 1:]
        # The box's own closing edge, and the three rows of the status bar.
        assert len(after) == 4, after
        assert after[0].strip().startswith(ui.g.bl)
        assert "model:" in after[2] and after[3].strip().endswith(ui.g.br)


class TestTheStatusBar:
    def test_it_says_which_model_what_mode_and_what_the_companion_is_doing(self):
        ui = screen(120)
        line = "".join(drawn(ui, shell.status_bar(
            ui, shell.Metrics(model="qwen3:4b", mode="agent",
                              companion="thinking"))))
        assert "model: qwen3:4b" in line
        assert "mode: agent" in line
        assert "companion: thinking" in line

    def test_the_state_marks_sit_on_the_right(self):
        ui = screen(120)
        lines = drawn(ui, shell.status_bar(ui, shell.Metrics(model="m")))
        body = [line for line in lines if "model:" in line][0]
        assert body.rstrip("│ ").endswith(ui.g.gear)


class TestTheShellGivesWayWhereItCannotBeDrawn:
    def test_a_redirected_stream_gets_the_one_line_banner(self, capsys):
        ui = UI()
        ui.width = 100
        assert ui.can_draw_shell() is False
        ui.home("qwen3:4b", "/tmp/p")
        lines = [line for line in capsys.readouterr().out.splitlines()
                 if line.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("wynxo") and "qwen3:4b" in lines[0]

    def test_a_phone_width_terminal_gets_the_banner_too(self, monkeypatch):
        ui = UI()
        ui.narrow = True
        monkeypatch.setattr(type(ui.console), "is_terminal",
                            property(lambda self: True))
        monkeypatch.setattr("wynxo.platforms.is_dumb_terminal", lambda: False)
        assert ui.can_draw_shell() is False

    def test_a_dumb_terminal_gets_the_banner_too(self, monkeypatch):
        ui = UI()
        ui.narrow = False
        monkeypatch.setattr(type(ui.console), "is_terminal",
                            property(lambda self: True))
        monkeypatch.setattr("wynxo.platforms.is_dumb_terminal", lambda: True)
        assert ui.can_draw_shell() is False
