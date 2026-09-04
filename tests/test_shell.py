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

from wynxo import portrait, raster, shell
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


# -- the painter ------------------------------------------------------------

class TestColourArithmetic:
    def test_a_palette_without_hex_is_left_alone(self):
        """The plain and minimal palettes name ANSI colours, which have no
        numeric value to blend. The painter has to decline rather than
        guess, and the artwork falls back to a fixed reference set."""
        assert not raster.is_hex("bright_magenta")
        assert not raster.is_hex("default")
        assert raster.is_hex("#b47cff")

    def test_mixing_moves_towards_the_second_colour(self):
        black, white = raster.BLACK, raster.WHITE
        assert raster.mix(black, white, 0.0) == black
        assert raster.mix(black, white, 1.0) == white
        assert raster.hexed(raster.mix(black, white, 0.5)) == "#808080"

    def test_shading_is_one_axis(self):
        grey = (128.0, 128.0, 128.0)
        assert raster.shade(grey, -1.0) == raster.BLACK
        assert raster.shade(grey, 1.0) == raster.WHITE
        assert raster.shade(grey, 0.0) == grey


class TestTheCanvas:
    def test_a_span_is_composited_not_stamped(self):
        art = raster.Raster(4, 1, raster.BLACK, alpha=1.0)
        art.span(0, 0, 4, raster.WHITE, 0.5)
        assert art.r[0] == pytest.approx(127.5)
        assert art.a[0] == pytest.approx(1.0)

    def test_nothing_is_painted_outside_the_canvas(self):
        art = raster.Raster(4, 2)
        art.span(9, -10, 40, raster.WHITE)
        art.ellipse(-20, -20, 3, 3, raster.WHITE)
        assert max(art.a) == 0.0

    def test_resampling_averages_the_block_it_covers(self):
        """The two factors differ because a quarter of a terminal cell is
        about twice as tall as it is wide, so the painting is done square
        and squeezed at the end."""
        art = raster.Raster(4, 8, raster.BLACK, alpha=1.0)
        art.span(0, 0, 2, raster.WHITE)
        small = art.resampled(2, 4)
        assert small.w == 2 and small.h == 2
        # The first output pixel covers two columns and four rows, and the
        # white run filled two of those eight.
        assert small.r[0] == pytest.approx(255.0 * 2 / 8)
        assert small.r[1] == pytest.approx(0.0)

    def test_glow_adds_light_rather_than_painting_over(self):
        art = raster.Raster(8, 8, (10.0, 10.0, 10.0), alpha=1.0)
        art.glow(4, 4, 4, 4, (100.0, 0.0, 0.0), 1.0)
        assert art.r[4 * 8 + 4] > 10.0
        assert art.g[4 * 8 + 4] == pytest.approx(10.0)

    def test_the_edges_fade_to_nothing(self):
        """The picture has to end somewhere, and a hard edge makes it read
        as a photograph pasted into the terminal."""
        art = raster.Raster(24, 24, raster.WHITE, alpha=1.0)
        art.vignette(strength=0.5, fade=6)
        assert art.a[0] == pytest.approx(0.0)
        assert art.a[12 * 24 + 12] == pytest.approx(1.0)


class TestItIsSafeInALineOfText:
    def test_it_is_exactly_the_size_the_layout_measured(self):
        for cells in (32, 33, 40, 46, 52):
            drawn = portrait.rows(cells, PALETTES["catboy"])
            assert len(drawn) == portrait.rows_for(cells), cells
            for row in drawn:
                assert row.cell_len == cells, (cells, row.plain)

    def test_a_width_outside_the_range_is_clamped(self):
        """``paint`` clamps rather than producing a picture nobody should
        see, or one so large it is the same drawing with bigger pixels. The
        decision not to draw at all belongs to the layout, which asks
        ``fits`` first."""
        assert portrait.rows(4, PALETTES["catboy"])[0].cell_len \
            == portrait.MIN_CELLS
        assert portrait.rows(400, PALETTES["catboy"])[0].cell_len \
            == portrait.MAX_CELLS

    def test_only_block_glyphs_are_drawn(self):
        for row in portrait.rows(40, PALETTES["catboy"]):
            assert set(row.plain) <= set(raster.GLYPHS), row.plain

    def test_the_glyphs_are_one_cell_and_some_are_ambiguous(self):
        """Documenting the constraint the technique comes with.

        Most of the quadrants are East Asian Width Neutral and always cost
        one cell -- but the five that are halves or the full block (▀ ▄ ▌ ▐
        █) are Ambiguous and may be drawn two cells wide in a CJK locale.
        One is enough to shift every row of the picture, so the locale gate
        stays exactly as it was."""
        for char in raster.GLYPHS:
            assert cell_len(char) == 1, char
        widths = {unicodedata.east_asian_width(c)
                  for c in raster.GLYPHS.strip()}
        assert widths <= {"A", "N"}
        assert "A" in widths, "the gate would have nothing to protect"

    def test_the_picture_dissolves_at_its_edges(self):
        """It ends in the conversation rather than in a rectangle of
        near-black that is not quite the terminal's own."""
        art = portrait.paint(40, portrait.Ink.of(PALETTES["catboy"]))
        assert art.a[0] < 0.05                          # the corner pixel
        assert art.a[(art.h // 2) * art.w + art.w // 2] > 0.9   # and him
        drawn = portrait.rows(40, PALETTES["catboy"])
        for edge in (drawn[0], drawn[-1]):
            assert "█" not in edge.plain, edge.plain
        assert any("█" in row.plain for row in drawn)


# -- the illustration --------------------------------------------------------
#
# What these check is the composition, not the colours. The painting keeps a
# record of which part of the drawing owns each pixel (see ``Raster.label``),
# so "he has two eyes and they are under the fringe" can be asked of the
# drawing itself rather than of a screenshot of it -- which would be a test
# of the palette wearing a test of the anatomy's clothes.


class Anatomy:
    """Where each named part of the drawing ended up."""

    def __init__(self, cells: int, theme: str = "catboy") -> None:
        self.parts = portrait.parts(cells, PALETTES[theme])
        self.width = max(x for pixels in self.parts.values()
                         for x, _ in pixels) + 1

    def __contains__(self, name: str) -> bool:
        return bool(self.parts.get(name))

    def box(self, name: str):
        pixels = self.parts.get(name)
        assert pixels, f"nothing was drawn for {name!r}"
        xs = [x for x, _ in pixels]
        ys = [y for _, y in pixels]
        return min(xs), min(ys), max(xs), max(ys)

    def middle(self, name: str):
        x0, y0, x1, y1 = self.box(name)
        return (x0 + x1) / 2, (y0 + y1) / 2


class TestItIsTheSameCharacterAtEverySize:
    SIZES = (32, 36, 43, 52, 58)

    @pytest.mark.parametrize("cells", SIZES)
    def test_every_part_of_him_is_drawn(self, cells):
        him = Anatomy(cells)
        for part in ("room", "hair", "ears", "face", "features", "fringe",
                     "neck", "hood", "arms", "cheek hand", "keyboard hand",
                     "laptop", "eye left", "eye right"):
            assert part in him, (cells, part)

    @pytest.mark.parametrize("cells", SIZES)
    def test_the_ears_stand_on_top_of_his_head(self, cells):
        """Above the hair, not beside it. Painted before the hair mass, all
        that shows is the tip -- two spikes over the crown, which read as
        antennae rather than as ears."""
        him = Anatomy(cells)
        assert him.box("ears")[1] < him.box("hair")[1], cells
        assert him.box("ears")[3] < him.box("face")[3], cells

    @pytest.mark.parametrize("cells", SIZES)
    def test_he_has_two_eyes_either_side_of_his_nose(self, cells):
        him = Anatomy(cells)
        left, right = him.middle("eye left")[0], him.middle("eye right")[0]
        centre = him.middle("face")[0]
        assert left < centre < right, (cells, left, centre, right)
        assert him.box("eye left")[2] < him.box("eye right")[0], cells

    @pytest.mark.parametrize("cells", SIZES)
    def test_the_eyes_sit_below_a_forehead(self, cells):
        """The single thing that keeps this a person rather than an animal
        at terminal resolution. Eyes in the middle of a circle read as a
        cat's face however human the rest of the drawing is."""
        him = Anatomy(cells)
        _, top, _, bottom = him.box("face")
        eyes = him.middle("eye left")[1]
        assert eyes > top + (bottom - top) * 0.25, cells
        assert eyes < top + (bottom - top) * 0.65, cells

    @pytest.mark.parametrize("cells", SIZES)
    def test_the_laptop_is_in_front_of_him_and_below_his_chin(self, cells):
        """Between him and the viewer. Behind him it is a prop standing on a
        shelf, and the pose stops being someone sitting at a desk."""
        him = Anatomy(cells)
        assert him.box("laptop")[1] > him.box("face")[3], cells
        assert him.box("laptop")[1] > him.box("neck")[1], cells

    @pytest.mark.parametrize("cells", SIZES)
    def test_one_hand_holds_his_face_and_the_other_is_on_the_keys(self, cells):
        """A frame with one hand in it is a different pose from this one."""
        him = Anatomy(cells)
        cheek, keys = him.box("cheek hand"), him.box("keyboard hand")
        centre = him.middle("face")[0]
        assert cheek[2] < centre, cells             # at the face, to one side
        assert cheek[1] < him.box("face")[3], cells
        assert keys[0] > centre, cells              # and the other on the desk
        assert keys[1] > him.box("laptop")[1], cells

    @pytest.mark.parametrize("cells", SIZES)
    def test_the_fringe_is_over_the_face_and_not_over_the_eyes(self, cells):
        him = Anatomy(cells)
        assert him.box("fringe")[1] < him.box("face")[1], cells
        assert him.box("fringe")[3] < him.middle("eye left")[1] + 4, cells

    def test_the_drawing_is_deterministic(self):
        """It is repainted whenever the terminal changes width. A bookshelf
        that reshuffles itself when the window moves is a distraction rather
        than a room."""
        first = [row.plain for row in portrait.rows(40, PALETTES["catboy"])]
        portrait._painted.cache_clear()
        second = [row.plain for row in portrait.rows(40, PALETTES["catboy"])]
        assert first == second


class TestItKeepsItsProportions:
    def test_a_width_chosen_for_a_height_actually_fits_it(self):
        """The property the layout leans on. ``cells_for`` is asked how wide
        a picture may be given the rows available, and the answer has to be
        one that ``rows_for`` will not then exceed."""
        for rows in range(8, 60):
            cells = portrait.cells_for(rows)
            assert portrait.rows_for(cells) <= rows, (rows, cells)

    @pytest.mark.parametrize("cells", (32, 40, 46, 52, 58))
    def test_the_aspect_never_drifts(self, cells):
        """The composition is only isotropic on the shape it was painted at.
        Left free, a wider box gives him an oval head and horns for ears."""
        art = portrait.paint(cells, portrait.Ink.of(PALETTES["catboy"]))
        assert abs(art.w / art.h
                   - portrait.NATIVE_CELLS / portrait.NATIVE_PIXELS) < 0.03

    @pytest.mark.parametrize("cells", (32, 43, 58))
    def test_his_head_is_the_same_shape_at_every_size(self, cells):
        """The proportion that matters most, measured rather than trusted:
        a face wider than it is tall is a different character."""
        him = Anatomy(cells)
        x0, y0, x1, y1 = him.box("face")
        # The record is kept on the painting's own square-pixel grid, before
        # it is squeezed onto cells, so this is a true ratio.
        assert 1.25 < (y1 - y0) / (x1 - x0) < 1.75, cells


class TestTheColourComesFromTheTheme:
    def test_every_shade_is_a_colour(self):
        ink = portrait.Ink.of(PALETTES["catboy"])
        for name, value in ink.__dict__.items():
            assert len(value) == 3, name
            assert all(0.0 <= c <= 255.0 for c in value), (name, value)

    @pytest.mark.parametrize("theme", names())
    def test_every_palette_produces_a_drawable_scene(self, theme):
        drawn = portrait.rows(40, resolve(theme))
        assert len(drawn) == portrait.rows_for(40)
        assert any(str(span.style) for row in drawn for span in row.spans), theme

    def test_a_sixteen_colour_palette_falls_back_to_the_reference(self):
        """The plain palette has nothing to blend, so the painting cannot be
        derived from it. Drawing in a fixed violet beats not drawing: a rough
        picture is still the character, and a missing one is a hole in the
        layout."""
        assert portrait.Ink.of(PALETTES["plain"]) \
            == portrait.Ink.of(PALETTES["minimal"])

    def test_two_themes_do_not_paint_the_same_picture(self):
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
