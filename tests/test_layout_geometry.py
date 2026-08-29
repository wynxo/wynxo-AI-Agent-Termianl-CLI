"""The layout contract, measured on the rows prompt_toolkit actually gave out.

Nothing here asserts a constant. Every number below is read back from
``Window.render_info`` after the real layout has been written to a real
Screen at a real size, because the bug this file exists to prevent was a
composer that measured correctly and was then handed extra rows by the
allocator anyway.

The contract:

    HEADER      exactly one row, at the top
    TRANSCRIPT  every row the others did not claim
    RULE        exactly one row
    COMPOSER    its own content, capped, sitting on the footer
    FOOTER      exactly one row, on the last row of the screen

The mechanism it depends on is ``HSplit._divide_heights``: after every child
reaches its ``preferred``, the allocator keeps handing rows to any child
whose ``max`` is larger, cycling by weight, until the screen is full. So the
contract holds only while exactly one child reports ``max > preferred``.
``test_only_the_transcript_can_absorb_space`` is the one that would catch a
regression at the source.
"""

from __future__ import annotations

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput

from wynxo.layout import ChatLayout

SIZES = [(60, 20), (80, 24), (100, 30), (120, 40)]


class _Fixed(DummyOutput):
    def __init__(self, columns: int, rows: int):
        self._size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self._size


def measure(columns: int, rows: int, *, text: str = "", lines: int = 0,
            overlay: list[str] | None = None) -> tuple[ChatLayout, dict]:
    """Render the production layout and read back what each region got.

    Runs inside the caller's event loop: ``BufferControl.create_content``
    loads the buffer's history, which asks the application for a background
    task and therefore for a running loop.
    """
    ui = ChatLayout(width=columns, height=rows,
                    overlay=(lambda: overlay) if overlay is not None else None)
    ui.app.output = _Fixed(columns, rows)
    ui.buffer.text = text
    for i in range(lines):
        ui.transcript.console.print(f"transcript line {i}")
    ui.transcript.drain()

    screen = Screen(initial_width=columns, initial_height=rows)
    with set_app(ui.app):
        ui.app.layout.container.write_to_screen(
            screen, MouseHandlers(), WritePosition(0, 0, columns, rows),
            "", False, None)

    named = {id(window): name for name, window in ui.regions().items()}
    geometry = {}
    for window in ui.app.layout.walk():
        info = getattr(window, "render_info", None)
        if info is None or id(window) not in named:
            continue
        geometry[named[id(window)]] = (info._y_offset, info.window_height)
    return ui, geometry


def assert_contract(geometry: dict, rows: int) -> None:
    """Every region stacked, in order, filling the screen exactly once."""
    order = ["header", "transcript", "rule", "composer", "footer"]
    cursor = 0
    for name in order:
        top, height = geometry[name]
        assert top == cursor, f"{name} starts at {top}, expected {cursor}"
        assert height >= 0, f"{name} has negative height {height}"
        cursor += height
    assert cursor == rows, f"regions cover {cursor} rows, screen has {rows}"


@pytest.mark.parametrize("columns,rows", SIZES)
class TestTheContractHolds:
    async def test_an_empty_conversation_still_seats_the_composer(
            self, columns, rows):
        """The unused space belongs to the transcript, not to the composer.

        This is the failure from the bug report: with almost nothing said,
        the composer appeared as a tall empty box under the header.
        """
        _, geometry = measure(columns, rows)
        assert_contract(geometry, rows)
        assert geometry["composer"][1] == 1, "an empty composer is one row"
        assert geometry["transcript"][1] == rows - 4

    async def test_a_short_exchange_leaves_the_space_above(self, columns, rows):
        _, geometry = measure(columns, rows, lines=3)
        assert_contract(geometry, rows)
        assert geometry["composer"][1] == 1
        assert geometry["transcript"][1] == rows - 4

    async def test_a_huge_response_never_pushes_the_composer_down(
            self, columns, rows):
        _, geometry = measure(columns, rows, lines=5000)
        assert_contract(geometry, rows)
        assert geometry["composer"][1] == 1
        assert geometry["footer"][0] == rows - 1

    async def test_multiline_input_grows_upward(self, columns, rows):
        """The bottom edge is anchored; the composer takes rows from the
        transcript above it rather than from the footer below."""
        _, one = measure(columns, rows, lines=20)
        _, three = measure(columns, rows, text="one\ntwo\nthree", lines=20)
        assert_contract(three, rows)
        assert three["composer"][1] == 3
        assert three["footer"] == one["footer"], "the footer must not move"
        assert three["transcript"][1] == one["transcript"][1] - 2

    async def test_the_composer_is_bounded_and_then_scrolls_inside(
            self, columns, rows):
        """A pasted essay must not push the footer off the screen."""
        _, geometry = measure(columns, rows, text="x\n" * 400, lines=10)
        assert_contract(geometry, rows)
        assert geometry["composer"][1] == ChatLayout.COMPOSER_MAX_ROWS
        assert geometry["footer"][0] == rows - 1
        assert geometry["transcript"][1] > 0, "the transcript keeps some rows"


@pytest.mark.parametrize("columns,rows", SIZES)
class TestOverlaysAreNotStructural:
    """The plan, the pet and toasts are Floats. A Float reports no height
    into the vertical split, so it cannot move anything however tall it is."""

    @pytest.mark.parametrize("overlay_rows", [0, 3, 12, 60])
    async def test_an_overlay_never_moves_the_composer(
            self, columns, rows, overlay_rows):
        _, base = measure(columns, rows, text="a\nb", lines=40)
        _, with_overlay = measure(
            columns, rows, text="a\nb", lines=40,
            overlay=[f"[ ] task {i}" for i in range(overlay_rows)])
        assert with_overlay == base, (
            f"a {overlay_rows}-row overlay changed the structural geometry")


class TestTheAllocatorCannotInflateAnything:
    async def test_only_the_transcript_can_absorb_space(self):
        """The root cause, asserted directly.

        ``Dimension(min=1, max=8)`` reports preferred=1 and max=8, and the
        allocator's third pass grows any such child toward its max until the
        screen is full -- which is how the composer became a tall empty box.
        Exactly one child in the tree may report max > preferred.
        """
        ui, _ = measure(80, 24)
        absorbers = []
        for name, window in ui.regions().items():
            dimension = window.height
            if callable(dimension):
                dimension = dimension()
            if not isinstance(dimension, Dimension):
                continue
            if dimension.max > dimension.preferred:
                absorbers.append(name)
        assert absorbers == ["transcript"], (
            f"these regions can absorb spare rows: {absorbers}. Exactly one "
            "may, and it must be the transcript.")

    async def test_the_composer_reports_an_exact_dimension(self):
        """Not a range. A range is what the allocator inflates."""
        ui, _ = measure(80, 24, text="one\ntwo")
        dimension = ui._composer_window.height()
        assert dimension.min == dimension.preferred == dimension.max == 2

    async def test_resizing_keeps_the_composer_on_the_bottom(self):
        """Every size, idle and mid-input, including the narrow ones."""
        for columns, rows in SIZES + [(40, 10), (200, 60), (72, 21)]:
            for text in ("", "one", "one\ntwo\nthree"):
                _, geometry = measure(columns, rows, text=text, lines=30)
                assert_contract(geometry, rows)
                assert geometry["footer"][0] + geometry["footer"][1] == rows


class TestScrollingAndFollow:
    async def test_new_output_is_followed_when_at_the_bottom(self):
        ui, _ = measure(80, 24, lines=10)
        assert ui.following()
        ui.transcript.console.print("newest")
        ui.transcript.drain()
        assert ui.following(), "at the bottom, the view follows new output"
        assert "newest" in ui.transcript.visible(ui.transcript_rows())[-1]

    async def test_scrolling_back_stops_the_yank_to_bottom(self):
        ui, _ = measure(80, 24, lines=200)
        ui.scroll_by(30)
        before = ui.transcript.visible(ui.transcript_rows(), ui.scroll)
        ui.transcript.console.print("something new arrived")
        ui.transcript.drain()
        after = ui.transcript.visible(ui.transcript_rows(), ui.scroll)
        assert not ui.following()
        assert after == before, "the reader's place must be held"
        assert ui.unread > 0

    async def test_returning_to_the_bottom_resumes_following(self):
        ui, _ = measure(80, 24, lines=200)
        ui.scroll_by(30)
        ui.to_bottom()
        assert ui.following()
        assert ui.unread == 0

    async def test_scrolling_cannot_run_past_the_ends(self):
        ui, _ = measure(80, 24, lines=50)
        ui.scroll_by(10_000)
        assert ui.scroll == ui.transcript.max_offset(ui.transcript_rows())
        ui.scroll_by(-10_000)
        assert ui.scroll == 0


class TestTheOverlayFitsWhatItHolds:
    async def test_the_float_is_wide_enough_for_the_plan_panel(self):
        """Two independent widths that must agree.

        The plan is rendered by corner.py at its own inner width; the Float
        that carries it is sized by the layout. When the Float was the
        narrower of the two, every panel lost its right border and looked
        broken rather than narrow.
        """
        from rich.text import Text

        from wynxo.corner import CornerPlan, parse
        from wynxo.ui import UI

        panel = CornerPlan(UI())
        panel.items = parse("[x] read the parser\n[>] add the retry path")
        widest = max(len(Text.from_markup(line).plain) for line in panel.lines())
        assert widest <= ChatLayout.TODO_WIDTH, (
            f"the plan panel renders {widest} cells wide but the overlay "
            f"Float is only {ChatLayout.TODO_WIDTH}; the border is clipped")
