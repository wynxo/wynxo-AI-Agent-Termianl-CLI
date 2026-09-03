"""A line of code wider than the window.

rich wraps at the console edge and resumes at column zero, so the tail of a
long line landed flush left with no gutter -- reading as prose, in the
column the answer uses, in the middle of a code block. At forty columns,
which is an ordinary phone terminal, most of a Python file looks like that.

The two ways out that rich offers are both worse than the problem:
``no_wrap`` truncates, which answers "too long" by throwing characters away,
and ``overflow="fold"`` still resumes at column zero. So the line is cut
here, to the room inside the gutter, and every piece is drawn behind one.
"""

from __future__ import annotations

import io
import re

import pytest

from wynxo.ui import UI

SOURCE = (
    "for attempt in range(CONNECT_ATTEMPTS):\n"
    "        raise SomethingRatherLongIndeed(payload)\n"
    "    # a comment with several words in it that runs on and on\n"
)

WIDTHS = (24, 30, 40, 55, 80, 120)


def draw(width: int, source: str = SOURCE, language: str = "python"):
    ui = UI()
    ui.console.file = io.StringIO()
    ui.console.width = ui.width = width
    ui.code(source, language)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", ui.console.file.getvalue())
    return [line for line in plain.split("\n") if line.strip()]


@pytest.mark.parametrize("width", WIDTHS)
class TestEveryWidth:
    def test_every_row_keeps_the_gutter(self, width):
        for row in draw(width):
            assert row.startswith("│ "), (width, row)

    def test_nothing_runs_off_the_edge(self, width):
        for row in draw(width):
            assert len(row) <= width, (width, row)

    def test_not_one_character_is_lost(self, width):
        """The failure that matters. A line of code with a piece missing is
        a different line of code, and it looks like a real one."""
        recovered = "".join(row[2:] for row in draw(width))
        assert recovered == SOURCE.replace("\n", "")

    def test_indentation_survives(self, width):
        """Leading whitespace is what a Python line means."""
        rows = draw(width)
        assert rows[0][2:].startswith("for ")
        assert any(row[2:].startswith("        raise") for row in rows)


class TestItIsTheSameDrawingEitherWay:
    """A tool printing code and the model streaming it went through two
    different loops, so a long line kept its gutter one way and lost it the
    other."""

    def test_a_block_and_a_streamed_fence_agree(self):
        from wynxo.ui import CodeStreamer

        block = draw(34, "x = averylongidentifier_that_will_not_fit(alpha)\n")

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = 34
        streamer = CodeStreamer(ui)
        streamer.feed("```python\nx = averylongidentifier_that_will_not_fit(alpha)\n```\n")
        streamer.finish()
        plain = re.sub(r"\x1b\[[0-9;]*m", "", ui.console.file.getvalue())
        streamed = [line for line in plain.split("\n") if line.strip()]
        assert block == streamed


class TestWideCharacters:
    def test_a_line_of_wide_characters_still_fits(self):
        """Cells, not characters: a CJK glyph is two cells, so cutting by
        index puts twice the width on the row."""
        rows = draw(30, "x = '" + "漢" * 40 + "'\n")
        from rich.cells import cell_len

        for row in rows:
            assert cell_len(row) <= 30, row

    def test_a_wide_line_is_still_lossless(self):
        source = "x = '" + "漢" * 40 + "'\n"
        recovered = "".join(row[2:] for row in draw(30, source))
        assert recovered == source.replace("\n", "")


class TestTheCeilingStillApplies:
    def test_a_very_long_block_is_cut_and_counted(self):
        rows = draw(80, "".join(f"line_{i} = {i}\n" for i in range(400)))
        assert len(rows) == UI.MAX_CODE_LINES + 1
        assert "more" in rows[-1]
