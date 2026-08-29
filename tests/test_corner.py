"""The plan pinned to the top-right corner.

The panel lives in rows held out of the terminal's scrolling region, so the
things worth pinning down are the ones that outlive the process: the region
must always be released, and the number of rows reserved must match the
number of rows drawn.
"""

from __future__ import annotations

import io

import pytest

from wynxo import corner
from wynxo.corner import CornerPlan, Item, parse
from wynxo.ui import UI, Glyphs


def _ui(unicode_ok: bool = True, width: int = 100) -> tuple[UI, io.StringIO]:
    ui = UI()
    ui.g = Glyphs(unicode_ok)
    ui.width = width
    stream = io.StringIO()
    ui.console.file = stream
    ui.console._width = width
    return ui, stream


PLAN = "[x] read the files\n[>] edit the parser\n[ ] run the tests"


class TestParsing:
    def test_the_three_states_are_recognised(self):
        assert parse(PLAN) == [
            Item("read the files", "done"),
            Item("edit the parser", "active"),
            Item("run the tests", "todo"),
        ]

    def test_blank_lines_are_dropped(self):
        assert parse("[ ] one\n\n\n[ ] two") == [
            Item("one", "todo"), Item("two", "todo")]

    def test_an_unmarked_line_is_still_a_step(self):
        """A model that forgets the checkbox should not lose its step."""
        assert parse("just do the thing") == [Item("just do the thing", "todo")]

    def test_nothing_parses_to_nothing(self):
        assert parse("") == []
        assert parse("   \n  \n") == []


class TestTheReservedRows:
    """The count that decides how many rows are held out of the scroll."""

    @pytest.mark.parametrize("count", range(1, 14))
    def test_height_matches_what_is_drawn(self, count):
        """A panel taller than its region loses its last row to the scroll;
        shorter, and it leaves a dead band the transcript cannot use."""
        ui, _ = _ui()
        panel = CornerPlan(ui)
        panel.items = parse("\n".join(f"[ ] step {i}" for i in range(count)))
        assert panel.panel_height() == len(panel.lines())

    def test_an_empty_plan_reserves_nothing(self):
        ui, _ = _ui()
        assert CornerPlan(ui).panel_height() == 0

    def test_a_long_plan_is_summarised_rather_than_drawn_whole(self):
        ui, _ = _ui()
        panel = CornerPlan(ui)
        panel.items = parse("\n".join(f"[ ] step {i}" for i in range(30)))
        assert panel.panel_height() <= corner.MAX_ITEMS + 4
        assert any("more" in line for line in panel.lines())


class TestReleasingTheRegion:
    """A scrolling region outlives the process that set it. If wynxo exits
    without releasing one, the user's shell keeps the frozen top rows."""

    def test_release_emits_the_reset(self):
        ui, stream = _ui()
        panel = CornerPlan(ui)
        panel._released = False
        panel.release()
        assert "\x1b[r" in stream.getvalue()

    def test_release_is_idempotent(self):
        ui, stream = _ui()
        panel = CornerPlan(ui)
        panel._released = False
        panel.release()
        first = stream.getvalue()
        panel.release()
        panel.release()
        assert stream.getvalue() == first

    def test_a_dead_stream_does_not_take_the_turn_down(self):
        """Losing the panel is not worth raising through the agent loop."""
        ui, stream = _ui()
        stream.close()
        panel = CornerPlan(ui)
        panel.armed = True
        panel._write("anything")       # must not raise
        assert panel.armed is False

    def test_clearing_wipes_and_releases(self):
        ui, stream = _ui()
        panel = CornerPlan(ui)
        panel.items = parse(PLAN)
        panel.armed = True
        panel._released = False
        panel._rows = panel.panel_height()
        panel.clear()
        assert panel.items == []
        assert "\x1b[r" in stream.getvalue()
        assert panel.armed is False


class TestWhenItIsUsedAtAll:
    def test_a_dumb_terminal_never_gets_one(self, monkeypatch):
        ui, _ = _ui()
        monkeypatch.setattr(corner, "is_dumb_terminal", lambda: True)
        assert CornerPlan(ui).usable() is False

    def test_a_short_window_keeps_every_row_for_the_conversation(self, monkeypatch):
        ui, _ = _ui()
        monkeypatch.setattr(corner, "is_dumb_terminal", lambda: False)
        monkeypatch.setattr(corner, "terminal_height", lambda: 8)
        monkeypatch.setattr(corner, "terminal_width", lambda: 100)
        assert CornerPlan(ui).usable() is False

    def test_a_narrow_window_would_crowd_the_transcript(self, monkeypatch):
        ui, _ = _ui()
        monkeypatch.setattr(corner, "is_dumb_terminal", lambda: False)
        monkeypatch.setattr(corner, "terminal_height", lambda: 40)
        monkeypatch.setattr(corner, "terminal_width", lambda: 40)
        assert CornerPlan(ui).usable() is False

    def test_it_can_be_switched_off(self, monkeypatch):
        ui, _ = _ui()
        monkeypatch.setattr(corner, "is_dumb_terminal", lambda: False)
        monkeypatch.setattr(corner, "terminal_height", lambda: 40)
        monkeypatch.setattr(corner, "terminal_width", lambda: 120)
        monkeypatch.setenv("WYNXO_NO_CORNER", "1")
        assert CornerPlan(ui).usable() is False


class TestHowItLooks:
    def test_the_title_carries_the_count(self):
        ui, _ = _ui()
        panel = CornerPlan(ui)
        panel.items = parse(PLAN)
        assert "PLAN 1/3" in panel.lines()[0]

    def test_the_progress_bar_tracks_what_is_done(self):
        ui, _ = _ui()
        panel = CornerPlan(ui)
        panel.items = parse("[x] a\n[x] b\n[ ] c\n[ ] d")
        bar = panel.progress()
        assert bar.count("━") == len(bar) // 2      # half done, half the bar

    def test_an_ascii_terminal_gets_an_ascii_panel(self):
        ui, _ = _ui(unicode_ok=False)
        panel = CornerPlan(ui)
        panel.items = parse(PLAN)
        for line in panel.lines():
            assert line.isascii(), line

    def test_a_long_step_is_truncated_rather_than_wrapped(self):
        """The panel is a fixed box; a wrapped line would spill out of it."""
        ui, _ = _ui()
        panel = CornerPlan(ui, width=24)
        panel.items = parse("[ ] " + "x" * 200)
        for line in panel.lines():
            assert len(line.replace("[/]", "")) < 200

    def test_finishing_ticks_every_step(self):
        ui, _ = _ui()
        panel = CornerPlan(ui)
        panel.items = parse(PLAN)
        panel._pulse = 1
        drawn = "\n".join(panel.lines())
        assert panel.ui.g.dot not in drawn or drawn.count(panel.ui.g.tick) == 3
