"""How much the terminal is asked to do per streamed token.

Every repaint of the pinned block erases and redraws the plan, the line
being written and the status strip -- around a hundred and fifty bytes of
escape sequences on a two-row bar. Asked for once per streamed character, a
five-hundred-character answer spent seventy-five kilobytes of terminal
traffic to show five hundred characters, and on a slow terminal or over ssh
the display, not the model, is what you are waiting for.
"""

from __future__ import annotations

import pytest

from wynxo.ui import UI, ActivityBar


class _Counter:
    """Stands in for rich's Live, counting what it is asked to draw."""

    def __init__(self):
        self.paints = 0

    def refresh(self):
        self.paints += 1


@pytest.fixture
def bar():
    made = ActivityBar(UI(), effort="medium")
    made._live = _Counter()
    return made


class TestTheStreamCannotFloodTheTerminal:
    def test_a_burst_of_tokens_costs_one_paint(self, bar):
        """Five hundred characters arriving inside one frame."""
        for _ in range(500):
            bar.add_token()
        assert bar._live.paints == 1
        assert bar.tokens == 500, "the count itself must still be exact"

    def test_a_burst_of_partial_lines_costs_one_paint(self, bar):
        from rich.text import Text

        for i in range(200):
            bar.set_lead(Text("x" * i))
        assert bar._live.paints == 1

    def test_the_newest_state_is_what_the_next_frame_draws(self, bar):
        """Nothing is lost by skipping a paint: Live re-renders this object,
        so a coalesced update is shown by the next scheduled frame."""
        for i in range(50):
            bar.update(tokens=i)
        assert bar.tokens == 49

    def test_time_passing_allows_another_paint(self, bar):
        bar.add_token()
        bar._painted -= ActivityBar.REFRESH_INTERVAL * 2
        bar.add_token()
        assert bar._live.paints == 2


class TestEventsStillLandInTheirOwnFrame:
    def test_a_change_of_activity_is_not_coalesced(self, bar):
        bar.add_token()
        before = bar._live.paints
        bar.update(activity="reading")
        assert bar._live.paints == before + 1

    def test_a_new_plan_is_not_coalesced(self, bar):
        bar.add_token()
        before = bar._live.paints
        bar.set_plan("[ ] do the thing")
        assert bar._live.paints == before + 1

    def test_clearing_the_written_line_is_not_coalesced(self, bar):
        """It is cleared as the line is committed to the scrollback. A stale
        copy left in the live region is the same text on screen twice."""
        from rich.text import Text

        bar.set_lead(Text("half a line"))
        before = bar._live.paints
        bar.set_lead(None)
        assert bar._live.paints == before + 1


class TestTheThrottleIsNotVisible:
    def test_the_interval_is_faster_than_the_eye(self):
        """Above about fifteen frames a second, motion reads as continuous.
        Slower than that and the throttle would be something you notice."""
        assert ActivityBar.REFRESH_INTERVAL <= 1 / 15

    def test_a_bar_with_no_live_region_never_raises(self):
        made = ActivityBar(UI(), effort="medium")
        made.refresh()
        made.refresh(force=True)
        made.add_token()
