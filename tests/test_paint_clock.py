"""One clock driving the pinned region, and only one.

rich's Live runs a refresh thread of its own. With it on, the region was
painted on two unsynchronised schedules -- rich's, and ours whenever a
streamed character arrived -- so frames landed one millisecond apart and
then thirty-two, and every frame is an erase and a redraw of the whole
block. Irregular erase-and-redraw is what "less smooth than `ollama run`"
looks like from the outside.

Measured over a real turn against a scripted model: 869,563 bytes on the
wire before, 666,760 after, with the same answer on screen.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from wynxo.ui import UI, ActivityBar


class Recorder:
    """Stands in for rich's Live, remembering when it was asked to draw."""

    def __init__(self):
        self.at: list[float] = []

    def refresh(self):
        self.at.append(time.monotonic())

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def gaps(self) -> list[float]:
        return [b - a for a, b in zip(self.at, self.at[1:], strict=False)]


@pytest.fixture
def bar():
    made = ActivityBar(UI(), effort="medium")
    made._live = made.paints = Recorder()
    return made


class TestRichHasNoClockOfItsOwn:
    def test_the_live_region_does_not_refresh_itself(self):
        """The whole point. Anything rich paints on its own schedule
        bypasses the throttle here, and two schedules is the jitter."""
        import inspect

        source = inspect.getsource(ActivityBar.start)
        assert "auto_refresh=False" in source
        assert "refresh_per_second" not in source, \
            "a rate for a thread that should not be running"


class TestEveryPaintGoesThroughTheOneThrottle:
    async def test_the_heartbeat_cannot_paint_faster_than_the_throttle(self, bar):
        bar._beat = asyncio.ensure_future(bar._heartbeat())
        try:
            await asyncio.sleep(0.4)
        finally:
            bar.stop()
        assert bar.paints.at, "nothing kept the clock moving"
        for gap in bar.paints.gaps:
            assert gap >= ActivityBar.REFRESH_INTERVAL * 0.9, bar.paints.gaps

    async def test_a_stream_and_the_heartbeat_do_not_paint_twice_over(self, bar):
        """Both go through refresh(), so a heartbeat landing between two
        streamed characters is absorbed rather than adding a frame."""
        from rich.text import Text

        bar._beat = asyncio.ensure_future(bar._heartbeat())
        try:
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline:
                bar.set_lead(Text("x" * len(bar.paints.at)))
                await asyncio.sleep(0.005)
        finally:
            bar.stop()
        for gap in bar.paints.gaps:
            assert gap >= ActivityBar.REFRESH_INTERVAL * 0.9, bar.paints.gaps

    async def test_time_still_passes_with_nothing_arriving(self, bar):
        """Without rich's thread something has to keep the elapsed clock
        moving, or a long tool call freezes the strip mid-second."""
        bar._beat = asyncio.ensure_future(bar._heartbeat())
        try:
            await asyncio.sleep(0.3)
        finally:
            bar.stop()
        assert len(bar.paints.at) >= 2


class TestTheHeartbeatIsCleanedUp:
    async def test_stopping_the_bar_stops_the_beat(self, bar):
        bar._beat = asyncio.ensure_future(bar._heartbeat())
        bar.stop()
        assert bar._beat is None
        await asyncio.sleep(0.15)
        painted = len(bar.paints.at)
        await asyncio.sleep(0.15)
        assert len(bar.paints.at) == painted, "it went on painting after stop"

    def test_stopping_a_bar_that_never_started_is_survivable(self):
        ActivityBar(UI(), effort="medium").stop()

    def test_a_bar_with_no_event_loop_still_starts(self, monkeypatch):
        """-p, a pipe, a script: there is no loop to run a heartbeat in, and
        that is not a reason to fail."""
        made = ActivityBar(UI(), effort="medium")
        made.start()          # no live region here; must not raise
        made.stop()
