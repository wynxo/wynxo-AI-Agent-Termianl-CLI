"""Animation starts, stops, and leaves nothing behind.

Everything that moves in wynxo is driven by one repaint: the activity bar's
``Live`` recomputes the frame, and the mascot's counter advances because the
bar drew it. There is no scheduler, no timer thread and no task per
component -- which is the property worth pinning, because the failure it
prevents is invisible until it is not. A companion animated by a clock of
its own keeps animating through a stall, showing a cat typing while nothing
is being written, and keeps going after the turn it belonged to has ended.

The other half is the teardown: a live region left running after a turn is a
region drawing over the prompt.
"""

from __future__ import annotations

import inspect
import threading


from wynxo.companion import State
from wynxo.pet import Pet
from wynxo.ui import ActivityBar, UI


def _ui():
    ui = UI()
    ui.live_ok = False
    ui.width = 90
    return ui


class TestOneClockDrivesEverything:
    def test_the_bar_owns_the_only_repaint(self):
        """rich refuses a second live display on one console, and the
        refusal would land mid-turn."""
        source = inspect.getsource(ActivityBar)
        assert source.count("Live(") == 1

    def test_the_companion_has_no_clock_of_its_own(self):
        from wynxo import sprite as sprite_module

        source = inspect.getsource(sprite_module)
        for clock in ("import time", "asyncio", "threading", "Timer(",
                      "Thread(", "perf_counter", "sleep("):
            assert clock not in source, f"{clock} drives the companion"

    def test_the_frame_is_the_callers_and_only_the_callers(self):
        """A scheduler would keep animating through a stall, drawing typing
        while nothing is being written. The frame is an argument, so the
        picture moves exactly while the bar repaints."""
        from wynxo import sprite
        from wynxo.theme import PURPLE

        held = [r.plain for r in sprite.rows(State.SEARCHING, 0, PURPLE)]
        for _ in range(50):
            assert [r.plain for r in
                    sprite.rows(State.SEARCHING, 0, PURPLE)] == held
        moved = {tuple(r.plain for r in sprite.rows(State.SEARCHING, f, PURPLE))
                 for f in range(len(sprite.FRAMES[State.SEARCHING]))}
        assert len(moved) > 1, "the frame argument changes nothing"

    def test_reduced_motion_holds_the_frame_even_when_drawn(self):
        bar = ActivityBar(_ui(), "medium")
        bar.animate = False
        bar.state = "searching"
        bar.activity = "searching"
        assert len({tuple(t.plain for t in bar._scene())
                    for _ in range(20)}) == 1


class TestNothingKeepsRunningAfterwards:
    def test_stopping_the_bar_releases_the_live(self):
        bar = ActivityBar(_ui(), "medium")
        bar.start()
        bar.stop()
        assert bar._live is None

    def test_stopping_twice_is_harmless(self):
        bar = ActivityBar(_ui(), "medium")
        bar.start()
        bar.stop()
        bar.stop()

    def test_refreshing_a_stopped_bar_does_nothing(self):
        """The teardown order puts the bar down before the turn's last
        prints; a refresh arriving after that must not resurrect it."""
        bar = ActivityBar(_ui(), "medium")
        bar.start()
        bar.stop()
        bar.update(activity="thinking")
        bar.refresh()
        assert bar._live is None

    def test_no_thread_is_left_running(self):
        before = set(threading.enumerate())
        bar = ActivityBar(_ui(), "medium")
        bar.start()
        for _ in range(20):
            bar.refresh()
        bar.stop()
        leaked = [t for t in threading.enumerate()
                  if t not in before and t.is_alive()]
        assert leaked == [], leaked

    def test_the_turn_stops_the_bar_whatever_happens(self):
        from wynxo.cli import Repl

        source = inspect.getsource(Repl._turn_locked)
        finally_block = source.rsplit("finally:", 1)[-1]
        assert "bar.stop()" in finally_block


class TestStartupAnimationsAreBounded:
    def test_there_is_no_startup_animation_left_to_wait_through(self):
        """Start-up is one line and then the prompt. The block-art logo went
        first, then the wake-up the companion used to play before the
        header -- both were things you sat through before you could type."""
        assert not hasattr(UI, "WAKE")
        assert not hasattr(UI, "wake")

    def test_nothing_animates_without_a_terminal(self):
        """A Live where nothing can repaint writes its cursor moves into the
        output as literal text."""
        bar = ActivityBar(_ui(), "medium")
        bar.start()
        assert bar._live is None
        bar.stop()

class TestReducedMotionIsRealAndStillReadable:
    def _bar(self):
        bar = ActivityBar(_ui(), "medium", "^C stop", pet=Pet(animate=False))
        bar.animate = False
        bar.activity = "thinking"
        bar.state = "thinking"
        return bar

    def test_nothing_changes_between_frames(self):
        bar = self._bar()
        assert len({bar._render().plain for _ in range(30)}) == 1

    def test_it_still_says_what_is_happening(self):
        """In the scene, not the strip. The activity word moved up beside
        the companion; repeating it on the row underneath was the duplicate
        status the strip is meant to be free of."""
        bar = self._bar()
        assert "thinking" in "".join(t.plain for t in bar._scene())
        assert "thinking" not in bar._render().plain

    def test_what_is_happening_is_still_said(self):
        """Reduced motion means nothing moves, not that nothing is there.

        The companion is what goes: it is the moving part, and the words
        beside it carry the same fact. Turning motion off leaves the state
        written out rather than leaving a still picture of it."""
        bar = self._bar()
        assert bar._scene() == bar._scene_lines()
        assert "thinking" in bar._scene()[0].plain

    def test_the_layout_does_not_shift_when_motion_is_turned_off(self):
        """Animation off must not move anything: the still mark occupies the
        cells the spinner would have."""
        from rich.cells import cell_len

        moving = ActivityBar(_ui(), "medium", "^C stop")
        moving.activity = "thinking"
        still = ActivityBar(_ui(), "medium", "^C stop")
        still.activity = "thinking"
        still.animate = False
        assert cell_len(moving._render().plain) == cell_len(still._render().plain)
