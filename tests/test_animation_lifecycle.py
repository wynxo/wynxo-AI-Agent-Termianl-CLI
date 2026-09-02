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

import asyncio
import inspect
import threading


from wynxo.pet import Mood, Pet
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

    def test_the_mascot_has_no_clock_of_its_own(self):
        from wynxo import pet as pet_module

        source = inspect.getsource(pet_module)
        for clock in ("import time", "asyncio", "threading", "Timer(",
                      "Thread(", "perf_counter", "sleep("):
            assert clock not in source, f"{clock} drives the mascot"

    def test_the_frame_advances_only_when_something_draws(self):
        pet = Pet()
        pet.react(Mood.SEARCHING)
        before = tuple(pet.rows(advance=False))
        for _ in range(50):
            assert tuple(pet.rows(advance=False)) == before
        moved = {tuple(pet.rows()) for _ in range(12)}
        assert len(moved) > 1, "drawing never advances the frame"

    def test_reduced_motion_holds_the_frame_even_when_drawn(self):
        pet = Pet(animate=False)
        pet.react(Mood.SEARCHING)
        assert len({tuple(pet.rows()) for _ in range(20)}) == 1


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
    def test_the_wake_is_under_half_a_second(self):
        assert sum(pause for _, pause in UI.WAKE) <= 0.5

    def test_the_wake_starts_asleep_and_ends_settled(self):
        """It used to open on the ✕✕ face, so the first thing a session
        showed was a distressed cat."""
        moods = [name for name, _ in UI.WAKE]
        assert moods[0] == "sleepy"
        assert moods[-1] == "idle"
        assert "sad" not in moods

    def test_nothing_at_start_up_blocks_the_event_loop(self):
        """Both start-up animations run from the start-up coroutine. A
        blocking sleep there stops the loop for the length of the
        animation -- harmless four tenths of a second into a session, and
        exactly the habit that is not harmless anywhere else."""
        import ast
        import textwrap

        assert inspect.iscoroutinefunction(UI.wake)
        for func in (UI.wake,):
            tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
            # The calls themselves, not a docstring that mentions one.
            called = {ast.unparse(node.func) for node in ast.walk(tree)
                      if isinstance(node, ast.Call)}
            blocking = [name for name in called
                        if name.endswith("sleep") and "asyncio" not in name]
            assert blocking == [], f"{func.__name__}: {blocking}"

    def test_the_whole_startup_animation_is_short(self):
        """One animation at start-up, not two. The block-art logo that used
        to play before it is gone, so this is the whole of it."""
        total = sum(p for _, p in UI.WAKE)
        assert total < 1.0, f"{total:.2f}s of animation before the prompt"

    def test_nothing_animates_without_a_terminal(self):
        """A Live where nothing can repaint writes its cursor moves into the
        output as literal text."""
        ui = _ui()
        pet = Pet()
        asyncio.run(ui.wake(pet, "wyn"))   # live_ok is False: the still path
        bar = ActivityBar(ui, "medium")
        bar.start()
        assert bar._live is None
        bar.stop()

class TestReducedMotionIsRealAndStillReadable:
    def _bar(self):
        pet = Pet(animate=False)
        pet.react(Mood.THINKING)
        bar = ActivityBar(_ui(), "medium", "^C stop", pet=pet)
        bar.animate = False
        bar.activity = "thinking"
        return bar

    def test_nothing_changes_between_frames(self):
        bar = self._bar()
        assert len({bar._render().plain for _ in range(30)}) == 1

    def test_it_still_says_what_is_happening(self):
        assert "thinking" in self._bar()._render().plain

    def test_the_mascot_is_still_drawn(self):
        """Reduced motion means nothing moves, not that nothing is there.

        The strip carries the pet's one-cell mark rather than the drawing:
        the cat itself is three rows tall and belongs to the header, where
        it is identity rather than a status widget."""
        bar = self._bar()
        assert bar.pet.mark() in bar._render().plain

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
