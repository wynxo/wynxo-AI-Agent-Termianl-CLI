"""Ctrl-C.

There is one interactive surface now -- the scrolling prompt -- and Ctrl-C
has to mean the right thing at every point in it.

At the prompt prompt_toolkit raises KeyboardInterrupt: one press forgets the
line, two in a row leave. During a turn the key watcher holds the terminal
in cbreak mode (ISIG intact), so the press arrives twice over -- as SIGINT
and as a bound key -- and both land on the same idempotent ``interrupt()``.

The scrolling prompt's own historical gap: prompt_toolkit's teardown puts
SIGINT back to Python's default after each prompt, so a Ctrl-C during a slow
command raised KeyboardInterrupt out of the loop and ended the session.
"""

from __future__ import annotations

import asyncio
import types

from wynxo import cli
from wynxo.cli import LIVE_KEYS, Repl
from wynxo.journal import Journal
from wynxo.ui import UI


class TestTheScrollingPrompt:
    def test_a_bare_keyboard_interrupt_does_not_end_the_session(self):
        """It is a BaseException, so it walked past the guard's `except
        Exception` and out of the process, conversation and all."""
        async def explode():
            raise KeyboardInterrupt

        warned = []
        ui = UI()
        ui.warn = warned.append
        repl = types.SimpleNamespace(
            ui=ui, journal=Journal(session_id="t", path=None, enabled=False),
            callbacks=types.SimpleNamespace(_end_stream=lambda: None))
        assert asyncio.run(Repl._guarded(repl, explode())) is None
        assert warned and "Interrupted" in warned[0]

    def test_commands_run_with_the_handler_armed(self):
        import inspect

        source = inspect.getsource(Repl._prompt_loop)
        assert "_arm_interrupt()" in source, (
            "SIGINT is back to Python's default by the time a command runs")

    def prompt_loop_over(self, answers):
        """Drive the real loop, with prompt_async replaying `answers`.

        Returns (prompts asked, exit code) -- the count is how the tests
        tell "looped again" from "quit".
        """
        raised = iter(answers)
        asked = []

        async def prompt_async(*_args, **_kwargs):
            asked.append(1)
            answer = next(raised)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        repl = Repl.__new__(Repl)
        repl.prompt_session = types.SimpleNamespace(prompt_async=prompt_async)
        repl._dictation_draft = ""
        repl._prompt_message = ">"
        repl._bottom_toolbar = lambda: ""
        repl._interrupt_armed = 0.0
        repl._prompt_note = None
        repl.ui = UI()
        repl.ui.reset_prompt_lines = lambda: None
        repl.ui.console.print = lambda *a, **k: None
        code = asyncio.run(Repl._prompt_loop(repl))
        return len(asked), code

    def test_two_interrupts_in_a_row_leave_the_prompt(self):
        """There was no way out of the scrolling prompt but /quit or
        Ctrl-D; one Ctrl-C forgot the line and looped forever."""
        asked, code = self.prompt_loop_over(
            [KeyboardInterrupt(), KeyboardInterrupt(), "never reached"])
        assert (asked, code) == (2, 0)

    def test_one_interrupt_only_forgets_the_line(self):
        asked, code = self.prompt_loop_over([KeyboardInterrupt(), EOFError()])
        assert asked == 2, "the first Ctrl-C ended the session on its own"
        assert code == 0

    def test_a_line_in_between_disarms_the_quit(self):
        """Ctrl-C, then type something, then Ctrl-C much later: two presses,
        but not two in a row, and the session must survive."""
        asked, _ = self.prompt_loop_over(
            [KeyboardInterrupt(), "   ", KeyboardInterrupt(), EOFError()])
        assert asked == 4, "a Ctrl-C from before the line still counted"


class TestEveryAdvertisedKeyIsBound:
    def test_the_activity_bar_promises_nothing_it_cannot_do(self, tmp_path):
        """LIVE_KEYS is rendered into the bar during every turn. ^D was in
        it and bound nowhere -- mid-turn it fell through to type-ahead,
        which drops it for not being printable."""
        import inspect

        source = inspect.getsource(Repl._turn_locked)
        watcher = source.split("KeyWatcher(")[1].split("on_key=")[0]
        for key in LIVE_KEYS:
            assert f'"{key}"' in watcher, f"{key} is advertised but never bound"


class TestTheNoteIsVisible:
    """"press Ctrl-C again to quit" and the Ctrl-E effort change are both
    transient notes. They have one home -- the prompt's bottom border."""

    def test_a_fresh_note_is_shown_and_an_expired_one_is_dropped(self):
        import time

        fresh = ("effort: high", time.monotonic() + 5)
        assert cli.live_note(fresh) == ("effort: high", fresh)
        assert cli.live_note(("stale", time.monotonic() - 1)) == ("", None)
        assert cli.live_note(None) == ("", None)

    def test_the_bottom_border_carries_it(self):
        import time

        repl = types.SimpleNamespace(
            _prompt_note=("press Ctrl-C again to quit", time.monotonic() + 5),
            _status_line=lambda: "model  ctx 3%",
            ui=UI())
        rendered = str(Repl._bottom_toolbar(repl))
        assert "press Ctrl-C again to quit" in rendered

    def test_an_expired_note_leaves_the_border_alone(self):
        import time

        repl = types.SimpleNamespace(
            _prompt_note=("stale", time.monotonic() - 1),
            _status_line=lambda: "model  ctx 3%",
            ui=UI())
        rendered = str(Repl._bottom_toolbar(repl))
        assert "stale" not in rendered
        assert repl._prompt_note is None, "the note was never expired"


class TestInterruptIsIdempotent:
    """During a turn the press arrives as SIGINT *and* as a bound key from
    the watcher: cbreak leaves ISIG on. Cancelling twice must be harmless."""

    def test_two_interrupts_cancel_once(self):
        async def go():
            task = asyncio.ensure_future(asyncio.sleep(30))
            repl = Repl.__new__(Repl)
            repl._task = task
            repl.speaker = types.SimpleNamespace(stop=lambda: None)
            repl.interrupt()
            repl.interrupt()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert task.cancelled()

        import contextlib
        asyncio.run(go())

    def test_an_idle_interrupt_does_nothing(self):
        repl = Repl.__new__(Repl)
        repl._task = None
        stopped = []
        repl.speaker = types.SimpleNamespace(stop=lambda: stopped.append(1))
        repl.interrupt()
        assert stopped == [1], "the voice must stop even with no turn running"
