"""Ctrl-C.

The full-screen layout holds the terminal in prompt_toolkit's raw mode,
which clears ISIG: the driver never raises SIGINT there, so the handler the
session installs before every turn cannot fire, and the key has to be bound
like any other. It was not -- outside a picker or an open question nothing
claimed it at all, and Ctrl-C did nothing for a whole session while the
status bar said "^C stop".

The scrolling prompt has its own gap: prompt_toolkit's teardown puts SIGINT
back to Python's default after each prompt, so a Ctrl-C during a slow
command raised KeyboardInterrupt out of the loop and ended the session.
"""

from __future__ import annotations

import asyncio
import types


from prompt_toolkit.keys import Keys

from wynxo import cli
from wynxo.cli import LIVE_KEYS, Repl
from wynxo.journal import Journal
from wynxo.layout import ChatLayout
from wynxo.ui import UI


def bindings_for_ctrl_c(layout: ChatLayout):
    """Every c-c binding whose filter passes in the layout's current state."""
    return [b for b in layout.app.key_bindings.get_bindings_for_keys(
        (Keys.ControlC,)) if b.filter()]


class TestTheLayoutBindsIt:
    def test_ctrl_c_is_live_with_no_modal_open(self):
        """The regression: two bindings existed, both filtered off unless a
        picker or a question was open."""
        layout = ChatLayout(width=80, height=24, on_interrupt=lambda: None)
        assert bindings_for_ctrl_c(layout), (
            "Ctrl-C is unbound in the layout's normal state")

    def test_it_calls_the_owner(self):
        pressed = []
        layout = ChatLayout(width=80, height=24,
                            on_interrupt=lambda: pressed.append(1))
        for binding in bindings_for_ctrl_c(layout):
            binding.handler(None)
        assert pressed == [1]

    def test_an_open_question_still_owns_it(self):
        """cancel_ask is what lets Ctrl-C out of a blocking permission
        prompt. prompt_toolkit runs the last binding whose filter passes, so
        a broader filter would have quietly taken the key from it."""
        pressed = []
        layout = ChatLayout(width=80, height=24,
                            on_interrupt=lambda: pressed.append(1))
        layout._ask = {"question": "?", "future": None}
        live = bindings_for_ctrl_c(layout)
        assert len(live) == 1, "more than one binding wants Ctrl-C while asking"
        assert pressed == []

    def test_an_open_picker_still_owns_it(self):
        pressed = []
        layout = ChatLayout(width=80, height=24,
                            on_interrupt=lambda: pressed.append(1))
        layout._picker = {"title": "", "choices": [], "index": 0, "future": None}
        assert len(bindings_for_ctrl_c(layout)) == 1
        assert pressed == []

    def test_a_layout_without_an_owner_does_not_crash(self):
        layout = ChatLayout(width=80, height=24)
        for binding in bindings_for_ctrl_c(layout):
            binding.handler(None)


class FakeChat:
    """Just the surface _on_interrupt_key touches."""

    def __init__(self, text: str = ""):
        self.buffer = types.SimpleNamespace(
            text=text, reset=lambda: setattr(self.buffer, "text", ""))
        self.submitted: list[str] = []
        self.repaints = 0

    def submit(self, text: str) -> None:
        self.submitted.append(text)

    def invalidate(self) -> None:
        self.repaints += 1


def repl_with(chat: FakeChat | None, task=None) -> Repl:
    repl = Repl.__new__(Repl)
    repl.chat = chat
    repl._task = task
    repl._interrupt_armed = 0.0
    repl._prompt_note = None
    repl.speaker = types.SimpleNamespace(stop=lambda: None)
    return repl


class TestWhatCtrlCDoes:
    """In the order a terminal program is expected to do them."""

    def test_a_running_turn_is_stopped_first(self):
        async def go():
            task = asyncio.ensure_future(asyncio.sleep(30))
            repl = repl_with(FakeChat("half a sentence"), task)
            repl._on_interrupt_key()
            assert task.cancelled() or task.cancelling()
            assert repl.chat.buffer.text == "half a sentence", (
                "the draft was thrown away along with the turn")
            assert repl.chat.submitted == []
            task.cancel()

        asyncio.run(go())

    def test_a_typed_line_is_cleared_when_nothing_is_running(self):
        repl = repl_with(FakeChat("a message I changed my mind about"))
        repl._on_interrupt_key()
        assert repl.chat.buffer.text == ""
        assert repl.chat.submitted == []

    def test_one_press_on_an_empty_composer_only_warns(self):
        """A session is too expensive to lose to a stray keystroke."""
        repl = repl_with(FakeChat())
        repl._on_interrupt_key()
        assert repl.chat.submitted == []
        assert "again" in repl._prompt_note[0]

    def test_a_second_press_quits(self):
        repl = repl_with(FakeChat())
        repl._on_interrupt_key()
        repl._on_interrupt_key()
        assert repl.chat.submitted == ["/quit"], (
            "quitting must run the same shutdown /quit does")

    def test_the_second_press_has_to_be_soon(self):
        repl = repl_with(FakeChat())
        repl._on_interrupt_key()
        repl._interrupt_armed = 0.0          # the window elapsed
        repl._on_interrupt_key()
        assert repl.chat.submitted == []

    def test_clearing_a_draft_disarms_the_quit(self):
        """Ctrl-C to clear, then Ctrl-C again, must not end the session."""
        repl = repl_with(FakeChat())
        repl._on_interrupt_key()             # arms
        repl.chat.buffer.text = "typed something"
        repl._on_interrupt_key()             # clears the draft
        repl._on_interrupt_key()             # first press again
        assert repl.chat.submitted == []


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
        repl.chat = None
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
    def test_the_chat_footer_shows_a_transient_note(self):
        """The layout had nowhere to put one, so "press Ctrl-C again to
        quit" was written to a footer that never displayed it."""
        import time

        repl = types.SimpleNamespace(
            _prompt_note=("press Ctrl-C again to quit", time.monotonic() + 5),
            _status_line=lambda: "model  ctx 3%")
        assert "press Ctrl-C again to quit" in Repl._chat_footer(repl)

    def test_an_expired_note_leaves_the_footer_alone(self):
        import time

        repl = types.SimpleNamespace(
            _prompt_note=("stale", time.monotonic() - 1),
            _status_line=lambda: "model  ctx 3%")
        assert Repl._chat_footer(repl) == " model  ctx 3%"
        assert repl._prompt_note is None, "the note was never expired"

    def test_a_fresh_note_is_shown_and_an_expired_one_is_dropped(self):
        import time

        fresh = ("effort: high", time.monotonic() + 5)
        assert cli.live_note(fresh) == ("effort: high", fresh)
        assert cli.live_note(("stale", time.monotonic() - 1)) == ("", None)
        assert cli.live_note(None) == ("", None)
