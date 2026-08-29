"""Private reasoning must never leak into the user-visible transcript.

The model's chain of thought -- "The user is greeting me", "I should try...",
plan outlines -- is internal. When thinking display is off (the default, and
the saved preference), `on_thinking` must hold the reasoning back, never
print a cursor of it, and expose only a high-level activity state through the
bar. Toggling on mid-turn is an explicit user choice to peek, not a change to
what the transcript normally contains.
"""

from __future__ import annotations

import asyncio

from wynxo.cli import TerminalCallbacks
from wynxo.ui import UI, Glyphs


def _capturing_ui(*, show_thinking: bool) -> "tuple[UI, object]":
    import io

    ui = UI()
    ui.g = Glyphs(False)
    ui.show_thinking = show_thinking
    ui.width = 80
    stream = io.StringIO()
    ui.console.file = stream
    ui.console._width = ui.width
    return ui, stream


def _cb(ui: UI) -> TerminalCallbacks:
    cb = TerminalCallbacks(ui, prompt_session=None)
    return cb


def test_reasoning_is_held_back_when_thinking_is_off():
    ui, stream = _capturing_ui(show_thinking=False)
    cb = _cb(ui)
    asyncio.run(cb.on_thinking("The user is greeting me. I should respond "
                               "warmly and offer to help with their project."))
    # Nothing printed, and not one token of the reasoning reached the screen.
    assert stream.getvalue() == ""
    # It is kept around only in case the user explicitly toggles the panel
    # open mid-turn -- held out of the transcript, not thrown away.
    assert "".join(cb._thinking_unsent).startswith("The user is greeting me")


def test_reasoning_never_reaches_the_screen_after_many_chunks():
    ui, stream = _capturing_ui(show_thinking=False)
    cb = _cb(ui)
    for chunk in ["step one", "then step two", "maybe approach B instead",
                  "check the callers first"]:
        asyncio.run(cb.on_thinking(chunk))
    assert stream.getvalue() == ""
    assert len(cb._thinking_unsent) == 4


def test_the_config_and_ui_default_to_hidden():
    from wynxo.config import Config

    assert Config(verify_with_tests=False).show_thinking is False
    ui = UI()
    assert ui.show_thinking is False

class TestHidingNeverDiscards:
    """Hiding is a display state, not a decision to throw the thought away.

    The rule the UI promises: capture never stops, so showing can always go
    back over everything -- including the turns that have already finished.
    """

    def test_capture_continues_while_hidden(self):
        ui, stream = _capturing_ui(show_thinking=False)
        cb = _cb(ui)
        for chunk in ("weighing the ", "options here ", "carefully "):
            asyncio.run(cb.on_thinking(chunk))
        # Nothing drawn ...
        assert "weighing" not in stream.getvalue()
        # ... but all of it kept.
        assert "".join(cb._thinking_buffer) == (
            "weighing the options here carefully ")

    def test_showing_replays_what_was_hidden(self):
        ui, stream = _capturing_ui(show_thinking=False)
        cb = _cb(ui)
        asyncio.run(cb.on_thinking("the hidden part "))
        ui.show_thinking = True
        cb._open_thinking()
        assert "hidden part" in stream.getvalue()

    def test_a_finished_turn_is_retired_not_dropped(self):
        """The turn boundary is a divider in the record, not the end of it."""
        ui, _ = _capturing_ui(show_thinking=False)
        cb = _cb(ui)
        asyncio.run(cb.on_thinking("first turn thought "))
        # What Repl.turn does at the top of each turn.
        cb._thinking_turns.append("".join(cb._thinking_buffer))
        cb._thinking_buffer.clear()
        cb._thinking_unsent.clear()
        asyncio.run(cb.on_thinking("second turn thought "))
        assert cb._thinking_turns == ["first turn thought "]

    def test_replaying_the_session_shows_every_turn(self):
        ui, stream = _capturing_ui(show_thinking=False)
        cb = _cb(ui)
        cb._thinking_turns = ["thought about parsers ", "thought about tests "]
        asyncio.run(cb.on_thinking("thinking about this one "))
        cb._open_thinking(whole_session=True)
        out = stream.getvalue()
        assert "parsers" in out
        assert "tests" in out
        assert "this one" in out

    def test_replaying_with_no_history_shows_only_the_backlog(self):
        ui, stream = _capturing_ui(show_thinking=False)
        cb = _cb(ui)
        asyncio.run(cb.on_thinking("only this "))
        cb._open_thinking(whole_session=True)
        out = stream.getvalue()
        assert "only this" in out
        assert "earlier turn" not in out

    def test_the_note_keeps_counting_while_hidden(self):
        """The rising count is the promise that ^O still has something."""
        ui, _ = _capturing_ui(show_thinking=False)
        cb = _cb(ui)
        asyncio.run(cb.on_thinking("one two three four five "))
        note = cb._thinking_note()
        assert "words thought" in note and "^O" in note
