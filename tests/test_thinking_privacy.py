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