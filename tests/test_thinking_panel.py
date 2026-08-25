"""Reasoning is collapsed by default, never discarded.

Collapsed is a display state, not a decision to throw the thought away.
Before this, hidden meant not kept -- so opening the panel part-way through
a turn could only ever show what came after the keypress, and the thought
you wanted to read was the one that had already gone by.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from wynxo.cli import TerminalCallbacks
from wynxo.tui import Transcript
from wynxo.ui import UI


def plain(page: Transcript) -> str:
    page.drain()
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(page.lines))


@pytest.fixture
def session():
    page = Transcript(width=90)
    ui = UI(show_thinking=False)          # collapsed, which is the default
    ui.console = page.console
    ui.live_ok = False
    return page, ui, TerminalCallbacks(ui)


def think(callbacks, *chunks):
    async def go():
        for chunk in chunks:
            await callbacks.on_thinking(chunk)

    asyncio.run(go())


class TestCollapsedByDefault:
    def test_the_setting_starts_off(self):
        from wynxo.config import Config

        assert Config().show_thinking is False

    def test_nothing_is_printed_while_collapsed(self, session):
        page, _ui, callbacks = session
        think(callbacks, "weighing it up. ", "the retry looks linear. ")
        assert "weighing it up" not in plain(page)

    def test_but_it_is_kept(self, session):
        _page, _ui, callbacks = session
        think(callbacks, "weighing it up. ")
        assert "".join(callbacks._thinking_buffer) == "weighing it up. "

    def test_collapsed_still_says_there_is_something_to_open(self, session):
        """Collapsed does not mean invisible: something has to say the model
        is reasoning, or the panel is a feature nobody discovers."""
        _page, _ui, callbacks = session
        think(callbacks, "one two three four five six seven ")
        note = callbacks._thinking_note()
        assert "words thought" in note and "^O" in note

    def test_a_bare_start_says_nothing(self, session):
        """Two words in, a counter is noise."""
        _page, _ui, callbacks = session
        think(callbacks, "so ")
        assert callbacks._thinking_note() == ""


class TestOpeningItMidTurn:
    def test_it_shows_what_came_before(self, session):
        """The whole point: the thought you want is the one already gone."""
        page, _ui, callbacks = session
        think(callbacks, "first I check the retry. ", "it is linear. ")
        callbacks.toggle_thinking()
        body = plain(page)
        assert "first I check the retry" in body
        assert "it is linear" in body

    def test_and_keeps_streaming_after(self, session):
        page, _ui, callbacks = session
        think(callbacks, "before the keypress. ")
        callbacks.toggle_thinking()
        think(callbacks, "after the keypress.")
        callbacks._end_thinking()
        body = plain(page)
        assert "before the keypress" in body and "after the keypress" in body

    def test_the_two_halves_are_not_glued_together(self, session):
        """Stripping the backlog's tail joins it to the next word."""
        page, _ui, callbacks = session
        think(callbacks, "before ")
        callbacks.toggle_thinking()
        think(callbacks, "after")
        callbacks._end_thinking()
        assert "beforeafter" not in plain(page)

    def test_the_note_does_not_land_mid_sentence(self, session):
        """Announced before the panel opens: printed after, it drops into
        the middle of whatever is streaming."""
        page, _ui, callbacks = session
        think(callbacks, "a thought that runs on for a little while. ")
        callbacks.toggle_thinking()
        body = plain(page)
        assert "thinking shown" in body
        assert body.index("thinking shown") < body.index("a thought that runs")

    def test_closing_it_again_does_not_lose_the_buffer(self, session):
        _page, _ui, callbacks = session
        think(callbacks, "kept regardless. ")
        callbacks.toggle_thinking()
        callbacks.toggle_thinking()
        assert "kept regardless" in "".join(callbacks._thinking_buffer)

    def test_opening_with_nothing_thought_yet_is_quiet(self, session):
        page, _ui, callbacks = session
        callbacks.toggle_thinking()
        assert "thinking\n" not in plain(page)


class TestEachTurnStartsFresh:
    def test_the_buffer_is_cleared_between_turns(self):
        """Opening the panel should show this answer's reasoning, not
        everything the model has thought all session."""
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl.turn)
        assert "_thinking_buffer.clear()" in source
