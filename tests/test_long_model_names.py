"""A model tag from the hub is a third of a terminal.

``huihui_ai/Qwen3.8-abliterated:27b`` is thirty-three characters, and it
sits on the line under the prompt for the whole session, competing with the
effort level, the context figure and every key hint for the same row.

Cut blindly from the right it kept the one string on that line that never
changes and dropped the two facts that do.
"""

from __future__ import annotations

import types

import pytest
from rich.cells import cell_len

from wynxo import cli
from wynxo.ui import UI, Glyphs

LONG = "huihui_ai/Qwen3.8-abliterated:27b"
SHORT = "qwen3:8b"


@pytest.fixture
def ui():
    made = UI()
    made.g = Glyphs(True)
    return made


class TestShorteningAModelName:
    def test_a_name_that_fits_is_left_alone(self, ui):
        assert ui.shorten_model(SHORT, 40) == SHORT
        assert ui.shorten_model(LONG, 40) == LONG

    def test_the_namespace_goes_first(self, ui):
        """Who published it is not something anybody reads while typing."""
        assert ui.shorten_model(LONG, 30) == "Qwen3.8-abliterated:27b"

    def test_the_tag_always_survives(self, ui):
        """The tag is the size, and the size is what tells you what to
        expect of the thing you are talking to."""
        for room in range(5, 34):
            assert ui.shorten_model(LONG, room).endswith("27b"), room

    def test_it_never_returns_more_than_it_was_given_room_for(self, ui):
        for room in range(1, 40):
            assert cell_len(ui.shorten_model(LONG, room)) <= max(room, 2), room

    def test_no_room_at_all_is_survivable(self, ui):
        ui.shorten_model(LONG, 0)
        ui.shorten_model("", 10)


class TestTheStatusKeepsTheFactsThatChange:
    def _border(self, width, model, effort="high", pct=11):
        made = UI()
        made.g = Glyphs(True)
        made.width = made.console.width = width
        room = max(12, width // 3)
        status = f"{made.shorten_model(model, room)} · {effort} · ctx {pct}%"
        repl = types.SimpleNamespace(ui=made)
        repl._status_line = lambda: status
        repl._prompt_note = None
        repl.session_prompt = types.SimpleNamespace(
            default_buffer=types.SimpleNamespace(text=""))
        repl._bottom_toolbar = cli.Repl._bottom_toolbar.__get__(repl, cli.Repl)
        repl._border_plain = cli.Repl._border_plain.__get__(repl, cli.Repl)
        return repl._border_plain()

    @pytest.mark.parametrize("width", [40, 50, 60, 80, 98, 120, 200])
    def test_the_effort_and_the_context_always_survive(self, width):
        """They are the two things on the line that move. At forty columns
        a long name kept `huihui_ai/Qwen3.8-abliterated:` and lost both."""
        border = self._border(width, LONG)
        assert "high" in border, border
        assert "ctx 11%" in border, border

    @pytest.mark.parametrize("width", [40, 50, 60, 80, 98, 120, 200])
    def test_the_border_is_exactly_the_terminal_wide(self, width):
        for model in (SHORT, LONG):
            assert cell_len(self._border(width, model)) == width, model

    @pytest.mark.parametrize("width", [50, 60, 80, 98, 120])
    def test_a_long_name_does_not_cost_the_stop_hint(self, width):
        """It is the one binding you most need to remember, and it was the
        first casualty of somebody's choice of model."""
        assert "^C stop" in self._border(width, LONG), width

    def test_a_wide_terminal_still_shows_the_whole_name(self):
        assert LONG in self._border(120, LONG)

    def test_a_short_name_is_not_made_worse(self):
        assert SHORT in self._border(40, SHORT)


class TestTheStatusLineItselfShortens:
    """The shortener is only worth having if the line that competes for the
    space actually calls it."""

    def _status(self, width, model):
        from wynxo.effort import resolve
        from wynxo.scope import Mode

        made = UI()
        made.g = Glyphs(True)
        made.width = made.console.width = width
        repl = types.SimpleNamespace(ui=made)
        repl.config = types.SimpleNamespace(model=model, num_ctx=32768)
        repl.policy = resolve("high")
        repl.pending = []
        repl.callbacks = None
        repl.agent = types.SimpleNamespace(
            session=types.SimpleNamespace(token_estimate=lambda: 3600),
            permissions=types.SimpleNamespace(mode=Mode.MANUAL))
        repl._context_limit = lambda: (32768, "num_ctx")
        repl._status_line = cli.Repl._status_line.__get__(repl, cli.Repl)
        return repl._status_line()

    def test_a_narrow_terminal_gets_a_shortened_name(self):
        line = self._status(60, LONG)
        assert "huihui_ai/" not in line, line
        assert "27b" in line and "high" in line and "ctx 11%" in line

    def test_a_wide_terminal_keeps_the_whole_name(self):
        assert LONG in self._status(120, LONG)

    def test_a_short_name_is_untouched_at_any_width(self):
        for width in (40, 60, 98, 200):
            assert SHORT in self._status(width, SHORT), width
