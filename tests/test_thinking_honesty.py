"""Turning the reasoning display on has to mean something.

Showing reasoning and *having* reasoning are two different settings, and
only one of them is the toggle. Below "high" effort the policy sends no
``think`` to the model at all, so the reasoning never exists -- and the
default effort is "medium". Turning the display on there produced a session
that said "thinking shown" and then never showed a word, indefinitely, with
nothing on screen to say why. From the outside that is indistinguishable
from a broken feature.

Both routes to the setting -- Ctrl-O mid-turn and /thinking -- now say so.
"""

from __future__ import annotations

import asyncio
import io

import pytest

from wynxo.cli import Repl, TerminalCallbacks
from wynxo.effort import resolve
from wynxo.ui import SafeConsole, UI


def _ui():
    ui = UI()
    ui.live_ok = False
    ui.console = SafeConsole(file=io.StringIO(), width=100, highlight=False,
                             soft_wrap=False)
    return ui


def _written(ui) -> str:
    """One line, so a message that wrapped still reads as one sentence."""
    import re

    return re.sub(r"\s+", " ", ui.console.file.getvalue())


class TestCtrlO:
    def _callbacks(self, effort: str):
        ui = _ui()
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        policy = resolve(effort)
        callbacks.thinking_asked_for = lambda: bool(policy.thinking)
        return callbacks, ui

    @pytest.mark.parametrize("effort", ["low", "medium"])
    def test_it_says_there_is_nothing_to_reveal(self, effort):
        callbacks, ui = self._callbacks(effort)
        assert resolve(effort).thinking is False, "premise changed"
        callbacks.toggle_thinking()
        assert "does not ask the model to think" in _written(ui)

    @pytest.mark.parametrize("effort", ["high", "ultra"])
    def test_a_thinking_level_says_nothing_extra(self, effort):
        callbacks, ui = self._callbacks(effort)
        assert resolve(effort).thinking is True, "premise changed"
        callbacks.toggle_thinking()
        assert "does not ask the model" not in _written(ui)

    def test_turning_it_back_off_is_quiet(self):
        callbacks, ui = self._callbacks("medium")
        callbacks.toggle_thinking()          # on, warns
        ui.console.file.truncate(0)
        ui.console.file.seek(0)
        callbacks.toggle_thinking()          # off
        assert "does not ask the model" not in _written(ui)

    def test_an_unwired_hook_is_harmless(self):
        """The callbacks are built before the Repl finishes wiring them, and
        several tests build them alone."""
        ui = _ui()
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.toggle_thinking()
        assert "does not ask the model" not in _written(ui)

    def test_a_hook_that_raises_does_not_take_the_keypress_down(self):
        callbacks, ui = self._callbacks("medium")

        def broken():
            raise RuntimeError("boom")

        callbacks.thinking_asked_for = broken
        callbacks.toggle_thinking()
        assert ui.show_thinking is True


class TestTheCommand:
    def _repl(self, effort: str):
        repl = Repl.__new__(Repl)
        repl.ui = _ui()
        repl.policy = resolve(effort)
        repl.config = type("C", (), {"show_thinking": False,
                                     "save": lambda self: None})()
        repl.callbacks = TerminalCallbacks(repl.ui, prompt_session=None)
        return repl

    def test_it_warns_at_an_effort_that_does_not_think(self):
        repl = self._repl("medium")
        asyncio.run(Repl.cmd_thinking(repl, ["on"]))
        assert repl.ui.show_thinking is True
        assert "nothing to show" in _written(repl.ui)

    def test_it_names_the_remedy(self):
        repl = self._repl("medium")
        asyncio.run(Repl.cmd_thinking(repl, ["on"]))
        assert "/effort high" in _written(repl.ui)

    def test_a_thinking_level_is_not_warned_about(self):
        repl = self._repl("high")
        asyncio.run(Repl.cmd_thinking(repl, ["on"]))
        assert "nothing to show" not in _written(repl.ui)

    def test_turning_it_off_never_warns(self):
        repl = self._repl("medium")
        asyncio.run(Repl.cmd_thinking(repl, ["off"]))
        assert "nothing to show" not in _written(repl.ui)
