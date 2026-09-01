"""Ctrl-C stops the queue as well as the turn, and what is left is visible.

Type-ahead collected during a turn drains the moment the turn ends. That is
right for a turn that finished, and wrong for one that was interrupted: you
press stop because you want to look at something, and having the next queued
message start immediately is the opposite of what the key means.

The other half of the same bug is that a held queue used to be invisible.
The activity bar shows it while a turn runs; at the prompt nothing did, so
messages you had forgotten about fired on the next thing you typed.
"""

from __future__ import annotations

import asyncio
import io
import types

from wynxo.cli import Repl
from wynxo.queue import Pending
from wynxo.ui import SafeConsole, UI


def _repl(outcomes):
    """A Repl whose turns succeed or fail per ``outcomes``."""
    ui = UI()
    ui.live_ok = False
    ui.console = SafeConsole(file=io.StringIO(), width=100, highlight=False,
                             soft_wrap=False)
    ui.width = 100

    repl = Repl.__new__(Repl)
    repl.ui = ui
    repl.pending = Pending()
    repl.ran = []
    results = iter(outcomes)

    async def turn(text):
        repl.ran.append(text)
        return next(results)

    async def command(text):
        repl.ran.append(text)
        return True

    repl.turn = turn
    repl.command = command
    return repl


def _written(repl) -> str:
    import re

    return re.sub(r"\s+", " ", repl.ui.console.file.getvalue())


class TestAnInterruptHoldsTheRest:
    def test_the_next_message_does_not_start(self):
        repl = _repl([False])
        for message in ("first", "second", "third"):
            repl.pending.items.append(message)
        asyncio.run(Repl._drain_queue(repl))
        assert repl.ran == ["first"], repl.ran

    def test_what_is_left_stays_queued(self):
        repl = _repl([False])
        for message in ("first", "second", "third"):
            repl.pending.items.append(message)
        asyncio.run(Repl._drain_queue(repl))
        assert repl.pending.summary() == ["second", "third"]

    def test_it_says_what_is_waiting(self):
        repl = _repl([False])
        for message in ("first", "second"):
            repl.pending.items.append(message)
        asyncio.run(Repl._drain_queue(repl))
        assert "1 queued message still waiting" in _written(repl)

    def test_it_only_offers_things_that_exist(self):
        """The note used to say "Enter to run the next", and an empty Enter
        at the prompt does nothing at all -- a promise the prompt does not
        keep is worse than no note."""
        from wynxo.cli import COMMANDS

        repl = _repl([False])
        for message in ("first", "second"):
            repl.pending.items.append(message)
        asyncio.run(Repl._drain_queue(repl))
        note = _written(repl)
        assert "/queue run" in note and "/queue clear" in note
        assert "/queue" in COMMANDS


class TestTheQueueCommand:
    def _repl(self):
        repl = _repl([True, True])
        return repl

    def test_run_sends_everything_held(self):
        repl = self._repl()
        for message in ("first", "second"):
            repl.pending.items.append(message)
        asyncio.run(Repl.cmd_queue(repl, ["run"]))
        assert repl.ran == ["first", "second"]
        assert repl.pending.summary() == []

    def test_clear_drops_everything(self):
        repl = self._repl()
        for message in ("first", "second"):
            repl.pending.items.append(message)
        asyncio.run(Repl.cmd_queue(repl, ["clear"]))
        assert repl.ran == []
        assert repl.pending.summary() == []

    def test_bare_queue_only_lists(self):
        repl = self._repl()
        repl.pending.items.append("first")
        asyncio.run(Repl.cmd_queue(repl, []))
        assert repl.ran == []
        assert repl.pending.summary() == ["first"]
        assert "first" in _written(repl)

    def test_an_empty_queue_says_so(self):
        repl = self._repl()
        for args in ([], ["run"], ["clear"]):
            asyncio.run(Repl.cmd_queue(repl, args))
        assert repl.ran == []

    def test_an_unknown_argument_does_not_run_anything(self):
        repl = self._repl()
        repl.pending.items.append("first")
        asyncio.run(Repl.cmd_queue(repl, ["banana"]))
        assert repl.ran == []
        assert repl.pending.summary() == ["first"]

    def test_the_session_is_not_ended_by_it(self):
        """Returning False from the drain means "quit", which an interrupt
        emphatically does not."""
        repl = _repl([False])
        repl.pending.items.append("first")
        assert asyncio.run(Repl._drain_queue(repl)) is True


class TestAFinishedTurnStillDrains:
    def test_every_message_runs_in_order(self):
        repl = _repl([True, True, True])
        for message in ("first", "second", "third"):
            repl.pending.items.append(message)
        asyncio.run(Repl._drain_queue(repl))
        assert repl.ran == ["first", "second", "third"]
        assert repl.pending.summary() == []

    def test_a_queued_command_runs_too(self):
        repl = _repl([True])
        repl.pending.items.append("/stats")
        repl.pending.items.append("a question")
        asyncio.run(Repl._drain_queue(repl))
        assert repl.ran == ["/stats", "a question"]

    def test_a_quitting_command_ends_the_drain(self):
        async def command(_text):
            return False

        repl = _repl([True])
        repl.command = command
        repl.pending.items.append("/quit")
        repl.pending.items.append("never reached")
        assert asyncio.run(Repl._drain_queue(repl)) is False
        assert repl.pending.summary() == ["never reached"]


class TestTheQueueIsVisibleAtThePrompt:
    def _status(self, queued):
        repl = Repl.__new__(Repl)
        repl.ui = UI()
        repl.pending = Pending()
        for message in queued:
            repl.pending.items.append(message)
        repl.pet = types.SimpleNamespace(enabled=False)
        repl.config = types.SimpleNamespace(model="m", num_ctx=8192)
        repl.policy = types.SimpleNamespace(name="medium", context_budget=8192)
        repl._last_elapsed = 0.0
        usage = types.SimpleNamespace(completion_tokens=0,
                                      tokens_per_second=lambda: 0)
        session = types.SimpleNamespace(usage=usage,
                                        token_estimate=lambda: 100)
        from wynxo.scope import Mode

        repl.agent = types.SimpleNamespace(
            session=session,
            permissions=types.SimpleNamespace(mode=Mode.MANUAL))
        repl.callbacks = types.SimpleNamespace(bar=None)
        return Repl._status_line(repl)

    def test_a_held_queue_is_named(self):
        assert "2 queued" in self._status(["one", "two"])

    def test_an_empty_queue_says_nothing(self):
        assert "queued" not in self._status([])

    def test_it_comes_before_the_numbers(self):
        """Something you typed and have not seen run outranks a token
        count."""
        line = self._status(["one"])
        assert line.index("queued") < line.index("ctx ")
