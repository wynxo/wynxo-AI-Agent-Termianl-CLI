"""Running read-only tool calls together.

Every tool declared `concurrency_safe` and nothing ever read it, and the
loop's own comment claimed read-only calls "can run together" while running
them one at a time. Both are now true.

The value is modest -- a local file read is fast and the model is the
bottleneck -- but a turn that greps a large repo three times pays for it,
and a declared contract that nothing honours is worse than no contract.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from wynxo.agent import Agent
from wynxo.config import Config
from wynxo.effort import resolve
from wynxo.parsing import ToolCall
from wynxo.scope import Mode


@pytest.fixture
def agent(tmp_path):
    for i in range(4):
        (tmp_path / f"f{i}.py").write_text(f"value = {i}\n", encoding="utf-8")
    return Agent(client=MagicMock(), config=Config(),
                 policy=resolve("medium"), workspace=tmp_path)


def reads(*names) -> list[ToolCall]:
    return [ToolCall(name="read_file", arguments={"path": n}, call_id=str(i))
            for i, n in enumerate(names)]


class TestWhatMayGoTogether:
    def test_several_reads_are_batched(self, agent):
        calls = reads("f0.py", "f1.py", "f2.py")
        assert len(agent._parallel_batch(calls)) == 3

    def test_a_single_read_is_not_worth_batching(self, agent):
        assert agent._parallel_batch(reads("f0.py")) == []

    def test_a_write_stops_the_batch(self, agent):
        """Two writes to one file must never interleave."""
        calls = reads("f0.py", "f1.py")
        calls.append(ToolCall(name="write_file",
                              arguments={"path": "f0.py", "content": "x"},
                              call_id="9"))
        assert len(agent._parallel_batch(calls)) == 2

    def test_a_batch_never_starts_with_a_write(self, agent):
        calls = [ToolCall(name="write_file",
                          arguments={"path": "f0.py", "content": "x"},
                          call_id="0")] + reads("f1.py", "f2.py")
        assert agent._parallel_batch(calls) == []

    def test_shell_is_never_batched(self, agent):
        """It is marked concurrency_safe = False, and now that means
        something."""
        calls = [ToolCall(name="shell", arguments={"command": "ls"},
                          call_id=str(i)) for i in range(3)]
        assert agent._parallel_batch(calls) == []

    def test_an_unknown_tool_stops_the_batch(self, agent):
        calls = reads("f0.py", "f1.py")
        calls.insert(0, ToolCall(name="nonsense", arguments={}, call_id="9"))
        assert agent._parallel_batch(calls) == []

    def test_a_blocked_call_stops_the_batch(self, agent):
        """Plan mode refuses rather than asks, and that answer is per call."""
        agent.permissions.mode = Mode.PLAN
        calls = reads("f0.py", "f1.py")
        calls.append(ToolCall(name="write_file",
                              arguments={"path": "f0.py", "content": "x"},
                              call_id="9"))
        assert len(agent._parallel_batch(calls)) == 2


class TestRunningThem:
    def test_results_come_back_in_the_order_asked_for(self, agent):
        """Race-dependent ordering would make the same turn read differently
        each time, and the model's next step depends on that order."""
        calls = reads("f0.py", "f1.py", "f2.py")
        asyncio.run(agent._run_together(calls))
        bodies = [m["content"] for m in agent.session.messages
                  if m.get("role") == "tool"]
        assert len(bodies) == 3
        assert "value = 0" in bodies[0]
        assert "value = 1" in bodies[1]
        assert "value = 2" in bodies[2]

    def test_they_really_do_overlap(self, agent, monkeypatch):
        started, finished = [], []

        async def slow(self, args):
            from wynxo.tools.base import ToolResult

            started.append(time.monotonic())
            await asyncio.sleep(0.25)
            finished.append(time.monotonic())
            return ToolResult.success("done")

        monkeypatch.setattr("wynxo.tools.files.ReadFile.run", slow)
        began = time.monotonic()
        asyncio.run(agent._run_together(reads("f0.py", "f1.py", "f2.py")))
        elapsed = time.monotonic() - began
        assert len(finished) == 3
        assert elapsed < 0.6, f"ran one at a time ({elapsed:.2f}s)"

    def test_one_failure_does_not_lose_the_others(self, agent):
        calls = reads("f0.py", "does-not-exist.py", "f2.py")
        asyncio.run(agent._run_together(calls))
        bodies = [m["content"] for m in agent.session.messages
                  if m.get("role") == "tool"]
        assert len(bodies) == 3
        assert "value = 0" in bodies[0] and "value = 2" in bodies[2]

    def test_a_crashing_tool_becomes_a_result_not_a_crash(self, agent,
                                                          monkeypatch):
        async def explode(self, args):
            raise RuntimeError("fell over")

        monkeypatch.setattr("wynxo.tools.files.ReadFile.run", explode)
        asyncio.run(agent._run_together(reads("f0.py", "f1.py")))
        bodies = [m["content"] for m in agent.session.messages
                  if m.get("role") == "tool"]
        assert len(bodies) == 2

    def test_an_interrupt_is_not_turned_into_a_result(self, agent,
                                                      monkeypatch):
        """Ctrl-C has to reach the REPL, not be recorded as a tool failure."""
        async def interrupted(self, args):
            raise asyncio.CancelledError

        monkeypatch.setattr("wynxo.tools.files.ReadFile.run", interrupted)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(agent._run_together(reads("f0.py", "f1.py")))

    def test_the_context_budget_is_split_between_them(self, agent):
        """They land in the same context together, so three reads each sized
        against the whole of it would overflow between them."""
        big = "\n".join(f"line {i} of a long file" for i in range(20_000))
        for i in range(3):
            (agent.workspace / f"f{i}.py").write_text(big, encoding="utf-8")

        asyncio.run(agent._run_together(reads("f0.py", "f1.py", "f2.py")))
        bodies = [m["content"] for m in agent.session.messages
                  if m.get("role") == "tool"]
        from wynxo.session import estimate_tokens

        total = sum(estimate_tokens(b) for b in bodies)
        assert total < agent.policy.context_budget

    def test_the_originals_are_not_left_holding_a_budget(self, agent):
        asyncio.run(agent._run_together(reads("f0.py", "f1.py")))
        assert agent.tools.get("read_file").context_left == 0


class TestTheWholeLoopStillWorks:
    def test_a_mixed_turn_keeps_every_result_in_order(self, agent):
        from wynxo.parsing import ParsedTurn

        turn = ParsedTurn()
        turn.tool_calls = reads("f0.py", "f1.py") + [
            ToolCall(name="write_file",
                     arguments={"path": "new.py", "content": "written\n"},
                     call_id="9")] + reads("f2.py")
        agent.permissions.mode = Mode.YOLO
        assert asyncio.run(agent._run_tool_calls(turn)) is True

        bodies = [m["content"] for m in agent.session.messages
                  if m.get("role") == "tool"]
        assert len(bodies) == 4
        assert "value = 0" in bodies[0] and "value = 1" in bodies[1]
        assert "value = 2" in bodies[3]
        assert (agent.workspace / "new.py").read_text() == "written\n"
