"""Not letting one read quietly eat the context window.

read_file defaults to 2000 lines, which is roughly 22k tokens -- more than
the entire budget at low effort. Nothing used to notice: the read succeeded,
the oldest messages fell out of the window, and the model got quietly stupid
halfway through a task with no error anywhere to explain it.
"""

from __future__ import annotations

import asyncio

import pytest

from wynxo.session import estimate_tokens
from wynxo.tools.files import MIN_READ_TOKENS, READ_SHARE, ReadFile, ReadInput


@pytest.fixture
def big(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(
        f"def function_{i}(argument_one, argument_two):  # line {i}"
        for i in range(2000)), encoding="utf-8")
    return tmp_path


def read(workspace, context_left, path="big.py", **kwargs):
    tool = ReadFile(workspace=workspace)
    tool.context_left = context_left
    return asyncio.run(tool.run(ReadInput(path=path, **kwargs)))


class TestTrimmingToFit:
    def test_a_big_read_is_cut_down_when_context_is_tight(self, big):
        result = read(big, context_left=8_000)
        assert estimate_tokens(result.output) < 8_000 * READ_SHARE + 200

    def test_it_scales_with_what_is_actually_left(self, big):
        roomy = estimate_tokens(read(big, context_left=20_000).output)
        cramped = estimate_tokens(read(big, context_left=4_000).output)
        assert cramped < roomy

    def test_it_says_how_to_get_the_rest(self, big):
        """A refusal leaves a weaker model nowhere to go; it just asks for
        the same file again."""
        output = read(big, context_left=8_000).output
        assert "offset=" in output and "grep" in output
        assert "of 2000" in output

    def test_the_offset_it_suggests_continues_where_it_stopped(self, big):
        import re

        output = read(big, context_left=8_000).output
        shown = re.search(r"showing lines 1-(\d+) of", output)
        suggested = re.search(r"offset=(\d+)", output)
        assert shown and suggested
        assert int(suggested.group(1)) == int(shown.group(1))

    def test_the_lines_it_returns_are_still_numbered_from_the_offset(self, big):
        output = read(big, context_left=8_000, offset=100).output
        assert "\n  101\t" in output or output.lstrip().startswith("101\t")


class TestNotInterfering:
    def test_an_unknown_budget_changes_nothing(self, big):
        """Zero means unknown. A guard that fires on missing information
        would be worse than no guard."""
        assert estimate_tokens(read(big, context_left=0).output) > 20_000

    def test_plenty_of_room_means_no_trimming(self, big):
        assert estimate_tokens(read(big, context_left=500_000).output) > 20_000

    def test_a_small_file_is_never_touched(self, tmp_path):
        (tmp_path / "small.py").write_text("print('hi')\n", encoding="utf-8")
        result = read(tmp_path, context_left=1_000, path="small.py")
        assert "left out" not in result.output
        assert "print('hi')" in result.output

    def test_it_never_trims_to_nothing(self, big):
        """A read cut to nothing teaches the model only that reading does
        not work, and it tries something worse."""
        result = read(big, context_left=1)
        assert estimate_tokens(result.output) >= MIN_READ_TOKENS * 0.5
        assert "function_0" in result.output

    def test_a_trimmed_read_still_succeeds(self, big):
        assert read(big, context_left=2_000).ok is True


class TestTheAgentSuppliesTheNumber:
    def _agent(self, tmp_path):
        from unittest.mock import MagicMock

        from wynxo.agent import Agent
        from wynxo.config import Config
        from wynxo.effort import resolve

        return Agent(client=MagicMock(), config=Config(),
                     policy=resolve("medium"), workspace=tmp_path)

    def test_it_reports_what_is_free(self, tmp_path):
        agent = self._agent(tmp_path)
        left = agent._context_left()
        assert 0 < left <= agent.policy.context_budget

    def test_it_shrinks_as_the_conversation_grows(self, tmp_path):
        agent = self._agent(tmp_path)
        before = agent._context_left()
        agent.session.add_user("x" * 20_000)
        assert agent._context_left() < before

    def test_it_never_goes_negative(self, tmp_path):
        """A negative budget would read as "unknown" and switch the guard
        off exactly when it is needed most."""
        agent = self._agent(tmp_path)
        agent.session.add_user("x" * 5_000_000)
        assert agent._context_left() == 0

    def test_the_budget_is_cleared_after_the_call(self, tmp_path):
        """A stale budget on a reused tool would size the next read against
        a number from the wrong turn."""
        from wynxo.parsing import ToolCall

        agent = self._agent(tmp_path)
        (tmp_path / "x.py").write_text("print(1)\n", encoding="utf-8")
        tool = agent.tools.get("read_file")
        asyncio.run(agent._run_one(
            ToolCall(name="read_file", arguments={"path": "x.py"}, call_id="1")))
        assert tool.context_left == 0
