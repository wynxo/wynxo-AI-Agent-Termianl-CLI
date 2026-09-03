"""Reasoning shown on screen, and reasoning that must not be.

Some model calls in a turn are not the turn's answer: the planner, the
verifier, the pass that repairs a malformed tool call, and -- the one that
matters -- a safety-sensitive turn, which is generated in full and then
decided about before any of it is shown.

Those calls pass ``stream_content=False``. It used to mean only that the
*answer* was withheld: the model's reasoning went to the terminal a
character at a time regardless. That is two things at once. It puts a
second "thinking" block in the transcript with no answer under it, for a
question the user never asked; and on the safety path it streams the
model's working out through the boundary that exists to hold the whole
turn back until it has been looked at.
"""

from __future__ import annotations

import pytest

from wynxo.agent import Callbacks

from test_agent import make_agent


class Watcher(Callbacks):
    """Records what actually reached the screen."""

    def __init__(self):
        self.thinking, self.content, self.stages = [], [], []

    async def on_thinking(self, text):
        self.thinking.append(text)

    async def on_content(self, text):
        self.content.append(text)

    async def on_stage(self, name, detail=""):
        self.stages.append(name)


TURN = {"content": "The answer.", "thinking": "Working through it."}


@pytest.fixture
def watcher():
    return Watcher()


class TestWhatIsStreamedIsStreamedWhole:
    async def test_an_ordinary_call_shows_its_reasoning(self, tmp_path, watcher):
        agent, *_ = make_agent(tmp_path, [TURN], callbacks=watcher)
        await agent._call_model(messages=[{"role": "user", "content": "hi"}],
                                use_tools=False)
        assert "".join(watcher.thinking) == "Working through it."
        assert "".join(watcher.content) == "The answer."


class TestAHiddenAnswerHidesItsReasoning:
    async def test_a_withheld_call_shows_nothing(self, tmp_path, watcher):
        agent, *_ = make_agent(tmp_path, [TURN], callbacks=watcher)
        await agent._call_model(messages=[{"role": "user", "content": "hi"}],
                                use_tools=False, stream_content=False)
        assert watcher.content == []
        assert watcher.thinking == [], \
            "the working of a call whose conclusion is hidden"

    async def test_the_reasoning_is_still_recorded(self, tmp_path, watcher):
        """Not shown is not the same as thrown away: the parsed turn keeps
        it, so anything written afterwards has the whole call."""
        agent, *_ = make_agent(tmp_path, [TURN], callbacks=watcher)
        turn = await agent._call_model(
            messages=[{"role": "user", "content": "hi"}],
            use_tools=False, stream_content=False)
        assert turn.thinking == "Working through it."
        assert turn.content == "The answer."

    async def test_a_silent_call_shows_nothing_either(self, tmp_path, watcher):
        agent, *_ = make_agent(tmp_path, [TURN], callbacks=watcher)
        await agent._call_model(messages=[{"role": "user", "content": "hi"}],
                                use_tools=False, stream_content=False,
                                silent=True)
        assert watcher.thinking == []
        assert watcher.stages == [], "infrastructure narrates no progress"

    async def test_a_streaming_call_still_announces_its_stage(self, tmp_path, watcher):
        agent, *_ = make_agent(tmp_path, [TURN], callbacks=watcher)
        await agent._call_model(messages=[{"role": "user", "content": "hi"}],
                                use_tools=False)
        assert watcher.stages == ["thinking"]


class TestOneAnswerOneThinkingBlock:
    async def test_a_planning_turn_narrates_its_reasoning_once(self, tmp_path, watcher):
        """At an effort level that plans first, the turn made two model
        calls and streamed the reasoning of both -- so the transcript
        carried two identical-looking "thinking" heads, the first of them
        belonging to a plan the transcript deliberately does not show.
        """
        agent, *_ = make_agent(
            tmp_path,
            [{"content": "1. do the thing", "thinking": "planning it out"},
             {"content": "Done.", "thinking": "writing the answer"}],
            effort="high", callbacks=watcher)
        plan = await agent._plan("do the thing")
        assert plan, "the plan itself is still produced"
        assert watcher.thinking == []
