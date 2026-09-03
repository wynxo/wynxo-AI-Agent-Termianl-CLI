"""What the conversation remembers when you press Ctrl-C mid-answer.

Half an answer is still something the model said, and it is still on the
screen above the prompt. Without a record of it the next question goes to a
model that would deny having written those words, and "carry on from there"
has nothing to carry on from.
"""

from __future__ import annotations

import asyncio

import pytest

from wynxo.agent import Callbacks, Interrupted
from wynxo.provider import Chunk

from test_agent import make_agent

ANSWER = "for attempt in range(3):\n    try_it()\n"


class Impatient(Callbacks):
    """Presses Ctrl-C once some of the answer is on screen."""

    def __init__(self, after: int = 10):
        self.seen = ""
        self.after = after
        self.task: asyncio.Task | None = None

    async def on_content(self, text):
        self.seen += text
        if len(self.seen) >= self.after and self.task is not None:
            self.task.cancel()
            self.task = None


def chunked(text: str, size: int = 4, thinking: str = ""):
    """A backend that streams, with a real await between chunks -- which is
    what gives a cancellation somewhere to land."""

    def chat(*args, **kwargs):
        async def gen():
            if thinking:
                yield Chunk(thinking=thinking)
                await asyncio.sleep(0.01)
            for i in range(0, len(text), size):
                yield Chunk(content=text[i:i + size])
                await asyncio.sleep(0.01)
            yield Chunk(done=True, prompt_tokens=10, completion_tokens=10)
        return gen()

    return chat


async def interrupted(tmp_path, *, text=ANSWER, thinking="", after=10):
    cb = Impatient(after)
    agent, _, _ = make_agent(tmp_path, [{"content": text}], callbacks=cb)
    agent.permissions.yolo = True
    agent.backend.chat = chunked(text, thinking=thinking)
    task = asyncio.ensure_future(agent.run("write a loop"))
    cb.task = task
    with pytest.raises((asyncio.CancelledError, Interrupted)):
        await task
    return agent, cb


class TestTheTranscriptMatchesTheScreen:
    async def test_what_was_streamed_is_in_the_conversation(self, tmp_path):
        agent, cb = await interrupted(tmp_path)
        said = [m for m in agent.session.messages
                if m.get("role") == "assistant"]
        assert said, "the half-answer on screen was not recorded at all"
        assert said[-1]["content"] in cb.seen
        assert said[-1]["content"].startswith("for attempt")

    async def test_the_question_survives_too(self, tmp_path):
        agent, _ = await interrupted(tmp_path)
        assert agent.session.messages[0]["content"] == "write a loop"

    async def test_nothing_is_invented_when_nothing_was_shown(self, tmp_path):
        """Cancelled while the model was still reading the prompt: there is
        no half-answer, and an empty assistant message is worse than none --
        it puts the model on record as having said nothing."""
        cb = Impatient()
        agent, _, _ = make_agent(tmp_path, [{"content": ANSWER}], callbacks=cb)
        agent.permissions.yolo = True

        started = asyncio.Event()

        def chat(*args, **kwargs):
            async def gen():
                started.set()
                await asyncio.sleep(10)       # thinking, nothing emitted yet
                yield Chunk(done=True)
            return gen()

        agent.backend.chat = chat
        task = asyncio.ensure_future(agent.run("write a loop"))
        await started.wait()
        task.cancel()
        with pytest.raises((asyncio.CancelledError, Interrupted)):
            await task
        assert cb.seen == ""
        assert [m for m in agent.session.messages
                if m.get("role") == "assistant"] == []

    async def test_reasoning_is_not_recorded_as_the_answer(self, tmp_path):
        """The raw stream carries the model's scratchpad and any half-
        written tool call. What goes in the session is what went on the
        screen."""
        agent, _ = await interrupted(
            tmp_path, thinking="I should write a loop here.")
        for message in agent.session.messages:
            assert "I should write a loop" not in str(message.get("content"))


class TestAFinishedTurnIsUnaffected:
    async def test_the_whole_answer_is_recorded_once(self, tmp_path):
        cb = Impatient(after=10 ** 6)      # never interrupts
        agent, _, _ = make_agent(tmp_path, [{"content": ANSWER}], callbacks=cb)
        agent.permissions.yolo = True
        agent.backend.chat = chunked(ANSWER)
        await agent.run("write a loop")
        said = [m["content"] for m in agent.session.messages
                if m.get("role") == "assistant"]
        assert said.count(ANSWER.strip()) <= 1
        assert agent._streaming_shown is None, \
            "a finished stream must leave nothing for a later cancel to add"
