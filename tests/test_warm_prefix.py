"""What the warm-up reads, and whether the first question can reuse it.

Loading the weights is only half the wait. wynxo's system prompt and tool
schemas come to somewhere north of five thousand tokens, and a local model
reads every one of them before it writes a word -- where `ollama run` sends
your message and nothing else. That is most of the gap between the two.

Ollama keeps the KV cache between requests and reuses whatever prefix
matches, so the warm-up reads that prompt while you are still typing. Which
is worth exactly nothing unless what it reads really is a prefix of what
the first question sends -- so that is what is checked here.
"""

from __future__ import annotations

import json

import httpx
import pytest

from test_agent import make_agent


def rendered(body: dict) -> str:
    """Roughly what the chat template produces, in order: the system turn
    with the tool schemas inside it, then the conversation."""
    system, rest = "", []
    for message in body.get("messages", []):
        if message.get("role") == "system" and not system:
            system = str(message.get("content", ""))
        else:
            rest.append(json.dumps(message))
    return system + json.dumps(body.get("tools") or []) + "".join(rest)


@pytest.fixture
def wired(tmp_path):
    agent, fake, _ = make_agent(tmp_path, [{"content": "Done."}])
    return agent, fake


class TestTheWarmUpReadsTheRealPrompt:
    async def test_it_sends_the_system_prompt_and_the_tools(self, wired):
        agent, fake = wired
        await agent.backend.client.warm(
            agent.config.model, messages=agent.session.wire(),
            tools=agent.tools.ollama_schemas())
        body = fake.requests[-1]
        assert body["messages"], "an empty warm-up caches nothing"
        assert body["tools"], "the schemas are most of the prompt"
        assert "wynxo" in str(body["messages"][0]["content"])

    async def test_it_generates_almost_nothing(self, wired):
        """One token, to be sure the prompt was really evaluated. A warm-up
        that quietly skipped the prefill would look exactly like one that
        worked."""
        agent, fake = wired
        await agent.backend.client.warm(agent.config.model,
                                 messages=agent.session.wire())
        assert fake.requests[-1]["options"]["num_predict"] == 1

    async def test_it_carries_the_window_every_later_request_will(self, wired):
        """Loading under a different num_ctx is worse than not loading:
        Ollama evicts and reloads on the first real question, so the wait is
        paid twice."""
        agent, fake = wired
        await agent.backend.client.warm(agent.config.model,
                                 messages=agent.session.wire())
        warm = fake.requests[-1]
        await agent.run("hello")
        real = fake.requests[-1]
        assert warm["options"]["num_ctx"] == real["options"]["num_ctx"]
        assert warm["keep_alive"] == real["keep_alive"]

    async def test_with_no_messages_it_still_just_loads(self, wired):
        """The old behaviour, kept: a caller with no prompt to offer gets a
        plain load rather than an error."""
        agent, fake = wired
        await agent.backend.client.warm(agent.config.model)
        body = fake.requests[-1]
        assert body["messages"] == []
        assert "num_predict" not in body["options"]


class TestTheFirstQuestionCanReuseIt:
    async def test_what_was_warmed_is_a_prefix_of_what_is_asked(self, wired):
        """The whole point. A prompt that differs by one character shares
        nothing: the cache is keyed on the prefix, and the first mismatch
        throws away everything after it."""
        agent, fake = wired
        await agent.backend.client.warm(
            agent.config.model, messages=agent.session.wire(),
            tools=agent.tools.ollama_schemas())
        warm = rendered(fake.requests[-1])
        await agent.run("what does a.txt say")
        real = rendered(fake.requests[-1])
        assert real.startswith(warm), (
            "the warm-up read something the question cannot reuse")
        assert len(warm) > 1000, "a prefix worth caching"

    async def test_the_question_itself_is_all_that_is_left_to_read(self, wired):
        agent, fake = wired
        await agent.backend.client.warm(
            agent.config.model, messages=agent.session.wire(),
            tools=agent.tools.ollama_schemas())
        warm = rendered(fake.requests[-1])
        await agent.run("hello")
        real = rendered(fake.requests[-1])
        assert len(real) - len(warm) < 400, \
            f"{len(real) - len(warm)} characters still to read"


class TestAWarmUpCannotFailAStartUp:
    async def test_a_server_that_refuses_is_not_an_error(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [{"content": "Done."}])

        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "no"})

        agent.backend.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(refuse),
            base_url="http://fake:11434")
        assert await agent.backend.client.warm(
            agent.config.model, messages=agent.session.wire()) is False

    async def test_a_server_that_is_not_there_is_not_an_error(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [{"content": "Done."}])

        def die(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        agent.backend.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(die), base_url="http://fake:11434")
        assert await agent.backend.client.warm(agent.config.model) is False
