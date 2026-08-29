"""The OpenAI-compatible backend and the client factory.

OpenAI is exercised offline: the httpx transport is faked so the SSE parse
and message translation are what get tested, not the network.
"""

from __future__ import annotations

import asyncio

import pytest

from wynxo.config import Config, Endpoint
from wynxo.provider import (
    OpenAIClient, OllamaClient, _openai_messages, make_client,
)


class _FakeStream:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aread(self):
        return b"error body"

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()


class _FakeTransport:
    def __init__(self, lines, status=200):
        self.lines = lines
        self.status = status
        self.closed = False
        self.last: tuple | None = None

    async def aclose(self):
        self.closed = True

    async def get(self, url, **kwargs):
        return _FakeStream([], status=self.status)

    def stream(self, method, url, **kwargs):
        self.last = (method, url, kwargs)
        return _FakeStream(self.lines, self.status)


def _client(lines, transport=None):
    config = Config(endpoints=[Endpoint(url="https://api.example.com",
                                        api_key="k", kind="openai")])
    client = OpenAIClient(config)
    client._client = transport or _FakeTransport(lines)
    return client


def _chat(client, **kw):
    async def run():
        return [chunk async for chunk
                in client.chat([{"role": "user", "content": "hi"}], **kw)]
    return asyncio.run(run())


def test_factory_picks_the_client_by_kind():
    openai = Config(endpoints=[Endpoint(url="https://api.example.com", kind="openai")])
    assert isinstance(make_client(openai), OpenAIClient)
    ollama = Config(endpoints=[Endpoint(url="http://127.0.0.1:11434")])
    assert isinstance(make_client(ollama), OllamaClient)
    auto = Config(endpoints=[Endpoint(url="http://127.0.0.1:11434", kind="auto")])
    assert isinstance(make_client(auto), OllamaClient)


def test_chat_text_turn_yields_content_and_done():
    lines = [
        'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        'data: {"usage":{"prompt_tokens":7,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    chunks = _chat(_client(lines))
    assert [c.content for c in chunks if c.content] == ["Hel", "lo"]
    final = chunks[-1]
    assert final.done is True
    assert final.tool_calls == []
    assert final.prompt_tokens == 7
    assert final.completion_tokens == 2


def test_chat_reasoning_becomes_thinking():
    lines = [
        'data: {"choices":[{"index":0,"delta":{"reasoning":"pondering..."}}]}',
        'data: {"choices":[{"index":0,"delta":{"content":"answer"}}]}',
        "data: [DONE]",
    ]
    chunks = _chat(_client(lines))
    assert any(c.thinking == "pondering..." for c in chunks)
    assert [c.content for c in chunks if c.content] == ["answer"]


def test_chat_accumulates_tool_calls_across_deltas():
    lines = [
        'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":"}}]},"finish_reason":null}]}',
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"a.py\\"}"}}]},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
    chunks = _chat(_client(lines), tools=[{"type": "function"}])
    final = chunks[-1]
    assert final.done is True
    assert len(final.tool_calls) == 1
    call = final.tool_calls[0]
    assert call["id"] == "call_9"
    assert call["function"]["name"] == "read_file"
    assert call["function"]["arguments"] == '{"path":"a.py"}'


def test_transport_uses_the_v1_path():
    transport = _FakeTransport(['data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}', "data: [DONE]"])
    _chat(_client([], transport=transport))
    method, url, kwargs = transport.last
    assert method == "POST"
    assert url == "/v1/chat/completions"
    assert kwargs["json"]["temperature"] == 0.4


def test_openai_message_translation():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "check"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "x"}}}]},
        {"role": "tool", "content": "content here", "tool_call_id": "call_1"},
    ]
    out = _openai_messages(messages)
    assert out[0] == {"role": "system", "content": "sys"}
    assistant = out[2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"path": "x"}'
    assert assistant["tool_calls"][0]["type"] == "function"
    tool = out[3]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_1"
    assert tool["content"] == "content here"


def test_pull_is_not_supported():
    client = _client([])
    with pytest.raises(Exception) as exc:
        asyncio.run(client.pull("model"))
    assert "OpenAI-compatible" in str(exc.value)