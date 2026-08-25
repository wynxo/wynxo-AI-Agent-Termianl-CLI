"""Surviving a connection that drops before the model has said anything.

Ollama refuses connections while it loads a model, and loading a 30B from
cold takes long enough that the first request of a session is the one most
likely to be hit. That used to end the turn.

The hazard is the other half: retrying after tokens have reached the user
replays the answer from the top and prints it twice, which is worse than the
error it was hiding. So the retry window is exactly "nothing emitted yet".
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from wynxo.config import Config, Endpoint
from wynxo.provider import CONNECT_ATTEMPTS, Chunk, OllamaClient, ProviderError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("wynxo.provider.RETRY_BACKOFF", 0.0)
    config = Config()
    config.endpoints = [Endpoint(name="t", url="http://127.0.0.1:99999")]
    config.active_endpoint = "t"
    return OllamaClient(config)


def drive(client) -> list[Chunk]:
    async def go():
        return [c async for c in client.chat([{"role": "user", "content": "hi"}])]

    return asyncio.run(go())


def failing(times: int, exc, then=("recovered",)):
    """A _stream_chat that fails `times` times, then answers."""
    state = {"left": times}

    async def stream(payload):
        if state["left"] > 0:
            state["left"] -= 1
            raise exc
        for piece in then:
            yield Chunk(content=piece)
        yield Chunk(content="", done=True)

    return stream, state


class TestRetryingBeforeAnythingWasSaid:
    @pytest.mark.parametrize("exc", [
        httpx.ConnectError("refused"),
        httpx.ReadError("reset"),
        httpx.RemoteProtocolError("closed early"),
        httpx.ConnectTimeout("timed out connecting"),
        httpx.PoolTimeout("no connection free"),
    ])
    def test_a_dropped_connection_is_retried(self, client, monkeypatch, exc):
        stream, _ = failing(1, exc)
        monkeypatch.setattr(client, "_stream_chat", stream)
        assert [c.content for c in drive(client) if c.content] == ["recovered"]

    def test_it_keeps_trying_up_to_the_limit(self, client, monkeypatch):
        stream, state = failing(CONNECT_ATTEMPTS - 1, httpx.ConnectError("no"))
        monkeypatch.setattr(client, "_stream_chat", stream)
        drive(client)
        assert state["left"] == 0

    def test_it_gives_up_with_something_useful(self, client, monkeypatch):
        stream, _ = failing(99, httpx.ConnectError("no"))
        monkeypatch.setattr(client, "_stream_chat", stream)
        with pytest.raises(ProviderError) as caught:
            drive(client)
        message = str(caught.value)
        assert "ollama serve" in message.lower()
        assert str(CONNECT_ATTEMPTS) in message

    def test_it_backs_off_between_attempts(self, client, monkeypatch):
        slept = []

        async def record(seconds):
            slept.append(seconds)

        monkeypatch.setattr("wynxo.provider.RETRY_BACKOFF", 0.5)
        monkeypatch.setattr("wynxo.provider.asyncio.sleep", record)
        stream, _ = failing(99, httpx.ConnectError("no"))
        monkeypatch.setattr(client, "_stream_chat", stream)
        with pytest.raises(ProviderError):
            drive(client)
        assert slept and slept == sorted(slept), "backoff should not shrink"


class TestNotRetryingOnceItHasSpoken:
    def test_a_mid_answer_drop_is_not_replayed(self, client, monkeypatch):
        """Replaying would print the answer twice -- worse than the error."""
        attempts = {"n": 0}

        async def stream(payload):
            attempts["n"] += 1
            yield Chunk(content="half an answer")
            raise httpx.RemoteProtocolError("dropped")

        monkeypatch.setattr(client, "_stream_chat", stream)
        with pytest.raises(ProviderError):
            drive(client)
        assert attempts["n"] == 1, "it retried after emitting"

    def test_it_says_the_answer_is_incomplete(self, client, monkeypatch):
        async def stream(payload):
            yield Chunk(content="half an answer")
            raise httpx.ReadError("dropped")

        monkeypatch.setattr(client, "_stream_chat", stream)
        with pytest.raises(ProviderError) as caught:
            drive(client)
        assert "incomplete" in str(caught.value)


class TestWhatIsNotRetried:
    def test_a_read_timeout_keeps_its_own_advice(self, client, monkeypatch):
        """A slow model is not a broken connection, and the advice for it is
        different -- raise request_timeout, not 'check the server'."""
        async def stream(payload):
            raise httpx.ReadTimeout("too slow")
            yield  # pragma: no cover

        monkeypatch.setattr(client, "_stream_chat", stream)
        with pytest.raises(ProviderError) as caught:
            drive(client)
        assert "request_timeout" in str(caught.value)

    def test_a_provider_error_is_not_retried(self, client, monkeypatch):
        """A 400 will be a 400 again. Retrying only delays the message."""
        attempts = {"n": 0}

        async def stream(payload):
            attempts["n"] += 1
            raise ProviderError("model not found")
            yield  # pragma: no cover

        monkeypatch.setattr(client, "_stream_chat", stream)
        with pytest.raises(ProviderError):
            drive(client)
        assert attempts["n"] == 1
