from __future__ import annotations

import asyncio

from wynxo.config import Config
from wynxo.provider import OllamaClient
import wynxo.runtime_compat  # noqa: F401  # installs the compatibility shim


class _Response:
    status_code = 200


class _Client:
    def __init__(self):
        self.payload = None

    async def post(self, path, *, json, timeout):
        assert path == "/api/chat"
        self.payload = json
        return _Response()


def test_warm_accepts_messages_and_tools():
    config = Config()
    client = OllamaClient(config)
    fake = _Client()
    client._client = fake

    ok = asyncio.run(client.warm(
        "test-model",
        messages=[{"role": "system", "content": "hello"}],
        tools=[{"type": "function"}],
    ))

    assert ok is True
    assert fake.payload["model"] == "test-model"
    assert fake.payload["messages"] == [{"role": "system", "content": "hello"}]
    assert fake.payload["tools"] == [{"type": "function"}]
    assert fake.payload["options"]["num_predict"] == 1


def test_warm_without_prompt_keeps_legacy_load_only_behavior():
    config = Config()
    client = OllamaClient(config)
    fake = _Client()
    client._client = fake

    ok = asyncio.run(client.warm("test-model"))

    assert ok is True
    assert fake.payload["messages"] == []
    assert "num_predict" not in fake.payload["options"]
