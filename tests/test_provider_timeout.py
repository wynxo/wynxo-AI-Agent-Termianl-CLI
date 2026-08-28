import asyncio

from wynxo.config import Config
from wynxo.provider import OllamaClient


class FakeResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def aiter_lines(self):
        yield '{"message":{"content":"ok"},"done":true}'


class FakeHTTPClient:
    def __init__(self):
        self.timeout = None

    def stream(self, method, path, *, json, timeout):
        self.timeout = timeout
        return FakeResponse()



def test_stream_chat_uses_configured_request_timeout():
    config = Config()
    config.request_timeout = 123.0
    client = OllamaClient(config)
    fake = FakeHTTPClient()
    client._client = fake

    async def run():
        chunks = [c async for c in client._stream_chat({"model": "test"})]
        assert chunks[0].content == "ok"

    asyncio.run(run())
    assert fake.timeout == 123.0
