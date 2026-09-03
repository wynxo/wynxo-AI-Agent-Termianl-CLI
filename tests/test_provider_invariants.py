import httpx
import pytest

from wynxo.config import Config, Endpoint
from wynxo.provider import OllamaClient, ProviderError


@pytest.mark.asyncio
async def test_tolerates_missing_load_duration_without_corrupting_total_duration():
    config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")], active_endpoint="t")
    client = OllamaClient(config)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text='{"message":{"role":"assistant","content":"ok"},"done":true,'
                     '"total_duration":900}',
            )
        ),
        base_url="http://fake:11434",
    )
    chunks = [chunk async for chunk in client.chat([])]
    assert chunks[-1].total_duration_ns == 900
    assert chunks[-1].load_duration_ns == 0
    await client.aclose()


def test_provider_error_is_not_exposed_as_raw_httpx_error():
    assert issubclass(ProviderError, RuntimeError)
