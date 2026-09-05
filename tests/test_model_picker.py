"""The model picker is now the only place a model is chosen, so it has to be
right: it must reflect the server rather than any built-in opinion, and it
must say which models can actually drive an agent."""

import json

import httpx
from unittest.mock import AsyncMock

from prompt_toolkit import PromptSession

from wynxo.config import Config, Endpoint
from wynxo.provider import OllamaClient, inspect_all

CATALOGUE = {
    "qwen3-coder:30b": (18_600_000_000, "30.5B", "Q4_K_M", ["completion", "tools"], 262144),
    "gemma4:latest": (9_600_000_000, "12.2B", "Q4_K_M", ["completion"], 8192),
    "qwen3.8:27b": (17_700_000_000, "27.2B", "Q4_K_M",
                    ["completion", "tools", "thinking"], 131072),
}


def transport(catalogue=None, tags_only=False):
    catalogue = CATALOGUE if catalogue is None else catalogue

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.32.14"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": n, "size": v[0],
                 "details": {"parameter_size": v[1], "quantization_level": v[2]}}
                for n, v in catalogue.items()]})
        if request.url.path == "/api/show":
            if tags_only:
                return httpx.Response(500, json={"error": "nope"})
            name = json.loads(request.content).get("model")
            entry = catalogue.get(name)
            if entry is None:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(200, json={
                "capabilities": entry[3],
                "details": {"parameter_size": entry[1], "quantization_level": entry[2]},
                "model_info": {"qwen3.context_length": entry[4]}})
        return httpx.Response(404, json={"error": "no route"})

    return httpx.MockTransport(handler)


def make_client(catalogue=None, tags_only=False):
    config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")],
                    active_endpoint="t")
    client = OllamaClient(config)
    client._client = httpx.AsyncClient(transport=transport(catalogue, tags_only),
                                       base_url="http://fake:11434")
    return client, config


def fake_prompt(answer=""):
    session = PromptSession.__new__(PromptSession)
    session.prompt_async = AsyncMock(return_value=answer)
    return session


class TestCapabilityDiscovery:
    async def test_capabilities_are_filled_in(self):
        client, _ = make_client()
        models = await inspect_all(client, await client.list_models())
        by_name = {m.name: m for m in models}
        assert by_name["qwen3-coder:30b"].supports_tools
        assert not by_name["gemma4:latest"].supports_tools
        assert by_name["qwen3.8:27b"].supports_thinking
        await client.aclose()

    async def test_a_failing_show_does_not_break_the_list(self):
        """One unreadable model must not cost you the whole picker."""
        client, _ = make_client(tags_only=True)
        models = await inspect_all(client, await client.list_models())
        assert [m.name for m in models] == sorted(CATALOGUE)
        await client.aclose()
