"""The model picker is now the only place a model is chosen, so it has to be
right: it must reflect the server rather than any built-in opinion, and it
must say which models can actually drive an agent."""

import json

import httpx
import pytest
from unittest.mock import AsyncMock

from prompt_toolkit import PromptSession

from wynxo.config import Config, Endpoint
from wynxo.provider import OllamaClient, inspect_all
from wynxo.ui import UI
from wynxo.wizard import _badge, _humanise_context, _print_model_rows, ask_model

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
        models = await inspect_all(client, await client.list_models(), timeout=1)
        assert len(models) == 3
        assert all(not m.capabilities_known for m in models)
        await client.aclose()


class TestBadges:
    def test_badge_reflects_capability(self):
        class M:
            def __init__(self, caps):
                self.capabilities = caps

            capabilities_known = property(lambda self: self.capabilities is not None)
            supports_tools = property(lambda self: "tools" in (self.capabilities or []))
            supports_thinking = property(
                lambda self: "thinking" in (self.capabilities or []))

        assert _badge(M(["tools"]))[0] == "tools"
        assert _badge(M(["tools", "thinking"]))[0] == "tools + think"
        assert _badge(M(["completion"]))[0] == "no tools"
        assert _badge(M(None))[0] == "unknown"

    def test_context_is_humanised(self):
        assert _humanise_context(262144) == "256k ctx"
        assert _humanise_context(8192) == "8k ctx"
        assert _humanise_context(0) == ""


class TestPicker:
    async def test_lists_only_what_the_server_has(self, capsys):
        client, config = make_client()
        chosen = await ask_model(UI(), fake_prompt(""), config, client)
        await client.aclose()
        out = capsys.readouterr().out
        assert "qwen3-coder:30b" in out and "gemma4:latest" in out
        # No built-in catalogue leaking in as if it were installed.
        assert "devstral" not in out and "qwen3:8b" not in out
        assert chosen in CATALOGUE

    async def test_tool_capable_models_come_first(self, capsys):
        client, config = make_client()
        chosen = await ask_model(UI(), fake_prompt(""), config, client)
        await client.aclose()
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.strip().startswith(("1", "2", "3"))]
        assert "gemma4" in lines[-1], "the model that cannot call tools should sort last"
        assert chosen != "gemma4:latest"

    async def test_non_tool_models_are_flagged(self, capsys):
        client, config = make_client()
        await ask_model(UI(), fake_prompt(""), config, client)
        await client.aclose()
        out = capsys.readouterr().out
        assert "no tools" in out
        assert "cannot drive the agent" in out

    async def test_choosing_by_number(self):
        client, config = make_client()
        assert await ask_model(UI(), fake_prompt("2"), config, client) in CATALOGUE
        await client.aclose()

    async def test_choosing_by_name(self):
        client, config = make_client()
        assert await ask_model(UI(), fake_prompt("gemma4:latest"), config,
                               client) == "gemma4:latest"
        await client.aclose()

    async def test_choosing_by_unique_prefix(self):
        client, config = make_client()
        assert await ask_model(UI(), fake_prompt("gemma"), config,
                               client) == "gemma4:latest"
        await client.aclose()

    async def test_warns_when_nothing_can_call_tools(self, capsys):
        client, config = make_client({"gemma4:latest": CATALOGUE["gemma4:latest"]})
        await ask_model(UI(), fake_prompt(""), config, client)
        await client.aclose()
        out = capsys.readouterr().out
        assert "None of these advertise tool calling" in out
        assert "Hermes" in out

    async def test_empty_server_does_not_pull_anything(self, capsys):
        """It explains how to get a model; it does not download one."""
        client, config = make_client({})
        with pytest.raises(SystemExit):
            await ask_model(UI(), fake_prompt(""), config, client)
        await client.aclose()
        out = capsys.readouterr().out
        assert "no models installed" in out
        assert "ollama pull" in out


class TestRowLayout:
    class Row:
        def __init__(self, name, caps, ctx=32768):
            self.name = name
            self.capabilities = caps
            self.context_length = ctx
            self.parameter_size = "30.5B"
            self.quantization = "Q4_K_M"

        capabilities_known = property(lambda self: self.capabilities is not None)
        supports_tools = property(lambda self: "tools" in (self.capabilities or []))
        supports_thinking = property(lambda self: "thinking" in (self.capabilities or []))

        def human_size(self):
            return "18.6GB"

    @pytest.mark.parametrize("width", [40, 56, 72, 100, 160])
    def test_rows_never_wrap(self, width, capsys):
        ui = UI()
        ui.width = width
        ui.console.width = width
        _print_model_rows(ui, [
            self.Row("qwen3-coder:30b", ["tools"]),
            self.Row("huihui_ai/qwen3.5-abliterated-long:4B", ["completion"]),
        ])
        for line in capsys.readouterr().out.splitlines():
            assert len(line) <= width, f"wrapped at {width}: {line!r}"

    def test_badge_survives_a_very_long_name(self, capsys):
        ui = UI()
        ui.width = 44
        ui.console.width = 44
        _print_model_rows(ui, [
            self.Row("some/absurdly-long-vendor-prefix/model-name:70b-instruct-q8",
                     ["completion"])])
        assert "no tools" in capsys.readouterr().out
