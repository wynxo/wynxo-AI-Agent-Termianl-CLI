"""Why `ollama run` feels faster than wynxo.

`ollama run` uses the model's own default context window; wynxo asks for
num_ctx, which defaults to 32768. The KV cache scales with that window, so
the same model on the same machine can be entirely on the GPU under one and
spilled onto the CPU under the other -- several times slower, with nothing
anywhere saying why. /api/ps reports where the weights actually went.
"""

from __future__ import annotations

import pytest

from wynxo.provider import Loaded


class TestWhereTheWeightsWent:
    def test_a_model_wholly_on_the_gpu_is_not_split(self):
        model = Loaded("m", size=10, size_vram=10)
        assert model.on_gpu == 1.0
        assert model.split is False

    def test_a_partial_offload_is_the_case_worth_reporting(self):
        model = Loaded("m", size=10, size_vram=6)
        assert model.on_gpu == pytest.approx(0.6)
        assert model.split is True

    def test_a_machine_with_no_gpu_has_nothing_wrong_with_it(self):
        """size_vram of zero is CPU-only, not a spill. There is no faster
        arrangement to suggest, so it must not be reported as a problem."""
        model = Loaded("m", size=10, size_vram=0)
        assert model.split is False

    def test_an_unknown_size_never_reads_as_a_problem(self):
        """A guard that fires on missing information is worse than none."""
        assert Loaded("m", size=0, size_vram=0).on_gpu == 1.0
        assert Loaded("m", size=0, size_vram=0).split is False

    def test_more_vram_than_size_is_still_capped(self):
        assert Loaded("m", size=10, size_vram=12).on_gpu == 1.0


class TestTheEndpointDegrades:
    @pytest.mark.asyncio
    async def test_a_server_without_the_endpoint_reports_nothing(self):
        """Older Ollama has no /api/ps. That is not an error and must never
        be able to break a turn."""
        import httpx

        from wynxo.config import Config
        from wynxo.provider import OllamaClient

        client = OllamaClient(Config())
        client._client = httpx.AsyncClient(
            base_url="http://ps.invalid",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(404, json={"error": "not found"})))
        try:
            assert await client.running() == []
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_junk_in_the_listing_costs_that_entry_and_no_more(self):
        import httpx

        from wynxo.config import Config
        from wynxo.provider import OllamaClient

        body = {"models": ["a string", None, 7,
                           {"name": "good:30b", "size": 100, "size_vram": 40}]}
        client = OllamaClient(Config())
        client._client = httpx.AsyncClient(
            base_url="http://ps.invalid",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)))
        try:
            loaded = await client.running()
        finally:
            await client.aclose()
        assert [m.name for m in loaded] == ["good:30b"]
        assert loaded[0].split is True


class TestToolsThatCannotWorkAreNotOffered:
    """Every schema is sent with every request. A tool certain to fail is
    paying prompt-processing time to be offered, tried, and refused."""

    def test_the_github_tools_need_the_gh_cli(self, tmp_path, monkeypatch):
        from wynxo.tools import build_registry

        monkeypatch.setattr("wynxo.gh.shutil.which", lambda name: None)
        registry = build_registry(tmp_path)
        assert "github_read" not in registry
        assert "github_write" not in registry
        assert set(registry.withheld) == {"github_read", "github_write"}
        assert "gh" in registry.withheld["github_read"]

    def test_they_come_back_once_gh_is_installed(self, tmp_path, monkeypatch):
        from wynxo.tools import build_registry

        monkeypatch.setattr("wynxo.gh.shutil.which", lambda name: "/usr/bin/gh")
        registry = build_registry(tmp_path)
        assert "github_read" in registry
        assert registry.withheld == {}

    def test_holding_them_back_is_worth_real_context(self, tmp_path, monkeypatch):
        """The saving is the reason this exists, so it is worth asserting."""
        import json

        from wynxo.session import estimate_tokens
        from wynxo.tools import build_registry

        monkeypatch.setattr("wynxo.gh.shutil.which", lambda name: "/usr/bin/gh")
        with_gh = estimate_tokens(json.dumps(build_registry(tmp_path).ollama_schemas()))
        monkeypatch.setattr("wynxo.gh.shutil.which", lambda name: None)
        without = estimate_tokens(json.dumps(build_registry(tmp_path).ollama_schemas()))
        assert with_gh - without > 800

    def test_a_withheld_tool_is_not_described_to_the_model_either(
            self, tmp_path, monkeypatch):
        """Hermes prompted mode renders describe(), not the schemas."""
        from wynxo.tools import build_registry

        monkeypatch.setattr("wynxo.gh.shutil.which", lambda name: None)
        assert "github_read" not in build_registry(tmp_path).describe()
