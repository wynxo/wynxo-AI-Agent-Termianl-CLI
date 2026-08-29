"""The first-run wizard's testable surface: the recommended-model table,
the probe that checks an Ollama server, and the pure formatting helpers.
The interactive ask_* steps need a terminal and are covered by the flow
tests instead."""

from __future__ import annotations

import asyncio

from wynxo import wizard
from wynxo.wizard import RECOMMENDED, _humanise_context, describe_model, probe


class TestRecommendedModels:
    def test_every_entry_is_a_tag_and_a_note(self):
        assert RECOMMENDED
        for tag, why in RECOMMENDED:
            assert isinstance(tag, str) and tag
            assert isinstance(why, str) and why

    def test_tags_are_unique(self):
        tags = [tag for tag, _ in RECOMMENDED]
        assert len(tags) == len(set(tags))


class TestDescribeModel:
    def test_exact_tag_matches(self):
        why = describe_model("qwen3-coder:30b")
        assert why and "MoE" in why

    def test_prefix_does_not_match(self):
        # "qwen3-coder" without the size, and a different size of the same
        # family, must not borrow the note of the exact tag.
        assert describe_model("qwen3-coder") == ""
        assert describe_model("qwen3-coder:30b-abliterated") == ""

    def test_unknown_model_is_empty(self):
        assert describe_model("some-other-model:7b") == ""


class TestHumaniseContext:
    def test_zero_is_empty(self):
        assert _humanise_context(0) == ""

    def test_under_a_thousand_is_plain(self):
        assert _humanise_context(500) == "500 ctx"

    def test_thousands_are_k(self):
        assert _humanise_context(8192) == "8k ctx"


class TestProbe:
    def test_probe_returns_the_server_version(self, monkeypatch):
        async def fake_verify(url, timeout=None):
            return "0.5.1"

        monkeypatch.setattr(wizard, "verify", fake_verify)
        assert asyncio.run(probe("http://127.0.0.1:11434")) == "0.5.1"

    def test_probe_passes_its_timeout_through(self, monkeypatch):
        seen = {}

        async def fake_verify(url, timeout=None):
            seen["timeout"] = timeout
            return None

        monkeypatch.setattr(wizard, "verify", fake_verify)
        asyncio.run(probe("http://127.0.0.1:11434", timeout=7.5))
        assert seen["timeout"] == 7.5
