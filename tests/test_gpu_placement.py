"""Why a local model is slower under wynxo than under `ollama run`."""

from __future__ import annotations

import io

import pytest

from wynxo.provider import Loaded, same_model

GB = 1_000_000_000


class TestWorkingOutTheWindowThatWouldFit:
    def test_a_model_just_over_the_card(self):
        m = Loaded("m", size=21 * GB, size_vram=20 * GB)
        assert 20_000 <= m.context_that_fits(18 * GB, 32768) <= 23_000

    def test_the_answer_is_a_round_number(self):
        m = Loaded("m", size=21 * GB, size_vram=20 * GB)
        assert m.context_that_fits(18 * GB, 32768) % 1024 == 0

    def test_more_spare_vram_means_more_context(self):
        weights, small, big = 18 * GB, 19 * GB, 22 * GB
        at_small = Loaded("m", size=21 * GB, size_vram=small)
        at_big = Loaded("m", size=21 * GB, size_vram=big)
        assert at_small.context_that_fits(weights, 32768) < at_big.context_that_fits(weights, 32768)

    def test_a_card_too_small_for_the_weights_gets_no_number(self):
        m = Loaded("m", size=21 * GB, size_vram=8 * GB)
        assert m.context_that_fits(18 * GB, 32768) == 0

    @pytest.mark.parametrize("weights,num_ctx,size", [
        (0, 32768, 21 * GB),
        (18 * GB, 0, 21 * GB),
        (18 * GB, 32768, 18 * GB),
        (21 * GB, 32768, 18 * GB),
    ])
    def test_nonsense_in_gets_nothing_out(self, weights, num_ctx, size):
        assert Loaded("m", size=size, size_vram=17 * GB).context_that_fits(weights, num_ctx) == 0


class TestWhatCountsAsTheSameModel:
    def test_exact_tags_match(self):
        assert same_model("qwen3-coder:30b", "qwen3-coder:30b")
        assert same_model("Qwen3-Coder:30B", "qwen3-coder:30b")

    def test_same_base_different_tags_do_not_match(self):
        assert not same_model("qwen3:8b", "qwen3:30b")
        assert not same_model("qwen3:latest", "qwen3:30b")

    def test_two_different_models_are_not(self):
        assert not same_model("llama3:8b", "qwen3-coder:30b")

    def test_nothing_is_not_a_model(self):
        assert not same_model("", "qwen3-coder:30b")
