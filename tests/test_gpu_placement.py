"""Why a local model is slower under wynxo than under `ollama run`.

Almost always one thing: `ollama run` uses the model's own default context
window, and wynxo asks for num_ctx, which defaults to 32,768. The KV cache
grows with that window, so a model that fits entirely on the GPU under one
has layers pushed onto the CPU under the other -- same model, same machine,
several times slower.

The server reports enough to work out both that it happened and what window
would avoid it. Nothing was reading it outside /doctor, which you have to
already suspect the problem to run.
"""

from __future__ import annotations

import io

import pytest

from wynxo.provider import Loaded, same_model

GB = 1_000_000_000


class TestWorkingOutTheWindowThatWouldFit:
    """size is what the model needs at the window it was loaded under, and
    the weights do not grow with the window -- so everything above the file
    size is KV cache, and the KV cache is linear in tokens."""

    def test_a_model_just_over_the_card(self):
        """18 GB of weights, 3 GB of KV at 32k, on a card that took 20 GB.
        Two spare gigabytes at ~92 KB a token is about 21k tokens."""
        m = Loaded("m", size=21 * GB, size_vram=20 * GB)
        assert 20_000 <= m.context_that_fits(18 * GB, 32768) <= 23_000

    def test_the_answer_is_a_round_number(self):
        """An estimate that says 21,417 claims a precision it has not got."""
        m = Loaded("m", size=21 * GB, size_vram=20 * GB)
        assert m.context_that_fits(18 * GB, 32768) % 1024 == 0

    def test_more_spare_vram_means_more_context(self):
        weights, small, big = 18 * GB, 19 * GB, 22 * GB
        at_small = Loaded("m", size=21 * GB, size_vram=small)
        at_big = Loaded("m", size=21 * GB, size_vram=big)
        assert (at_small.context_that_fits(weights, 32768)
                < at_big.context_that_fits(weights, 32768))

    def test_a_card_too_small_for_the_weights_gets_no_number(self):
        """No window is small enough, and "try /ctx 0" is not advice."""
        m = Loaded("m", size=21 * GB, size_vram=8 * GB)
        assert m.context_that_fits(18 * GB, 32768) == 0

    @pytest.mark.parametrize("weights,num_ctx,size", [
        (0, 32768, 21 * GB),
        (18 * GB, 0, 21 * GB),
        (18 * GB, 32768, 18 * GB),
        (21 * GB, 32768, 18 * GB),
    ])
    def test_nonsense_in_gets_nothing_out(self, weights, num_ctx, size):
        """Zero means "say nothing", which every caller honours -- rather
        than a number computed from a division nobody checked."""
        assert Loaded("m", size=size, size_vram=17 * GB) \
            .context_that_fits(weights, num_ctx) == 0


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


class TestItSaysSoWithoutBeingAsked:
    def _repl(self, loaded, model_size=18 * GB, num_ctx=32768, also=()):
        from types import SimpleNamespace
        from wynxo.cli import Repl
        from wynxo.provider import ModelInfo
        from wynxo.ui import UI

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = 100

        async def running():
            return loaded

        async def list_models():
            return [ModelInfo(name="qwen3-coder:30b", size=model_size),
                    *(ModelInfo(name=n, size=z) for n, z in also)]

        repl = Repl.__new__(Repl)
        repl.ui = ui
        repl._placement_checked = 0
        repl.config = SimpleNamespace(model="qwen3-coder:30b", num_ctx=num_ctx)
        repl.client = SimpleNamespace(running=running, list_models=list_models)
        return repl

    async def _said(self, repl) -> str:
        await repl._report_placement()
        return repl.ui.console.file.getvalue()

    async def test_a_split_model_is_reported_with_the_window_that_fits(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=20 * GB)])
        said = await self._said(repl)
        assert "GPU" in said
        assert "/ctx" in said

    async def test_a_model_wholly_on_the_gpu_says_nothing(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=21 * GB)])
        assert await self._said(repl) == ""

    async def test_a_machine_with_no_gpu_says_nothing(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=0)])
        assert await self._said(repl) == ""

    async def test_a_server_that_cannot_say_says_nothing(self):
        assert await self._said(self._repl([])) == ""

    async def test_it_is_said_once(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=20 * GB)])
        first = await self._said(repl)
        assert first
        repl.ui.console.file = io.StringIO()
        assert await self._said(repl) == ""

    async def test_a_card_too_small_for_the_weights_is_told_the_truth(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=8 * GB)])
        said = await self._said(repl)
        assert "not the lever" in said, said
        assert "8.0 GB" in said

    async def test_it_names_a_model_that_would_fit(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)],
                          also=[("qwen2.5-coder:7b", 4 * GB), ("llama3.2:3b", 2 * GB)])
        said = await self._said(repl)
        assert "/model qwen2.5-coder:7b" in said

    async def test_it_recommends_nothing_when_nothing_fits(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=2 * GB)],
                          also=[("llama3:70b", 40 * GB)])
        assert "/model" not in await self._said(repl)

    async def test_it_does_not_recommend_the_model_already_loaded(self):
        repl = self._repl([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=20 * GB)])
        assert "/model qwen3-coder:30b" not in await self._said(repl)

    async def test_a_diagnostic_cannot_fail_a_turn(self):
        from types import SimpleNamespace
        repl = self._repl([])

        async def boom():
            raise RuntimeError("server went away")

        repl.client = SimpleNamespace(running=boom, list_models=boom)
        assert await self._said(repl) == ""


class TestALoadingModelIsNotAnAnswer:
    def _repl(self, answers):
        from types import SimpleNamespace
        from wynxo.cli import Repl
        from wynxo.provider import ModelInfo
        from wynxo.ui import UI

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = 100

        async def running():
            return answers.pop(0) if answers else []

        async def list_models():
            return [ModelInfo(name="qwen3-coder:30b", size=18 * GB)]

        repl = Repl.__new__(Repl)
        repl.ui = ui
        repl._placement_checked = 0
        repl.config = SimpleNamespace(model="qwen3-coder:30b", num_ctx=32768)
        repl.client = SimpleNamespace(running=running, list_models=list_models)
        return repl

    async def test_it_looks_again_next_turn(self):
        split = Loaded("qwen3-coder:30b", size=21 * GB, size_vram=20 * GB)
        repl = self._repl([[], [split]])
        assert await self._say(repl) == ""
        assert "/ctx" in await self._say(repl)

    async def test_it_stops_looking_eventually(self):
        from wynxo.cli import Repl
        repl = self._repl([])
        for _ in range(Repl.PLACEMENT_TRIES + 3):
            await repl._report_placement()
        assert repl._placement_checked == Repl.PLACEMENT_TRIES

    async def test_a_definite_answer_settles_it(self):
        from wynxo.cli import Repl
        whole = Loaded("qwen3-coder:30b", size=21 * GB, size_vram=21 * GB)
        repl = self._repl([[whole], [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=1 * GB)]])
        await repl._report_placement()
        assert repl._placement_checked == Repl.PLACEMENT_TRIES
        assert await self._say(repl) == ""

    async def _say(self, repl) -> str:
        repl.ui.console.file = io.StringIO()
        await repl._report_placement()
        return repl.ui.console.file.getvalue()


class TestItReadsAsOneBlock:
    async def _lines(self, width=100):
        from types import SimpleNamespace
        from wynxo.cli import Repl
        from wynxo.provider import ModelInfo
        from wynxo.ui import UI

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = width

        async def running():
            return [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)]

        async def list_models():
            return [ModelInfo(name="qwen3-coder:30b", size=18 * GB),
                    ModelInfo(name="qwen2.5-coder:7b", size=4 * GB)]

        repl = Repl.__new__(Repl)
        repl.ui = ui
        repl._placement_checked = 0
        repl.config = SimpleNamespace(model="qwen3-coder:30b", num_ctx=32768)
        repl.client = SimpleNamespace(running=running, list_models=list_models)
        await repl._report_placement()
        return [ln for ln in ui.console.file.getvalue().splitlines() if ln.strip()]

    async def test_one_marker_at_the_top_and_nothing_at_column_zero_after(self):
        lines = await self._lines()
        assert lines[0].startswith("!"), lines[0]
        for line in lines[1:]:
            assert line.startswith("  "), f"fell out of the block: {line!r}"

    async def test_only_the_headline_is_marked(self):
        lines = await self._lines()
        assert sum(1 for ln in lines if ln.lstrip().startswith("!")) == 1

    @pytest.mark.parametrize("width", [50, 62, 80, 100, 140])
    async def test_it_holds_together_at_any_width(self, width):
        from wynxo.ui import cell_len
        lines = await self._lines(width)
        for line in lines:
            assert cell_len(line) <= width, (width, line)
        for line in lines[1:]:
            assert line.startswith("  "), (width, line)


class TestTheDoctorAgreesWithTheLiveWarning:
    def _doctor(self, loaded, installed, num_ctx=32768, ram=64 * GB):
        from types import SimpleNamespace
        from wynxo.doctor import Doctor
        from wynxo.ui import UI

        async def running():
            return loaded

        async def list_models():
            return installed

        made = Doctor.__new__(Doctor)
        made.ui = UI()
        made.checks = []
        made.config = SimpleNamespace(model="qwen3-coder:30b", num_ctx=num_ctx)
        made.client = SimpleNamespace(running=running, list_models=list_models)
        made.ram = ram
        return made

    async def _check(self, doc):
        from unittest.mock import patch
        with patch("wynxo.platforms.total_memory", return_value=int(doc.ram)):
            await doc.check_memory()
        return doc.checks[-1]

    async def test_it_does_not_recommend_a_window_that_cannot_help(self):
        from wynxo.provider import ModelInfo
        doc = self._doctor([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)],
                           [ModelInfo(name="qwen3-coder:30b", size=18 * GB)])
        fix = (await self._check(doc)).fix
        assert "not the lever" in fix
        assert "smaller model" in fix

    async def test_it_names_the_same_model_the_live_warning_would(self):
        from wynxo.provider import ModelInfo
        installed = [ModelInfo(name="qwen3-coder:30b", size=18 * GB),
                     ModelInfo(name="qwen2.5-coder:7b", size=4 * GB)]
        doc = self._doctor([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)], installed)
        assert "/model qwen2.5-coder:7b" in (await self._check(doc)).fix

    async def test_a_window_that_would_fit_is_still_offered(self):
        from wynxo.provider import ModelInfo
        doc = self._doctor([Loaded("qwen3-coder:30b", size=21 * GB, size_vram=20 * GB)],
                           [ModelInfo(name="qwen3-coder:30b", size=18 * GB)])
        assert "/ctx 21504" in (await self._check(doc)).fix


class TestAnOfferIsSomethingYouCanType:
    async def _lines(self, width, loaded, installed):
        from types import SimpleNamespace
        from wynxo.cli import Repl
        from wynxo.ui import UI

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = width

        async def running():
            return loaded

        async def list_models():
            return installed

        repl = Repl.__new__(Repl)
        repl.ui = ui
        repl._placement_checked = 0
        repl.config = SimpleNamespace(model="qwen3-coder:30b", num_ctx=32768)
        repl.client = SimpleNamespace(running=running, list_models=list_models)
        await repl._report_placement()
        return ui.console.file.getvalue().splitlines()

    @pytest.mark.parametrize("width", [50, 62, 72, 80, 98, 120])
    async def test_a_command_is_never_split_from_its_argument(self, width):
        from wynxo.provider import ModelInfo
        lines = await self._lines(
            width,
            [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)],
            [ModelInfo(name="qwen3-coder:30b", size=18 * GB),
             ModelInfo(name="qwen2.5-coder:7b", size=4 * GB)])
        body = "\n".join(lines)
        assert "/model" in body, body
        for line in lines:
            stripped = line.strip()
            assert stripped != "/model", (width, lines)
            if stripped.startswith("/model"):
                assert "qwen2.5-coder:7b" in stripped, (width, line)


class TestTheCardIsNotTheOnlyCeiling:
    async def _said(self, ram, *, size=20.1, vram=3.6, weights=17.7,
                    num_ctx=32768, width=98):
        from types import SimpleNamespace
        from unittest.mock import patch
        from wynxo.cli import Repl
        from wynxo.provider import ModelInfo
        from wynxo.ui import UI

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = width

        async def running():
            return [Loaded("m", size=int(size * GB), size_vram=int(vram * GB))]

        async def list_models():
            return [ModelInfo(name="m", size=int(weights * GB))]

        repl = Repl.__new__(Repl)
        repl.ui = ui
        repl._placement_checked = 0
        repl.config = SimpleNamespace(model="m", num_ctx=num_ctx)
        repl.client = SimpleNamespace(running=running, list_models=list_models)
        with patch("wynxo.platforms.total_memory", return_value=int(ram * GB)):
            await repl._report_placement()
        return repl.ui.console.file.getvalue()

    async def test_a_model_bigger_than_the_machine_is_named_as_such(self):
        said = await self._said(16.9)
        assert "off disk" in said
        assert "20.1 GB" in said

    async def test_the_window_that_stops_the_paging_is_offered(self):
        said = await self._said(16.9)
        assert "/ctx 10240" in said

    async def test_a_roomy_machine_is_not_told_it_is_paging(self):
        said = await self._said(64)
        assert "off disk" not in said
        assert "not the lever" in said

    async def test_a_machine_that_will_not_say_its_memory_says_nothing(self):
        said = await self._said(0)
        assert "off disk" not in said

    async def test_the_gpu_line_is_still_the_headline(self):
        said = await self._said(16.9)
        assert said.lstrip().startswith("!")
        assert "18% on the GPU" in said
