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
        (0, 32768, 21 * GB),           # the server did not say
        (18 * GB, 0, 21 * GB),         # no window to divide by
        (18 * GB, 32768, 18 * GB),     # nothing above the weights: no KV
        (21 * GB, 32768, 18 * GB),     # weights larger than the whole
    ])
    def test_nonsense_in_gets_nothing_out(self, weights, num_ctx, size):
        """Zero means "say nothing", which every caller honours -- rather
        than a number computed from a division nobody checked."""
        assert Loaded("m", size=size, size_vram=17 * GB) \
            .context_that_fits(weights, num_ctx) == 0


class TestWhatCountsAsTheSameModel:
    def test_a_tag_and_latest_are_the_same_weights(self):
        """/api/ps answers with what the server loaded, which is not always
        the string configured here -- and a check that misses that reports
        nothing rather than reporting the wrong thing."""
        assert same_model("qwen3-coder:latest", "qwen3-coder:30b")
        assert same_model("Qwen3-Coder:30B", "qwen3-coder:30b")

    def test_two_different_models_are_not(self):
        assert not same_model("llama3:8b", "qwen3-coder:30b")

    def test_nothing_is_not_a_model(self):
        assert not same_model("", "qwen3-coder:30b")


class TestItSaysSoWithoutBeingAsked:
    """The report reached the user only through /doctor, which you have to
    already suspect the problem to run. "wynxo is slower than ollama run" is
    the report that arrived instead."""

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
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=20 * GB)])
        said = await self._said(repl)
        assert "GPU" in said
        assert "/ctx" in said, "a diagnosis with no remedy is a complaint"

    async def test_a_model_wholly_on_the_gpu_says_nothing(self):
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=21 * GB)])
        assert await self._said(repl) == ""

    async def test_a_machine_with_no_gpu_says_nothing(self):
        """Nothing has gone wrong and there is nothing to fix. A warning
        here would be a warning on every CPU-only machine, forever."""
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=0)])
        assert await self._said(repl) == ""

    async def test_a_server_that_cannot_say_says_nothing(self):
        assert await self._said(self._repl([])) == ""

    async def test_it_is_said_once(self):
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=20 * GB)])
        first = await self._said(repl)
        assert first
        repl.ui.console.file = io.StringIO()
        assert await self._said(repl) == "", "one line a turn is noise"

    async def test_a_card_too_small_for_the_weights_is_told_the_truth(self):
        """Recommending a smaller window there would be advice that cannot
        work: the weights alone do not fit, whatever the window."""
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=8 * GB)])
        said = await self._said(repl)
        assert "/ctx" not in said
        assert "8.0 GB" in said, "say how much card there actually is"

    async def test_it_names_a_model_that_would_fit(self):
        """"A smaller model is the fix" is advice, not help: it leaves the
        one question that matters to somebody who has just been told their
        setup is slow. wynxo knows what is installed."""
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=6 * GB)],
                          also=[("qwen2.5-coder:7b", 4 * GB),
                                ("llama3.2:3b", 2 * GB)])
        said = await self._said(repl)
        assert "/model qwen2.5-coder:7b" in said, \
            "the largest that fits is the most capable one that stays fast"

    async def test_it_recommends_nothing_when_nothing_fits(self):
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=2 * GB)],
                          also=[("llama3:70b", 40 * GB)])
        assert "/model" not in await self._said(repl)

    async def test_it_does_not_recommend_the_model_already_loaded(self):
        """It is the one that does not fit. Suggesting it is the shape of a
        check that has not understood its own answer."""
        repl = self._repl([Loaded("qwen3-coder:30b",
                                  size=21 * GB, size_vram=20 * GB)])
        assert "/model qwen3-coder:30b" not in await self._said(repl)

    async def test_a_diagnostic_cannot_fail_a_turn(self):
        from types import SimpleNamespace

        repl = self._repl([])

        async def boom():
            raise RuntimeError("server went away")

        repl.client = SimpleNamespace(running=boom, list_models=boom)
        assert await self._said(repl) == ""


class TestALoadingModelIsNotAnAnswer:
    """A 30B is not in /api/ps until it has finished being read off disk,
    which on the first turn of a session it may not have. Settling for
    "could not tell" there costs the one answer this exists to give."""

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
        assert await self._say(repl) == "", "nothing loaded yet"
        assert "/ctx" in await self._say(repl), "the second look found it"

    async def test_it_stops_looking_eventually(self):
        """An old server with no /api/ps must not be polled once a turn for
        the rest of the session."""
        from wynxo.cli import Repl

        repl = self._repl([])
        for _ in range(Repl.PLACEMENT_TRIES + 3):
            await repl._report_placement()
        assert repl._placement_checked == Repl.PLACEMENT_TRIES

    async def test_a_definite_answer_settles_it(self):
        """Found and whole: there is nothing to say and nothing to re-ask."""
        from wynxo.cli import Repl

        whole = Loaded("qwen3-coder:30b", size=21 * GB, size_vram=21 * GB)
        repl = self._repl([[whole], [Loaded("qwen3-coder:30b",
                                            size=21 * GB, size_vram=1 * GB)]])
        await repl._report_placement()
        assert repl._placement_checked == Repl.PLACEMENT_TRIES
        assert await self._say(repl) == ""

    async def _say(self, repl) -> str:
        repl.ui.console.file = io.StringIO()
        await repl._report_placement()
        return repl.ui.console.file.getvalue()


class TestItReadsAsOneBlock:
    """It went out as a warn() followed by two info()s, and info() has no
    marker -- so the warning wrapped under its own "!" at column two while
    the explanation under it started hard against column zero. Three ragged
    paragraphs for one fact, in the message whose whole job is to be read
    carefully."""

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
    """Two places say the same thing and must not disagree: the line at the
    end of a slow turn, and the check you run when you go looking."""

    def _doctor(self, loaded, installed, num_ctx=32768):
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
        return made

    async def test_it_does_not_recommend_a_window_that_cannot_help(self):
        """It fell back to halving num_ctx when the weights themselves do
        not fit -- the same instruction, offered again with no more reason,
        for a problem no window solves."""
        from wynxo.provider import ModelInfo

        doc = self._doctor(
            [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)],
            [ModelInfo(name="qwen3-coder:30b", size=18 * GB)])
        await doc.check_memory()
        fix = doc.checks[-1].fix
        assert "/ctx" not in fix
        assert "smaller model" in fix

    async def test_it_names_the_same_model_the_live_warning_would(self):
        from wynxo.provider import ModelInfo

        installed = [ModelInfo(name="qwen3-coder:30b", size=18 * GB),
                     ModelInfo(name="qwen2.5-coder:7b", size=4 * GB)]
        doc = self._doctor(
            [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)],
            installed)
        await doc.check_memory()
        assert "/model qwen2.5-coder:7b" in doc.checks[-1].fix

    async def test_a_window_that_would_fit_is_still_offered(self):
        from wynxo.provider import ModelInfo

        doc = self._doctor(
            [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=20 * GB)],
            [ModelInfo(name="qwen3-coder:30b", size=18 * GB)])
        await doc.check_memory()
        assert "/ctx 21504" in doc.checks[-1].fix
