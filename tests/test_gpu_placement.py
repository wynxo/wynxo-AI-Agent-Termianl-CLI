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
        assert "not the lever" in said, said
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
        """With the machine's RAM pinned. Otherwise the branch taken
        depends on how much memory the developer happens to have."""
        from unittest.mock import patch

        with patch("wynxo.platforms.total_memory", return_value=int(doc.ram)):
            await doc.check_memory()
        return doc.checks[-1]

    async def test_it_does_not_recommend_a_window_that_cannot_help(self):
        """It fell back to halving num_ctx when the weights themselves do
        not fit -- the same instruction, offered again with no more reason,
        for a problem no window solves."""
        from wynxo.provider import ModelInfo

        doc = self._doctor(
            [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)],
            [ModelInfo(name="qwen3-coder:30b", size=18 * GB)])
        fix = (await self._check(doc)).fix
        assert "not the lever" in fix, fix
        assert "smaller model" in fix

    async def test_it_names_the_same_model_the_live_warning_would(self):
        from wynxo.provider import ModelInfo

        installed = [ModelInfo(name="qwen3-coder:30b", size=18 * GB),
                     ModelInfo(name="qwen2.5-coder:7b", size=4 * GB)]
        doc = self._doctor(
            [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=6 * GB)],
            installed)
        assert "/model qwen2.5-coder:7b" in (await self._check(doc)).fix

    async def test_a_window_that_would_fit_is_still_offered(self):
        from wynxo.provider import ModelInfo

        doc = self._doctor(
            [Loaded("qwen3-coder:30b", size=21 * GB, size_vram=20 * GB)],
            [ModelInfo(name="qwen3-coder:30b", size=18 * GB)])
        assert "/ctx 21504" in (await self._check(doc)).fix


class TestAnOfferIsSomethingYouCanType:
    """A command wrapped between itself and its argument is not a command
    any more: "/model qwen2.5-coder:7b" came out as "/model" at the end of
    one line and the name at the start of the next, which cannot be copied
    and does not read as one thing."""

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
    """A model needs its weights and its cache resident somewhere. What
    will not fit in the card and the RAM together is read off disk on every
    token, which is slower than the CPU by a long way -- and it is the one
    condition where a smaller window helps a machine with no GPU room left
    to gain.

    The numbers here are from a real machine: a 27B at Q4_K_M, 17.7 GB of
    weights, 20.1 GB in memory at 32,768 tokens, 3.6 GB of it on a small
    card, in a box with 16 GB of RAM.
    """

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
        assert "off disk" in said, said
        assert "20.1 GB" in said, "say what it needs"

    async def test_the_window_that_stops_the_paging_is_offered(self):
        """This is the one recommendation that helps a machine which has no
        GPU room left to win: 20.1 GB down to about 18.4."""
        said = await self._said(16.9)
        assert "/ctx 10240" in said, said

    async def test_a_roomy_machine_is_not_told_it_is_paging(self):
        """The same card in a 64 GB box holds the whole model in RAM. The
        GPU share is unchanged and disk has nothing to do with it."""
        said = await self._said(64)
        assert "off disk" not in said
        assert "not the lever" in said

    async def test_a_machine_that_will_not_say_its_memory_says_nothing(self):
        """Zero is "could not tell", and a diagnosis from a number that was
        never read is worse than no diagnosis."""
        said = await self._said(0)
        assert "off disk" not in said

    async def test_the_gpu_line_is_still_the_headline(self):
        said = await self._said(16.9)
        assert said.lstrip().startswith("!")
        assert "18% on the GPU" in said
