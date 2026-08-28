"""Speaking the answer, and splitting talking from coding.

The part worth testing hardest is what *never* happens: the talker must not
be handed tools, and a machine with no synthesiser must not fail to start.
"""

import json
import os
import tempfile

import httpx
import pytest

from wynxo import speech
from wynxo.duo import Talker, _tidy
from wynxo.provider import OllamaClient
from wynxo.config import Config, Endpoint


class TestSpeakable:
    """A coding answer read literally is unlistenable. This is the filter."""

    def test_a_plain_sentence_is_unchanged(self):
        assert speech.speakable("I fixed the bug.") == "I fixed the bug."

    def test_code_fences_are_dropped_entirely(self):
        out = speech.speakable("Here it is:\n```python\nx = 1\n```\nDone.")
        assert "x = 1" not in out
        assert "Done." in out

    def test_paths_are_dropped_but_the_sentence_survives(self):
        """Read aloud a path is a stream of punctuation names, and the
        sentence around it already says what the file is."""
        out = speech.speakable("The bug was in /home/me/src/auth.py at line 42.")
        assert "/" not in out
        assert "at line 42" in out

    def test_inline_code_keeps_its_text(self):
        assert "check_token" in speech.speakable("Call `check_token` first.")

    def test_markdown_furniture_is_removed(self):
        out = speech.speakable("## Heading\n- one\n- two\n**bold** and _italic_")
        for junk in ("#", "-", "*", "_"):
            assert junk not in out
        assert "bold" in out and "italic" in out

    def test_link_text_survives_and_the_url_does_not(self):
        out = speech.speakable("See [the docs](https://example.com/x) for more.")
        assert "the docs" in out
        assert "example.com" not in out

    def test_tables_and_rules_are_dropped(self):
        out = speech.speakable("Results:\n| a | b |\n|---|---|\n----\nAll good.")
        assert "|" not in out
        assert "All good." in out

    def test_tool_and_think_markup_never_reaches_the_voice(self):
        out = speech.speakable(
            '<think>hmm</think>Right.<tool_call>{"name":"x"}</tool_call>')
        assert "think" not in out.lower()
        assert "tool_call" not in out
        assert "Right." in out

    def test_long_answers_are_cut_at_a_sentence_end(self):
        text = ("This is a sentence. " * 60)
        out = speech.speakable(text, limit=100)
        assert len(out) <= 100
        assert out.endswith(".")

    def test_an_answer_that_is_only_code_says_nothing(self):
        """Better silent than reading a diff out character by character."""
        assert speech.speakable("```\nx = 1\n```") == ""

    def test_empty_input_is_survivable(self):
        assert speech.speakable("") == ""


class TestEngineCommands:
    """Text is always an argv element, never interpolated into a shell."""

    def _engine(self, name):
        return next(e for e in speech.ENGINES if e.name == name)

    def test_espeak_gets_a_female_voice_by_default(self):
        argv = speech.command(self._engine("espeak-ng"), "hi")
        assert argv[:3] == ["espeak-ng", "-v", "en+f3"]
        assert argv[-1] == "hi"

    def test_say_gets_a_female_voice_by_default(self):
        argv = speech.command(self._engine("say"), "hi")
        assert "Samantha" in argv
        assert argv[-1] == "hi"

    def test_text_after_a_double_dash_cannot_look_like_a_flag(self):
        """An answer starting with '-' must not be parsed as an option."""
        argv = speech.command(self._engine("espeak-ng"), "-v evil")
        assert "--" in argv
        assert argv.index("--") < argv.index("-v evil")

    def test_powershell_escapes_quotes_in_the_text(self, monkeypatch):
        monkeypatch.setattr(speech, "_powershell", lambda: "powershell")
        argv = speech.command(self._engine("powershell"), "it's \"fine\"")
        script = argv[-1]
        # Doubling ' is PowerShell's entire escaping rule for a literal.
        assert "it''s" in script
        assert script.count("'") % 2 == 0

    def test_powershell_prefers_a_natural_voice_over_zira(self, monkeypatch):
        """A stock Windows box only has the robotic Zira; a free natural
        voice (Aria, Jenny, ...) should win automatically when installed,
        with Zira and then the generic hint as fallbacks."""
        monkeypatch.setattr(speech, "_powershell", lambda: "powershell")
        argv = speech.command(self._engine("powershell"), "hi")
        script = argv[-1]
        assert "GetInstalledVoices" in script
        assert "Aria" in script                 # natural names tried first
        assert "Zira" in script                 # known fallback
        assert "SelectVoiceByHints('Female')" in script   # last resort

    def test_powershell_honours_an_explicit_voice(self, monkeypatch):
        """speech_voice used to be ignored entirely on Windows -- whatever
        SelectVoiceByHints picked was what you got. A named voice now wins,
        with the female hint as the fallback if the name is wrong."""
        monkeypatch.setattr(speech, "_powershell", lambda: "powershell")
        argv = speech.command(self._engine("powershell"), "hi",
                              voice="Microsoft Jenny Natural")
        script = argv[-1]
        assert "SelectVoice('Microsoft Jenny Natural')" in script
        assert "catch" in script
        assert "SelectVoiceByHints('Female')" in script

    def test_powershell_escapes_the_voice_name_too(self, monkeypatch):
        monkeypatch.setattr(speech, "_powershell", lambda: "powershell")
        argv = speech.command(self._engine("powershell"), "hi",
                              voice="it's")
        script = argv[-1]
        assert script.count("'") % 2 == 0

    def test_piper_without_a_model_refuses_rather_than_guesses(self):
        assert speech.command(self._engine("piper"), "hi") is None

    def test_edge_tts_builds_the_synthesis_command(self):
        """edge-tts is the one voice worth installing: Microsoft's neural
        voices, which sound like a person. Its command is the synthesis
        half; the Speaker appends the output file and plays it."""
        engine = next(e for e in speech.ENGINES if e.name == "edge-tts")
        argv = speech.command(engine, "hello", voice="en-US-AriaNeural")
        assert argv[:4] == ["edge-tts", "--voice", "en-US-AriaNeural",
                            "--text"]
        assert argv[4] == "hello"
        assert argv[-1] == "--write-media"   # path appended by the Speaker

    def test_a_player_is_found_for_edge_tts(self, monkeypatch):
        """Synthesised audio must be played back; without any player the
        engine would be silence, so one is required and found."""
        monkeypatch.setattr(speech.shutil, "which", lambda n: n == "ffplay"
                            and "/usr/bin/ffplay" or None)
        assert speech._player() is not None


class TestSpeakerDegrades:
    def test_no_engine_means_silence_not_a_crash(self):
        speaker = speech.Speaker(None)
        assert speaker.enabled is False
        assert speaker.say("anything") is False
        assert speaker.describe() == "off"

    def test_a_missing_binary_disables_itself_instead_of_raising(self):
        engine = speech.Engine("espeak-ng", "espeak-ng", "x")
        speaker = speech.Speaker(engine)
        # Nothing on PATH by this name in the test environment.
        speaker.say("hello")
        assert speaker.enabled is False

    def test_nothing_speakable_says_nothing(self):
        engine = speech.Engine("espeak-ng", "espeak-ng", "x")
        speaker = speech.Speaker(engine)
        assert speaker.say("```\ncode only\n```") is False

    def test_stop_is_safe_when_nothing_is_speaking(self):
        speech.Speaker(None).stop()   # must not raise

    def test_edge_tts_synthesises_then_plays_and_cleans_up(self, monkeypatch):
        """The edge-tts path is two steps: synth to a temp file, then play
        it with the quietest available player. The file must be removed
        when she is stopped, or temp files pile up."""
        import os

        engine = next(e for e in speech.ENGINES if e.name == "edge-tts")
        speaker = speech.Speaker(engine, voice="en-US-AriaNeural")
        created = []
        real_mkstemp = tempfile.mkstemp

        def _mkstemp(suffix):
            # The real one, tracked -- pytest's own temp machinery must not
            # see a fake mkstemp.
            fd, path = real_mkstemp(suffix=suffix)
            created.append(path)
            return fd, path

        played = []

        class _Proc:
            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        def _popen(argv, **kw):
            played.append(argv)
            return _Proc()

        monkeypatch.setattr("tempfile.mkstemp", _mkstemp)
        monkeypatch.setattr(speech, "_player",
                            lambda: ["ffplay", "-nodisp", "-autoexit"])
        monkeypatch.setattr(speech.subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 0})())
        monkeypatch.setattr(speech.subprocess, "Popen", _popen)

        assert speaker.say("hello there") is True
        assert played and played[0][:3] == ["ffplay", "-nodisp", "-autoexit"]
        assert os.path.exists(created[0])
        speaker.stop()
        assert not os.path.exists(created[0])   # temp file removed

    def test_edge_tts_without_a_player_disables_cleanly(self, monkeypatch):
        engine = next(e for e in speech.ENGINES if e.name == "edge-tts")
        speaker = speech.Speaker(engine)
        monkeypatch.setattr(speech, "_player", lambda: None)
        assert speaker.say("hello") is False
        assert speaker.enabled is False


class FakeDuoServer:
    """Records every request so the test can assert what the talker was sent."""

    def __init__(self, reply="Okay, on it~"):
        self.reply = reply
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            self.requests.append(body)
            lines = [
                json.dumps({"message": {"role": "assistant",
                                        "content": self.reply}, "done": False}),
                json.dumps({"message": {"role": "assistant", "content": ""},
                            "done": True, "prompt_eval_count": 5,
                            "eval_count": 5, "total_duration": 10 ** 9}),
            ]
            return httpx.Response(200, text="\n".join(lines))
        return httpx.Response(404, json={"error": "no"})


def make_talker(reply="Okay, on it~"):
    fake = FakeDuoServer(reply)
    config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")],
                    active_endpoint="t")
    client = OllamaClient(config)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(fake.handler), base_url="http://fake:11434")
    return Talker(client, "tiny:0.8b"), fake, client


class TestTalkerHasNoTools:
    """The whole safety argument for running a 1B model in the loop."""

    async def test_the_opener_is_sent_without_tools(self):
        talker, fake, client = make_talker()
        await talker.opening("make a file")
        assert fake.requests[0].get("tools") in (None, [])
        assert "tools" not in fake.requests[0] or not fake.requests[0]["tools"]
        await client.aclose()

    async def test_the_report_is_sent_without_tools(self):
        talker, fake, client = make_talker()
        await talker.report("make a file", "Created out.txt.")
        assert not fake.requests[0].get("tools")
        await client.aclose()

    async def test_the_talker_does_not_ask_for_thinking(self):
        """A one-line acknowledgement does not need a reasoning budget, and
        waiting for one would defeat the point of using a fast model."""
        talker, fake, client = make_talker()
        await talker.opening("hi")
        assert "think" not in fake.requests[0]
        await client.aclose()


class TestTalkerBehaviour:
    async def test_a_failure_is_reported_as_a_failure(self):
        """The talker must never turn an error into good news."""
        talker, fake, client = make_talker()
        await talker.report("do it", "permission denied", failed=True)
        sent = fake.requests[0]["messages"][-1]["content"]
        assert "FAILED" in sent
        assert "permission denied" in sent
        system = fake.requests[0]["messages"][0]["content"]
        assert "Never claim something worked" in system
        await client.aclose()

    async def test_the_voice_block_reaches_the_talker(self):
        """So kawaii sounds like the same character in both halves."""
        talker, fake, client = make_talker()
        talker.voice_block = "## Voice\nBe cheerful."
        await talker.opening("hi")
        assert "Be cheerful." in fake.requests[0]["messages"][0]["content"]
        await client.aclose()

    async def test_a_provider_error_yields_silence_not_a_crash(self):
        """A broken talker must not take the coder down with it."""
        config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")],
                        active_endpoint="t")
        client = OllamaClient(config)
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(500, json={"error": "boom"})),
            base_url="http://fake:11434")
        talker = Talker(client, "tiny:0.8b")
        assert await talker.opening("hi") == ""
        assert talker.last_error
        await client.aclose()


class TestTidy:
    """Small models pad, label and quote themselves. Take it off."""

    @pytest.mark.parametrize("raw,expected", [
        ("Assistant: hello there", "hello there"),
        ('"just a line"', "just a line"),
        ("Voice:   spaced   out  ", "spaced out"),
        ("<think>hmm</think> the answer", "the answer"),
        ("uses `backticks` here", "uses backticks here"),
        ("```py\nx=1\n```ok", "ok"),
    ])
    def test_padding_is_stripped(self, raw, expected):
        assert _tidy(raw) == expected

    def test_a_clean_line_is_left_alone(self):
        assert _tidy("All done, nya~") == "All done, nya~"


class TestEffortMeter:
    """A gauge that fills as the effort level rises, so the change is
    visible without reading a word."""

    def test_it_rises_monotonically_across_the_ladder(self):
        from wynxo.effort import ORDER
        from wynxo.ui import METER_BLOCKS, effort_meter

        def weight(meter):
            return sum(METER_BLOCKS.index(c) + 1 for c in meter if c in METER_BLOCKS)

        weights = [weight(effort_meter(name)) for name in ORDER]
        assert weights == sorted(weights), weights
        assert weights[0] < weights[-1]

    def test_the_lowest_level_still_shows_something(self):
        """A blank meter reads as broken rather than as low."""
        from wynxo.ui import effort_meter

        assert effort_meter("low").strip() != ""

    def test_width_is_fixed_so_the_bar_does_not_shift(self):
        from rich.cells import cell_len

        from wynxo.effort import ORDER
        from wynxo.ui import METER_WIDTH, effort_meter

        for name in ORDER:
            for unicode_ok in (True, False):
                assert cell_len(effort_meter(name, unicode_ok)) == METER_WIDTH

    def test_ascii_terminals_get_a_one_cell_ramp(self):
        from wynxo.ui import effort_meter

        assert effort_meter("ultra", False).isascii()
        assert effort_meter("low", False).isascii()

    def test_an_unknown_level_is_blank_rather_than_an_error(self):
        from wynxo.ui import METER_WIDTH, effort_meter

        assert effort_meter("nonsense") == " " * METER_WIDTH


class TestPetPace:
    """More effort, more visible energy."""

    def test_higher_effort_animates_faster(self):
        from wynxo.effort import ORDER
        from wynxo.pet import Pet

        pet = Pet()
        paces = []
        for name in ORDER:
            pet.set_pace(name)
            paces.append(pet.pace)
        assert paces == sorted(paces, reverse=True), paces
        assert paces[-1] < paces[0]

    def test_pace_never_reaches_zero(self):
        """It divides the frame counter."""
        from wynxo.effort import ORDER
        from wynxo.pet import Pet

        pet = Pet()
        for name in ORDER:
            pet.set_pace(name)
            assert pet.pace >= 1

    def test_an_unknown_level_leaves_the_pace_alone(self):
        from wynxo.pet import Pet

        pet = Pet()
        pet.set_pace("ultra")
        before = pet.pace
        pet.set_pace("nonsense")
        assert pet.pace == before


class TestSakuraPalette:
    def test_it_is_selectable_by_name(self):
        from wynxo.theme import resolve

        assert resolve("sakura").name == "sakura"

    def test_its_accent_is_not_the_error_colour(self):
        """Pink drifting into the red that `bad` uses would make a failure
        and a heading look the same."""
        from wynxo.theme import resolve

        palette = resolve("sakura")
        assert palette.accent != palette.bad


class TestSheStopsWhenWynxoDoes:
    """A speech process is a child that outlives its parent. Quitting
    mid-sentence left the voice talking to an empty terminal."""

    def test_both_exits_silence_her(self):
        import inspect

        from wynxo.cli import Repl

        for loop in (Repl._chat_loop, Repl._loop):
            source = inspect.getsource(loop)
            assert "self.speaker.stop()" in source, loop.__name__

    def test_stop_terminates_a_running_process(self):
        import subprocess

        from wynxo.speech import Speaker

        speaker = Speaker()
        speaker._process = subprocess.Popen(
            [__import__("sys").executable, "-c", "import time; time.sleep(30)"])
        assert speaker.is_speaking() is True
        held = speaker._process
        speaker.stop()
        assert speaker._process is None
        assert speaker.is_speaking() is False
        # stop() reaps rather than abandoning: nothing is left running.
        assert held.poll() is not None

    def test_stopping_when_silent_is_harmless(self):
        from wynxo.speech import Speaker

        Speaker().stop()          # must not raise


class TestSpeakableStaysFastOnLongAnswers:
    """Preparing an answer for speech must not cost more than saying it.

    The path rule was written \\S*[/\\\\]\\S*, which at every position ate to
    the end of the answer, failed to find a slash, and gave the characters
    back one at a time. A 40k answer took ten seconds -- with the UI
    waiting on it.
    """

    BUDGET = 5.0    # linear does this in milliseconds; quadratic in seconds

    def test_a_long_answer_with_no_paths_at_all(self):
        import time

        start = time.perf_counter()
        speech.speakable("x" * 40_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_a_long_answer_of_path_ish_characters(self):
        # Worst case: one unbroken run of the characters a path is made of,
        # with no slash anywhere to end the search early.
        import time

        start = time.perf_counter()
        speech.speakable("a.b-c_d" * 6_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_a_long_answer_full_of_unclosed_brackets(self):
        # What the link rule chokes on: an opening bracket everywhere and a
        # closing one nowhere. A log dump or raw escape sequences look like
        # this. CPython 3.11 hides most of the cost; 3.10 does not.
        import time

        start = time.perf_counter()
        speech.speakable("\x1b[31ma" * 27_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_links_still_read_as_their_text(self):
        assert speech.speakable("see [the readme](http://x/y) now") == (
            "see the readme now")

    def test_a_long_answer_full_of_unclosed_brackets(self):
        # What the link rule chokes on: an opening bracket everywhere and a
        # closing one nowhere. A log dump or raw escape sequences look like
        # this. CPython 3.11 hides most of the cost; 3.10 does not.
        import time

        start = time.perf_counter()
        speech.speakable("\x1b[31ma" * 7_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_links_still_read_as_their_text(self):
        assert speech.speakable("see [the readme](http://x/y) now") == (
            "see the readme now")

    def test_a_long_answer_full_of_unclosed_brackets(self):
        # What the link rule chokes on: an opening bracket everywhere and a
        # closing one nowhere. A log dump or raw escape sequences look like
        # this. CPython 3.11 hides most of the cost; 3.10 does not.
        import time

        start = time.perf_counter()
        speech.speakable("\x1b[31ma" * 7_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_links_still_read_as_their_text(self):
        assert speech.speakable("see [the readme](http://x/y) now") == (
            "see the readme now")

    def test_a_long_answer_full_of_unclosed_brackets(self):
        # What the link rule chokes on: an opening bracket everywhere and a
        # closing one nowhere. A log dump or raw escape sequences look like
        # this. CPython 3.11 hides most of the cost; 3.10 does not.
        import time

        start = time.perf_counter()
        speech.speakable("\x1b[31ma" * 7_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_links_still_read_as_their_text(self):
        assert speech.speakable("see [the readme](http://x/y) now") == (
            "see the readme now")

    def test_a_long_answer_full_of_unclosed_brackets(self):
        # What the link rule chokes on: an opening bracket everywhere and a
        # closing one nowhere. A log dump or raw escape sequences look like
        # this. CPython 3.11 hides most of the cost; 3.10 does not.
        import time

        start = time.perf_counter()
        speech.speakable("\x1b[31ma" * 7_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_links_still_read_as_their_text(self):
        assert speech.speakable("see [the readme](http://x/y) now") == (
            "see the readme now")

    def test_a_long_answer_full_of_unclosed_brackets(self):
        # What the link rule chokes on: an opening bracket everywhere and a
        # closing one nowhere. A log dump or raw escape sequences look like
        # this. CPython 3.11 hides most of the cost; 3.10 does not.
        import time

        start = time.perf_counter()
        speech.speakable("\x1b[31ma" * 7_000, limit=10 ** 9)
        assert time.perf_counter() - start < self.BUDGET

    def test_links_still_read_as_their_text(self):
        assert speech.speakable("see [the readme](http://x/y) now") == (
            "see the readme now")

    def test_paths_still_come_out_of_a_long_answer(self):
        text = "x" * 20_000 + " open src/main.py now " + "y" * 20_000
        spoken = speech.speakable(text, limit=10 ** 9)
        assert "src/main.py" not in spoken
        assert "open" in spoken and "now" in spoken
