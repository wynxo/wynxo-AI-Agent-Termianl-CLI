"""Untrusted input reaching parsers, bounds and the conversation record.

Four defects, each found by driving the real code rather than reading it:
a model or server can crash the turn with a long number or deep nesting; a
server can crash the code that explains its own error; a Ctrl-C can leave
the conversation permanently malformed; and NaN walks through every bound
the schema declares.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from wynxo.agent import Agent, Callbacks, Interrupted
from wynxo.coerce import loads
from wynxo.config import LOAD_PROBLEMS, _validate_forgivingly
from wynxo.effort import resolve
from wynxo.parsing import parse_turn, repair_json
from wynxo.provider import Chunk, OllamaClient, OpenAIClient, _openai_messages
from wynxo.schema import Field
from wynxo.scope import Boundary, Scope
from wynxo.session import Session
from wynxo.tools import build_registry

# Under the interpreter's 4300-digit limit for int parsing, and past the
# decoder's recursion limit. Both are decoder failures that are not
# JSONDecodeError, which is all every handler used to catch.
LONG_NUMBER = "9" * 5000
DEEP = "[" * 60000


class TestTheDecoderFailuresNobodyCaught:
    def test_a_long_number_is_not_a_crash(self):
        assert loads('{"n":' + LONG_NUMBER + "}") is None

    def test_deep_nesting_is_not_a_crash(self):
        assert loads(DEEP) is None

    def test_a_json_value_that_is_not_an_object_is_none(self):
        """Callers immediately .get() the result. A bare string decoded
        fine and produced AttributeError one frame later."""
        for text in ('"a string"', "[1,2,3]", "123", "null", "true"):
            assert loads(text) is None, text

    def test_ordinary_objects_still_decode(self):
        assert loads('{"a": 1}') == {"a": 1}
        assert loads(b'{"a": 1}') == {"a": 1}

    def test_non_text_is_none_rather_than_a_typeerror(self):
        for value in (None, 123, [], {"a": 1}):
            assert loads(value) is None


class TestAToolCallCannotKillTheTurn:
    """repair_json promises None when the text is hopeless. Two decoder
    failures escaped instead, and the turn died on model output."""

    def test_a_long_number_in_a_tool_call(self):
        raw = ('<tool_call>{"name":"shell","arguments":{"n":'
               + LONG_NUMBER + "}}</tool_call>")
        turn = parse_turn(raw)
        assert turn.malformed, "it should be reported as malformed, not raised"

    def test_deep_nesting_in_a_tool_call(self):
        raw = '<tool_call>{"name":"shell","arguments":' + DEEP + "}</tool_call>"
        turn = parse_turn(raw)
        assert turn.malformed

    def test_repair_json_returns_none_rather_than_raising(self):
        assert repair_json('{"n":' + LONG_NUMBER + "}") is None
        assert repair_json(DEEP) is None

    def test_a_good_tool_call_still_parses(self):
        turn = parse_turn(
            '<tool_call>{"name":"shell","arguments":{"command":"ls"}}</tool_call>')
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].arguments == {"command": "ls"}
        assert not turn.malformed


class TestExplainingAnErrorCannotItselfError:
    """_explain_error turns a server failure into something readable. It
    reached straight for .get on whatever the body decoded to, so four
    realistic bodies replaced the server's own diagnosis with an
    AttributeError about explaining it."""

    BODIES = ['"a string"', "[1,2,3]", "123", "null", "not json at all", "",
              '{"error":' + LONG_NUMBER + "}", DEEP]

    @pytest.mark.parametrize("client", [OllamaClient, OpenAIClient])
    def test_no_body_shape_raises(self, client):
        instance = client.__new__(client)
        for body in self.BODIES:
            message = client._explain_error(instance, 500, body, {"model": "m"})
            assert isinstance(message, str) and message.strip()

    @pytest.mark.parametrize("client", [OllamaClient, OpenAIClient])
    def test_a_real_error_message_still_comes_through(self, client):
        instance = client.__new__(client)
        message = client._explain_error(
            instance, 500, '{"error":"the real reason"}', {"model": "m"})
        assert "the real reason" in message

    @pytest.mark.parametrize("client", [OllamaClient, OpenAIClient])
    def test_an_unparseable_body_is_shown_verbatim(self, client):
        instance = client.__new__(client)
        message = client._explain_error(
            instance, 500, "plain text failure", {"model": "m"})
        assert "plain text failure" in message


class Quiet(Callbacks):
    pass


class TwoReads:
    """Announces two read_file calls, then answers."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        if self.calls == 1:
            return self._iter("", [
                {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
                {"function": {"name": "read_file", "arguments": {"path": "b.txt"}}}])
        return self._iter("done", [])

    async def _iter(self, content, calls):
        yield Chunk(content=content, tool_calls=calls, done=True)


def build_agent(tmp: Path) -> Agent:
    (tmp / "a.txt").write_text("A")
    (tmp / "b.txt").write_text("B")
    from wynxo.config import Config

    config = Config(verify_with_tests=False, allow_shell=False,
                    auto_approve=["*"])
    return Agent(TwoReads(), config, resolve("low"), tmp, Quiet(),
                 registry=build_registry(tmp),
                 boundary=Boundary(scope=Scope.FOLDER, root=tmp))


def pairing(session) -> tuple[int, int]:
    announced = sum(len(m["tool_calls"]) for m in session.messages
                    if m.get("role") == "assistant" and m.get("tool_calls"))
    answered = sum(1 for m in session.messages if m.get("role") == "tool")
    return announced, answered


class TestAnInterruptedTurnLeavesAValidConversation:
    """A tool call is a question the conversation has to answer. The
    announcement is written before the calls run, so a cancellation in
    between left them unanswered for good -- and the user was told the
    conversation was intact. An OpenAI-compatible server rejects that shape,
    so one Ctrl-C made every later request in the session fail.
    """

    def run_cancelled_at(self, steps: int) -> Agent:
        tmp = Path(tempfile.mkdtemp())
        agent = build_agent(tmp)

        async def go():
            task = asyncio.ensure_future(agent.run("read them"))
            for _ in range(steps):
                await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Interrupted):
                pass

        asyncio.run(go())
        return agent

    def test_every_cancellation_point_stays_well_formed(self):
        """Swept rather than pinned to one timing: the window was six of
        the first fourteen scheduler positions."""
        for steps in range(1, 20):
            announced, answered = pairing(self.run_cancelled_at(steps).session)
            assert announced == answered, (
                f"cancelled after {steps} steps: {announced} calls announced, "
                f"{answered} answered")

    def test_no_tool_result_dangles_on_the_openai_wire(self):
        for steps in (1, 4, 8, 15):
            session = self.run_cancelled_at(steps).session
            wire = _openai_messages(session.wire())
            ids = {call["id"] for m in wire for call in m.get("tool_calls") or []}
            refs = [m["tool_call_id"] for m in wire if m["role"] == "tool"]
            assert not [r for r in refs if r not in ids], (
                f"cancelled after {steps}: tool results reference calls that "
                f"are not on the assistant message")

    def test_the_stand_in_says_the_call_never_ran(self):
        session = self.run_cancelled_at(2).session
        notes = [m["content"] for m in session.messages if m.get("role") == "tool"]
        assert any("interrupted" in n.lower() for n in notes)

    def test_an_uninterrupted_turn_is_untouched(self):
        tmp = Path(tempfile.mkdtemp())
        agent = build_agent(tmp)
        asyncio.run(agent.run("read them"))
        announced, answered = pairing(agent.session)
        assert (announced, answered) == (2, 2), "the repair invented results"


class TestCloseOpenToolCalls:
    def session_with(self, messages) -> Session:
        session = Session(workspace=Path("."))
        session.messages = list(messages)
        return session

    def test_it_is_idempotent(self):
        session = self.session_with([
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "read_file"}}]},
            {"role": "tool", "tool_name": "read_file", "content": "ok"},
        ])
        assert session.close_open_tool_calls() == 0
        assert session.close_open_tool_calls() == 0
        assert len(session.messages) == 2

    def test_it_answers_only_what_is_missing(self):
        session = self.session_with([
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "grep"}},
                {"function": {"name": "list_dir"}}]},
            {"role": "tool", "tool_name": "read_file", "content": "ok"},
        ])
        assert session.close_open_tool_calls() == 2
        assert [m["role"] for m in session.messages] == [
            "assistant", "tool", "tool", "tool"]

    def test_the_stand_ins_land_before_the_next_message(self):
        """Order is the pairing: results answer the calls above them."""
        session = self.session_with([
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a"}}, {"function": {"name": "b"}}]},
            {"role": "user", "content": "next question"},
        ])
        session.close_open_tool_calls()
        assert [m["role"] for m in session.messages] == [
            "assistant", "tool", "tool", "user"]

    def test_several_exchanges_are_each_repaired(self):
        session = self.session_with([
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "a"}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "b"}}]},
        ])
        assert session.close_open_tool_calls() == 2
        assert [m["role"] for m in session.messages] == [
            "assistant", "tool", "assistant", "tool"]

    def test_a_conversation_with_no_tool_calls_is_untouched(self):
        session = self.session_with([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert session.close_open_tool_calls() == 0

    def test_the_stand_in_carries_the_id_the_wire_expects(self):
        """_openai_messages numbers calls by position; the stand-in has to
        use the same convention or it dangles."""
        session = self.session_with([
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a"}}, {"function": {"name": "b"}}]},
        ])
        session.close_open_tool_calls()
        assert [m["tool_call_id"] for m in session.messages
                if m["role"] == "tool"] == ["call_0", "call_1"]


class TestNaNCannotWalkThroughABound:
    """Every comparison against NaN is false, so `value < self.ge` passed it
    through every bounded field. Python's json decoder accepts the literal
    NaN, and a NaN reaching asyncio.wait_for is not a short timeout but no
    timeout at all."""

    def test_json_really_does_decode_the_literal(self):
        """The premise, so this does not rest on a claim about the decoder."""
        assert json.loads("NaN") != json.loads("NaN")

    def test_wait_for_does_not_time_out_on_nan(self):
        """Why it matters: the wait never expires."""
        async def probe():
            await asyncio.wait_for(asyncio.sleep(0), timeout=float("nan"))
            return "no timeout enforced"

        assert asyncio.run(probe()) == "no timeout enforced"

    @pytest.mark.parametrize("field_name,default", [
        ("request_timeout", 600.0),
        ("stt_silence_timeout", 1.25),
        ("stt_max_duration", 30.0),
        ("stt_transcription_timeout", 60.0),
    ])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_no_bounded_float_accepts_a_non_finite_value(self, field_name,
                                                         default, value):
        LOAD_PROBLEMS.clear()
        config = _validate_forgivingly({field_name: value})
        assert getattr(config, field_name) == default
        assert LOAD_PROBLEMS, "it was accepted silently"

    def test_a_field_with_no_bounds_stays_forgiving(self):
        """The check is what declaring a range asks for; it must not
        tighten fields that never declared one."""
        errors: list = []
        Field(float, "x", default=1.0)._bounded(float("nan"), "x", errors)
        assert errors == [], "an unbounded field started rejecting values"

    def test_a_bounded_field_records_why(self):
        errors: list = []
        Field(float, "x", default=1.0, ge=0.0, le=9.0)._bounded(
            float("nan"), "x", errors)
        assert errors and "finite" in errors[0][1]

    def test_ordinary_values_still_pass(self):
        LOAD_PROBLEMS.clear()
        assert _validate_forgivingly({"request_timeout": 42.0}).request_timeout == 42.0
        assert not LOAD_PROBLEMS


class TestTheTwoUnboundedSettings:
    """Every other number in the config carries ge/le. These two did not,
    and a zero request_timeout fails every request the instant it is made --
    telling the user to raise the setting they had just set."""

    @pytest.mark.parametrize("value", [-1, 0, 0.5, 86401, 1e9])
    def test_an_unusable_timeout_falls_back_to_the_default(self, value):
        LOAD_PROBLEMS.clear()
        assert _validate_forgivingly({"request_timeout": value}).request_timeout == 600.0
        assert LOAD_PROBLEMS

    @pytest.mark.parametrize("value", [1, 30, 600, 86400])
    def test_real_timeouts_are_kept(self, value):
        LOAD_PROBLEMS.clear()
        assert _validate_forgivingly(
            {"request_timeout": value}).request_timeout == float(value)
        assert not LOAD_PROBLEMS

    def test_the_rest_of_the_file_survives_a_bad_timeout(self):
        LOAD_PROBLEMS.clear()
        config = _validate_forgivingly({"request_timeout": 0, "model": "KEPT"})
        assert config.model == "KEPT"
        assert config.request_timeout == 600.0

    @pytest.mark.parametrize("value,kept", [(-5, False), (0, True),
                                            (200, True), (5000, False)])
    def test_speech_rate_is_bounded_too(self, value, kept):
        LOAD_PROBLEMS.clear()
        got = _validate_forgivingly({"speech_rate": value}).speech_rate
        assert (got == value) is kept


class TestEveryBoundedNumberInTheConfig:
    def test_no_numeric_setting_is_left_unbounded(self):
        """The class, not the two instances: config.py's own opening comment
        says a num_ctx of -5 used to load and be sent to Ollama, so bounding
        is the established pattern. These were the two it had missed."""
        from wynxo.config import Config

        unbounded = [
            name for name, field in Config._fields.items()
            if field.type in (int, float)
            and field.ge is None and field.le is None
            and field.gt is None and field.lt is None
        ]
        assert not unbounded, f"numeric settings with no range: {unbounded}"
