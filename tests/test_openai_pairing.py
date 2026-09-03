"""Tool calls and their answers have to line up on the OpenAI wire.

Ollama's shape carries no call ids at all, so most conversations reach the
translation without them and both sides have to invent the same ones. The
announcing side numbered its calls by position within the message; the
answering side did not, and fell back to a flat "call_0" for every result.

So a turn that called two tools sent two answers both claiming to answer the
first, and left the second call unanswered. A strict server (vLLM, OpenAI
itself) rejects that outright; a lenient one acts on it with the results
attributed to the wrong calls, which is worse.

The convention was already written down -- close_open_tool_calls invents
"the same id the translation would invent for the call at this position" --
and only one of the two sides implemented it.
"""

from __future__ import annotations

import pytest

from wynxo.provider import _openai_messages
from wynxo.session import Session


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr("wynxo.session.data_dir", lambda: tmp_path)
    return Session(workspace=tmp_path)


def pairs(session: Session) -> tuple[list[str], list[str]]:
    wire = _openai_messages(session.wire())
    announced = [call["id"] for message in wire
                 if message.get("tool_calls")
                 for call in message["tool_calls"]]
    answered = [message.get("tool_call_id") for message in wire
                if message.get("role") == "tool"]
    return announced, answered


def announce(session: Session, *names: str, ids: tuple[str, ...] = ()) -> None:
    calls = []
    for index, name in enumerate(names):
        call = {"function": {"name": name, "arguments": {}}}
        if ids:
            call["id"] = ids[index]
        calls.append(call)
    session.add_assistant("", calls)


class TestEveryCallGetsItsOwnAnswer:
    @pytest.mark.parametrize("count", [1, 2, 3, 5])
    def test_however_many_were_called(self, session, count):
        session.add_user("go")
        announce(session, *[f"tool{i}" for i in range(count)])
        for i in range(count):
            session.add_tool_result(f"tool{i}", f"result {i}")

        announced, answered = pairs(session)
        assert announced == answered
        assert len(set(answered)) == count, "two answers share an id"

    def test_across_several_turns(self, session):
        """The counter has to reset at each announcing message, not run on
        through the conversation."""
        session.add_user("go")
        announce(session, "a", "b")
        session.add_tool_result("a", "1")
        session.add_tool_result("b", "2")
        session.add_assistant("thinking")
        announce(session, "c", "d", "e")
        for name in "cde":
            session.add_tool_result(name, name)

        announced, answered = pairs(session)
        assert announced == answered

    def test_real_ids_are_carried_through_untouched(self, session):
        """A server that does give ids must have its own used, not ours."""
        session.add_user("go")
        announce(session, "a", "b", ids=("call_abc", "call_def"))
        session.add_tool_result("a", "1", call_id="call_abc")
        session.add_tool_result("b", "2", call_id="call_def")

        announced, answered = pairs(session)
        assert announced == answered == ["call_abc", "call_def"]

    def test_a_turn_unwound_after_an_interrupt_still_pairs(self, session):
        """close_open_tool_calls writes the ids it expects the translation
        to invent. If either side changes alone, this is what catches it."""
        session.add_user("go")
        announce(session, "a", "b", "c")
        assert session.close_open_tool_calls() == 3

        announced, answered = pairs(session)
        assert announced == answered


class TestTheRestOfTheTranslation:
    def test_arguments_become_a_json_string(self, session):
        session.add_assistant("", [{"function": {"name": "grep",
                                                 "arguments": {"pattern": "x"}}}])
        call = _openai_messages(session.wire())[0]["tool_calls"][0]
        assert call["arguments" if "arguments" in call else "function"]
        assert isinstance(call["function"]["arguments"], str)
        assert call["type"] == "function"

    def test_an_ordinary_exchange_is_left_alone(self, session):
        session.add_user("hello")
        session.add_assistant("hi")
        assert _openai_messages(session.wire()) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_a_system_prompt_leads(self, session):
        session.system_prompt = "be helpful"
        session.add_user("hello")
        wire = _openai_messages(session.wire())
        assert wire[0] == {"role": "system", "content": "be helpful"}
