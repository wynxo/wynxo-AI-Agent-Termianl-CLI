"""The conversation has to survive being interrupted.

Ctrl-C while the model is generating is the ordinary way a turn ends -- you
saw enough, or it went the wrong way -- and until autosave the on-disk copy
was whatever the *last completed turn* left behind. A turn that ran for two
minutes across a dozen tool calls and was then interrupted left no trace at
all, not even a record of having been asked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wynxo.session import Session


@pytest.fixture
def store(tmp_path, monkeypatch):
    (tmp_path / "sessions").mkdir(parents=True)
    monkeypatch.setattr("wynxo.session.data_dir", lambda: tmp_path)
    return tmp_path / "sessions"


def on_disk(store: Path, session: Session) -> dict:
    path = store / f"{session.session_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class TestAutosave:
    def test_off_by_default_so_nothing_touches_the_real_store(self, store):
        """A Session built in a test or a tool must not write anywhere."""
        session = Session(workspace=store.parent)
        session.add_user("hello")
        assert not list(store.glob("*.json"))

    def test_the_request_is_on_disk_before_a_single_token_comes_back(self, store):
        session = Session(workspace=store.parent, autosave=True)
        session.add_user("fix the failing test")
        assert [m["content"] for m in on_disk(store, session)["messages"]] \
            == ["fix the failing test"]

    def test_every_step_of_a_long_turn_is_recorded_as_it_happens(self, store):
        session = Session(workspace=store.parent, autosave=True)
        session.add_user("go")
        for step in range(4):
            session.add_assistant("", [{"function": {"name": "grep"}}])
            session.add_tool_result("grep", f"hit {step}")
            # Interrupting here, at any step, must leave everything so far.
            assert len(on_disk(store, session)["messages"]) == 1 + 2 * (step + 1)

    def test_unwinding_an_interrupted_turn_is_persisted(self, store):
        """The repaired shape, not the malformed one, is what reloads."""
        session = Session(workspace=store.parent, autosave=True)
        session.add_user("go")
        session.add_assistant("", [{"function": {"name": "read_file"}},
                                   {"function": {"name": "grep"}}])
        assert session.close_open_tool_calls() == 2

        saved = on_disk(store, session)
        assert [m["role"] for m in saved["messages"]] \
            == ["user", "assistant", "tool", "tool"]
        # And it comes back the same way.
        back = Session.load(session.session_id, store.parent)
        assert [m["role"] for m in back.messages] \
            == ["user", "assistant", "tool", "tool"]

    def test_compaction_is_persisted(self, store):
        session = Session(workspace=store.parent, autosave=True)
        for i in range(8):
            session.add_user(f"message {i}")
        _, kept = session.slice_for_summary()
        session.apply_compaction("they talked about numbers", kept)
        assert "they talked about numbers" in json.dumps(on_disk(store, session))

    def test_the_store_is_swept_once_rather_than_on_every_message(self, store):
        """Autosave writes constantly; pruning on each one is wasted work."""
        session = Session(workspace=store.parent, autosave=True)
        swept = []
        session.prune = lambda *a, **k: swept.append(1)
        for i in range(10):
            session.add_user(f"m{i}")
        assert swept == [1]


class TestTheAgentAlwaysOwnsADurableConversation:
    def test_replacing_the_conversation_keeps_it_durable(self, store, tmp_path):
        """/clear, /new and /resume all assign agent.session.

        A resumed conversation that quietly stopped saving would be worse
        than one never saved: you carry on talking into it and lose the lot.
        """
        from wynxo.agent import Agent

        agent = Agent.__new__(Agent)
        agent.session = Session(workspace=tmp_path)
        assert agent.session.autosave is True

        agent.session = Session(workspace=tmp_path)
        assert agent.session.autosave is True


class TestTitles:
    def test_a_conversation_is_named_by_what_was_asked(self, store):
        session = Session(workspace=store.parent)
        session.add_user("why does the parser drop the last token")
        assert session.title() == "why does the parser drop the last token"

    def test_wynxos_own_words_never_name_the_conversation(self, store):
        """The inline plan note and the compaction preamble wear the user
        role. Neither is something the user said."""
        session = Session(workspace=store.parent)
        session.add_user("(If this takes more than a couple of steps...)")
        session.add_user("[Earlier conversation, condensed...]")
        session.add_user("actually fix the parser")
        assert session.title() == "actually fix the parser"

    def test_a_title_survives_the_round_trip(self, store):
        session = Session(workspace=store.parent, autosave=True)
        session.add_user("teach me about  \n  embeddings")
        rows = Session.recent()
        assert rows[0]["preview"] == "teach me about embeddings"


class TestListingAcrossWorkspaces:
    def test_conversations_from_other_projects_are_listed_too(self, store):
        """The point of resuming is often that it happened somewhere else."""
        for name, workspace in (("a", "/one"), ("b", "/two")):
            (store / f"{name}.json").write_text(json.dumps({
                "session_id": name, "workspace": workspace,
                "title": f"about {name}", "updated_at": 1.0,
                "messages": [{"role": "user", "content": f"about {name}"}],
            }), encoding="utf-8")
        assert {r["workspace"] for r in Session.recent()} == {"/one", "/two"}

    def test_the_conversation_you_are_in_is_not_offered(self, store):
        for name in ("a", "b"):
            (store / f"{name}.json").write_text(json.dumps({
                "session_id": name, "workspace": "/w", "messages": [],
            }), encoding="utf-8")
        assert [r["session_id"] for r in Session.recent(exclude="a")] == ["b"]
