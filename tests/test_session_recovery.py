"""Reading session files that are not what wynxo last wrote.

Session files outlive the process that made them. A crash mid-write, a full
disk, or a different version of wynxo all leave a file that still parses as
JSON but holds the wrong shapes -- and those used to travel inland and die
several frames later.

The listing case is the one that matters most to the user: /resume and
/sessions read every file, so one bad file must cost them that one session
rather than every conversation they have ever had.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from wynxo.session import Session


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    directory = tmp_path / "sessions"
    directory.mkdir(parents=True)
    monkeypatch.setattr("wynxo.session.data_dir", lambda: tmp_path)
    return directory


def write(directory: Path, name: str, doc) -> None:
    (directory / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")


def good(subject: str = "hello") -> dict:
    return {
        "session_id": "good", "workspace": "/w", "created_at": time.time(),
        "updated_at": time.time(), "compactions": 0,
        "messages": [{"role": "user", "content": subject}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "requests": 1, "tool_calls": 0},
    }


class TestOneBadFileDoesNotHideTheRest:
    """The bug that would actually cost someone their work."""

    BROKEN = [
        ("not_json", "{ truncated"),
        ("a_list", []),
        ("a_string", "session"),
        ("a_number", 5),
        ("null", None),
        ("messages_not_a_list", {"messages": "hello"}),
        ("messages_of_junk", {"messages": ["a string", None, 7]}),
        ("message_without_content", {"messages": [{"role": "user"}]}),
        ("usage_not_a_dict", {"messages": [], "usage": True}),
    ]

    @pytest.mark.parametrize("name,doc", BROKEN)
    def test_a_broken_file_leaves_the_good_ones_listed(self, sessions, name, doc):
        write(sessions, "keeper", good("the work I want back"))
        if isinstance(doc, str) and doc.startswith("{"):
            (sessions / f"{name}.json").write_text(doc, encoding="utf-8")
        else:
            write(sessions, name, doc)

        listed = Session.recent()
        previews = [row["preview"] for row in listed]
        assert "the work I want back" in previews, (
            f"{name} hid the user's real session")

    def test_every_broken_file_at_once_still_lists_the_keeper(self, sessions):
        write(sessions, "keeper", good("still here"))
        for name, doc in self.BROKEN:
            if isinstance(doc, str) and doc.startswith("{"):
                (sessions / f"{name}.json").write_text(doc, encoding="utf-8")
            else:
                write(sessions, name, doc)

        assert any(r["preview"] == "still here" for r in Session.recent())

    def test_a_file_that_vanishes_mid_listing_is_survivable(self, sessions,
                                                            monkeypatch):
        """/new prunes old sessions, and a second wynxo may be running."""
        write(sessions, "keeper", good("still here"))
        write(sessions, "ghost", good("gone"))

        real = Path.stat

        def vanishing(self, *args, **kwargs):
            if self.name == "ghost.json":
                raise OSError("no such file")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", vanishing)
        assert any(r["preview"] == "still here" for r in Session.recent())

    def test_rows_always_have_the_types_the_ui_formats(self, sessions):
        """The listing is rendered with %-formatting and sorted on
        updated_at, so a str where a float belongs breaks the screen."""
        write(sessions, "odd", {"session_id": 5, "workspace": [],
                                "updated_at": "not a time",
                                "messages": [{"role": "user", "content": 9}]})
        for row in Session.recent():
            assert isinstance(row["session_id"], str)
            assert isinstance(row["workspace"], str)
            assert isinstance(row["updated_at"], float)
            assert isinstance(row["messages"], int)
            assert isinstance(row["preview"], str)


class TestLoadingOne:
    def test_a_document_that_is_not_an_object_is_not_a_session(self, sessions,
                                                               tmp_path):
        for name, doc in (("a", []), ("b", "text"), ("c", 5), ("d", None)):
            write(sessions, name, doc)
            assert Session.load(name, tmp_path) is None

    def test_junk_messages_are_dropped_not_carried(self, sessions, tmp_path):
        """Carrying them would break the next request rather than this one,
        which is a much harder bug to trace back to a bad file."""
        write(sessions, "s", {"messages": [
            {"role": "user", "content": "keep me"}, "junk", None, 7, []]})
        loaded = Session.load("s", tmp_path)
        assert [m["content"] for m in loaded.messages] == ["keep me"]

    def test_counts_written_as_strings_still_add_up(self, sessions, tmp_path):
        write(sessions, "s", {"messages": [], "usage": {
            "prompt_tokens": "10", "completion_tokens": 5.0,
            "requests": "1", "tool_calls": None}})
        usage = Session.load("s", tmp_path).usage
        assert usage.prompt_tokens == 10 and usage.completion_tokens == 5
        assert usage.requests == 1 and usage.tool_calls == 0

    def test_a_missing_creation_time_means_now_not_1970(self, sessions,
                                                        tmp_path):
        """A session dated 1970 sorts to the bottom and looks lost."""
        write(sessions, "s", {"messages": []})
        loaded = Session.load("s", tmp_path)
        assert loaded.created_at > time.time() - 60

    def test_a_loaded_session_can_still_estimate_its_tokens(self, sessions,
                                                            tmp_path):
        """token_estimate() iterates messages, so a non-list used to raise
        here rather than at load -- far from the file that caused it."""
        write(sessions, "s", {"messages": 0, "usage": ""})
        loaded = Session.load("s", tmp_path)
        assert isinstance(loaded.token_estimate(), int)


class TestTheCoercionsThemselves:
    def test_text_keeps_what_it_can_rather_than_dropping_it(self):
        from wynxo.coerce import as_text

        assert as_text({"text": "inner"}) == "inner"
        assert as_text(["a", "b"]) == "ab"
        assert as_text(None) == "" and as_text(True) == ""

    def test_a_float_that_is_not_a_number_falls_back(self):
        from wynxo.coerce import as_float

        assert as_float(float("nan"), 7.0) == 7.0
        assert as_float(float("inf"), 7.0) == 7.0
        assert as_float("1.5") == 1.5

    def test_int_refuses_bools_which_would_count_as_one(self):
        from wynxo.coerce import as_int

        assert as_int(True) == 0 and as_int(False) == 0
