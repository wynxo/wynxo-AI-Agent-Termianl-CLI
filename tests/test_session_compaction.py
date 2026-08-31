from pathlib import Path

from wynxo.session import Session


def make_session(messages):
    session = Session(workspace=Path("."))
    session.messages = messages
    return session


def test_compaction_never_orphans_tool_result():
    session = make_session([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "x.py"}}}
        ]},
        {"role": "tool", "tool_name": "read_file", "content": "result"},
        {"role": "user", "content": "recent 1"},
        {"role": "assistant", "content": "recent 2"},
        {"role": "user", "content": "recent 3"},
        {"role": "assistant", "content": "recent 4"},
        {"role": "user", "content": "recent 5"},
    ])

    older, kept = session.slice_for_summary(keep_recent=6)

    assert not any(m.get("role") == "tool" for m in older)
    assert kept[0]["role"] == "assistant"
    assert kept[0].get("tool_calls")
    assert kept[1]["role"] == "tool"


def test_compaction_keeps_all_sibling_tool_results_together():
    session = make_session([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a"}}},
            {"function": {"name": "read_file", "arguments": {"path": "b"}}},
        ]},
        {"role": "tool", "tool_name": "read_file", "content": "a"},
        {"role": "tool", "tool_name": "read_file", "content": "b"},
        {"role": "user", "content": "recent 1"},
        {"role": "assistant", "content": "recent 2"},
        {"role": "user", "content": "recent 3"},
        {"role": "assistant", "content": "recent 4"},
        {"role": "user", "content": "recent 5"},
    ])

    _, kept = session.slice_for_summary(keep_recent=6)

    assert [m["role"] for m in kept[:3]] == ["assistant", "tool", "tool"]


# -- superseding stale tool output -------------------------------------------
#
# A read-edit-verify loop reads the same file two or three times in one turn.
# Keeping every copy in full is how a small context window gets spent on
# describing one file repeatedly. These tests pin the collapse and, just as
# importantly, the things it must not do.


def big(marker: str, size: int = 5000) -> str:
    return f"{marker}:" + "x" * size


def test_a_re_read_collapses_the_earlier_copy():
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", big("first"), "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", big("second"), "c2", subject="read_file:a.py")

    earlier, current = session.messages
    assert earlier["superseded"] is True
    assert "superseded" in earlier["content"]
    assert "first" not in earlier["content"]
    assert current["content"].startswith("second:")
    assert not current.get("superseded")


def test_the_note_says_where_the_content_went():
    # A bare "[dropped]" reads to the model as "the file is empty".
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", big("first"), "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", big("second"), "c2", subject="read_file:a.py")

    note = session.messages[0]["content"]
    assert "a.py" in note
    assert "further down" in note


def test_superseding_replaces_and_never_removes():
    # A tool result answers a specific tool call. Deleting it leaves that call
    # unanswered, which is a malformed conversation, not a smaller one.
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", big("first"), "call-1", subject="read_file:a.py")
    session.add_tool_result("read_file", big("second"), "call-2", subject="read_file:a.py")

    assert len(session.messages) == 2
    assert [m["tool_call_id"] for m in session.messages] == ["call-1", "call-2"]
    assert all(m["role"] == "tool" for m in session.messages)
    assert all(m["content"] for m in session.messages)


def test_a_different_file_is_untouched():
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", big("a1"), "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", big("b1"), "c2", subject="read_file:b.py")
    session.add_tool_result("read_file", big("a2"), "c3", subject="read_file:a.py")

    assert session.messages[0].get("superseded")
    assert not session.messages[1].get("superseded")
    assert session.messages[1]["content"].startswith("b1:")
    assert not session.messages[2].get("superseded")


def test_results_without_a_subject_are_never_collapsed():
    # A shell result is about a moment, not a subject; a later run does not
    # make an earlier run untrue.
    session = Session(workspace=Path("."))
    session.add_tool_result("run_shell", big("run one"), "c1")
    session.add_tool_result("run_shell", big("run two"), "c2")

    assert not any(m.get("superseded") for m in session.messages)
    assert session.superseded_chars == 0


def test_two_small_results_are_left_alone():
    # The note costs more than the result. Collapsing here makes the
    # conversation harder to read and no smaller.
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", "ok", "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", "ok still", "c2", subject="read_file:a.py")

    assert not any(m.get("superseded") for m in session.messages)


def test_a_large_earlier_copy_collapses_even_when_the_new_one_is_small():
    # The file was truncated to nothing. The 20k description of what it used
    # to contain is exactly what has to go.
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", big("was huge"), "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", "", "c2", subject="read_file:a.py")

    assert session.messages[0].get("superseded")
    assert session.superseded_chars > 4000


def test_a_third_read_does_not_re_collapse_the_first():
    # Double-counting the reclaimed bytes would make /stats lie, and rewriting
    # a note that is already a note reclaims nothing.
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", big("one"), "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", big("two"), "c2", subject="read_file:a.py")
    after_two = session.superseded_chars
    session.add_tool_result("read_file", big("three"), "c3", subject="read_file:a.py")

    assert [bool(m.get("superseded")) for m in session.messages] == [True, True, False]
    assert session.superseded_chars == after_two * 2


def test_reclaimed_bytes_are_counted_net_of_the_note():
    session = Session(workspace=Path("."))
    body = big("first")
    session.add_tool_result("read_file", body, "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", big("second"), "c2", subject="read_file:a.py")

    note = len(session.messages[0]["content"])
    assert session.superseded_chars == len(body) - note


def test_the_tally_survives_a_save_and_reload(tmp_path, monkeypatch):
    import wynxo.session as session_module

    monkeypatch.setattr(session_module, "data_dir", lambda: tmp_path)
    session = Session(workspace=Path("."))
    session.add_tool_result("read_file", big("first"), "c1", subject="read_file:a.py")
    session.add_tool_result("read_file", big("second"), "c2", subject="read_file:a.py")
    assert session.save() is not None

    back = Session.load(session.session_id, Path("."))
    assert back is not None
    assert back.superseded_chars == session.superseded_chars


# -- deciding what a result is about ------------------------------------------
#
# _subject_of is the half of the collapse that decides whether two results can
# both be true. Getting it wrong loses information, so every "no" below is
# load-bearing.


def subject(name, arguments, ok=True):
    from wynxo.agent import _subject_of
    from wynxo.parsing import ToolCall
    from wynxo.tools.base import ToolResult

    result = (ToolResult.success("body") if ok else ToolResult.failure("no"))
    return _subject_of(ToolCall(name=name, arguments=arguments), result)


def test_a_whole_file_read_has_a_subject():
    assert subject("read_file", {"path": "a.py"}) == "read_file:/:a.py"


def test_the_same_path_read_twice_gets_the_same_subject():
    assert subject("read_file", {"path": "a.py", "limit": 2000}) == \
           subject("read_file", {"path": "a.py"})


def test_a_failed_read_is_not_a_view_of_the_file():
    # It describes an error, not contents, and it must not silently erase the
    # last successful read.
    assert subject("read_file", {"path": "a.py"}, ok=False) == ""


def test_a_ranged_read_does_not_supersede():
    assert subject("read_file", {"path": "a.py", "start_line": 10, "end_line": 20}) == ""
    assert subject("read_file", {"path": "a.py", "offset": 500}) == ""


def test_a_range_sent_as_a_string_still_counts_as_a_range():
    # Models send "10" as readily as 10.
    assert subject("read_file", {"path": "a.py", "start_line": "10"}) == ""


def test_an_explicit_zero_range_is_a_whole_file_read():
    assert subject("read_file", {"path": "a.py", "start_line": 0, "end_line": 0}) != ""


def test_tools_that_are_not_file_views_have_no_subject():
    for name, arguments in [("grep", {"path": "a.py"}),
                            ("run_shell", {"path": "a.py"}),
                            ("list_dir", {"path": "."}),
                            ("write_file", {"path": "a.py"}),
                            ("edit_file", {"path": "a.py"})]:
        assert subject(name, arguments) == "", name


def test_only_the_github_read_operation_is_a_file_view():
    common = {"repo": "o/r", "path": "a.py"}
    assert subject("github_read", {**common, "operation": "read"}) != ""
    for operation in ("search", "tree", "stat", ""):
        assert subject("github_read", {**common, "operation": operation}) == "", operation


def test_the_same_path_on_two_branches_is_two_files():
    read = {"operation": "read", "repo": "o/r", "path": "a.py"}
    assert subject("github_read", {**read, "branch": "main"}) != \
           subject("github_read", {**read, "branch": "feature"})


def test_the_same_path_in_two_repositories_is_two_files():
    read = {"operation": "read", "path": "a.py"}
    assert subject("github_read", {**read, "repo": "o/one"}) != \
           subject("github_read", {**read, "repo": "o/two"})


def test_a_local_read_and_a_remote_read_are_not_the_same_file():
    assert subject("read_file", {"path": "a.py"}) != \
           subject("github_read", {"operation": "read", "repo": "o/r", "path": "a.py"})


def test_a_missing_or_junk_path_has_no_subject():
    assert subject("read_file", {}) == ""
    assert subject("read_file", {"path": "   "}) == ""
    assert subject("read_file", None) == ""
