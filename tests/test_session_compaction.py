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
