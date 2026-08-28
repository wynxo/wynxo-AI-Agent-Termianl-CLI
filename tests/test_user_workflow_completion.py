from pathlib import Path

from wynxo.cli import CommandCompleter
from wynxo.session import Session


def test_model_completion_uses_dynamic_models():
    completer = CommandCompleter()
    completer._model_names = ["qwen:7b", "llama3:8b"]
    completions = list(completer.get_completions(type("D", (), {"text_before_cursor": "/model qwen"})(), None))
    assert [item.text for item in completions] == ["qwen:7b"]


def test_compaction_preserves_recent_objective_and_tool_result(tmp_path: Path):
    session = Session(tmp_path, system_prompt="coding agent")
    session.add_user("Find and fix the Windows bug")
    session.add_assistant("I found the subprocess issue")
    session.add_tool_result("read_file", "shell uses POSIX assumptions")
    session.apply_compaction("Objective: fix the Windows subprocess issue. File: shell.py.", session.messages[-1:])
    wire = session.wire()
    assert "Objective: fix the Windows subprocess issue" in wire[1]["content"]
    assert any("shell uses POSIX assumptions" in str(message) for message in wire)
    assert session.compactions == 1
