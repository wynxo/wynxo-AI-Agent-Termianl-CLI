from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.agent import Agent, Callbacks
from wynxo.checkpoints import Checkpoints
from wynxo.config import Config
from wynxo.effort import resolve
from wynxo.provider import Chunk
from wynxo.scope import Boundary, Scope
from wynxo.tools import build_registry


class FakeBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        reply = next(self.replies)

        async def stream():
            yield Chunk(content=reply.get("content", ""),
                        tool_calls=reply.get("tool_calls", []), done=True)

        return stream()


class Recorder(Callbacks):
    def __init__(self):
        self.events = []

    async def on_tool_start(self, name, summary):
        self.events.append(("start", name, summary))

    async def on_tool_result(self, name, ok, display, output):
        self.events.append(("result", name, ok, display))


def call(name, **arguments):
    return [{"function": {"name": name, "arguments": arguments}}]


def make_agent(tmp_path, replies, recorder=None, **kwargs):
    config = Config(verify_with_tests=False, auto_approve=["*"], **kwargs)
    backend = FakeBackend(replies)
    agent = Agent(backend, config, resolve("low"), tmp_path,
                  recorder or Callbacks(),
                  registry=build_registry(tmp_path),
                  boundary=Boundary(scope=Scope.FOLDER, root=tmp_path))
    agent.backend = backend
    return agent, backend


def test_agent_recovers_from_failed_edit_then_tests_then_finishes(tmp_path: Path):
    (tmp_path / "bug.py").write_text("def answer():\n    return 1\n")
    replies = [
        {"tool_calls": call("grep", pattern="return", path=".")},
        {"tool_calls": call("edit_file", path="bug.py", old_text="return 99", new_text="return 2")},
        {"tool_calls": call("read_file", path="bug.py")},
        {"tool_calls": call("edit_file", path="bug.py", old_text="return 1", new_text="return 2")},
        {"content": "Fixed the bug after rereading the file."},
    ]
    recorder = Recorder()
    agent, backend = make_agent(tmp_path, replies, recorder)
    result = asyncio.run(agent.run("Find and fix the bug."))
    assert "return 2" in (tmp_path / "bug.py").read_text()
    assert "Fixed the bug" in result.content
    assert backend.calls == 5
    assert [event[1] for event in recorder.events if event[0] == "start"] == [
        "grep", "edit_file", "read_file", "edit_file"
    ]


def test_checkpoint_undo_refuses_to_destroy_user_changes(tmp_path: Path):
    path = tmp_path / "x.py"
    path.write_text("before\n")
    checkpoints = Checkpoints()
    checkpoints.capture(path, "edit_file", "x.py")
    path.write_text("agent change\n")
    path.write_text("user change after agent\n")
    ok, message = checkpoints.undo()
    assert ok
    assert path.read_text() == "before\n"
    assert "Reverted" in message


def test_session_limits_are_reported_in_command_output(tmp_path: Path, monkeypatch):
    """/session says what the limits actually are, not what was configured.

    The iteration limit is the lower of the effort policy's and the
    config's, and it is the number that decides whether a turn stops early
    -- so reporting the config's alone would tell you 7 while the run gives
    you 6.
    """
    import io

    from wynxo.cli import Repl
    from wynxo.session import Session
    from wynxo.ui import UI

    monkeypatch.setattr("wynxo.session.data_dir", lambda: tmp_path)

    session = Session(workspace=tmp_path, session_id="abc")
    session.add_user("why do the tests fail")
    session.usage.requests = 1
    session.usage.tool_calls = 2

    repl = Repl.__new__(Repl)
    repl.agent = type("A", (), {"session": session, "tools": [1, 2]})()
    repl.config = Config(max_tool_iterations=7, num_ctx=1000)
    repl.policy = resolve("low")
    repl.workspace = tmp_path

    ui = UI()
    ui.console.file = io.StringIO()
    ui.console.width = ui.width = 100
    repl.ui = ui
    assert repl._describe_session() is True

    shown = ui.console.file.getvalue()
    assert "up to 6 calls a turn" in shown, shown
    assert "2 tool calls" in shown, shown
    assert "why do the tests fail" in shown, shown
    assert "abc" in shown, shown
