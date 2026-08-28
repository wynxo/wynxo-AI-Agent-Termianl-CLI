from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.agent import Agent, Callbacks
from wynxo.config import Config
from wynxo.effort import resolve
from wynxo.parsing import ToolCall
from wynxo.provider import Chunk
from wynxo.scope import Boundary, Scope
from wynxo.tools import build_registry


class Events(Callbacks):
    def __init__(self):
        self.events = []

    async def on_tool_start(self, name, summary):
        self.events.append(("start", name))

    async def on_tool_result(self, name, ok, display, output):
        self.events.append(("result", name, ok))


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        if self.calls == 1:
            return self._chunks("", [{"function": {"name": "read_file", "arguments": {"path": "bug.py"}}}])
        if self.calls == 2:
            return self._chunks("", [{"function": {"name": "edit_file", "arguments": {"path": "bug.py", "old_text": "return 1", "new_text": "return 2"}}}])
        if self.calls == 3:
            return self._chunks("", [{"function": {"name": "run_tests", "arguments": {"command": "python -m pytest -q"}}}])
        return self._chunks("Fixed the bug and verified the tests.", [])

    async def _iter(self, content, calls):
        yield Chunk(content=content, tool_calls=calls, done=True)

    def _chunks(self, content, calls):
        return self._iter(content, calls)


def test_real_multi_step_coding_loop(tmp_path: Path):
    (tmp_path / "bug.py").write_text("def answer():\n    return 1\n")
    (tmp_path / "test_bug.py").write_text("from bug import answer\n\ndef test_answer():\n    assert answer() == 2\n")
    config = Config(verify_with_tests=False, allow_shell=True, auto_approve=["*"])
    events = Events()
    backend = FakeBackend()
    agent = Agent(
        backend, config, resolve("low"), tmp_path, events,
        registry=build_registry(tmp_path),
        boundary=Boundary(scope=Scope.FOLDER, root=tmp_path),
    )
    agent.backend = backend
    result = asyncio.run(agent.run("Find the bug, fix it, and run the tests."))
    assert (tmp_path / "bug.py").read_text().endswith("return 2\n")
    assert "Fixed the bug" in result.content
    assert [e[1] for e in events.events if e[0] == "start"] == ["read_file", "edit_file", "run_tests"]
