"""Agent-level behaviour of the Python hardening pass: focused-first test
selection, the compileall syntax gate, and turning a failing suite into a
classified, model-visible instruction."""

from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.agent import Agent, Callbacks
from wynxo.config import Config
from wynxo.effort import resolve
from wynxo.provider import Chunk
from wynxo.scope import Boundary, Scope
from wynxo.tools import build_registry


class Events(Callbacks):
    def __init__(self):
        self.tool_results = []

    async def on_tool_result(self, name, ok, display, output, event=None):
        self.tool_results.append((name, ok, display))


class RecordingBackend:
    """Scripted model that records every request's messages."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append(messages)
        if not self.turns:
            return self._chunks("Done.", [])
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            return self._chunks(turn, [])
        content, calls = turn
        return self._chunks(content, calls)

    async def _iter(self, content, calls):
        yield Chunk(content=content, tool_calls=calls, done=True)

    def _chunks(self, content, calls):
        return self._iter(content, calls)


def make_agent(tmp_path, turns):
    backend = RecordingBackend(turns)
    config = Config(
        verify_with_tests=True, allow_shell=True, auto_approve=["*"]
    )
    events = Events()
    agent = Agent(
        backend, config, resolve("low"), tmp_path, events,
        registry=build_registry(tmp_path),
        boundary=Boundary(scope=Scope.FOLDER, root=tmp_path),
    )
    agent.backend = backend
    return agent, backend, events


def test_focused_tests_run_first_then_the_full_suite(tmp_path: Path):
    """The verify pass must run the affected tests before the whole suite:
    a focused pass followed by a full-suite failure is the news the model
    needs, and it must arrive classified."""
    (tmp_path / "bug.py").write_text("def answer():\n    return 1\n")
    (tmp_path / "test_bug.py").write_text(
        "from bug import answer\n\ndef test_answer():\n    assert answer() == 2\n")
    (tmp_path / "test_unrelated.py").write_text(
        "def test_unrelated():\n    assert False\n")

    agent, backend, events = make_agent(tmp_path, [
        ("", [{"function": {"name": "read_file", "arguments": {"path": "bug.py"}}}]),
        ("", [{"function": {"name": "edit_file",
                            "arguments": {"path": "bug.py",
                                          "old_text": "return 1",
                                          "new_text": "return 2"}}}]),
        "Fixed it.",
    ])
    result = asyncio.run(agent.run("Fix the failing test and run the tests."))
    assert (tmp_path / "bug.py").read_text().endswith("return 2\n")
    # The focused run passed, then the full suite surfaced the unrelated
    # failure -- and the model was told about it in structured form.
    failure_prompt = next(
        (m.get("content", "") for m in backend.requests[-1]
         if m.get("role") == "user" and "failed" in str(m.get("content", ""))),
        "")
    assert "structured failure analysis" in failure_prompt, failure_prompt
    assert "test_unrelated" in failure_prompt, failure_prompt
    assert result.interrupted is False


def test_missing_dev_tool_is_reported_as_an_environment_problem(tmp_path: Path):
    """A test suite that cannot collect because a dev tool is missing must
    be classified as an environment problem -- never as a reason to edit
    application source to paper over it."""
    import importlib.util
    if importlib.util.find_spec("pytest_cov"):
        pytest.skip("pytest-cov installed; the environment problem is gone")
    (tmp_path / "app.py").write_text("def value():\n    return 1\n")
    (tmp_path / "test_cov_flow.py").write_text(
        "import pytest_cov\n\n"
        "def test_flow():\n    assert value() == 1\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    agent, backend, _ = make_agent(tmp_path, [
        ("", [{"function": {"name": "edit_file",
                            "arguments": {"path": "app.py",
                                          "old_text": "return 1",
                                          "new_text": "return 2"}}}]),
        "Done.",
    ])
    asyncio.run(agent.run("Make value return 2."))
    failure_prompt = next(
        (m.get("content", "") for m in backend.requests[-1]
         if m.get("role") == "user" and "failed" in str(m.get("content", ""))),
        "")
    assert "environment" in failure_prompt, failure_prompt
    assert "pytest-cov" in failure_prompt, failure_prompt


def test_compileall_gate_catches_a_broken_edit_without_a_runner(tmp_path: Path):
    """A project with no test runner still gets a syntax gate when a Python
    file changed -- a broken edit must not sail through verification."""
    (tmp_path / "app.py").write_text("def ok():\n    return 1\n")
    agent, backend, events = make_agent(tmp_path, [
        ("", [{"function": {"name": "edit_file",
                            "arguments": {"path": "app.py",
                                          "old_text": "def ok():",
                                          "new_text": "def broken(:"}}}]),
        "Done.",
    ])
    asyncio.run(agent.run("Refactor the function."))
    assert ("tests", False, "syntax check failed") in events.tool_results


def test_async_bug_fix_benchmark(tmp_path: Path):
    """Benchmark: an async bug. The agent must find it, fix it, and see the
    focused async test pass -- exercising async test execution end to end."""
    (tmp_path / "app.py").write_text(
        "async def double(x):\n    return x\n")
    (tmp_path / "test_async_bench.py").write_text(
        "import pytest\nfrom app import double\n\n"
        "@pytest.mark.asyncio\nasync def test_double():\n"
        "    assert await double(2) == 4\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    agent, _, events = make_agent(tmp_path, [
        ("", [{"function": {"name": "read_file", "arguments": {"path": "app.py"}}}]),
        ("", [{"function": {"name": "edit_file",
                            "arguments": {"path": "app.py",
                                          "old_text": "return x",
                                          "new_text": "return x * 2"}}}]),
        "Fixed the async bug.",
    ])
    result = asyncio.run(agent.run(
        "The async double function is wrong; find the bug and fix it."))
    assert "x * 2" in (tmp_path / "app.py").read_text()
    assert any(name == "tests" and ok for name, ok, _ in events.tool_results)
    assert result.interrupted is False


def test_windows_posix_assumption_benchmark(tmp_path: Path):
    """Benchmark: POSIX-only logic. os.uname() does not exist on Windows;
    the agent must replace it with the portable platform API and verify."""
    # newline="\n": text-mode writing on Windows would translate to CRLF and
    # break the LF old_text the scripted model uses.
    (tmp_path / "tool.py").write_text(
        "import os\n\ndef system_name():\n    return os.uname().sysname\n",
        newline="\n")
    (tmp_path / "test_windows_bench.py").write_text(
        "from tool import system_name\n\n"
        "def test_system_name():\n    assert isinstance(system_name(), str)\n",
        newline="\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    agent, _, events = make_agent(tmp_path, [
        ("", [{"function": {"name": "read_file", "arguments": {"path": "tool.py"}}}]),
        ("", [{"function": {"name": "edit_file",
                            "arguments": {"path": "tool.py",
                                          "old_text": "import os\n\ndef system_name():\n    return os.uname().sysname",
                                          "new_text": "import platform\n\ndef system_name():\n    return platform.uname().system"}}}]),
        "Replaced the POSIX-only call.",
    ])
    asyncio.run(agent.run(
        "system_name crashes on Windows; fix it so the test passes."))
    fixed = (tmp_path / "tool.py").read_text()
    assert "os.uname" not in fixed
    assert "platform.uname" in fixed
    assert any(name == "tests" and ok for name, ok, _ in events.tool_results)


def test_compileall_gate_passes_on_a_valid_edit(tmp_path: Path):
    """The syntax gate must approve a clean edit -- verification succeeding
    quietly is as important as failing loudly."""
    (tmp_path / "app.py").write_text("def ok():\n    return 1\n")
    agent, _, events = make_agent(tmp_path, [
        ("", [{"function": {"name": "edit_file",
                            "arguments": {"path": "app.py",
                                          "old_text": "def ok():\n    return 1",
                                          "new_text": "def ok():\n    return 2"}}}])
    ])
    asyncio.run(agent.run("Make ok return 2."))
    assert ("tests", True, "syntax check passed (compileall)") in events.tool_results


def test_repeated_identical_failure_is_flagged_as_no_progress(tmp_path: Path):
    """The same failure signature twice in a row must read as no progress,
    so the model changes strategy instead of re-attempting the same edit."""
    agent, backend, _ = make_agent(tmp_path, ["Done."])
    result = type("R", (), {"ok": False, "output": "E   AssertionError: x",
                            "metadata": {"exit_code": 1}})()
    asyncio.run(agent._report_test_failure("python -m pytest", result))
    asyncio.run(agent._report_test_failure("python -m pytest", result))
    all_text = " ".join(
        str(m.get("content", "")) for req in backend.requests for m in req)
    assert "No meaningful progress" in all_text
