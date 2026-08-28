"""End-to-end agent tests against a scripted fake Ollama server.

These are the tests that matter: they exercise the real loop, real tools and
real files, with only the model replaced. If these pass, the agent works.
"""

import asyncio
import dataclasses
import json

import httpx
import pytest

from wynxo.agent import Agent, Callbacks
from wynxo.config import Config, Endpoint
from wynxo.effort import resolve
from wynxo.permissions import Decision
from wynxo.provider import OllamaClient
from wynxo.tools import build_registry


class FakeOllama:
    """Serves a scripted list of assistant turns over Ollama's wire format."""

    def __init__(self, turns, model_capabilities=("tools",), think_levels=True):
        self.turns = list(turns)
        self.requests = []
        self.capabilities = list(model_capabilities)
        self.think_levels = think_levels
        """False emulates an older Ollama that only understands think: bool."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.0-fake"})

        if path == "/api/show":
            return httpx.Response(200, json={
                "capabilities": self.capabilities,
                "details": {"parameter_size": "30B"},
                "model_info": {"qwen3.context_length": 40960},
            })

        if path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3-coder:30b", "size": 18_000_000_000,
                 "details": {"parameter_size": "30B", "quantization_level": "Q4_K_M"}},
            ]})

        if path == "/api/chat":
            body = json.loads(request.content)
            self.requests.append(body)
            if not self.think_levels and isinstance(body.get("think"), str):
                return httpx.Response(400, json={
                    "error": 'invalid think value: "medium" (must be "high", '
                             '"medium", "low", "max", true, or false)'})
            turn = self.turns.pop(0) if self.turns else {"content": "Done."}
            message = {"role": "assistant", "content": turn.get("content", "")}
            if turn.get("thinking"):
                message["thinking"] = turn["thinking"]
            if turn.get("tool_calls"):
                message["tool_calls"] = turn["tool_calls"]
            lines = [
                json.dumps({"message": message, "done": False}),
                json.dumps({
                    "message": {"role": "assistant", "content": ""},
                    "done": True, "prompt_eval_count": 100, "eval_count": 50,
                    "total_duration": 1_000_000_000,
                }),
            ]
            return httpx.Response(200, text="\n".join(lines))

        return httpx.Response(404, json={"error": f"no route {path}"})


class RecordingCallbacks(Callbacks):
    def __init__(self, permission=Decision.ALLOW):
        self.stages, self.tools, self.warnings = [], [], []
        self.content = []
        self.permission = permission
        self.permission_asks = []

    async def on_stage(self, name, detail=""):
        self.stages.append(name)

    async def on_tool_start(self, name, summary):
        self.tools.append((name, summary))

    async def on_content(self, text):
        self.content.append(text)

    async def on_warning(self, message):
        self.warnings.append(message)

    async def ask_permission(self, name, summary, preview):
        self.permission_asks.append((name, summary))
        return self.permission


def make_agent(tmp_path, turns, effort="low", capabilities=("tools",),
               callbacks=None, think_levels=True):
    fake = FakeOllama(turns, capabilities, think_levels)
    config = Config(
        endpoints=[Endpoint(name="t", url="http://fake:11434")],
        active_endpoint="t",
        model="qwen3-coder:30b",
        num_ctx=32768,
    )
    client = OllamaClient(config)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(fake.handler), base_url="http://fake:11434"
    )
    cb = callbacks or RecordingCallbacks()
    agent = Agent(
        client, config, resolve(effort), tmp_path, cb,
        registry=build_registry(tmp_path, allow_shell=True),
    )
    agent.permissions.yolo = callbacks is None
    return agent, fake, cb


class TestBasicLoop:
    async def test_plain_answer_no_tools(self, tmp_path):
        agent, _, cb = make_agent(tmp_path, [{"content": "It is 42."}])
        result = await agent.run("what is the answer")
        assert result.content == "It is 42."
        assert result.tool_calls == 0
        await agent.client.aclose()

    async def test_native_tool_call_then_answer(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('hi')\n")
        agent, _, cb = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "hello.py"}}}]},
            {"content": "It prints hi."},
        ])
        result = await agent.run("what does hello.py do")
        assert result.content == "It prints hi."
        assert result.tool_calls == 1
        assert cb.tools[0][0] == "read_file"
        await agent.client.aclose()

    async def test_hermes_text_tool_call(self, tmp_path):
        """A model whose template does not wire up native tool calls."""
        (tmp_path / "a.txt").write_text("content here\n")
        agent, _, cb = make_agent(tmp_path, [
            {"content": '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'},
            {"content": "It says content here."},
        ], capabilities=())
        await agent.detect_capabilities()
        assert agent.native_tools is False
        result = await agent.run("read a.txt")
        assert "content here" in result.content
        assert cb.tools[0][0] == "read_file"
        await agent.client.aclose()

    async def test_hermes_tool_call_markup_is_not_streamed_to_the_user(self, tmp_path):
        """The raw <tool_call>{...}</tool_call> text used to be streamed
        straight to the terminal before being parsed out, so every response
        from a non-tool-tuned model showed broken-looking protocol markup."""
        (tmp_path / "a.txt").write_text("content here\n")
        agent, _, cb = make_agent(tmp_path, [
            {"content": 'Let me check.\n<tool_call>{"name":"read_file",'
                        '"arguments":{"path":"a.txt"}}</tool_call>'},
            {"content": "It says content here."},
        ], capabilities=())
        await agent.detect_capabilities()
        await agent.run("read a.txt")
        streamed = "".join(cb.content)
        assert "<tool_call>" not in streamed
        assert "</tool_call>" not in streamed
        assert "Let me check." in streamed
        await agent.client.aclose()

    async def test_inline_think_tags_are_not_streamed_to_the_user(self, tmp_path):
        """A model with no native `thinking` field writes <think> straight
        into content instead; that must not leak into the live view either."""
        agent, _, cb = make_agent(tmp_path, [
            {"content": "<think>I should just answer directly.</think>The answer is 42."},
        ], capabilities=())
        await agent.detect_capabilities()
        await agent.run("what is the answer")
        streamed = "".join(cb.content)
        assert "<think>" not in streamed
        assert "The answer is 42." in streamed
        await agent.client.aclose()

    async def test_file_is_actually_written(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "new.py", "content": "x = 1\n"}}}]},
            {"content": "Created new.py."},
        ])
        await agent.run("create new.py")
        assert (tmp_path / "new.py").read_text() == "x = 1\n"
        await agent.client.aclose()

    async def test_edit_applies_and_reports_diff(self, tmp_path):
        target = tmp_path / "e.py"
        target.write_text("a = 1\nb = 2\n")
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "edit_file", "arguments": {
                "path": "e.py", "old_text": "b = 2", "new_text": "b = 3"}}}]},
            {"content": "Changed b."},
        ])
        await agent.run("set b to 3")
        assert target.read_text() == "a = 1\nb = 3\n"
        await agent.client.aclose()


class TestEffortShapesTheLoop:
    async def test_low_effort_does_not_plan_or_verify(self, tmp_path):
        agent, fake, cb = make_agent(tmp_path, [{"content": "Done."}], effort="low")
        result = await agent.run("do it")
        assert "planning" not in cb.stages
        assert result.verify_rounds == 0
        assert len(fake.requests) == 1
        await agent.client.aclose()

    async def test_high_effort_plans_then_verifies(self, tmp_path):
        target = tmp_path / "output.txt"
        target.write_text("old\n")
        agent, fake, cb = make_agent(tmp_path, [
            {"content": "Plan: change the thing."},   # planning pass
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": str(target), "content": "new\n"}}}]},  # execution
            {"content": "Changed it."},               # after tool
            {"content": "VERIFIED"},                  # verification
        ], effort="high")
        result = await agent.run("do it")
        assert "planning" in cb.stages
        assert "verifying" in cb.stages
        assert result.verify_rounds == 1
        # The VERIFIED marker must not become the user-facing answer.
        assert result.content == "Changed it."
        await agent.client.aclose()

    async def test_max_effort_critiques_its_own_plan(self, tmp_path):
        turns = [{"content": f"plan {i}"} for i in range(3)]      # parallel samples
        turns += [{"content": "merged plan"}, {"content": "critique"}]
        turns += [{"content": "Done."}, {"content": "VERIFIED"}]
        agent, _, cb = make_agent(tmp_path, turns, effort="max")
        await agent.run("do it")
        assert "planning" in cb.stages
        assert "reconciling" in cb.stages
        assert "critiquing plan" in cb.stages
        await agent.client.aclose()

    async def test_iteration_ceiling_is_enforced(self, tmp_path):
        """A model stuck in a tool loop must be stopped, not run forever."""
        turns = [{"tool_calls": [{"function": {"name": "list_dir",
                                              "arguments": {"path": "."}}}]}] * 30
        agent, _, cb = make_agent(tmp_path, turns, effort="low")
        result = await agent.run("loop forever")
        assert result.iterations == resolve("low").max_iterations
        assert any("ceiling" in w for w in cb.warnings)
        await agent.client.aclose()

    async def test_verify_round_can_fix_and_continue(self, tmp_path):
        (tmp_path / "f.py").write_text("old\n")
        agent, _, _ = make_agent(tmp_path, [
            {"content": "Plan."},
            # Execution writes a first version.
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "f.py", "content": "first\n"}}}]},
            {"content": "Wrote it."},
            # Verification notices a problem and fixes it with a tool.
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "f.py", "content": "fixed\n"}}}]},
            {"content": "Now correct."},
            {"content": "VERIFIED"},
        ], effort="high")
        await agent.run("do it")
        assert (tmp_path / "f.py").read_text() == "fixed\n"
        await agent.client.aclose()


class TestResilience:
    async def test_malformed_tool_call_is_repaired(self, tmp_path):
        (tmp_path / "r.txt").write_text("ok\n")
        agent, _, cb = make_agent(tmp_path, [
            {"content": "<tool_call>{utter nonsense}</tool_call>"},
            {"content": '<tool_call>{"name":"read_file","arguments":{"path":"r.txt"}}</tool_call>'},
            {"content": "Says ok."},
        ], effort="medium", capabilities=())
        result = await agent.run("read it")
        assert any("repairing" in s for s in cb.stages)
        assert cb.tools and cb.tools[0][0] == "read_file"
        assert result.content == "Says ok."
        await agent.client.aclose()

    async def test_unknown_tool_gets_a_suggestion(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_fil", "arguments": {"path": "x"}}}]},
            {"content": "Sorry."},
        ])
        await agent.run("go")
        tool_messages = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert "read_file" in tool_messages[0]["content"]
        await agent.client.aclose()

    async def test_tool_failure_is_reported_to_the_model(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "nope.py"}}}]},
            {"content": "That file does not exist."},
        ])
        result = await agent.run("read nope.py")
        tool_messages = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert "ERROR" in tool_messages[0]["content"]
        assert result.content == "That file does not exist."
        await agent.client.aclose()

    async def test_path_escape_is_refused(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "../../escaped.txt", "content": "nope"}}}]},
            {"content": "Cannot."},
        ])
        await agent.run("escape")
        assert not (tmp_path.parent.parent / "escaped.txt").exists()
        tool_messages = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert "outside the project directory" in tool_messages[0]["content"]
        await agent.client.aclose()

    async def test_denied_permission_stops_the_write(self, tmp_path):
        cb = RecordingCallbacks(permission=Decision.DENY)
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "denied.py", "content": "x"}}}]},
            {"content": "Understood."},
        ], callbacks=cb)
        await agent.run("write it")
        assert not (tmp_path / "denied.py").exists()
        assert cb.permission_asks
        tool_messages = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert "declined" in tool_messages[0]["content"]
        await agent.client.aclose()

    async def test_abort_ends_the_turn(self, tmp_path):
        cb = RecordingCallbacks(permission=Decision.ABORT)
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "a.py", "content": "x"}}}]},
        ], callbacks=cb)
        result = await agent.run("write it")
        assert result.interrupted
        assert not (tmp_path / "a.py").exists()
        await agent.client.aclose()

    async def test_reads_do_not_prompt_for_permission(self, tmp_path):
        (tmp_path / "x.txt").write_text("hi\n")
        cb = RecordingCallbacks(permission=Decision.DENY)
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "x.txt"}}}]},
            {"content": "It says hi."},
        ], callbacks=cb)
        await agent.run("read x")
        assert cb.permission_asks == []
        await agent.client.aclose()


class TestWireFormat:
    async def test_tools_are_sent_when_supported(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [{"content": "hi"}])
        await agent.run("hello")
        assert "tools" in fake.requests[0]
        assert any(t["function"]["name"] == "read_file" for t in fake.requests[0]["tools"])
        await agent.client.aclose()

    async def test_tools_are_not_sent_in_hermes_mode(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [{"content": "hi"}], capabilities=())
        await agent.detect_capabilities()
        await agent.run("hello")
        assert "tools" not in fake.requests[-1]
        assert "<tool_call>" in agent.session.system_prompt
        await agent.client.aclose()

    async def test_num_ctx_and_keep_alive_are_sent(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [{"content": "hi"}])
        await agent.run("hello")
        assert fake.requests[0]["options"]["num_ctx"] == 32768
        assert fake.requests[0]["keep_alive"] == "30m"
        await agent.client.aclose()

    async def test_thinking_is_omitted_at_low_effort(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [{"content": "hi"}], effort="low")
        await agent.run("x")
        assert "think" not in fake.requests[0]
        await agent.client.aclose()

    async def test_think_level_is_a_string_at_high_effort(self, tmp_path):
        """Ollama's `think` takes "low"|"medium"|"high"|"max" as well as a bool."""
        agent, fake, _ = make_agent(tmp_path, [
            {"content": "plan"}, {"content": "done"}, {"content": "VERIFIED"},
        ], effort="high")
        await agent.run("x")
        assert fake.requests[0]["think"] == "medium"
        await agent.client.aclose()

    async def test_max_effort_sends_the_top_think_level(self, tmp_path):
        turns = [{"content": f"p{i}"} for i in range(3)] + [
            {"content": "merged"}, {"content": "critique"},
            {"content": "done"}, {"content": "VERIFIED"}]
        agent, fake, _ = make_agent(tmp_path, turns, effort="max")
        await agent.run("x")
        assert fake.requests[0]["think"] == "max"
        await agent.client.aclose()

    async def test_tool_results_use_tool_name_not_name(self, tmp_path):
        """`name` is silently ignored by Ollama; the field is `tool_name`."""
        (tmp_path / "a.txt").write_text("hi\n")
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "a.txt"}}}]},
            {"content": "done"},
        ])
        await agent.run("read it")
        tool_messages = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert tool_messages[0]["tool_name"] == "read_file"
        assert "name" not in tool_messages[0]
        await agent.client.aclose()


class TestCapabilityStaysCurrent:
    """detect_capabilities() must never only ratchet one direction: a
    session that switches models has to pick up both a downgrade (a
    tool-tuned model -> a plain one) and an upgrade (the reverse) as the
    model actually in use changes."""

    async def test_thinking_downgrade_survives_an_effort_change(self, tmp_path):
        """Switching to a non-thinking model turns policy.thinking off; a
        later /effort change used to silently turn it back on, because
        set_effort() started from the freshly-resolved named policy instead
        of accounting for what the current model can do."""
        agent, fake, _ = make_agent(
            tmp_path, [{"content": "hi"}], effort="medium", capabilities=())
        await agent.detect_capabilities()
        assert agent.policy.thinking is False

        agent.set_effort(resolve("high"))
        assert agent.policy.thinking is False, (
            "an effort change on a non-thinking model must not re-enable "
            "the native `think` request")

        await agent.run("x")
        assert "think" not in fake.requests[-1]
        await agent.client.aclose()

    async def test_thinking_still_applies_normally_on_a_capable_model(self, tmp_path):
        """The downgrade guard must not suppress thinking for a model that
        genuinely supports it."""
        agent, fake, _ = make_agent(
            tmp_path, [{"content": "hi"}], effort="medium",
            capabilities=("tools", "thinking"))
        await agent.detect_capabilities()
        agent.set_effort(resolve("high"))
        assert agent.policy.thinking is True

        await agent.run("x")
        assert fake.requests[-1]["think"] == "medium"
        await agent.client.aclose()

    async def test_native_tools_recovers_after_switching_to_a_tool_model(self, tmp_path):
        """The reverse direction: a session that starts on a model with no
        tool support (Hermes fallback) and then switches to one that has
        native tools must stop using the prompted fallback. Without a reset,
        detect_capabilities() could only ever turn native_tools off."""
        agent, fake, _ = make_agent(tmp_path, [{"content": "hi"}], capabilities=())
        await agent.detect_capabilities()
        assert agent.native_tools is False

        # Same server, but the newly-selected model advertises native tools.
        fake.capabilities = ["tools"]
        await agent.detect_capabilities()
        assert agent.native_tools is True
        await agent.client.aclose()


class TestOlderServerCompatibility:
    async def test_string_think_level_downgrades_to_bool(self, tmp_path):
        """An older Ollama rejects think:"medium". Retry as think:true, once."""
        agent, fake, _ = make_agent(tmp_path, [
            {"content": "plan"}, {"content": "done"}, {"content": "VERIFIED"},
        ], effort="high", think_levels=False)
        result = await agent.run("x")
        assert result.content == "done"
        assert fake.requests[0]["think"] == "medium"   # first attempt
        assert fake.requests[1]["think"] is True       # retried
        assert agent.client.think_levels_supported is False
        await agent.client.aclose()

    async def test_downgrade_is_remembered_for_the_session(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [
            {"content": "plan"}, {"content": "done"}, {"content": "VERIFIED"},
        ], effort="high", think_levels=False)
        await agent.run("x")
        before = len(fake.requests)
        await agent.run("again")
        # Every later request goes straight out as a boolean: no repeat 400s.
        assert all(r["think"] is True for r in fake.requests[before:])
        await agent.client.aclose()


class TestModesAndScopeInTheLoop:
    """Scope and mode enforcement, exercised through the real agent loop
    against a model that genuinely tries to write."""

    def _agent(self, tmp_path, turns, mode=None, scope=None):
        from wynxo.memory import Memory
        from wynxo.scope import Mode, Scope, resolve as resolve_scope

        boundary = resolve_scope(tmp_path, scope or Scope.FOLDER)
        fake = FakeOllama(turns)
        config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")],
                        active_endpoint="t", model="qwen3-coder:30b", num_ctx=32768)
        client = OllamaClient(config)
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://fake:11434")
        cb = RecordingCallbacks()
        agent = Agent(client, config, resolve("low"), tmp_path, cb,
                      boundary=boundary, memory=Memory(tmp_path, tmp_path / "u"))
        agent.permissions.mode = mode or Mode.MANUAL
        return agent, cb

    async def test_plan_mode_refuses_a_real_write(self, tmp_path):
        from wynxo.scope import Mode

        target = tmp_path / "a.py"
        target.write_text("original\n")
        agent, _ = self._agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "a.py", "content": "rewritten\n"}}}]},
            {"content": "I cannot write in plan mode."},
        ], mode=Mode.PLAN)

        await agent.run("rewrite it")
        assert target.read_text() == "original\n", "plan mode let a write through"
        tool_messages = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert "plan mode" in tool_messages[0]["content"]
        await agent.client.aclose()

    async def test_plan_mode_still_allows_reads(self, tmp_path):
        from wynxo.scope import Mode

        (tmp_path / "a.py").write_text("contents here\n")
        agent, cb = self._agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "a.py"}}}]},
            {"content": "It contains that."},
        ], mode=Mode.PLAN)
        await agent.run("what is in a.py")
        assert cb.tools and cb.tools[0][0] == "read_file"
        await agent.client.aclose()

    async def test_auto_mode_writes_without_a_prompt(self, tmp_path):
        from wynxo.scope import Mode

        agent, cb = self._agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "new.py", "content": "x = 1\n"}}}]},
            {"content": "Written."},
        ], mode=Mode.AUTO)
        await agent.run("create new.py")
        assert (tmp_path / "new.py").read_text() == "x = 1\n"
        assert cb.permission_asks == [], "auto mode should not have asked"
        await agent.client.aclose()

    async def test_auto_mode_still_asks_before_running_a_command(self, tmp_path):
        from wynxo.scope import Mode

        agent, cb = self._agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "shell",
                                          "arguments": {"command": "make install"}}}]},
            {"content": "Done."},
        ], mode=Mode.AUTO)
        await agent.run("build it")
        assert cb.permission_asks, "auto mode must still gate shell commands"
        await agent.client.aclose()

    async def test_yolo_cannot_escape_the_scope(self, tmp_path):
        """The invariant: approving everything is not being allowed everywhere."""
        from wynxo.scope import Mode

        work = tmp_path / "work"
        work.mkdir()
        agent, _ = self._agent(work, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "../escaped.txt", "content": "nope"}}}]},
            {"content": "Blocked."},
        ], mode=Mode.YOLO)
        await agent.run("escape")
        assert not (tmp_path / "escaped.txt").exists()
        await agent.client.aclose()


class TestUndoInTheLoop:
    async def test_a_write_can_be_undone(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("before\n")
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "a.py", "content": "after\n"}}}]},
            {"content": "Changed."},
        ])
        await agent.run("change it")
        assert target.read_text() == "after\n"

        done, _ = agent.checkpoints.undo()
        assert done and target.read_text() == "before\n"
        await agent.client.aclose()

    async def test_a_created_file_is_removed_by_undo(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "write_file", "arguments": {
                "path": "fresh.py", "content": "x\n"}}}]},
            {"content": "Created."},
        ])
        await agent.run("create it")
        assert (tmp_path / "fresh.py").exists()
        agent.checkpoints.undo()
        assert not (tmp_path / "fresh.py").exists()
        await agent.client.aclose()

    async def test_reads_are_not_checkpointed(self, tmp_path):
        (tmp_path / "a.py").write_text("x\n")
        agent, _, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "a.py"}}}]},
            {"content": "Read."},
        ])
        await agent.run("read it")
        assert len(agent.checkpoints) == 0
        await agent.client.aclose()


class TestMemoryInTheLoop:
    async def test_the_agent_can_write_to_memory(self, tmp_path):
        from wynxo.memory import Memory

        memory = Memory(tmp_path, tmp_path / "u")
        fake = FakeOllama([
            {"tool_calls": [{"function": {"name": "remember", "arguments": {
                "note": "Tests run with pytest -q", "scope": "project"}}}]},
            {"content": "Noted."},
        ])
        config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")],
                        active_endpoint="t", model="qwen3-coder:30b", num_ctx=32768)
        client = OllamaClient(config)
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://fake:11434")
        agent = Agent(client, config, resolve("low"), tmp_path,
                      RecordingCallbacks(), memory=memory)
        agent.permissions.yolo = True

        await agent.run("remember how tests run")
        assert memory.counts()[0] == 1
        assert "pytest" in memory.project.body()
        await client.aclose()

    async def test_memory_reaches_the_system_prompt(self, tmp_path):
        from wynxo.memory import Memory

        memory = Memory(tmp_path, tmp_path / "u")
        memory.remember("Never edit files under generated/")
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.memory = memory
        agent.refresh_system_prompt()
        assert "generated/" in agent.session.system_prompt
        await agent.client.aclose()

    async def test_no_memory_costs_no_prompt_space(self, tmp_path):
        from wynxo.memory import Memory

        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.memory = Memory(tmp_path, tmp_path / "u")
        agent.refresh_system_prompt()
        assert "## Memory" not in agent.session.system_prompt
        await agent.client.aclose()


class TestSmallTalkIsNotWork:
    """At high effort a turn is plan -> execute -> verify, and the plan
    prompt asks for "a plan for this task". Handed "hello" a model does as
    it is told: it invents a task, and the next message tells it to carry
    the plan out. That is how saying hello created hello_world.py."""

    @pytest.mark.parametrize("text", [
        "hello", "hi", "hi!", "hey", "yo", "hello!!", "heya",
        "thanks", "thank you", "ty", "cheers",
        "ok", "okay", "cool", "nice", "lol", "haha",
        "bye", "gn", "good morning", "good night",
        "who are you", "what are you", "how are you", "what's up",
        "are you there", "hmm", "  hello  ", "hello~",
    ])
    def test_conversation_is_recognised(self, text):
        from wynxo.agent import is_small_talk

        assert is_small_talk(text) is True, text

    @pytest.mark.parametrize("text", [
        "add a retry to the upload path",
        "fix the parser",
        "read main.py",
        "what does src/auth.py do?",
        "hello, now fix the parser",     # a greeting stuck on a task
        "hi can you add tests",
        "make it faster",
        "write hello world in python",
        "explain this",
        "run the tests",
        "check /etc/hosts",
        "```python\nx=1\n```",
        "test",                          # 'test' is a verb here, not chatter
    ])
    def test_real_work_is_never_mistaken_for_chatter(self, text):
        """The dangerous direction. A task read as chat merely loses the
        planning scaffold; chat read as a task invents work."""
        from wynxo.agent import is_small_talk

        assert is_small_talk(text) is False, text

    def test_a_long_message_is_never_chatter(self):
        from wynxo.agent import is_small_talk

        assert is_small_talk("hello " * 40) is False

    def test_empty_input_is_not_chatter(self):
        from wynxo.agent import is_small_talk

        assert is_small_talk("") is False
        assert is_small_talk("   ") is False

    async def test_greeting_at_ultra_does_not_plan_or_use_tools(self, tmp_path):
        """The actual bug, end to end: one model call, no planning stage,
        no tools, and nothing written to disk."""
        agent, fake, cb = make_agent(
            tmp_path, [{"content": "Hey! How can I help?"}], effort="ultra")
        result = await agent.run("hello")

        assert result.content == "Hey! How can I help?"
        assert result.tool_calls == 0
        assert "planning" not in cb.stages
        assert len(fake.requests) == 1, "a greeting is one call, not a pipeline"
        assert list(tmp_path.iterdir()) == []
        await agent.client.aclose()

    async def test_a_real_task_at_ultra_still_plans(self, tmp_path):
        """The guard must not disarm the effort levels for actual work."""
        turns = [{"content": f"p{i}"} for i in range(3)] + [
            {"content": "merged plan"}, {"content": "critique"},
            {"content": "done"}, {"content": "VERIFIED"}]
        agent, fake, cb = make_agent(tmp_path, turns, effort="ultra")
        await agent.run("add a retry to the upload path")

        assert "planning" in cb.stages
        assert len(fake.requests) > 1
        await agent.client.aclose()

    async def test_a_plan_saying_no_plan_needed_is_not_executed(self, tmp_path):
        """Belt and braces for phrasings the heuristic misses: the plan
        prompt may answer NO PLAN NEEDED, and that must not become work."""
        agent, fake, cb = make_agent(tmp_path, [
            {"content": "NO PLAN NEEDED"},
            {"content": "Nice to meet you."},
        ], effort="high")
        result = await agent.run("tell me about yourself please")

        assert result.tool_calls == 0
        carried = [r for r in fake.requests
                   if any("carry out that plan" in str(m.get("content", ""))
                          for m in r.get("messages", []))]
        assert carried == [], "NO PLAN NEEDED must not be executed"
        await agent.client.aclose()


class TestAnswerNeverGoesMissing:
    """"it answers me in thinking mode, or doesn't answer at all" -- both
    symptoms of a chat template that pre-fills the opening <think>."""

    async def test_a_dangling_close_tag_splits_correctly(self, tmp_path):
        agent, _, cb = make_agent(tmp_path, [
            {"content": "Working it out. 2+2 is 4.</think>\n\nThe answer is 4."},
        ])
        result = await agent.run("what is 2+2")
        assert result.content == "The answer is 4."
        assert "</think>" not in "".join(cb.content)
        await agent.client.aclose()

    async def test_the_reasoning_does_not_become_the_answer(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [
            {"content": "Let me think about this.</think>\n\nParis."},
        ])
        result = await agent.run("capital of france")
        assert "Let me think" not in result.content
        await agent.client.aclose()

    async def test_a_turn_that_streamed_nothing_still_shows_its_answer(self, tmp_path):
        """The filter can be started inside a think block the raw text never
        mentions, so it and parse_turn can disagree. An answer nobody saw is
        the one outcome worth any amount of care."""
        agent, _, cb = make_agent(tmp_path, [
            {"content": "Working it out.</think>\n\nFirst answer."},
            {"content": "No tags here at all, just the answer."},
        ])
        await agent.run("one")
        assert agent._template_prefills_think is True

        await agent.run("two")
        streamed = "".join(cb.content)
        assert "No tags here at all" in streamed, "the second turn showed nothing"
        await agent.client.aclose()

    async def test_a_thought_only_response_never_leaks_as_an_answer(self, tmp_path):
        """Reasoning is not a user-visible fallback for a missing answer.
        The empty answer gets its one nudge, then warns as before."""
        agent, _, cb = make_agent(tmp_path, [
            {"content": "", "thinking": "The capital is Paris."},
            {"content": "", "thinking": "It is Paris."},
        ])
        result = await agent.run("capital of france")
        assert result.content == ""
        assert "Paris" not in "".join(cb.content)
        assert any("empty answer" in warning for warning in cb.warnings)
        await agent.client.aclose()

    async def test_a_thought_only_response_recovers_when_the_retry_answers(self, tmp_path):
        """A nudge after empty content gets a real answer, and the reasoning
        still never leaks into it."""
        agent, _, cb = make_agent(tmp_path, [
            {"content": "", "thinking": "The capital is Paris."},
            {"content": "The capital of France is Paris."},
        ])
        result = await agent.run("capital of france")
        assert result.content == "The capital of France is Paris."
        assert not any("empty answer" in warning for warning in cb.warnings)
        await agent.client.aclose()

    async def test_a_normal_answer_is_unaffected(self, tmp_path):
        agent, _, cb = make_agent(tmp_path, [{"content": "Plain answer."}])
        result = await agent.run("hi there friend")
        assert result.content == "Plain answer."
        assert "".join(cb.content) == "Plain answer."
        await agent.client.aclose()


class TestProviderErrorsNeverEscape:
    """A crash here ended the process and took the conversation with it.
    Every cause is something a local model does on a bad day."""

    def failing(self, tmp_path, fail_after: int, message="boom"):
        """A server that streams normally, then errors mid-stream."""
        import httpx

        from wynxo.agent import Agent
        from wynxo.config import Config, Endpoint
        from wynxo.effort import resolve
        from wynxo.provider import OllamaClient
        from wynxo.tools import build_registry

        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/version":
                return httpx.Response(200, json={"version": "0.5.0-fake"})
            if path == "/api/show":
                return httpx.Response(200, json={
                    "capabilities": ["tools"], "details": {},
                    "model_info": {"q.context_length": 40960}})
            if path != "/api/chat":
                return httpx.Response(404, json={"error": "no"})

            index = calls["n"]
            calls["n"] += 1
            if index == fail_after:
                return httpx.Response(200, text=json.dumps({"error": message}))
            return httpx.Response(200, text="\n".join([
                json.dumps({"message": {"role": "assistant",
                                        "content": "An answer."}, "done": False}),
                json.dumps({"message": {"role": "assistant", "content": ""},
                            "done": True, "prompt_eval_count": 10,
                            "eval_count": 5, "total_duration": 10 ** 9}),
            ]))

        config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")],
                        active_endpoint="t", model="m", num_ctx=32768)
        client = OllamaClient(config)
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://fake:11434")
        agent = Agent(client, config, resolve("high"), tmp_path,
                      registry=build_registry(tmp_path, allow_shell=True))
        agent.permissions.yolo = True
        return agent

    async def test_a_failure_during_verification_is_caught(self, tmp_path):
        """The reported crash: run() guarded _act() but not _verify(), so a
        provider error there escaped and killed the REPL."""
        # fail_after=2 means the 3rd call errors. The agent must write a
        # file in the execution phase so that verification is triggered.
        target = tmp_path / "out.py"
        agent = self.failing(tmp_path, fail_after=2)
        # Patch the mock to write a file on the 2nd call (index 1).
        original_handler = agent.client._client._transport.handler
        calls = {"n": 0}
        _fail_after = 2
        def handler_with_write(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/version":
                return httpx.Response(200, json={"version": "0.5.0-fake"})
            if path == "/api/show":
                return httpx.Response(200, json={
                    "capabilities": ["tools"], "details": {},
                    "model_info": {"q.context_length": 40960}})
            if path != "/api/chat":
                return httpx.Response(404, json={"error": "no"})
            index = calls["n"]
            calls["n"] += 1
            if index == _fail_after:
                return httpx.Response(200, text=json.dumps({"error": "boom"}))
            if index == 1:
                # Execution phase: write a file so verification runs.
                return httpx.Response(200, text="\n".join([
                    json.dumps({"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": "write_file",
                            "arguments": {"path": str(target), "content": "x\n"}}}]},
                        "done": False}),
                    json.dumps({"message": {"role": "assistant", "content": ""},
                        "done": True, "prompt_eval_count": 10,
                        "eval_count": 5, "total_duration": 10 ** 9}),
                ]))
            return httpx.Response(200, text="\n".join([
                json.dumps({"message": {"role": "assistant",
                                        "content": "An answer."}, "done": False}),
                json.dumps({"message": {"role": "assistant", "content": ""},
                            "done": True, "prompt_eval_count": 10,
                            "eval_count": 5, "total_duration": 10 ** 9}),
            ]))
        agent.client._client._transport.handler = handler_with_write
        result = await agent.run("check the retry path")
        assert result.errors, "the error was not reported"
        await agent.client.aclose()

    async def test_the_answer_survives_a_failed_verification(self, tmp_path):
        """The work was already done; losing it as well would be worse."""
        agent = self.failing(tmp_path, fail_after=2)
        result = await agent.run("check the retry path")
        assert "An answer." in result.content
        await agent.client.aclose()

    async def test_a_failure_during_planning_does_not_lose_the_turn(self, tmp_path):
        """Planning is a convenience. Failing it should carry on without one."""
        agent = self.failing(tmp_path, fail_after=0)
        result = await agent.run("do the thing")
        assert result.content or result.errors
        await agent.client.aclose()

    async def test_the_session_is_usable_afterwards(self, tmp_path):
        agent = self.failing(tmp_path, fail_after=2)
        await agent.run("first")
        second = await agent.run("second")
        assert second.errors == []
        assert "An answer." in second.content
        await agent.client.aclose()

    async def test_a_template_parse_error_is_explained(self, tmp_path):
        """"XML syntax error on line 6" tells you nothing about what to do,
        and reads like your machine is broken rather than the model."""
        agent = self.failing(
            tmp_path, fail_after=1,
            message="XML syntax error on line 6: element <function> "
                    "closed by </parameter>")
        result = await agent.run("check it")
        blob = " ".join(result.errors).lower()
        assert "malformed output" in blob
        assert "not a problem with your setup" in blob
        await agent.client.aclose()


class TestGreetingsAreNotTasks:
    """Alternation takes the first branch that matches, and h[ei]y? claimed
    the "he" of "hello" before hello+ was ever tried -- leaving "llo there",
    which is not small talk. So "hello there" was run as a task: at max
    effort that means a plan, a document hunt and a verify round, for a
    greeting.
    """

    @pytest.mark.parametrize("text", [
        "hello", "hi", "hey", "yo", "sup", "hiya", "howdy",
        "hello there", "hi there", "hey there", "hello again",
        "hello!", "hey whats up", "good morning", "thanks", "cheers",
    ])
    def test_these_are_chatter(self, text):
        from wynxo.agent import is_small_talk

        assert is_small_talk(text) is True, f"{text!r} would start work"

    @pytest.mark.parametrize("text", [
        "hello world program",
        "write hello world in python",
        "hey can you refactor upload.py",
        "hi, add a test for the parser",
        "fix the retry helper",
        "hello.py is broken",
    ])
    def test_these_are_work(self, text):
        from wynxo.agent import is_small_talk

        assert is_small_talk(text) is False, f"{text!r} would be brushed off"


class TestAnEmptyAnswerIsNotSilence:
    """A model that sends back nothing must not look like a model still working.

    Local models do this: a chat template that swallows the reply, a window
    the conversation just outgrew, a stop token emitted straight away.
    Nothing errors and the turn succeeds, so the screen showed the question
    and then nothing at all -- no answer, no warning, no way to tell whether
    wynxo was thinking or had quietly given up.
    """

    @pytest.mark.asyncio
    async def test_it_says_so_when_the_answer_is_empty(self, tmp_path):
        agent, _, cb = make_agent(
            tmp_path, [{"content": ""}, {"content": ""}])
        await agent.run("add retries to the fetch helper")
        assert any("empty answer" in w for w in cb.warnings), cb.warnings

    @pytest.mark.asyncio
    async def test_it_says_so_for_whitespace_too(self, tmp_path):
        agent, _, cb = make_agent(
            tmp_path, [{"content": "   \n\t  \n"}, {"content": "   \n\t  \n"}])
        await agent.run("add retries to the fetch helper")
        assert any("empty answer" in w for w in cb.warnings), cb.warnings

    @pytest.mark.asyncio
    async def test_small_talk_gets_the_same_warning(self, tmp_path):
        # Greetings never reach the tool loop, so they need their own guard.
        agent, _, cb = make_agent(tmp_path, [{"content": ""}, {"content": ""}])
        await agent.run("hello")
        assert any("empty answer" in w for w in cb.warnings), cb.warnings

    @pytest.mark.asyncio
    async def test_an_ordinary_answer_is_not_warned_about(self, tmp_path):
        agent, _, cb = make_agent(tmp_path, [{"content": "Added a retry loop."}])
        await agent.run("add retries to the fetch helper")
        assert not any("empty answer" in w for w in cb.warnings), cb.warnings

    @pytest.mark.asyncio
    async def test_a_thought_only_response_warns_about_the_missing_answer(self, tmp_path):
        agent, _, cb = make_agent(
            tmp_path, [{"content": "", "thinking": "The file already retries."},
                       {"content": "", "thinking": "It already retries."}])
        await agent.run("does the fetch helper retry?")
        assert any("empty answer" in w for w in cb.warnings), cb.warnings

    @pytest.mark.asyncio
    async def test_the_warning_says_what_to_do_about_it(self, tmp_path):
        agent, _, cb = make_agent(
            tmp_path, [{"content": ""}, {"content": ""}])
        await agent.run("add retries to the fetch helper")
        said = " ".join(cb.warnings)
        assert "/doctor" in said and "/compact" in said


class TestAnEmptyAnswerRecovers:
    """Before answering "nothing", the agent gives the model one explicit
    nudge. A single glitched reply should not kill the turn."""

    @pytest.mark.asyncio
    async def test_a_single_empty_answer_recovers(self, tmp_path):
        agent, fake, cb = make_agent(tmp_path, [{"content": ""}])
        result = await agent.run("add retries to the fetch helper")
        assert result.content == "Done."          # the retry answered
        assert not any("empty answer" in w for w in cb.warnings), cb.warnings
        assert len(fake.requests) == 2            # the nudge cost one retry
        await agent.client.aclose()

    @pytest.mark.asyncio
    async def test_the_nudge_is_visible_to_the_model(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [{"content": ""}])
        await agent.run("add retries to the fetch helper")
        second = fake.requests[1]["messages"]
        assert any("came back empty" in str(m.get("content", ""))
                   for m in second), second
        await agent.client.aclose()

    @pytest.mark.asyncio
    async def test_it_never_nudges_twice(self, tmp_path):
        agent, fake, cb = make_agent(
            tmp_path, [{"content": ""}, {"content": ""}, {"content": ""}])
        await agent.run("add retries to the fetch helper")
        assert any("empty answer" in w for w in cb.warnings), cb.warnings
        assert len(fake.requests) == 2            # one retry, then warn
        await agent.client.aclose()


class TestRepeatedActionsReachTheModel:
    """A repeat warning that only the UI sees cannot stop a loop, because
    the model is the one stuck in it. The repeat must be visible in the
    conversation the model reads next."""

    async def _run_two_reads(self, tmp_path, first, second):
        (tmp_path / "a.txt").write_text("one\n")
        agent, fake, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": first}}]},
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": second}}]},
            {"content": "Done."},
        ])
        await agent.run("read a.txt")
        return agent, fake

    @pytest.mark.asyncio
    async def test_an_identical_action_is_flagged_in_the_conversation(self, tmp_path):
        agent, fake = await self._run_two_reads(
            tmp_path, {"path": "a.txt"}, {"path": "a.txt"})
        tool_msgs = [m for m in fake.requests[-1]["messages"]
                     if m.get("role") == "tool"]
        assert any("already performed" in m.get("content", "")
                   for m in tool_msgs), tool_msgs
        await agent.client.aclose()

    @pytest.mark.asyncio
    async def test_formatting_noise_still_counts_as_a_repeat(self, tmp_path):
        # Same action, different key order and quoting: the raw string would
        # miss it, the canonical fingerprint must not.
        agent, fake = await self._run_two_reads(
            tmp_path, {"path": "a.txt", "offset": 1},
            {"offset": 1, "path": "a.txt"})
        tool_msgs = [m for m in fake.requests[-1]["messages"]
                     if m.get("role") == "tool"]
        assert any("already performed" in m.get("content", "")
                   for m in tool_msgs), tool_msgs
        await agent.client.aclose()

    @pytest.mark.asyncio
    async def test_a_different_action_is_not_flagged(self, tmp_path):
        agent, fake = await self._run_two_reads(
            tmp_path, {"path": "a.txt"}, {"path": "a.txt", "offset": 5})
        tool_msgs = [m for m in fake.requests[-1]["messages"]
                     if m.get("role") == "tool"]
        assert not any("already performed" in m.get("content", "")
                       for m in tool_msgs), tool_msgs
        await agent.client.aclose()


class TestNoProgressRecovery:
    """The hard repeat cap: the same action repeated with no progress event
    between repeats trips a structured recovery prompt once, then stops."""

    def _shell_call(self, command="echo hi"):
        return {"function": {"name": "shell",
                             "arguments": {"command": command}}}

    def test_recovery_prompt_then_strategy_change_completes(self, tmp_path):
        agent, fake, cb = make_agent(tmp_path, [
            {"tool_calls": [self._shell_call()]},     # fresh
            {"tool_calls": [self._shell_call()]},     # repeat 1
            {"tool_calls": [self._shell_call()]},     # repeat 2
            {"tool_calls": [self._shell_call()]},     # repeat 3 -> cap, recovery
            {"tool_calls": [self._shell_call("echo changed-strategy")]},
            {"content": "Done."},
        ])
        agent.policy = dataclasses.replace(agent.policy, max_iterations=20)
        result = asyncio.run(agent.run("Inspect the launcher"))
        assert result.content == "Done."
        assert result.recovered, "the cap must have tripped once"
        # The recovery block reached the model as a user message.
        recovery = next((m.get("content", "") for m in fake.requests[-1]["messages"]
                         if m.get("role") == "user" and "RECOVERY" in str(m.get("content", ""))), "")
        assert "RECOVERY" in recovery
        assert "Inspect the launcher" in recovery
        # The 4th identical call was never executed: only 3 ran.
        tool_msgs = [m for m in fake.requests[-1]["messages"]
                     if m.get("role") == "tool"]
        assert len(tool_msgs) == 4, tool_msgs   # 3 blocked-by-cap shell + strategy change
        assert any("stopping" in w or "Repeated" in w for w in cb.warnings)
        assert agent.task_state.state.name == "COMPLETED"
        asyncio.run(agent.client.aclose())

    def test_repeat_cap_stops_the_loop_cleanly(self, tmp_path):
        agent, fake, cb = make_agent(tmp_path, [
            {"tool_calls": [self._shell_call()]} for _ in range(8)
        ])
        agent.policy = dataclasses.replace(agent.policy, max_iterations=20)
        result = asyncio.run(agent.run("Inspect the launcher"))
        assert result.content.startswith("(stopped:")
        assert "repeated the same action" in result.content
        assert result.recovered, "the first cap trip inserted recovery"
        # 3 executed per batch; the 4th of each batch was blocked.
        tool_msgs = [m for m in fake.requests[-1]["messages"]
                     if m.get("role") == "tool"]
        assert len(tool_msgs) == 6, tool_msgs
        assert any("stopping the tool loop" in w for w in cb.warnings)
        asyncio.run(agent.client.aclose())

    def test_progress_between_repeats_never_trips_the_cap(self, tmp_path):
        """A re-read after an edit is verification, not a stuck loop."""
        (tmp_path / "a.txt").write_text("one\n")
        agent, fake, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "a.txt"}}}]},
            {"tool_calls": [{"function": {"name": "edit_file",
                                          "arguments": {"path": "a.txt",
                                                        "old_text": "one",
                                                        "new_text": "two"}}}]},
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "a.txt"}}}]},
            {"tool_calls": [{"function": {"name": "read_file",
                                          "arguments": {"path": "a.txt"}}}]},
            {"content": "Done."},
        ])
        result = asyncio.run(agent.run("Update a.txt"))
        assert result.content == "Done."
        assert not result.recovered, "progress must reset the repeat counts"
        asyncio.run(agent.client.aclose())


class TestCompletionReportFromState:
    """The final report is built from recorded task state, never model
    prose: changed files, checks that ran, failures that remain."""

    def test_report_reflects_an_actual_edit(self, tmp_path):
        (tmp_path / "calc.py").write_text(
            "def add(a, b):\n    return a - b\n", newline="\n")
        agent, fake, _ = make_agent(tmp_path, [
            {"tool_calls": [{"function": {"name": "edit_file",
                                          "arguments": {"path": "calc.py",
                                                        "old_text": "return a - b",
                                                        "new_text": "return a + b"}}}]},
            {"content": "Fixed."},
        ])
        agent.policy = dataclasses.replace(agent.policy, max_iterations=20)
        result = asyncio.run(agent.run("Fix the add function"))
        assert result.content == "Fixed."
        report = agent.task_state.completion_report()
        assert report is not None
        assert "calc.py" in report
        assert "✓ completed" in report
        # No verification ran (no test runner in the scratch project), so no
        # verification line -- and no invented claims.
        assert "verification" not in report
        asyncio.run(agent.client.aclose())

    def test_small_talk_gets_no_report(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [{"content": "hey!"}])
        result = asyncio.run(agent.run("hello"))
        assert result.content == "hey!"
        assert agent.task_state.completion_report() is None
        asyncio.run(agent.client.aclose())
