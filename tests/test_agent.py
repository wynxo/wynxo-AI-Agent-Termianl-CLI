"""End-to-end agent tests against a scripted fake Ollama server.

These are the tests that matter: they exercise the real loop, real tools and
real files, with only the model replaced. If these pass, the agent works.
"""

import json

import httpx

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
        agent, fake, cb = make_agent(tmp_path, [
            {"content": "Plan: change the thing."},   # planning pass
            {"content": "Changed it."},               # execution
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
