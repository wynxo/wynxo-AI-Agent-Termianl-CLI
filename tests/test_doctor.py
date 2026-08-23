"""Doctor must be right about what is wrong -- a false 'all clear' is worse
than no check at all."""

import json

import httpx

from wynxo.config import Config, Endpoint
from wynxo.doctor import Doctor, Status
from wynxo.provider import OllamaClient
from wynxo.ui import UI


class Server:
    """A configurable fake, so each check can be driven to each verdict."""

    def __init__(self, *, version="0.12.0", models=("qwen3-coder:30b",),
                 capabilities=("completion", "tools", "thinking"),
                 context_length=262144, think_levels=True,
                 native_tool_call=True, hermes_tool_call=False,
                 malformed_tool_call=False, no_tool_call=False,
                 stream_chunks=3, reachable=True):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.reachable:
            raise httpx.ConnectError("refused", request=request)
        path = request.url.path

        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": n, "size": 18_000_000_000,
                 "details": {"parameter_size": "30B", "quantization_level": "Q4_K_M"}}
                for n in self.models]})
        if path == "/api/show":
            body = {"details": {"parameter_size": "30B"},
                    "model_info": {"qwen3.context_length": self.context_length}}
            if self.capabilities is not None:
                body["capabilities"] = list(self.capabilities)
            return httpx.Response(200, json=body)

        if path == "/api/chat":
            payload = json.loads(request.content)
            if isinstance(payload.get("think"), str) and not self.think_levels:
                return httpx.Response(400, json={"error": 'invalid think value: "medium"'})
            return self._chat(payload)

        return httpx.Response(404, json={"error": "no route"})

    def _chat(self, payload):
        asked_for_tool = bool(payload.get("tools"))
        thinking = payload.get("think") is not None

        messages: list[dict] = []
        content = "OK"
        if asked_for_tool or "README.md" in json.dumps(payload["messages"]):
            if self.native_tool_call and asked_for_tool:
                messages.append({"role": "assistant", "content": "",
                                 "tool_calls": [{"function": {
                                     "name": "get_file_size",
                                     "arguments": {"path": "README.md"}}}]})
                content = None
            elif self.hermes_tool_call:
                content = '<tool_call>{"name":"get_file_size","arguments":{"path":"README.md"}}</tool_call>'
            elif self.malformed_tool_call:
                content = "<tool_call>{name: get_file_size,,}</tool_call>"
            elif self.no_tool_call:
                content = "I cannot check file sizes."

        if content is not None:
            size = max(1, len(content) // self.stream_chunks)
            parts = [content[i:i + size] for i in range(0, len(content), size)]
            for part in parts:
                msg = {"role": "assistant", "content": part}
                if thinking:
                    msg["thinking"] = "reasoning "
                messages.append(msg)

        lines = [json.dumps({"message": m, "done": False}) for m in messages]
        lines.append(json.dumps({"message": {"role": "assistant", "content": ""},
                                 "done": True, "prompt_eval_count": 50,
                                 "eval_count": 20, "total_duration": 500_000_000}))
        return httpx.Response(200, text="\n".join(lines))


def make_doctor(server: Server, model="qwen3-coder:30b", num_ctx=32768):
    config = Config(endpoints=[Endpoint(name="t", url="http://fake:11434")],
                    active_endpoint="t", model=model, num_ctx=num_ctx)
    client = OllamaClient(config)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://fake:11434")
    return Doctor(client, config, UI()), client


def verdict(doctor, name):
    return next(c.status for c in doctor.checks if c.name == name)


class TestHealthyServer:
    async def test_everything_passes(self, tmp_path):
        doctor, client = make_doctor(Server())
        code = await doctor.run()
        assert code == 0
        assert all(c.status is Status.PASS for c in doctor.checks), \
            [(c.name, c.status, c.detail) for c in doctor.checks]
        await client.aclose()


class TestBrokenServer:
    async def test_unreachable_fails_fast(self):
        doctor, client = make_doctor(Server(reachable=False))
        code = await doctor.run()
        assert code == 1
        assert verdict(doctor, "server reachable") is Status.FAIL
        # It must not pretend to have checked anything further.
        assert len(doctor.checks) == 1
        await client.aclose()

    async def test_missing_model_fails_with_a_pull_command(self):
        doctor, client = make_doctor(Server(models=("llama3:8b",)))
        code = await doctor.run()
        assert code == 1
        check = next(c for c in doctor.checks if c.name == "model installed")
        assert check.status is Status.FAIL
        assert "ollama pull" in check.fix
        await client.aclose()

    async def test_wrong_tag_suggests_the_installed_one(self):
        doctor, client = make_doctor(Server(models=("qwen3-coder:7b",)))
        await doctor.run()
        check = next(c for c in doctor.checks if c.name == "model installed")
        assert "qwen3-coder:7b" in check.fix
        await client.aclose()


class TestContextWindow:
    async def test_tiny_context_is_a_failure(self):
        """The silent killer: Ollama's small default."""
        doctor, client = make_doctor(Server(), num_ctx=2048)
        await doctor.run()
        check = next(c for c in doctor.checks if c.name == "context window")
        assert check.status is Status.FAIL
        assert "32768" in check.fix
        await client.aclose()

    async def test_context_beyond_native_warns(self):
        doctor, client = make_doctor(Server(context_length=32768), num_ctx=131072)
        await doctor.run()
        assert verdict(doctor, "context window") is Status.WARN
        await client.aclose()


class TestToolCalling:
    async def test_native_tool_call_passes(self):
        doctor, client = make_doctor(Server(native_tool_call=True))
        await doctor.run()
        check = next(c for c in doctor.checks if c.name == "tool calling")
        assert check.status is Status.PASS and "native" in check.detail
        await client.aclose()

    async def test_hermes_text_call_passes(self):
        doctor, client = make_doctor(Server(
            capabilities=("completion",), native_tool_call=False, hermes_tool_call=True))
        await doctor.run()
        check = next(c for c in doctor.checks if c.name == "tool calling")
        assert check.status is Status.PASS and "Hermes" in check.detail
        await client.aclose()

    async def test_malformed_call_warns_rather_than_fails(self):
        # wynxo repairs these, so it is a caveat and not a blocker.
        doctor, client = make_doctor(Server(
            capabilities=("completion",), native_tool_call=False,
            malformed_tool_call=True))
        await doctor.run()
        assert verdict(doctor, "tool calling") is Status.WARN
        await client.aclose()

    async def test_a_model_that_cannot_call_tools_fails(self):
        doctor, client = make_doctor(Server(
            capabilities=("completion",), native_tool_call=False, no_tool_call=True))
        code = await doctor.run()
        check = next(c for c in doctor.checks if c.name == "tool calling")
        assert check.status is Status.FAIL
        assert "qwen3-coder" in check.fix
        assert code == 1
        await client.aclose()


class TestCapabilitiesAndThinking:
    async def test_no_native_tools_warns_about_hermes_mode(self):
        doctor, client = make_doctor(Server(
            capabilities=("completion",), native_tool_call=False, hermes_tool_call=True))
        await doctor.run()
        check = next(c for c in doctor.checks if c.name == "model capabilities")
        assert check.status is Status.WARN
        assert "Hermes" in check.fix
        await client.aclose()

    async def test_unknown_capabilities_warns_but_continues(self):
        doctor, client = make_doctor(Server(capabilities=None))
        await doctor.run()
        assert verdict(doctor, "model capabilities") is Status.WARN
        # Still reaches the checks that matter.
        assert any(c.name == "tool calling" for c in doctor.checks)
        await client.aclose()

    async def test_old_server_without_think_levels_warns(self):
        doctor, client = make_doctor(Server(think_levels=False))
        await doctor.run()
        check = next(c for c in doctor.checks if c.name == "thinking mode")
        assert check.status is Status.WARN
        assert "boolean" in check.detail
        await client.aclose()

    async def test_model_without_thinking_still_usable(self):
        doctor, client = make_doctor(Server(capabilities=("completion", "tools")))
        code = await doctor.run()
        assert verdict(doctor, "thinking mode") is Status.WARN
        assert code == 0   # a caveat, not a blocker
        await client.aclose()
