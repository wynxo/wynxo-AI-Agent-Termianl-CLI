"""Live coding: real incremental code, and a companion that follows it.

The two failure modes this file exists to prevent are both forms of the same
lie -- showing something that did not happen when it appeared to happen:

* taking a finished edit and revealing it slowly, which looks like streaming
  and is an animation;
* running a coding animation on a timer, which looks like the agent is
  writing and is a clip.

So the assertions here are about *provenance*, not appearance: the code
shown must be the fragments the provider actually sent, and the frame shown
must be the state the agent is actually in.
"""

from __future__ import annotations

import json

import httpx

from wynxo.config import Config, Endpoint
from wynxo.provider import Chunk, OpenAIClient


class TestTheCodeStreamIsReal:
    """Providers that stream tool-call arguments send them in fragments.
    Those fragments were accumulated and yielded only at stream end, so the
    real incremental data existed and was thrown away."""

    def _client(self, handler) -> OpenAIClient:
        config = Config(
            endpoints=[Endpoint(name="t", url="http://fake/v1", kind="openai")],
            active_endpoint="t", model="m", num_ctx=8192)
        client = OpenAIClient(config)
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://fake")
        return client

    def _streaming_edit(self, code: str, chunk_size: int = 24):
        arguments = json.dumps({"path": "a.py", "content": code})
        pieces = [arguments[i:i + chunk_size]
                  for i in range(0, len(arguments), chunk_size)]

        def handler(request):
            lines = ['data: ' + json.dumps({"choices": [{"delta": {
                "tool_calls": [{"index": 0, "id": "c1", "function": {
                    "name": "write_file", "arguments": ""}}]}}]})]
            for piece in pieces:
                lines.append('data: ' + json.dumps({"choices": [{"delta": {
                    "tool_calls": [{"index": 0, "function": {
                        "arguments": piece}}]}}]}))
            lines.append("data: [DONE]")
            return httpx.Response(200, text="\n".join(lines))

        return handler, len(pieces)

    async def test_argument_fragments_are_emitted_as_they_arrive(self):
        code = "def fixed(x):\n    return x + 1\n" * 6
        handler, pieces = self._streaming_edit(code)
        client = self._client(handler)
        deltas = [c.arguments_delta async for c in client.chat(
            [{"role": "user", "content": "go"}], model="m")
            if c.arguments_delta]
        await client.aclose()
        assert len(deltas) == pieces, "each fragment must be surfaced, not pooled"
        assert json.loads("".join(deltas))["content"] == code

    async def test_a_finished_call_still_arrives_intact(self):
        """Surfacing the fragments must not cost the assembled call."""
        code = "x = 1\n"
        handler, _ = self._streaming_edit(code)
        client = self._client(handler)
        calls = [c.tool_calls async for c in client.chat(
            [{"role": "user", "content": "go"}], model="m") if c.tool_calls]
        await client.aclose()
        assert calls, "the completed tool call must survive"
        arguments = json.loads(calls[-1][0]["function"]["arguments"])
        assert arguments["content"] == code

    def test_an_atomic_provider_emits_no_fragments(self):
        """Ollama's native tool_calls carry their arguments complete. There
        is no partial data, and manufacturing some by revealing a finished
        string slowly would be an animation pretending to be a stream."""
        assert Chunk(tool_calls=[{"function": {"name": "write_file"}}]) \
            .arguments_delta == ""

    async def test_the_agent_shows_the_code_as_it_is_generated(self, tmp_path):
        """End to end: the deltas the agent hands the UI reconstruct exactly
        the code the provider sent, and arrive in more than one piece."""
        from wynxo.agent import Agent
        from wynxo.effort import resolve
        from wynxo.tools import build_registry

        code = "def parse(text):\n    return text.strip()\n" * 5
        handler, _ = self._streaming_edit(code)
        client = self._client(handler)
        shown: list[str] = []

        class Callbacks:
            def __getattr__(self, _name):
                async def anything(*a, **k):
                    return None
                return anything

            async def on_code(self, text):
                shown.append(text)

        agent = Agent(client, client.config, resolve("low"), tmp_path,
                      Callbacks(),
                      registry=build_registry(tmp_path, allow_shell=False))
        await agent._call_model()
        await client.aclose()
        assert len(shown) > 1, "the edit must appear progressively"
        assert "".join(shown) == code, "what was shown must be what was sent"

    async def test_nothing_is_shown_when_the_ui_is_not_streaming(self, tmp_path):
        """An infrastructure call (the intent router, compaction) must not
        paint code into the conversation."""
        from wynxo.agent import Agent
        from wynxo.effort import resolve
        from wynxo.tools import build_registry

        handler, _ = self._streaming_edit("y = 2\n")
        client = self._client(handler)
        shown: list[str] = []

        class Callbacks:
            def __getattr__(self, _name):
                async def anything(*a, **k):
                    return None
                return anything

            async def on_code(self, text):
                shown.append(text)

        agent = Agent(client, client.config, resolve("low"), tmp_path,
                      Callbacks(),
                      registry=build_registry(tmp_path, allow_shell=False))
        await agent._call_model(stream_content=False)
        await client.aclose()
        assert shown == []


class TestTheLiveRegionStaysTransient:
    """Everything provisional -- the edit card, the plan, the half-written
    line, the strip itself -- is drawn by one transient ``Live``. Transient
    is what makes it a layer rather than a record: it erases its own render
    area on stop, so nothing it drew can end up in the scrollback."""

    def test_the_bar_owns_a_transient_live(self):
        import inspect

        from wynxo.ui import ActivityBar

        source = inspect.getsource(ActivityBar.start)
        assert "transient=True" in source, (
            "the status strip would be committed to the scrollback on every "
            "repaint")

    def test_it_is_silent_when_the_pet_is_off(self):
        import inspect

        from wynxo.ui import ActivityBar

        source = inspect.getsource(ActivityBar._render)
        assert "self.pet.enabled" in source

    def test_there_is_exactly_one_live_region(self):
        """Two rich Live displays on one console fight for the same rows."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli)
        assert "Live(" not in source, (
            "cli started a second live region; the bar owns the only one")
