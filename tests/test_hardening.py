"""Production hardening: bugs found by driving the running system hard.

Same discipline as tests/test_qa_regressions.py -- every test here failed
before the fix beside it, and each was reproduced against the real thing
(a real socket, the real asyncio types, the real layout) rather than a
mock built to agree with the code.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from wynxo.config import Config, Endpoint
from wynxo.provider import OllamaClient, OpenAIClient


def _stream(kind: str, text: str):
    """A client whose next stream is exactly ``text``."""
    url = "http://fake/v1" if kind == "openai" else "http://fake"
    config = Config(endpoints=[Endpoint(name="t", url=url, kind=kind)],
                    active_endpoint="t", model="m", num_ctx=8192)
    client = (OpenAIClient if kind == "openai" else OllamaClient)(config)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=text)),
        base_url=url)
    return client


def _truncated(kind: str, text: str) -> bool:
    async def go():
        client = _stream(kind, text)
        flags = [chunk.truncated async for chunk
                 in client.chat([{"role": "user", "content": "x"}], model="m")
                 if chunk.done]
        await client.aclose()
        return any(flags)

    return asyncio.run(go())


def _delta(content: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": content}}]})


class TestACutOffGenerationIsNotAFinishedOne:
    """A server that dies, is killed, or unloads a model mid-generation
    closes its connection cleanly. From the client's side that is
    indistinguishable from a well-formed stream except for the missing end
    marker -- and nothing looked for one, so half an answer was handed back
    as a whole one and the agent went on to act on it. With local models this
    is ordinary rather than exotic: an OOM during generation looks exactly
    like this.
    """

    def test_a_stream_that_just_stops_is_flagged(self):
        assert _truncated("openai", "data: %s\n" % _delta("half an answer"))

    def test_ollama_native_too(self):
        assert _truncated("ollama", json.dumps(
            {"message": {"content": "half"}, "done": False}))

    def test_done_terminates_a_stream(self):
        assert not _truncated(
            "openai", "data: %s\ndata: [DONE]" % _delta("all of it"))

    def test_a_finish_reason_also_terminates_it(self):
        """Some compat shims send one and not the other."""
        assert not _truncated("openai", "data: " + json.dumps({"choices": [
            {"delta": {"content": "all of it"}, "finish_reason": "stop"}]}))

    def test_ollamas_done_flag_terminates_it(self):
        assert not _truncated("ollama", json.dumps(
            {"message": {"content": "all of it"}, "done": True}))

    def test_a_stream_that_generated_nothing_is_not_truncated(self):
        """That is the empty-answer case, which has its own handling and its
        own message. Two warnings for one event is worse than one."""
        assert not _truncated("openai", "data: [DONE]")
        assert not _truncated("openai", "")
        assert not _truncated("ollama", "")

    def test_a_cut_off_tool_call_counts_as_generation(self):
        assert _truncated("openai", "data: " + json.dumps({"choices": [{"delta": {
            "tool_calls": [{"index": 0, "id": "c1", "function": {
                "name": "write_file", "arguments": '{"path": "a.p'}}]}}]}))


class TestTheUserIsToldTheAnswerIsCutOff:
    """Reported rather than raised: half an answer is still worth reading,
    and discarding it would lose the only evidence of what the model was
    doing. It just must not be mistaken for a whole one."""

    def _run(self, text, **kwargs):
        import pathlib
        import tempfile

        from wynxo.agent import Agent
        from wynxo.effort import resolve
        from wynxo.tools import build_registry

        warnings: list[str] = []
        shown: list[str] = []

        class Callbacks:
            def __getattr__(self, _name):
                async def anything(*a, **k):
                    return None
                return anything

            async def on_warning(self, message):
                warnings.append(message)

            async def on_content(self, message):
                shown.append(message)

        async def go():
            workspace = pathlib.Path(tempfile.mkdtemp())
            client = _stream("openai", text)
            agent = Agent(client, client.config, resolve("low"), workspace,
                          Callbacks(),
                          registry=build_registry(workspace, allow_shell=False))
            turn = await agent._call_model(**kwargs)
            await client.aclose()
            return turn

        return asyncio.run(go()), warnings, shown

    def test_a_cut_stream_warns(self):
        _turn, warnings, _shown = self._run("data: %s\n" % _delta("half"))
        assert warnings, "the answer was cut off and nobody said so"
        assert "cut off" in warnings[0]

    def test_what_did_arrive_is_kept(self):
        turn, _warnings, shown = self._run("data: %s\n" % _delta("half"))
        assert turn.content == "half"
        assert "".join(shown) == "half"

    def test_a_whole_answer_does_not_warn(self):
        _turn, warnings, _shown = self._run(
            "data: %s\ndata: [DONE]" % _delta("all of it"))
        assert warnings == []

    def test_an_infrastructure_call_stays_quiet(self):
        """The intent router and compaction run turns of their own. A warning
        from one of those is noise about work the user did not ask for."""
        _turn, warnings, _shown = self._run(
            "data: %s\n" % _delta("half"), silent=True)
        assert warnings == []


class TestAnEndpointThatIsNotTheApiDoesNotCrashStartup:
    """Every discovery call assumed a 200 meant JSON.

    A wrong port, a proxy's HTML error page, a captive-portal login
    redirect, or a server answering 200 with nothing at all produced a bare
    JSONDecodeError -- uncaught, on the start-up path, before the UI even
    exists. Pointing wynxo at the wrong address killed it with "Expecting
    value: line 1 column 1 (char 0)" and a crash file, rather than telling
    anybody what was wrong.
    """

    BODIES = {
        "an HTML error page": "<html><body>401 Unauthorized</body></html>",
        "an empty body": "",
        "plain text": "proxy error: upstream unavailable",
        "a JSON scalar": '"not an object"',
    }

    def _client(self, kind: str, body: str):
        url = "http://fake/v1" if kind == "openai" else "http://fake"
        config = Config(endpoints=[Endpoint(name="t", url=url, kind=kind)],
                        active_endpoint="t", model="m", num_ctx=8192)
        client = (OpenAIClient if kind == "openai" else OllamaClient)(config)
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, text=body)),
            base_url=url)
        return client

    def _expect_provider_error(self, kind, body, call_name):
        from wynxo.provider import ProviderError

        async def go():
            client = self._client(kind, body)
            try:
                await getattr(client, call_name)()
            finally:
                await client.aclose()

        try:
            asyncio.run(go())
        except ProviderError as exc:
            return str(exc)
        except Exception as exc:                       # noqa: BLE001
            raise AssertionError(
                f"{kind}.{call_name} on {body!r} raised {type(exc).__name__} "
                f"instead of a ProviderError: {exc}") from exc
        raise AssertionError(
            f"{kind}.{call_name} accepted {body!r} without complaint")

    def test_ping_explains_rather_than_crashing(self):
        for label, body in self.BODIES.items():
            message = self._expect_provider_error("ollama", body, "ping")
            assert "not the API wynxo expects" in message, label

    def test_listing_models_explains_rather_than_crashing(self):
        for kind in ("ollama", "openai"):
            for label, body in self.BODIES.items():
                message = self._expect_provider_error(kind, body, "list_models")
                assert "not the API wynxo expects" in message, f"{kind} {label}"

    def test_the_message_names_what_came_back(self):
        """A message that says only "invalid response" sends somebody to the
        wrong place. The shape of the body is the clue."""
        assert "an HTML page" in self._expect_provider_error(
            "ollama", "<html>nope</html>", "ping")
        assert "an empty body" in self._expect_provider_error(
            "ollama", "", "ping")

    def test_a_bare_array_of_models_is_still_understood(self):
        """Some compat shims answer /v1/models with the array itself rather
        than {"data": [...]}. Reporting "no models" for a server that has
        plenty would be a different bug."""
        async def go():
            client = self._client("openai", '[{"id": "llama3"}]')
            models = await client.list_models()
            await client.aclose()
            return models

        assert [m.name for m in asyncio.run(go())] == ["llama3"]

    def test_a_well_formed_answer_is_untouched(self):
        async def go():
            client = self._client("ollama", '{"version": "0.5.7"}')
            version = await client.ping()
            await client.aclose()
            return version

        assert asyncio.run(go()) == "0.5.7"
