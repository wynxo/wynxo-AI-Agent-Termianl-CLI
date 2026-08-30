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


class TestTheOverlayCannotContradictTheTranscript:
    """The companion typed underneath the word "Interrupted".

    The running tool and the live edit card are both forgotten by a turn's
    teardown -- but the teardown is not the first thing that happens. On
    Ctrl-C the REPL prints "Interrupted. The conversation is intact" *before*
    its finally block, and that print repaints the layout. So for that frame
    the overlay drew the cat coding and a card saying "streaming...", under
    a message saying the work had stopped.

    Fixed where the two facts are combined rather than by reordering the
    teardown: a state that says the task is over is a statement about the
    whole task, and outranks a tool that by then is not running. That makes
    the contradiction impossible regardless of what order anything else
    happens in.
    """

    def _repl(self):
        import pathlib
        import tempfile

        from wynxo.cli import Repl, TerminalCallbacks
        from wynxo.layout import Transcript
        from wynxo.pet import Pet
        from wynxo.task_state import TaskStateMachine
        from wynxo.ui import UI

        ui = UI()
        ui.attach(Transcript(80))
        repl = Repl.__new__(Repl)
        repl.ui = ui
        repl.callbacks = TerminalCallbacks(ui, prompt_session=None)
        repl.callbacks.workspace = pathlib.Path(tempfile.mkdtemp())
        repl.pet = Pet()
        repl._pet_frame = 0
        repl.config = type("Config", (), {"animations": True})()
        repl.agent = type("Agent", (), {"task_state": TaskStateMachine()})()
        return repl

    def _mid_edit(self):
        from wynxo.task_state import TaskState

        repl = self._repl()
        repl.agent.task_state.begin("fix the bug")
        repl.agent.task_state.transition(TaskState.EXECUTING)
        asyncio.run(repl.callbacks.on_tool_start("write_file", "a.py"))
        asyncio.run(repl.callbacks.on_code("half an edit\n"))
        return repl

    def test_the_card_is_shown_while_the_edit_is_running(self):
        """The fix must not cost the thing it is protecting."""
        assert "streaming" in "\n".join(self._mid_edit()._chat_overlay())

    def test_the_pet_codes_while_the_edit_is_running(self):
        from wynxo import motion

        assert motion.scene_for_state("executing", "write_file").name == "coding"

    def test_the_card_stops_the_moment_the_task_is_over(self):
        from wynxo.task_state import TaskState

        repl = self._mid_edit()
        repl.agent.task_state.transition(TaskState.CANCELLED)
        drawn = "\n".join(repl._chat_overlay())
        assert "streaming" not in drawn
        assert "write_file" not in drawn

    def test_a_leftover_tool_cannot_animate_a_finished_task(self):
        from wynxo import motion

        for state in ("cancelled", "failed", "completed", "idle"):
            for tool in ("write_file", "grep", "run_tests", "shell"):
                scene = motion.scene_for_state(state, tool)
                assert scene.name not in {"coding", "reading", "searching",
                                          "running", "testing", "working"}, \
                    f"{state} + {tool} showed {scene.name}"

    def test_a_running_task_still_follows_the_tool(self):
        from wynxo import motion

        for state in ("executing", "thinking", "planning", "recovering"):
            assert motion.scene_for_state(state, "edit_file").name == "coding"

    def test_task_is_over_agrees_with_the_state_machine(self):
        """The two must not drift apart: the overlay uses this to decide
        whether anything is in flight at all."""
        from wynxo import motion
        from wynxo.task_state import TaskState

        over = {TaskState.IDLE, TaskState.COMPLETED, TaskState.FAILED,
                TaskState.CANCELLED}
        for state in TaskState:
            assert motion.task_is_over(state.value) is (state in over), state


class TestAPermissionPromptCanBeAbandoned:
    """Ctrl-C at a blocking permission prompt did nothing at all.

    ``_ask`` has always had ``except (EOFError, KeyboardInterrupt): return
    Decision.ABORT`` written for exactly this. Under the layout that handler
    was unreachable, because ChatLayout.ask() had no way to end other than
    an answer -- so the agent sat waiting on a question with no way out
    except typing one of four letters, at the moment somebody most wants
    out.
    """

    def _layout(self):
        from wynxo.layout import ChatLayout

        return ChatLayout(width=80, height=24)

    def test_cancelling_raises_into_the_asker(self):
        async def go():
            layout = self._layout()
            asked = asyncio.ensure_future(layout.ask("[y] yes [n] no:"))
            await asyncio.sleep(0)
            layout.cancel_ask()
            return await asked

        try:
            asyncio.run(go())
        except EOFError:
            return
        raise AssertionError("the question could not be abandoned")

    def test_it_raises_something_asyncio_can_contain(self):
        """KeyboardInterrupt is a BaseException, and asyncio does not contain
        those the way it contains ordinary exceptions: set on a future it
        propagates out through the event loop itself and takes the whole
        application down rather than the question."""
        import inspect

        from wynxo.layout import ChatLayout

        source = inspect.getsource(ChatLayout.cancel_ask)
        assert "set_exception(EOFError" in source
        assert "set_exception(KeyboardInterrupt" not in source

    def test_the_question_is_closed_afterwards(self):
        async def go():
            layout = self._layout()
            asked = asyncio.ensure_future(layout.ask("[y] yes [n] no:"))
            await asyncio.sleep(0)
            layout.cancel_ask()
            try:
                await asked
            except EOFError:
                pass
            return layout.asking(), layout.buffer.text

        asking, text = asyncio.run(go())
        assert asking is False, "the composer is still borrowed"
        assert text == "", "the abandoned answer was left in the composer"

    def test_cancelling_when_nothing_is_asked_is_harmless(self):
        self._layout().cancel_ask()

    def test_an_answer_still_wins_over_a_later_cancel(self):
        async def go():
            layout = self._layout()
            asked = asyncio.ensure_future(layout.ask("[y] yes [n] no:"))
            await asyncio.sleep(0)
            layout.buffer.text = "n"
            layout._accept(layout.buffer)
            layout.cancel_ask()
            return await asked

        assert asyncio.run(go()) == "n"

    def test_the_permission_prompt_turns_it_into_an_abort(self):
        from wynxo.cli import TerminalCallbacks
        from wynxo.layout import Transcript
        from wynxo.permissions import Decision
        from wynxo.ui import UI

        ui = UI()
        ui.attach(Transcript(80))

        class Chat:
            async def ask(self, question, default=""):
                raise EOFError("abandoned")

            def invalidate(self):
                pass

        # A prompt_session must exist or ask_permission short-circuits to
        # ALLOW -- that is the non-interactive path, not this one.
        callbacks = TerminalCallbacks(ui, prompt_session=Chat())
        callbacks.chat = Chat()
        assert asyncio.run(
            callbacks.ask_permission("shell", "rm -rf build", "")) \
            is Decision.ABORT


class TestOnlyOneThingReadsTheKeyboard:
    """Every permission prompt left a second reader racing the composer.

    The KeyWatcher holds the terminal in cbreak mode and reads stdin
    directly. That is right for the scrolling prompt, where nothing else is
    reading during a turn, and catastrophic under the layout, where
    prompt_toolkit's application runs the whole time and owns the input --
    which is why the turn only ever starts it when there is no layout.
    ``_resume_live`` started it unconditionally, so answering (or
    abandoning) a permission prompt armed a rival reader for the rest of the
    turn, and it won often enough that the first character of the user's
    next message was simply gone.
    """

    def _callbacks(self, chat):
        from wynxo.cli import TerminalCallbacks
        from wynxo.layout import Transcript
        from wynxo.ui import UI

        ui = UI()
        ui.attach(Transcript(80))
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.chat = chat
        callbacks.watcher = self._Watcher()
        return callbacks

    class _Watcher:
        def __init__(self):
            self.running = False
            self.starts = 0

        def start(self):
            self.running = True
            self.starts += 1

        def stop(self):
            self.running = False

    def test_the_layout_keeps_the_keyboard_to_itself(self):
        callbacks = self._callbacks(chat=object())
        callbacks._resume_live()
        assert callbacks.watcher.starts == 0, (
            "a second stdin reader was started while the layout owned the "
            "input")

    def test_the_scrolling_prompt_still_gets_its_watcher_back(self):
        """The fix must not cost the fallback path, where the watcher *is*
        the reader and stopping it for good would kill the mid-turn keys."""
        callbacks = self._callbacks(chat=None)
        callbacks._suspend_live()
        callbacks._resume_live()
        assert callbacks.watcher.running

    def test_the_turn_and_the_prompt_agree_on_the_condition(self):
        """Two sites start the watcher. They disagreed once; a test is
        cheaper than finding out again."""
        import inspect

        from wynxo.cli import Repl, TerminalCallbacks

        resume = inspect.getsource(TerminalCallbacks._resume_live)
        turn = inspect.getsource(Repl._turn_locked)
        assert "self.chat is None" in resume
        import re

        assert re.search(r"if self\.chat is None:\s*\n\s*watcher\.start\(\)",
                         turn), "the turn no longer guards the watcher"
