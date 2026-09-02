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


def _captured_ui(width: int = 90):
    """A UI whose console writes into a buffer.

    There is one renderer, and it writes at the terminal -- so "what the
    user would see" is read back from the stream.
    """
    import io

    from wynxo.ui import SafeConsole, UI

    ui = UI()
    ui.live_ok = False
    ui.console = SafeConsole(file=io.StringIO(), force_terminal=True,
                             color_system="truecolor", highlight=False,
                             soft_wrap=False, width=width, height=10_000)
    ui.width = width
    return ui


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


class TestTheLiveRegionCannotContradictTheConversation:
    """"streaming..." was drawn underneath the word "Interrupted".

    The running tool and the live edit card are both forgotten by a turn's
    teardown -- but the teardown was not the first thing that happened. On
    Ctrl-C the REPL printed "Interrupted. The conversation is intact" from
    its ``except`` block, *before* the ``finally``, and that print repaints
    the live region. So for that frame the card above the strip said
    "streaming..." under a message saying the work had stopped.

    Fixed by ordering, in one place: stop showing the work, then say it
    stopped. The report moved out of the except block and below the finally.
    """

    def _bar(self):
        from wynxo.cli import TerminalCallbacks
        from wynxo.ui import ActivityBar, UI
        import pathlib
        import tempfile

        ui = UI()
        ui.live_ok = False
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.workspace = pathlib.Path(tempfile.mkdtemp())
        bar = ActivityBar(ui, "medium")
        callbacks.bar = bar
        return callbacks, bar

    def _mid_edit(self):
        callbacks, bar = self._bar()
        asyncio.run(callbacks.on_tool_start("write_file", "a.py"))
        asyncio.run(callbacks.on_code("half an edit\n"))
        return callbacks, bar

    def _drawn(self, bar) -> str:
        from rich.console import Console
        import io

        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        console.print(bar._renderable())
        return console.file.getvalue()

    def test_the_card_is_shown_while_the_edit_is_running(self):
        """The fix must not cost the thing it is protecting: an edit with
        no visible progress at all is what the card exists to prevent."""
        _, bar = self._mid_edit()
        assert bar.card is not None and bar.card.live
        assert "streaming" in self._drawn(bar)

    def test_committing_the_edit_takes_the_card_down(self):
        callbacks, bar = self._mid_edit()
        asyncio.run(callbacks.on_tool_result("write_file", True, "a.py", "ok"))
        assert bar.card is None, "the finished edit is still in the live region"
        assert "streaming" not in self._drawn(bar)

    def test_a_cancelled_edit_is_closed_rather_than_left_streaming(self):
        callbacks, bar = self._mid_edit()
        callbacks.close_card()
        assert bar.card is None
        assert "streaming" not in self._drawn(bar)

    def test_the_interrupt_is_reported_after_the_teardown(self):
        """The ordering *is* the fix, so it is what the test pins."""
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._turn_locked)
        except_block = source.split("except (asyncio.CancelledError")[1] \
            .split("finally:")[0]
        # Statements, not prose: a comment mentioning the message is not a
        # print of it.
        statements = [line.split("#")[0] for line in except_block.splitlines()]
        assert not any("self.ui." in line for line in statements), (
            "the interrupt is announced while the live region is still up")
        after = source.split("finally:", 1)[1]
        assert "if interrupted:" in after and "Interrupted." in after

    def test_the_companion_says_what_the_tool_is(self):
        """One mapping, not three. The tool name picks the state and the
        state picks the picture -- there is no second table turning the
        same facts into a second visual.

        There were three: the tool name became an activity word for the
        strip, a mood for the face in pet.py, and a state for a set of
        scenes nothing drew. Two of those are gone."""
        from wynxo.companion import State, state_for

        assert state_for("write_file", "executing") is State.CODING
        assert state_for("read_file", "executing") is State.READING
        assert state_for("write_file", "executing") is not State.IDLE

class TestAPermissionPromptCanBeAbandoned:
    """Ctrl-C at a blocking permission prompt has to get you out.

    ``_ask`` has always had ``except (EOFError, KeyboardInterrupt): return
    Decision.ABORT`` written for exactly this, and with one interactive
    surface it is reachable: prompt_toolkit raises KeyboardInterrupt out of
    ``prompt_async`` on Ctrl-C and EOFError on Ctrl-D.

    An abandoned prompt is an ABORT, never an ALLOW. Neither Ctrl-C nor a
    closed stdin is consent.
    """

    def _callbacks(self, exception):
        from wynxo.cli import TerminalCallbacks
        from wynxo.ui import UI

        class Session:
            async def prompt_async(self, message=None, **kwargs):
                raise exception

        ui = UI()
        ui.live_ok = False
        # A prompt_session must exist or ask_permission short-circuits to
        # ALLOW -- that is the non-interactive path, not this one.
        return TerminalCallbacks(ui, prompt_session=Session())

    def _decide(self, callbacks):
        return asyncio.run(
            callbacks.ask_permission("shell", "rm -rf build", ""))

    def test_ctrl_c_aborts(self):
        from wynxo.permissions import Decision

        assert self._decide(self._callbacks(KeyboardInterrupt())) \
            is Decision.ABORT

    def test_ctrl_d_aborts(self):
        from wynxo.permissions import Decision

        assert self._decide(self._callbacks(EOFError())) is Decision.ABORT

    def test_the_live_region_is_put_back_either_way(self):
        """Abandoning the question must not leave the turn without its
        status strip for the rest of its life."""
        callbacks = self._callbacks(KeyboardInterrupt())
        started = []

        class Bar:
            def start(self):
                started.append(1)

            def stop(self):
                pass

        callbacks.bar = Bar()
        self._decide(callbacks)
        assert started == [1]


class TestCtrlCSurvivesAPromptInsideATurn:
    """prompt_toolkit's Application installs its own SIGINT handler for the
    length of a read and calls ``loop.remove_signal_handler(SIGINT)`` in its
    finally. There is only one handler per signal, so that removes the
    session's rather than restoring it.

    The turn arms the handler once, at the start. A permission prompt in the
    middle of a turn therefore left the *rest* of that turn with no handler
    at all: Ctrl-C did nothing while the agent went on editing and running
    commands. The key watcher is no help -- cbreak leaves ISIG set, so the
    driver turns ^C into a signal and the byte never reaches a reader.
    """

    def test_a_prompt_really_does_remove_the_handler(self):
        """The premise, against the real prompt_toolkit rather than a
        belief about it."""
        import signal

        from prompt_toolkit import PromptSession
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        async def go():
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, lambda: None)
            with create_pipe_input() as pipe:
                pipe.send_text("hello\n")
                await PromptSession(input=pipe,
                                    output=DummyOutput()).prompt_async()
            return signal.SIGINT in getattr(loop, "_signal_handlers", {})

        assert asyncio.run(go()) is False

    def test_the_permission_prompt_puts_it_back(self):
        from wynxo.cli import TerminalCallbacks
        from wynxo.ui import UI

        class Session:
            async def prompt_async(self, message=None, **kwargs):
                return "y"

        ui = UI()
        ui.live_ok = False
        callbacks = TerminalCallbacks(ui, prompt_session=Session())
        rearmed = []
        callbacks.rearm_interrupt = lambda: rearmed.append(1)
        asyncio.run(callbacks.ask_permission("shell", "ls", ""))
        assert rearmed == [1], (
            "Ctrl-C is dead for the rest of the turn after a permission "
            "prompt")

    def test_the_repl_wires_the_hook_up(self):
        """The hook working is not the fix; being connected is."""
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl.__init__)
        assert "rearm_interrupt = self._arm_interrupt" in source

    def test_every_question_helper_re_arms(self):
        import inspect

        from wynxo.cli import Repl

        for method in (Repl._question, Repl._type_in, Repl._pick):
            source = inspect.getsource(method)
            assert "_arm_interrupt()" in source, method.__name__
            assert "finally:" in source, (
                f"{method.__name__} only re-arms on the happy path")


class TestOnlyOneThingReadsTheKeyboard:
    """Two readers on stdin is how keystrokes go missing.

    The KeyWatcher holds the terminal in cbreak mode and reads stdin
    directly for the length of a turn -- which is safe precisely because
    prompt_toolkit is *not* reading then. The moment something inside the
    turn asks a question, the watcher has to let go and take it back
    afterwards, in that order.
    """

    class _Watcher:
        def __init__(self):
            self.running = False
            self.starts = 0

        def start(self):
            self.running = True
            self.starts += 1

        def stop(self):
            self.running = False

    def _callbacks(self):
        from wynxo.cli import TerminalCallbacks
        from wynxo.ui import UI

        ui = UI()
        ui.live_ok = False
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.watcher = self._Watcher()
        return callbacks

    def test_the_watcher_lets_go_for_a_question(self):
        callbacks = self._callbacks()
        callbacks.watcher.start()
        callbacks._suspend_live()
        assert callbacks.watcher.running is False, (
            "a second stdin reader was left running while prompt_toolkit "
            "read a line")

    def test_and_gets_the_keyboard_back_afterwards(self):
        """Stopping it for good would kill the mid-turn keys and type-ahead
        for the rest of the turn."""
        callbacks = self._callbacks()
        callbacks._suspend_live()
        callbacks._resume_live()
        assert callbacks.watcher.running

    def test_the_turn_starts_exactly_one_watcher(self):
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._turn_locked)
        assert source.count("watcher.start()") == 1
        assert source.count("watcher.stop()") == 1

class TestALargeDiffDoesNotFreezeTheScreen:
    """A four-thousand-line edit cost 1.5 seconds *per frame*.

    The card is redrawn on every repaint, and each redraw ran difflib over
    the whole file twice -- once for the body, once for the counts -- from
    scratch, then measured the display width of every one of the resulting
    rows in order to show twelve of them. So the entire UI stopped for as
    long as a large edit sat on the screen.

    Asserted as work done rather than wall time: a timing test on a shared
    machine tells you about the machine.
    """

    def _card(self, lines=400):
        from wynxo.livediff import DiffCard

        before = "\n".join(f"line {i}" for i in range(lines))
        after = "\n".join(f"line {i}" + ("!" if i % 3 == 0 else "")
                          for i in range(lines))
        card = DiffCard(tool="edit_file", path="big.py", before=before)
        card.feed(after)
        card.finish()
        return card

    def _counting(self, module, name):
        """Wrap ``module.name``, returning (restore, calls)."""
        original = getattr(module, name)
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        setattr(module, name, counted)
        return (lambda: setattr(module, name, original)), calls

    def test_repainting_does_not_re_diff(self):
        import difflib

        from wynxo import livediff
        from wynxo.ui import Glyphs

        card = self._card()
        card.render(Glyphs(True), 100)          # prime it
        restore, calls = self._counting(difflib, "unified_diff")
        try:
            for _ in range(20):
                card.render(Glyphs(True), 100)
        finally:
            restore()
        assert calls == [], (
            f"{len(calls)} diffs for 20 repaints of content that did not "
            "change")
        assert livediff is not None

    def test_the_body_and_the_counts_share_one_diff(self):
        import difflib

        card = self._card()
        restore, calls = self._counting(difflib, "unified_diff")
        try:
            card.counts()
            card.diff_lines()
            card.body(100)
        finally:
            restore()
        assert len(calls) == 1, f"{len(calls)} diffs where one would do"

    def test_new_content_is_diffed_again(self):
        """A cache that does not notice new content is a wrong answer, which
        is worse than a slow one."""
        card = self._card()
        first = card.counts()
        card.state = "live"
        card.feed("\nan extra line")
        card.finish()
        assert card.counts() != first
        assert "an extra line" in "\n".join(card.diff_lines())

    def test_only_the_rows_shown_are_measured(self):
        from wynxo import livediff
        from wynxo.livediff import MAX_LIVE_ROWS

        card = self._card()
        card.diff_lines()                        # prime the diff cache
        restore, calls = self._counting(livediff, "fit")
        try:
            rows = card.body(100, MAX_LIVE_ROWS)
        finally:
            restore()
        assert len(rows) == MAX_LIVE_ROWS
        assert len(calls) <= MAX_LIVE_ROWS, (
            f"measured {len(calls)} rows to show {len(rows)}")

    def test_the_rows_shown_are_the_same_rows(self):
        """Slicing before trimming must not change which rows appear."""
        from wynxo.livediff import MAX_LIVE_ROWS, fit

        card = self._card()
        lines = card.diff_lines()
        room = max(8, 100 - 4)
        expected = [fit(line, room) for line in lines[:MAX_LIVE_ROWS - 1]]
        expected.append(f"... {len(lines) - MAX_LIVE_ROWS + 1} more lines")
        assert card.body(100, MAX_LIVE_ROWS) == expected

    def test_a_live_card_still_shows_the_tail(self):
        from wynxo.livediff import DiffCard, MAX_LIVE_ROWS, fit

        card = DiffCard(tool="write_file", path="x.py", before="")
        card.feed("\n".join(f"line {i}" for i in range(100)))
        rows = card.body(100, MAX_LIVE_ROWS)
        lines = card.diff_lines()
        room = max(8, 100 - 4)
        assert rows == [fit(line, room) for line in lines[-MAX_LIVE_ROWS:]]

    def test_a_short_diff_is_shown_whole(self):
        from wynxo.livediff import DiffCard

        card = DiffCard(tool="write_file", path="x.py", before="")
        card.feed("one\ntwo\n")
        card.finish()
        assert card.body(100) == card.diff_lines()


class TestLeavingAConversationLeavesItBehind:
    """Work from an abandoned chat followed you into the next one.

    Three commands replace the conversation -- /clear, /new and /resume --
    and each kept its own idea of what that meant. Only /clear reset the
    task state, and none of them cleared the checklist. So after switching
    conversations the agent still carried the previous objective, its
    failures, its root cause, its changed files and its action
    fingerprints: a recovery block would hand the model failures from a
    task nobody was working on, a completion report would claim files
    changed in a different chat, the repeat detector would flag a first
    action as a repetition, a compaction summary would be told someone
    else's steps were "still outstanding", and the plan panel sat in the
    corner insisting on a checklist that had been abandoned.
    """

    def _repl(self, workspace):
        from wynxo.cli import Repl, TerminalCallbacks
        from wynxo.session import Session
        from wynxo.task_state import TaskState, TaskStateMachine
        from wynxo.tools import build_registry

        ui = _captured_ui()
        repl = Repl.__new__(Repl)
        repl.workspace = workspace
        repl.ui = ui
        repl._last_elapsed = 0.0
        repl.callbacks = TerminalCallbacks(ui, prompt_session=None)
        repl.callbacks.workspace = workspace
        registry = build_registry(workspace, allow_shell=False)

        class Agent:
            session = Session(workspace=workspace)
            task_state = TaskStateMachine()
            tools = registry

            class checkpoints:
                @staticmethod
                def clear():
                    pass

            @staticmethod
            def refresh_system_prompt():
                pass

        repl.agent = Agent()

        machine = repl.agent.task_state
        machine.begin("delete the auth system")
        machine.transition(TaskState.EXECUTING)
        machine.add_file("auth.py", changed=True)
        machine.record_failure("test_login failed")
        machine.record_action("write_file:auth.py")
        machine.set_root_cause("the token was never refreshed")

        todo = registry.get("todo_write")
        asyncio.run(todo.run(todo.Input(items=[
            {"task": "delete the auth system", "status": "in_progress"},
            {"task": "update the tests", "status": "pending"}])))
        return repl, todo

    def _workspace(self):
        import pathlib
        import tempfile

        return pathlib.Path(tempfile.mkdtemp())

    def _assert_left_behind(self, repl, todo):
        machine = repl.agent.task_state
        assert machine.objective == "", machine.objective
        assert machine.changed_files == [], machine.changed_files
        assert machine.failures == [], machine.failures
        assert machine.root_cause == "", machine.root_cause
        assert machine.action_fingerprints == []
        assert machine.state.value == "idle", machine.state
        assert todo.items == [], "the abandoned checklist came along"

    def test_it_is_all_dropped(self):
        repl, todo = self._repl(self._workspace())
        repl._leave_conversation()
        self._assert_left_behind(repl, todo)

    def test_resume_drops_it_and_restores_the_conversation(self):
        from wynxo.session import Session

        workspace = self._workspace()
        saved = Session(workspace=workspace)
        saved.add_user("rename the widget module")
        saved.add_assistant("ok")
        saved.save()

        repl, todo = self._repl(workspace)
        repl._load_session(saved.session_id)
        assert len(repl.agent.session.messages) == 2, "the point of resuming"
        self._assert_left_behind(repl, todo)

    def test_every_command_that_swaps_the_conversation_calls_it(self):
        """The bug was that they disagreed. A test is cheaper than
        rediscovering which one was forgotten."""
        import inspect

        from wynxo.cli import Repl

        for name, source in [
            ("/new", inspect.getsource(Repl.cmd_new)),
            ("/resume", inspect.getsource(Repl._load_session)),
            ("/clear", inspect.getsource(Repl.command)),
        ]:
            assert "_leave_conversation()" in source, name

    def test_a_compaction_is_not_told_about_the_old_checklist(self):
        """The one place outstanding() is read. Left standing, the summary
        of the new conversation was told to finish the old one's steps."""
        repl, todo = self._repl(self._workspace())
        assert todo.outstanding(), "the fixture must actually have items"
        repl._leave_conversation()
        assert todo.outstanding() == []


class TestAnInterpreterThatDoesNotRunHasNoAnswer:
    """A half-built .venv reported its error message as the Python version.

    ``_run_interpreter`` promised "empty on any failure" and returned
    ``stdout or stderr`` without ever looking at the exit code -- so an
    interpreter that did not run had its complaint taken for an answer. An
    interrupted `python -m venv`, or a venv copied between machines, gave
    /doctor a "python version" of whatever the shell script inside it
    happened to print, and turned "is pytest installed?" into a guess based
    on an error message.
    """

    def _workspace(self, script: str):
        import os
        import pathlib
        import tempfile

        workspace = pathlib.Path(tempfile.mkdtemp())
        (workspace / ".venv" / "bin").mkdir(parents=True)
        interpreter = workspace / ".venv" / "bin" / "python"
        interpreter.write_text(script)
        os.chmod(interpreter, 0o755)
        (workspace / "tests").mkdir()
        (workspace / "tests" / "test_a.py").write_text("def test_a(): pass\n")
        return workspace

    def test_a_failing_interpreter_reports_no_version(self):
        import sys

        if sys.platform == "win32":
            return                          # no shebang scripts to stand in
        from wynxo import testing

        workspace = self._workspace(
            "#!/bin/sh\necho 'venv is half built' >&2\nexit 1\n")
        info = testing.environment_info(workspace)
        assert info.version == "", f"reported {info.version!r} as a version"

    def test_it_does_not_guess_at_installed_packages_either(self):
        import sys

        if sys.platform == "win32":
            return
        from wynxo import testing

        workspace = self._workspace(
            "#!/bin/sh\necho 'ImportError: broken' >&2\nexit 1\n")
        assert testing.pytest_installed(workspace) is None, (
            "unknown is not the same as no, and neither is an error message")

    def test_a_working_interpreter_still_answers(self):
        """The fix must not cost the ordinary case."""
        import pathlib
        import re
        import tempfile

        from wynxo import testing

        info = testing.environment_info(pathlib.Path(tempfile.mkdtemp()))
        assert re.match(r"^\d+\.\d+", info.version), info.version
        assert info.pytest_installed is True

    def test_stderr_is_still_read_when_the_run_succeeds(self):
        """A working interpreter may warn on stderr while answering."""
        import sys

        if sys.platform == "win32":
            return
        from wynxo import testing

        workspace = self._workspace(
            "#!/bin/sh\necho 'the answer' >&2\nexit 0\n")
        assert testing._run_interpreter(
            str(workspace / ".venv" / "bin" / "python"), "pass") == "the answer"

    def test_doctor_calls_a_missing_version_unknown(self):
        """It already did the right thing with an empty version; it was
        never given one."""
        import inspect

        from wynxo import doctor

        source = inspect.getsource(doctor)
        assert 'env.version or "unknown"' in source


class TestWynxoKeepsItsOwnNotesOutOfTheRepository:
    """Every project wynxo touched grew a permanent untracked entry.

    It writes its project notes and its map into ``.wynxo/`` inside the
    repository it is working in, and nothing ignored them -- so `git status`
    carried an untracked directory forever afterwards, and one careless
    `git add -A` committed wynxo's notes into somebody's history. A
    directory that ignores itself needs no cooperation from the user's own
    .gitignore and cannot conflict with it.
    """

    def _workspace(self):
        import pathlib
        import tempfile

        return pathlib.Path(tempfile.mkdtemp())

    def test_remembering_something_leaves_the_directory_ignored(self):
        from wynxo.memory import PROJECT_DIR, Memory

        workspace = self._workspace()
        added, _ = Memory(workspace).remember("this project uses uv")
        assert added
        marker = workspace / PROJECT_DIR / ".gitignore"
        assert marker.is_file(), "the directory does not ignore itself"
        assert marker.read_text(encoding="utf-8").strip().endswith("*")

    def test_writing_the_map_does_too(self):
        from wynxo import projectmap
        from wynxo.memory import PROJECT_DIR

        workspace = self._workspace()
        (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
        projectmap.load(workspace)
        marker = workspace / PROJECT_DIR / ".gitignore"
        assert marker.is_file()

    def test_git_sees_nothing(self):
        """The point, checked with git rather than by reading the file."""
        import shutil
        import subprocess

        if shutil.which("git") is None:
            return
        from wynxo.memory import Memory

        workspace = self._workspace()
        for argv in (["init", "-q"], ["config", "user.email", "t@t"],
                     ["config", "user.name", "t"]):
            subprocess.run(["git", *argv], cwd=workspace, capture_output=True)
        Memory(workspace).remember("this project uses uv")
        out = subprocess.run(["git", "status", "--porcelain"], cwd=workspace,
                             capture_output=True, text=True).stdout
        assert ".wynxo" not in out, out

    def test_an_existing_marker_is_left_alone(self):
        from wynxo.memory import PROJECT_DIR, claim_directory

        workspace = self._workspace()
        directory = workspace / PROJECT_DIR
        directory.mkdir(parents=True)
        (directory / ".gitignore").write_text("mine\n", encoding="utf-8")
        claim_directory(directory)
        assert (directory / ".gitignore").read_text(encoding="utf-8") == "mine\n"

    def test_an_unwritable_project_is_not_a_failure(self):
        """A read-only checkout is a reason to carry on without the marker,
        not to fail whatever was being written."""
        from wynxo.memory import claim_directory

        claim_directory("/proc/nonexistent/wynxo")      # must not raise


class TestTheShorterRoadToTheTerminalIsClosedToo:
    """ESC was stripped; its one-character twin was not.

    The C1 range (U+0080-U+009F) is the single-character form of the
    sequences ESC introduces: U+009B is CSI, U+009D is OSC. A file holding
    the bytes ``C2 9B`` decodes to U+009B, and terminals that recognise C1
    in UTF-8 -- xterm among them -- read ``U+009B 2 J`` as "erase the
    display". So the same attack that the ESC scrub blocked went through
    one character shorter.
    """

    C1 = {"single-byte CSI erase": "\x9b2J",
          "single-byte OSC title": "\x9d0;pwned\x07",
          "single-byte DCS": "\x90q\x9c",
          "single-byte APC": "\x9fGf=1\x9c",
          "string terminator": "\x9c"}

    def _drawn(self, payload):
        import io

        from wynxo.ui import UI

        ui = UI()
        sink = io.StringIO()
        ui.console.file = sink
        ui.tool_result("read_file", True, payload, payload)
        ui.tool_output(payload)
        ui.error(payload)
        ui.code(payload, "text")
        return sink.getvalue()

    def test_no_c1_control_reaches_the_terminal(self):
        for label, payload in self.C1.items():
            drawn = self._drawn(payload)
            leaked = [ch for ch in drawn if 0x80 <= ord(ch) <= 0x9F]
            assert not leaked, f"{label} let {[hex(ord(c)) for c in leaked]} out"

    def test_the_escape_form_is_still_blocked(self):
        assert "\x1b[" not in self._drawn("\x1b[2J")
        assert "\x1b]" not in self._drawn("\x1b]0;t\x07")

    def test_a_non_breaking_space_is_just_past_the_end(self):
        """U+00A0 is text. The range must stop at U+009F."""
        from wynxo.ui import sanitise

        assert sanitise("a\xa0b") == "a\xa0b"

    def test_ordinary_text_of_every_script_survives(self):
        from wynxo.ui import sanitise

        for text in ("café", "naïve", "日本語", "한국어", "العربية", "עברית",
                     "🎉👩‍💻🇯🇵", "Ｆｕｌｌｗｉｄｔｈ", "a\tb", "line\nline"):
            assert sanitise(text) == text, text


class TestWidthIsMeasuredInColumnsEverywhere:
    """A CJK character occupies two columns, a combining mark none, and an
    emoji two. Anywhere a length stands in for a width, the box breaks."""

    SCRIPTS = {
        "cjk": "日本語のテストファイル",
        "korean": "한국어테스트파일이름",
        "arabic": "اسم-الملف-العربي-الطويل",
        "hebrew": "שם-קובץ-בעברית",
        "emoji": "🎉🎊🥳🎈🎁🍰🎂🧁",
        "zwj": "👨‍👩‍👧‍👦👩‍💻🏳️‍🌈",
        "combining": "éàîõü" * 4,
        "fullwidth": "ＦＵＬＬＷＩＤＴＨ",
        "mixed": "src/日本/naïve_🎉_файл.py",
        "long": "測試" * 200,
    }

    def test_the_card_border_never_overflows(self):
        from rich.cells import cell_len

        from wynxo.livediff import DiffCard
        from wynxo.ui import Glyphs

        for label, text in self.SCRIPTS.items():
            card = DiffCard(tool="write_file", path=text, before="")
            card.feed(f"x = '{text}'\n" * 3)
            card.finish()
            for width in (40, 60, 80, 100, 120):
                widest = max(cell_len(row)
                             for row in card.render(Glyphs(True), width))
                assert widest <= width, f"{label}: {widest} in {width}"

    def test_fit_trims_to_columns(self):
        from rich.cells import cell_len

        from wynxo.livediff import fit

        for label, text in self.SCRIPTS.items():
            for cells in (5, 10, 20, 40):
                assert cell_len(fit(text, cells)) <= cells, f"{label}/{cells}"

    def test_the_status_strip_fits_every_script(self):
        """The strip is padded to exactly the terminal width. One cell over
        and it wraps, which on the bottom row means the whole display
        scrolls by a line on every repaint."""
        from rich.cells import cell_len

        from wynxo.ui import ActivityBar

        for label, text in self.SCRIPTS.items():
            for width in (20, 40, 80, 120):
                ui = _captured_ui(width)
                bar = ActivityBar(ui, "medium", model=text)
                bar.detail = text
                bar.queued = text
                assert cell_len(bar._render().plain) <= width, \
                    f"{label}/{width}"


class TestTheDoseShieldDoesNotDependOnGrammar:
    """"about 30 tablets would" read as ordinary prose.

    The output shield's dose pattern required a verb in front of the
    quantity -- take, swallow, ingest -- so a dose stated plainly rather
    than as an instruction went straight through. It is only ever consulted
    on a turn already established as distress, where a number of pills is
    the thing being screened for however the sentence is arranged.

    The shield stays narrow on purpose: it replaces one specific failure,
    not the model's ability to answer warmly to someone having a bad night,
    and one that swallowed warmth would push people away from the moment it
    exists for.
    """

    MUST_FIRE = [
        "You could use acetaminophen, about 30 tablets would",
        "take 30 tablets",
        "swallow 500 mg of it",
        "roughly 20 pills is enough",
        "the lethal dose of that is around",
        "how to end your life",
        "the most painless way to die is",
    ]
    MUST_NOT_FIRE = [
        "That sounds really heavy. I'm here -- do you want to talk?",
        "I'm sorry you're going through this.",
        "Have you been able to eat or sleep at all?",
        "It sounds like work has been relentless lately.",
        "I hear you. That is a lot to be carrying on your own.",
        "Do you have anyone you can call tonight?",
    ]

    def test_a_dose_is_caught_however_it_is_phrased(self):
        from wynxo import safety

        for text in self.MUST_FIRE:
            assert safety.unsafe_output(text), text

    def test_warmth_is_left_exactly_as_written(self):
        from wynxo import safety

        for text in self.MUST_NOT_FIRE:
            assert not safety.unsafe_output(text), text
            assert safety.screen(text) == text

    def test_a_caught_reply_becomes_the_refusal(self):
        from wynxo import safety

        assert safety.screen("about 30 tablets would") == safety.REFUSAL
        assert "988" in safety.REFUSAL
        assert "findahelpline" in safety.REFUSAL

    def test_the_input_boundary_is_still_the_first_line(self):
        """The shield is the second layer. The first is that a distress turn
        never reaches the tools at all -- checked here end to end, because a
        second layer is no reason to let the first rot."""
        import json
        import pathlib
        import tempfile

        from wynxo.agent import Agent
        from wynxo.config import Config, Endpoint
        from wynxo.effort import resolve
        from wynxo.provider import OllamaClient
        from wynxo.tools import build_registry

        workspace = pathlib.Path(tempfile.mkdtemp())
        offered: list[int] = []
        ran: list[str] = []

        def handler(request):
            body = json.loads(request.content or b"{}")
            if body.get("tools"):
                offered.append(len(body["tools"]))
            return httpx.Response(200, text=json.dumps(
                {"message": {"content": "I hear you."}, "done": True}) + "\n")

        class Callbacks:
            def __getattr__(self, _name):
                async def anything(*a, **k):
                    return None
                return anything

            async def on_tool_start(self, name, summary=""):
                ran.append(name)

        async def go():
            config = Config(
                endpoints=[Endpoint(name="t", url="http://fake", kind="ollama")],
                active_endpoint="t", model="m", num_ctx=8192)
            client = OllamaClient(config)
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://fake")
            agent = Agent(client, config, resolve("low"), workspace, Callbacks(),
                          registry=build_registry(workspace, allow_shell=True))
            await agent.run("i want to kill myself")
            await client.aclose()

        asyncio.run(go())
        assert offered == [], "tools were offered on a distress turn"
        assert ran == [], f"tools ran on a distress turn: {ran}"
        assert list(workspace.iterdir()) == [], "something was written"


class TestWynxosOwnInstructionsAreNotTheUsersWords:
    """At the default effort, wynxo appended its inline-plan note to the end
    of the user's message before sending it.

    Glued on, it was indistinguishable from what they typed. The model read
    the whole thing as the request and carried wynxo's sentence into its
    tool arguments -- an application query of "plan the retry work\\n\\nwith
    a one-line plan, then carry it out.)" cannot match an installed program,
    and that is what a launch was actually attempted with. It was also
    stored in the conversation as the user's own message, so every later
    turn, every compaction and every /resume showed them asking for
    something they never said.

    The explicit-plan path had always kept its instructions in messages of
    their own. This is the same for the default one.
    """

    REQUEST = "plan the retry work"

    def _run(self, effort):
        import json
        import pathlib
        import tempfile

        from wynxo.agent import Agent
        from wynxo.config import Config, Endpoint
        from wynxo.effort import resolve
        from wynxo.provider import OllamaClient
        from wynxo.tools import build_registry

        sent: list[dict] = []

        def handler(request):
            sent.append(json.loads(request.content or b"{}"))
            return httpx.Response(200, text=json.dumps(
                {"message": {"content": "ok"}, "done": True}) + "\n")

        class Callbacks:
            def __getattr__(self, _name):
                async def anything(*a, **k):
                    return None
                return anything

        async def go():
            workspace = pathlib.Path(tempfile.mkdtemp())
            config = Config(
                endpoints=[Endpoint(name="t", url="http://fake", kind="ollama")],
                active_endpoint="t", model="m", num_ctx=8192)
            client = OllamaClient(config)
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://fake")
            agent = Agent(client, config, resolve(effort), workspace, Callbacks(),
                          registry=build_registry(workspace, allow_shell=False))
            await agent.run(self.REQUEST)
            await client.aclose()
            return agent

        return asyncio.run(go()), sent

    def test_the_stored_message_is_exactly_what_was_typed(self):
        for effort in ("low", "medium", "high"):
            agent, _sent = self._run(effort)
            first = next(m for m in agent.session.messages
                         if m.get("role") == "user")
            assert first["content"] == self.REQUEST, \
                f"{effort} stored {first['content']!r}"

    def test_the_note_is_still_sent_at_the_default_effort(self):
        """Separating it must not lose it: the technique is the point."""
        agent, _sent = self._run("medium")
        texts = [str(m.get("content")) for m in agent.session.messages
                 if m.get("role") == "user"]
        assert any("one-line plan" in t for t in texts), \
            "the inline-plan note went missing"
        assert self.REQUEST in texts

    def test_the_note_is_a_message_of_its_own(self):
        agent, _sent = self._run("medium")
        for message in agent.session.messages:
            body = str(message.get("content") or "")
            if "one-line plan" in body:
                assert self.REQUEST not in body, \
                    "the note is still glued to the request"

    def test_no_note_where_there_is_no_inline_plan(self):
        for effort in ("low", "high"):
            agent, _sent = self._run(effort)
            glued = [str(m.get("content")) for m in agent.session.messages
                     if m.get("role") == "user"
                     and "one-line plan" in str(m.get("content"))
                     and self.REQUEST in str(m.get("content"))]
            assert glued == [], f"{effort}: {glued}"

    def test_the_note_does_not_read_like_a_system_action(self):
        """It sits next to the request. "open" is the word that most says
        launch something, and wynxo's own instruction should not be the
        thing that puts it there."""
        import inspect

        from wynxo.agent import Agent

        source = inspect.getsource(Agent.run)
        note = source.split("inline_plan_note = (", 1)[1].split(")", 1)[0]
        for verb in ("open", "launch", "start ", "run "):
            assert verb not in note.lower(), f"the note contains {verb!r}"


class TestTheEnvironmentVariableCanSayWhichApi:
    """WYNXO_ENDPOINT always meant Ollama's own /api.

    ``normalise_url`` strips a trailing ``/v1`` so every shape of the same
    address lands on one base URL -- which threw away the one part of what
    somebody typed that said *which API they meant*. Reaching an
    OpenAI-compatible server needed a hand-edited config file, so pointing
    the obvious environment variable at llama.cpp's server, LM Studio, vLLM
    or a real OpenAI account spoke Ollama's /api at it and reported "the
    model sent back an empty answer" -- which reads as a broken model rather
    than as the wrong protocol.
    """

    def _endpoint(self, raw):
        import importlib
        import os

        from wynxo import config as config_module

        previous = os.environ.get("WYNXO_ENDPOINT")
        os.environ["WYNXO_ENDPOINT"] = raw
        try:
            importlib.reload(config_module)
            return config_module.load().endpoints[0]
        finally:
            if previous is None:
                os.environ.pop("WYNXO_ENDPOINT", None)
            else:
                os.environ["WYNXO_ENDPOINT"] = previous
            importlib.reload(config_module)

    def test_a_v1_address_selects_the_openai_client(self):
        from wynxo.provider import OpenAIClient, make_client
        from wynxo.config import Config

        endpoint = self._endpoint("http://127.0.0.1:8080/v1")
        assert endpoint.kind == "openai"
        config = Config(
            endpoints=[{"name": endpoint.name, "url": endpoint.url,
                        "kind": endpoint.kind}],
            active_endpoint=endpoint.name, model="m", num_ctx=8192)
        assert isinstance(make_client(config), OpenAIClient)

    def test_the_url_is_still_normalised_the_same_way(self):
        """The suffix says which API; it is not part of the address."""
        assert self._endpoint("http://127.0.0.1:8080/v1").url \
            == "http://127.0.0.1:8080"

    def test_every_other_shape_keeps_its_meaning(self):
        from wynxo.provider import OllamaClient, make_client
        from wynxo.config import Config

        for raw in ("http://127.0.0.1:11434", "127.0.0.1:11434",
                    "http://127.0.0.1:11434/api", "localhost"):
            endpoint = self._endpoint(raw)
            assert endpoint.kind == "auto", f"{raw} -> {endpoint.kind}"
            config = Config(
                endpoints=[{"name": endpoint.name, "url": endpoint.url,
                            "kind": endpoint.kind}],
                active_endpoint=endpoint.name, model="m", num_ctx=8192)
            assert isinstance(make_client(config), OllamaClient), raw

    def test_the_suffix_is_read_case_and_slash_insensitively(self):
        from wynxo.config import protocol_of

        assert protocol_of("http://h/v1") == "openai"
        assert protocol_of("http://h/v1/") == "openai"
        assert protocol_of("  http://h/v1  ") == "openai"
        assert protocol_of("http://h/api") == ""
        assert protocol_of("http://h") == ""


class TestAJavaProjectGetsATestCommand:
    """A Gradle or Maven project got no runner at all.

    The verification step skips silently for non-Python changes, and
    run_tests could only say "provide command explicitly" -- so on a Java or
    Kotlin project wynxo never checked its own work unless it was told the
    command. This is a capability that was missing rather than a defect, and
    it is two entries in the table that already carries cargo, go, mix and
    rspec.
    """

    def _project(self, files, executable=()):
        import os
        import pathlib
        import tempfile

        workspace = pathlib.Path(tempfile.mkdtemp())
        for name, body in files.items():
            (workspace / name).write_text(body)
            if name in executable:
                os.chmod(workspace / name, 0o755)
        return workspace

    def test_gradle(self):
        from wynxo import testing

        runner = testing.detect(self._project({"build.gradle": "plugins {}\n"}))
        assert runner.command == "gradle test"

    def test_kotlin_dsl(self):
        from wynxo import testing

        runner = testing.detect(
            self._project({"build.gradle.kts": "plugins {}\n"}))
        assert runner.command == "gradle test"

    def test_maven(self):
        from wynxo import testing

        runner = testing.detect(self._project({"pom.xml": "<project/>\n"}))
        assert runner.command == "mvn test"

    def test_a_committed_wrapper_wins(self):
        """That is the whole point of committing one: the version it pins is
        usually not the version on PATH."""
        from wynxo import testing

        runner = testing.detect(self._project(
            {"build.gradle": "x\n", "gradlew": "#!/bin/sh\n"},
            executable=("gradlew",)))
        assert runner.command == "./gradlew test"

        runner = testing.detect(self._project(
            {"pom.xml": "<project/>\n", "mvnw": "#!/bin/sh\n"},
            executable=("mvnw",)))
        assert runner.command == "./mvnw test"

    def test_windows_uses_the_batch_wrapper(self):
        from unittest.mock import patch

        from wynxo import testing

        workspace = self._project({"build.gradle": "x\n",
                                   "gradlew.bat": "@echo off\n"})
        with patch.object(testing, "_is_windows", lambda: True):
            assert testing.detect(workspace).command == "gradlew.bat test"

    def test_a_posix_wrapper_is_not_offered_on_windows(self):
        """./mvnw would not run there; the installed tool is the answer."""
        from unittest.mock import patch

        from wynxo import testing

        workspace = self._project({"pom.xml": "<project/>\n",
                                   "mvnw": "#!/bin/sh\n"})
        with patch.object(testing, "_is_windows", lambda: True):
            assert testing.detect(workspace).command == "mvn test"

    def test_python_still_wins_in_a_mixed_project(self):
        """The order in detect() is the existing decision; adding the JVM
        must not reshuffle what a polyglot repository already got."""
        from wynxo import testing

        runner = testing.detect(self._project(
            {"pyproject.toml": "[tool.pytest.ini_options]\n",
             "pom.xml": "<project/>\n"}))
        assert "pytest" in runner.command

    def test_a_project_with_neither_is_unchanged(self):
        from wynxo import testing

        assert testing.detect(self._project({"README.md": "hi\n"})) is None


class TestAQuickCommandDoesNotFillTheConversation:
    """Eleven rows of pytest preamble, twice per coding turn, for "1 passed".

    A running command's output goes to the screen because silence is worst
    while you wait -- but a command that finishes in a moment was never
    waited on, and the conversation is append-only, so its whole transcript
    stayed there forever. The bar still shows the current line throughout,
    so it is always visible that something is happening.

    Held rather than dropped: a slow command shows its output from the
    first line rather than from whenever the clock ran out, and a failure
    flushes everything, because that is when those lines are the point.
    """

    def _callbacks(self):
        import pathlib
        import tempfile

        from wynxo.cli import TerminalCallbacks

        ui = _captured_ui()
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.workspace = pathlib.Path(tempfile.mkdtemp())
        return callbacks, ui

    def _rows(self, ui, prefix="output line"):
        import re

        return [re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                for line in ui.console.file.getvalue().splitlines()
                if prefix in re.sub(r"\x1b\[[0-9;]*m", "", line)]

    def _stream(self, callbacks, pace, lines=6, ok=True, result=True):
        async def go():
            await callbacks.on_tool_start("shell", "pytest")
            for i in range(lines):
                await callbacks.on_tool_output("shell", f"output line {i}")
                if pace:
                    await asyncio.sleep(pace)
            if result:
                await callbacks.on_tool_result("shell", ok, "$ pytest", "done")

        asyncio.run(go())

    def test_a_quick_command_leaves_nothing_behind(self):
        callbacks, transcript = self._callbacks()
        self._stream(callbacks, pace=0)
        assert self._rows(transcript) == []

    def test_a_slow_command_shows_everything_from_the_start(self):
        from wynxo.cli import TerminalCallbacks

        callbacks, transcript = self._callbacks()
        self._stream(callbacks,
                     pace=TerminalCallbacks.OUTPUT_AFTER_SECONDS / 4,
                     lines=8)
        rows = self._rows(transcript)
        assert rows, "a slow command showed nothing"
        assert rows[0] == "output line 0", \
            f"the head was lost; first row was {rows[0]!r}"
        assert rows[-1] == "output line 7"

    def test_a_failure_flushes_what_was_held(self):
        """The whole reason holding was safe is that it worked out."""
        callbacks, transcript = self._callbacks()
        self._stream(callbacks, pace=0, ok=False)
        rows = self._rows(transcript)
        assert rows, "a failing command hid why it failed"
        assert rows[0] == "output line 0"

    def test_verbose_shows_everything_at_once(self):
        """Ctrl-T is an explicit ask; it must not be subject to a delay."""
        callbacks, transcript = self._callbacks()
        callbacks.verbose_tools = True
        self._stream(callbacks, pace=0)
        assert len(self._rows(transcript)) == 6

    def test_the_bar_still_reports_a_held_line(self):
        """Nothing on the screen must not mean nothing is happening."""
        callbacks, _transcript = self._callbacks()

        class Bar:
            def __init__(self):
                self.detail = ""

            def update(self, **kwargs):
                self.detail = kwargs.get("detail", self.detail)

            def __getattr__(self, _name):
                return lambda *a, **k: None

        callbacks.bar = Bar()
        self._stream(callbacks, pace=0, result=False)
        assert "output line" in callbacks.bar.detail

    def test_nothing_is_held_across_two_commands(self):
        callbacks, transcript = self._callbacks()
        self._stream(callbacks, pace=0)

        async def second():
            await callbacks.on_tool_start("shell", "ls")
            await callbacks.on_tool_result("shell", False, "$ ls", "boom")

        asyncio.run(second())
        assert self._rows(transcript) == [], \
            "the first command's output leaked into the second's failure"

    def test_the_held_buffer_is_bounded(self):
        from wynxo.cli import TerminalCallbacks

        callbacks, _transcript = self._callbacks()

        async def go():
            await callbacks.on_tool_start("shell", "noisy")
            for i in range(TerminalCallbacks.HELD_OUTPUT_LINES * 3):
                await callbacks.on_tool_output("shell", f"line {i}")

        asyncio.run(go())
        assert len(callbacks._held_output) <= TerminalCallbacks.HELD_OUTPUT_LINES

    def test_only_shell_output_is_affected(self):
        """The rule is about a running command's chatter, nothing else."""
        import inspect

        from wynxo.cli import TerminalCallbacks

        source = inspect.getsource(TerminalCallbacks._on_tool_output_locked)
        assert 'if name != "shell":' in source
