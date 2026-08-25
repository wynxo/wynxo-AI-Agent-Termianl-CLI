"""Only one thing may own stdin at a time.

The KeyWatcher holds the terminal in cbreak mode for the whole turn. When a
permission prompt appears mid-turn -- the default manual-mode path --
prompt_toolkit also wants to read stdin, and both reading it means keystrokes
go to whichever wins the race. The observable symptom was typed characters
vanishing: "/quit" arriving as "quit".
"""


from prompt_toolkit import PromptSession

from wynxo.cli import TerminalCallbacks
from wynxo.permissions import Decision
from wynxo.ui import ActivityBar, UI


class FakeWatcher:
    def __init__(self):
        self.running = False
        self.transitions = []

    def start(self):
        self.running = True
        self.transitions.append("start")

    def stop(self):
        self.running = False
        self.transitions.append("stop")


class RecordingSession(PromptSession):
    """Records whether the watcher was running at the moment it read a line."""

    def __init__(self, watcher, answer="y"):
        self.watcher = watcher
        self.answer = answer
        self.watcher_running_during_prompt = None

    async def prompt_async(self, *args, **kwargs):
        self.watcher_running_during_prompt = self.watcher.running
        return self.answer


def build(answer="y"):
    ui = UI()
    watcher = FakeWatcher()
    session = RecordingSession(watcher, answer)
    callbacks = TerminalCallbacks(ui, session)
    callbacks.watcher = watcher
    callbacks.bar = ActivityBar(ui, "medium")
    watcher.start()
    return callbacks, watcher, session


class TestHandoff:
    async def test_watcher_is_not_reading_while_the_prompt_is(self):
        callbacks, watcher, session = build()
        await callbacks.ask_permission("write_file", "out.txt", "")
        assert session.watcher_running_during_prompt is False, (
            "the key watcher still held stdin while prompt_toolkit read a line"
        )

    async def test_watcher_is_restored_afterwards(self):
        callbacks, watcher, _ = build()
        await callbacks.ask_permission("write_file", "out.txt", "")
        assert watcher.running, "live keys were lost for the rest of the turn"
        assert watcher.transitions == ["start", "stop", "start"]

    async def test_restored_even_when_the_user_aborts(self):
        callbacks, watcher, _ = build(answer="q")
        decision = await callbacks.ask_permission("shell", "rm -rf x", "")
        assert decision is Decision.ABORT
        assert watcher.running

    async def test_restored_when_the_prompt_raises(self):
        callbacks, watcher, session = build()

        async def boom(*a, **k):
            raise EOFError

        session.prompt_async = boom
        await callbacks.ask_permission("write_file", "x", "")
        assert watcher.running, "an exception must not strand the terminal"

    async def test_answers_map_to_decisions(self):
        for answer, expected in [("y", Decision.ALLOW), ("", Decision.ALLOW),
                                 ("a", Decision.ALLOW_ALWAYS),
                                 ("n", Decision.DENY), ("q", Decision.ABORT)]:
            callbacks, _, _ = build(answer=answer)
            assert await callbacks.ask_permission("write_file", "x", "") is expected

    async def test_non_interactive_never_touches_the_terminal(self):
        """With no prompt session there is nobody to ask, so nothing suspends."""
        callbacks = TerminalCallbacks(UI())
        watcher = FakeWatcher()
        watcher.start()
        callbacks.watcher = watcher
        assert await callbacks.ask_permission("write_file", "x", "") is Decision.ALLOW
        assert watcher.transitions == ["start"]


class TestAToolResultClosesWhatWasStreaming:
    """The model says something, then calls a tool that cannot run.

    on_tool_start closes the open streamer, and three paths never reach it:
    an unknown tool, one blocked by the mode, one the user declined. So the
    sentence the model wrote before the call was still held in the streamer
    when the error printed, and arrived afterwards -- run into the first
    words of the next turn. In plan mode every write is refused, so that was
    every tool call.
    """

    def _callbacks(self):
        """With the activity bar up, which is where this goes wrong.

        A half-written line lives inside the bar while one is pinned, and
        only reaches the console when the line is finished. Without a bar it
        goes straight out and the ordering cannot break -- so a test without
        one would pass either way.
        """
        import io

        from rich.console import Console

        from wynxo.ui import ActivityBar

        ui = UI()
        ui.console = Console(file=io.StringIO(), force_terminal=True, width=80)
        ui.live_ok = False
        ui.bar = ActivityBar(ui, "low")
        ui.bar.set_lead = lambda line: None
        return TerminalCallbacks(ui), ui

    async def test_the_sentence_lands_before_the_error(self):
        callbacks, ui = self._callbacks()
        await callbacks.on_content("Let me check that for you.")
        await callbacks.on_tool_result("summon_demon", False,
                                       "No tool named that", "No tool named that")
        written = ui.console.file.getvalue()
        assert "Let me check" in written
        assert written.index("Let me check") < written.index("No tool named")

    async def test_the_streamer_is_closed(self):
        callbacks, _ = self._callbacks()
        await callbacks.on_content("half a sentence")
        await callbacks.on_tool_result("x", False, "no", "no")
        assert callbacks.streamer is None

    async def test_streamed_code_is_closed_too(self):
        import re

        callbacks, ui = self._callbacks()
        await callbacks.on_code("def f():\n    pass\n")
        await callbacks.on_tool_result("write_file", False, "refused", "refused")
        written = re.sub(r"\x1b\[[0-9;]*m", "", ui.console.file.getvalue())
        assert "def f()" in written
        assert written.index("def f()") < written.index("refused")

    async def test_a_normal_result_is_unaffected(self):
        callbacks, ui = self._callbacks()
        await callbacks.on_tool_start("read_file", "app.py")
        await callbacks.on_tool_result("read_file", True, "12 lines", "12 lines")
        assert "12 lines" in ui.console.file.getvalue()
