"""The chat layout: composer pinned at the bottom, conversation above it.

The classic REPL has to release the prompt before the agent can print
anything, which is why the input vanishes for the length of every answer.
Here the application runs for the whole session and the composer stays put.

The design that keeps this honest is that rich still draws everything
exactly as before -- into a buffer rather than onto the terminal -- so there
is no second renderer to drift out of step with the first.
"""

from __future__ import annotations

import asyncio

import pytest

from wynxo.tui import MAX_SCROLLBACK, ChatUI, Transcript, render_to_ansi


class TestTheTranscript:
    def test_rich_output_becomes_lines(self):
        page = Transcript(width=40)
        page.console.print("hello")
        page.console.print("world")
        page.drain()
        assert page.lines == ["hello", "world"]

    def test_colour_survives(self):
        """It is going to a screen, not a log; dropping styling here would
        make the whole layout look like a downgrade."""
        page = Transcript(width=40)
        page.console.print("[bold magenta]styled[/]")
        page.drain()
        assert "\x1b[" in page.lines[0]

    def test_a_trailing_newline_is_not_a_blank_line(self):
        """Splitting naively doubles every gap in the conversation."""
        page = Transcript(width=40)
        page.console.print("one")
        page.drain()
        assert page.lines == ["one"]

    def test_blank_lines_are_kept(self):
        page = Transcript(width=40)
        page.console.print("one")
        page.console.print()
        page.console.print("two")
        page.drain()
        assert page.lines == ["one", "", "two"]

    def test_rich_wraps_to_the_given_width(self):
        """Every stored line is at most one screen row, which is what makes
        'show the last N' the correct thing to do."""
        page = Transcript(width=20)
        page.console.print("word " * 40)
        page.drain()
        assert len(page.lines) > 1
        assert all(len(line) <= 20 for line in page.lines)

    def test_draining_twice_does_not_repeat(self):
        page = Transcript(width=40)
        page.console.print("once")
        page.drain()
        page.drain()
        assert page.lines == ["once"]

    def test_scrollback_is_capped(self):
        page = Transcript(width=40)
        for i in range(MAX_SCROLLBACK + 200):
            page.console.print(f"line {i}")
        page.drain()
        assert len(page.lines) == MAX_SCROLLBACK
        assert page.lines[-1] == f"line {MAX_SCROLLBACK + 199}"

    def test_the_newest_lines_are_the_visible_ones(self):
        page = Transcript(width=40)
        for i in range(50):
            page.console.print(f"line {i}")
        page.drain()
        assert page.visible(3) == ["line 47", "line 48", "line 49"]

    def test_scrolling_back_moves_the_window(self):
        page = Transcript(width=40)
        for i in range(50):
            page.console.print(f"line {i}")
        page.drain()
        assert page.visible(3, offset=10) == ["line 37", "line 38", "line 39"]

    def test_a_short_transcript_is_returned_whole(self):
        page = Transcript(width=40)
        page.console.print("only")
        page.drain()
        assert page.visible(20) == ["only"]

    def test_zero_height_asks_for_nothing(self):
        page = Transcript(width=40)
        page.console.print("x")
        page.drain()
        assert page.visible(0) == []

    def test_resizing_only_affects_what_comes_next(self):
        page = Transcript(width=20)
        page.console.print("a" * 30)
        page.drain()
        narrow = len(page.lines)
        page.resize(100)
        page.console.print("b" * 30)
        page.drain()
        assert len(page.lines) == narrow + 1

    def test_a_silly_width_is_clamped(self):
        page = Transcript(width=1)
        page.console.print("something")
        page.drain()          # must not divide by zero or hang
        assert page.lines


class TestTheLayout:
    @pytest.fixture
    def chat(self):
        return ChatUI(status=lambda: "status row")

    @pytest.fixture
    def chat_with_header(self):
        return ChatUI(status=lambda: "", header=lambda: "wynxo · qwen3 · medium")

    def test_the_transcript_gets_the_rows_the_furniture_does_not(self, chat):
        """Header, status row and composer are pinned; the conversation gets
        whatever is left."""
        width, rows = chat.size()
        furniture = (chat.HEADER_ROWS + chat.COMPOSER_ROWS + chat.STATUS_ROWS)
        assert chat.transcript_rows() == rows - furniture

    def test_the_header_stays_on_screen(self, chat_with_header):
        """It used to be printed into the conversation, so after a page the
        one line saying which model and which project you are talking to had
        scrolled away for the rest of the session."""
        for i in range(500):
            chat_with_header.transcript.console.print(f"line {i}")
        chat_with_header.flush()
        assert "qwen3" in str(chat_with_header._header_fragments().value)

    def test_a_short_conversation_hugs_the_bottom(self, chat):
        """A chat window puts three lines just above the composer, not
        stranded at the top of an empty screen."""
        chat.transcript.console.print("only line")
        chat.flush()
        rendered = str(chat._transcript_fragments().value)
        lines = rendered.split("\n")
        assert len(lines) == chat.transcript_rows()
        assert lines[-1].strip().endswith("only line")
        assert lines[0].strip() == ""

    def test_submitting_queues_the_text(self, chat):
        chat.buffer.text = "do the thing"
        chat.buffer.validate_and_handle()
        assert chat.submissions.get_nowait() == "do the thing"

    def test_the_composer_is_emptied_after_sending(self, chat):
        chat.buffer.text = "sent"
        chat.buffer.validate_and_handle()
        assert chat.buffer.text == ""

    def test_typing_while_busy_is_not_lost(self, chat):
        """The whole point of keeping the composer alive during a turn."""
        for text in ("first", "second"):
            chat.buffer.text = text
            chat.buffer.validate_and_handle()
        assert chat.submissions.get_nowait() == "first"
        assert chat.submissions.get_nowait() == "second"

    def test_scrolling_stops_at_the_top(self, chat):
        for i in range(5):
            chat.transcript.console.print(f"line {i}")
        chat.flush()
        for _ in range(50):
            chat.scroll = min(
                chat.transcript.max_offset(chat.transcript_rows()),
                chat.scroll + 10)
        assert chat.scroll == chat.transcript.max_offset(chat.transcript_rows())

    def test_a_stale_scroll_is_clamped_as_the_transcript_grows(self, chat):
        chat.scroll = 5_000
        chat.transcript.console.print("new line")
        chat.flush()
        chat._transcript_fragments()
        assert chat.scroll <= chat.transcript.max_offset(chat.transcript_rows())

    def test_the_status_row_says_when_you_have_scrolled_back(self, chat):
        for i in range(200):
            chat.transcript.console.print(f"line {i}")
        chat.flush()
        chat.scroll = 20
        assert "scrolled back" in str(chat._status_fragments().value)

    def test_the_status_row_is_plain_when_following(self, chat):
        assert "scrolled back" not in str(chat._status_fragments().value)

    def test_ctrl_d_on_an_empty_composer_quits(self, chat):
        chat.buffer.text = ""
        for binding in chat.app.key_bindings.bindings:
            if "c-d" in str(binding.keys):
                binding.handler(None)
        assert chat.submissions.get_nowait() == "/quit"

    def test_interrupting_does_not_end_the_session(self):
        """Ctrl-C cancels the turn. Killing the app here would throw away
        the conversation along with it."""
        interrupted = []
        chat = ChatUI(on_interrupt=lambda: interrupted.append(True))
        for binding in chat.app.key_bindings.bindings:
            if "c-c" in str(binding.keys):
                binding.handler(None)
        assert interrupted == [True]
        assert not chat.app.is_done

    def test_rendering_drains_without_being_asked(self, chat):
        """Nothing rich writes may be stranded in the buffer waiting for a
        caller that happens to remember to flush."""
        chat.transcript.console.print("written but never flushed")
        assert "written but never flushed" in str(
            chat._transcript_fragments().value)


class TestRenderingTheStatusRow:
    def test_a_renderable_becomes_one_line(self):
        from rich.text import Text

        out = render_to_ansi(Text("busy"), 40)
        assert "busy" in out and "\n" not in out

    def test_it_keeps_the_styling(self):
        from rich.text import Text

        assert "\x1b[" in render_to_ansi(Text("busy", style="bold red"), 40)

    def test_a_renderable_that_explodes_is_not_fatal(self):
        class Broken:
            def __rich_console__(self, *args):
                raise RuntimeError("no")

        assert render_to_ansi(Broken(), 40) == ""


class TestWhereItIsUsed:
    def test_a_pipe_cannot_host_it(self):
        """It needs a terminal at both ends; anywhere else falls back to the
        scrolling prompt, which works on anything."""
        import wynxo.tui as tui

        assert tui.usable() is False       # pytest's stdout is not a tty

    def test_a_dumb_terminal_falls_back(self, monkeypatch):
        import wynxo.tui as tui

        monkeypatch.setenv("TERM", "dumb")
        assert tui.usable() is False

    def test_it_is_the_default(self):
        from wynxo.config import Config

        assert Config().chat_layout is True

    def test_the_classic_prompt_is_still_reachable(self):
        from wynxo.cli import build_parser

        parser = build_parser()
        assert parser.parse_args(["--classic"]).classic is True
        assert parser.parse_args(["--chat"]).chat is True


class TestNoLiveWidgetWritesIntoTheBuffer:
    """A rich Live redraws in place with cursor moves and carriage returns.
    Sent to a buffer of finished lines those arrive as literal "?25l" and
    "^M" in the middle of the conversation -- which is exactly what the
    first run of this layout looked like.
    """

    def test_every_live_is_gated(self):
        import inspect

        from wynxo import ui as ui_module

        source = inspect.getsource(ui_module)
        assert source.count("live_ok") >= 4, "a Live was added without a gate"

    def test_the_flag_defaults_to_allowing_live(self):
        from wynxo.ui import UI

        assert UI().live_ok is True

    def test_the_activity_bar_starts_no_live_when_off(self):
        from wynxo.ui import UI
        from wynxo.ui import ActivityBar

        ui = UI()
        ui.live_ok = False
        bar = ActivityBar(ui, effort="medium")
        bar.start()
        assert bar._live is None
        bar.stop()

    def test_the_startup_animation_writes_plain_text(self):
        from wynxo.pet import Pet
        from wynxo.tui import Transcript
        from wynxo.ui import UI

        ui = UI()
        page = Transcript(width=60)
        ui.console = page.console
        ui.live_ok = False
        pet = Pet(enabled=True, name="wyn")
        pet.animate = True
        ui.wake(pet, "wyn")
        page.drain()
        body = "\n".join(page.lines)
        assert "?25" not in body and "\r" not in body


class TestAskingInsideTheRunningApp:
    """A second prompt_toolkit application cannot run inside this one.

    Trying left the layout half-drawn -- bottom border gone, the question
    typed over the composer -- so both the permission prompt and the arrow
    picker are drawn by the application that is already running.
    """

    @pytest.fixture
    def chat(self):
        return ChatUI(status=lambda: "")

    def _press(self, chat, key: str, data: str = ""):
        import types

        for binding in chat.app.key_bindings.bindings:
            if key in str(binding.keys):
                binding.handler(types.SimpleNamespace(data=data, app=chat.app))
                return True
        return False

    def test_a_single_key_answers(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[y] yes [n] no:", {"y": "yes", "n": "no"}))
            await asyncio.sleep(0)
            self._press(chat, "<any>", "y")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "y"

    def test_the_question_replaces_the_caret(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[y] yes [n] no:", {"y": "yes", "n": "no"}))
            await asyncio.sleep(0)
            prefix = chat._composer_prefix()
            self._press(chat, "<any>", "n")
            await asyncio.wait_for(pending, 1)
            return prefix

        assert "[y] yes" in asyncio.run(go())

    def test_the_caret_comes_back_afterwards(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.ask("?", {"y": "yes"}))
            await asyncio.sleep(0)
            self._press(chat, "<any>", "y")
            await asyncio.wait_for(pending, 1)

        asyncio.run(go())
        assert chat._composer_prefix() == "│ > "

    def test_a_typed_word_answers_too(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("?", {"y": "yes", "n": "no"}))
            await asyncio.sleep(0)
            chat.buffer.text = "yes"
            chat.buffer.validate_and_handle()
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "y"

    def test_an_unrelated_key_starts_a_typed_answer(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.ask("?", {"y": "yes"}))
            await asyncio.sleep(0)
            self._press(chat, "<any>", "z")
            assert not pending.done()
            assert chat.buffer.text == "z"
            chat.buffer.text = "y"
            chat.buffer.validate_and_handle()
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "y"

    def test_typing_a_sentence_cannot_grant_a_permission(self, chat):
        """The bug this rule exists for: with a composer that already has
        text in it, the "a" in "hello again" answered [a]lways and silently
        granted a permission nobody meant to grant."""
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[y] yes [a] always [n] no [q] stop:",
                         {"y": "yes", "a": "always", "n": "no", "q": "stop"}))
            await asyncio.sleep(0)
            for letter in "hello again":
                self._press(chat, "<any>", letter)
            assert not pending.done(), "a stray keystroke answered it"
            assert chat.buffer.text == "hello again"
            self._press(chat, "c-c")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "q"

    def test_a_single_key_still_answers_from_an_empty_composer(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("?", {"y": "yes", "a": "always"}))
            await asyncio.sleep(0)
            self._press(chat, "<any>", "a")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "a"

    def test_a_question_does_not_reach_the_turn_queue(self, chat):
        """Answering must not be mistaken for the next thing you said."""
        async def go():
            pending = asyncio.ensure_future(chat.ask("?", {"y": "yes"}))
            await asyncio.sleep(0)
            chat.buffer.text = "y"
            chat.buffer.validate_and_handle()
            await asyncio.wait_for(pending, 1)

        asyncio.run(go())
        assert chat.submissions.empty()


class TestThePickerInsideTheApp:
    @pytest.fixture
    def chat(self):
        return ChatUI(status=lambda: "")

    OPTIONS = [("purple", "deep violet"), ("sakura", "pink"),
               ("midnight", "cool blue")]

    # prompt_toolkit normalises "enter" to ControlM, so the binding does not
    # answer to the name it was registered under.
    ALIASES = {"enter": "c-m"}

    def _press(self, chat, key: str):
        import types

        wanted = self.ALIASES.get(key, key)
        for binding in chat.app.key_bindings.bindings:
            if wanted in str(binding.keys):
                binding.handler(types.SimpleNamespace(data="", app=chat.app))
                return True
        raise AssertionError(f"no binding for {key!r}")

    def test_arrows_move_and_enter_chooses(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.choose("theme", self.OPTIONS, "purple"))
            await asyncio.sleep(0)
            self._press(chat, "down")
            self._press(chat, "down")
            self._press(chat, "enter")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "midnight"

    def test_it_starts_on_the_current_value(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.choose("theme", self.OPTIONS, "sakura"))
            await asyncio.sleep(0)
            self._press(chat, "enter")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "sakura"

    def test_the_selection_wraps(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.choose("theme", self.OPTIONS, "purple"))
            await asyncio.sleep(0)
            self._press(chat, "up")          # off the top, round to the end
            self._press(chat, "enter")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "midnight"

    def test_escape_cancels_without_choosing(self, chat):
        """Cancelling and being unable to offer a choice are different, and
        printing the table anyway would ignore that."""
        async def go():
            pending = asyncio.ensure_future(
                chat.choose("theme", self.OPTIONS, "purple"))
            await asyncio.sleep(0)
            self._press(chat, "escape")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) is None

    def test_ctrl_c_cancels_the_picker_not_the_session(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.choose("theme", self.OPTIONS, "purple"))
            await asyncio.sleep(0)
            self._press(chat, "c-c")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) is None
        assert not chat.app.is_done

    def test_it_is_drawn_at_the_foot_of_the_conversation(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.choose("theme", self.OPTIONS, "purple"))
            await asyncio.sleep(0)
            body = str(chat._transcript_fragments().value)
            self._press(chat, "escape")
            await asyncio.wait_for(pending, 1)
            return body

        body = asyncio.run(go())
        assert "theme" in body and "midnight" in body
        assert "arrows move" in body

    def test_it_leaves_no_trace_afterwards(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.choose("theme", self.OPTIONS, "purple"))
            await asyncio.sleep(0)
            self._press(chat, "escape")
            await asyncio.wait_for(pending, 1)

        asyncio.run(go())
        assert chat.picker is None
        assert "arrows move" not in str(chat._transcript_fragments().value)

    def test_the_repl_routes_pickers_here(self):
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._pick)
        assert "self.chat.choose" in source
        assert source.index("self.chat") < source.index("arrows_supported")


class TestWithNoConsoleAtAll:
    """Windows without a console handle.

    Constructing an Application builds the platform's output object there
    and then, and on Windows that means opening a console. Without one -- a
    CI runner, a service, anything started by pythonw -- it raised
    NoConsoleScreenBufferError from the constructor, so merely building the
    layout was fatal. Twenty-eight of these took the Windows suite down.
    """

    @pytest.fixture
    def no_console(self, monkeypatch):
        import prompt_toolkit.output.defaults as defaults

        class NoConsoleScreenBufferError(Exception):
            pass

        def refuse(*args, **kwargs):
            raise NoConsoleScreenBufferError(
                "No Windows console found. Are you running cmd.exe?")

        monkeypatch.setattr(defaults, "create_output", refuse)
        return refuse

    def test_the_layout_can_still_be_built(self, no_console):
        assert ChatUI(status=lambda: "") is not None

    def test_it_falls_back_to_a_stand_in(self, no_console):
        from prompt_toolkit.output import DummyOutput

        assert isinstance(ChatUI().app.output, DummyOutput)

    def test_the_transcript_still_works(self, no_console):
        chat = ChatUI()
        chat.transcript.console.print("still fine")
        chat.flush()
        assert "still fine" in str(chat._transcript_fragments().value)

    def test_the_keys_still_resolve(self, no_console):
        """The bindings are what the tests and the picker rely on."""
        chat = ChatUI()
        assert chat.app.key_bindings.bindings

    def test_a_real_session_never_gets_here(self, monkeypatch):
        """usable() sends a session with no terminal down the scrolling
        path, so the stand-in is a safety net rather than a mode."""
        import wynxo.tui as tui

        assert tui.usable() is False       # pytest's stdout is not a tty
