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
import re

import pytest

from wynxo.tui import MAX_SCROLLBACK, ChatUI, Transcript, render_to_ansi


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(lines) -> str:
    """The transcript keeps styled text; assertions want what the eye sees."""
    return _ANSI.sub("", "\n".join(lines))


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

    def test_apply_theme_swaps_the_chrome_and_keeps_defaults(self, chat):
        from wynxo.tui import DEFAULT_CHROME

        chat.apply_theme({"edge": "#123456", "ok": "#00ff00"})
        assert chat._chrome["edge"] == "#123456"
        assert chat._chrome["ok"] == "#00ff00"
        assert chat._chrome["todo-title"] == DEFAULT_CHROME["todo-title"]

    def test_apply_theme_with_nothing_keeps_the_current_chrome(self, chat):
        chat.apply_theme({"edge": "#123456"})
        chat.apply_theme(None)
        assert chat._chrome["edge"] == "#123456"

    def test_the_footer_does_not_duplicate_live_activity(self, chat):
        """The bar already renders the current tool during a turn; the
        activity strip must not repeat it on the same row."""
        chat.set_activity("-> read_file x.py")
        chat._status = lambda: "\x1b[31m-> read_file x.py\x1b[0m   tokens"
        out = str(chat._footer_fragments().value)
        assert out.count("read_file") == 1

    def test_the_footer_prepends_activity_when_the_status_differs(self, chat):
        chat.set_activity("-> thinking")
        chat._status = lambda: "idle"
        out = str(chat._footer_fragments().value)
        assert "thinking" in out and "idle" in out

    def test_notifications_are_queued_and_do_not_change_geometry(self, chat):
        chat.notify("ok", ok=True)
        chat.notify("failed", ok=False)
        before = chat.transcript_rows()
        assert "ok" in str(chat._todo_fragments())
        assert chat.transcript_rows() == before
        chat._toast = ("ok", 0.0)
        chat._toast_life = -1
        assert "failed" in chat._toast_line()

    def test_activity_is_a_fixed_bounded_row(self, chat):
        chat.set_activity("-> testing " + "x" * 200)
        assert chat.ACTIVITY_ROWS == 1
        assert "\n" not in str(chat._activity_fragments().value)
        before = chat.transcript_rows()
        chat.set_activity("-> reading")
        assert chat.transcript_rows() == before

    def test_the_transcript_gets_the_rows_the_furniture_does_not(self, chat):
        """Header, status row and composer are pinned; the conversation gets
        whatever is left."""
        width, rows = chat.size()
        furniture = (chat.HEADER_ROWS + chat.FOOTER_ROWS
                     + chat.composer_frame_rows())
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

    def test_scrolled_back_counts_what_arrives_as_unread(self, chat):
        for i in range(50):
            chat.transcript.console.print(f"line {i}")
        chat.flush()
        chat.scroll = 10
        chat.transcript.console.print("new line")
        chat.flush()
        assert chat.scroll > 10
        assert chat._unread > 0
        assert "new" in str(chat._footer_fragments().value)
        assert "scrolled back" in str(chat._footer_fragments().value)

    def test_following_the_newest_resets_the_unread_count(self, chat):
        for i in range(50):
            chat.transcript.console.print(f"line {i}")
        chat.flush()
        chat.scroll = 10
        chat.transcript.console.print("while you were away")
        chat.flush()
        assert chat._unread > 0
        chat.scroll = 0
        chat.flush()
        assert chat._unread == 0

    def test_the_mouse_wheel_scrolls_the_transcript(self, chat):
        import types

        for i in range(50):
            chat.transcript.console.print(f"line {i}")
        chat.flush()

        def press(key):
            for binding in chat.app.key_bindings.bindings:
                if key in str(binding.keys):
                    binding.handler(types.SimpleNamespace(data="", app=chat.app))
                    return True
            return False

        assert press("scroll-up")
        assert chat.scroll > 0
        assert press("scroll-down")
        assert chat.scroll == 0

    def test_the_mouse_wheel_cannot_push_beyond_the_top(self, chat):
        import types

        for i in range(5):
            chat.transcript.console.print(f"line {i}")
        chat.flush()
        for binding in chat.app.key_bindings.bindings:
            if "scroll-up" in str(binding.keys):
                for _ in range(50):
                    binding.handler(types.SimpleNamespace(data="", app=chat.app))
        assert chat.scroll == chat.transcript.max_offset(chat.transcript_rows())

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
        assert "scrolled back" in str(chat._footer_fragments().value)

    def test_the_status_row_is_plain_when_following(self, chat):
        assert "scrolled back" not in str(chat._footer_fragments().value)

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

    def test_pet_frame_uses_elapsed_time_not_render_count(self, monkeypatch):
        from wynxo.tui import ChatUI

        now = [100.0]
        monkeypatch.setattr("wynxo.tui.time.monotonic", lambda: now[0])
        chat = ChatUI(pet_state=lambda: "coding", pet_enabled=lambda: True,
                      pet_animate=lambda: True)
        first = chat._pet_lines()
        now[0] += 0.5
        second = chat._pet_lines()
        assert first != second

    def test_exit_cancels_pending_interaction_futures(self, chat):
        async def go():
            pending = asyncio.create_task(chat.ask("?", {"y": "yes"}))
            await asyncio.sleep(0)
            chat.exit()
            with pytest.raises(asyncio.CancelledError):
                await pending

        asyncio.run(go())

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

    def test_classic_and_chat_flags_flip_the_saved_layout(self):
        """--classic and --chat decide the layout through apply_flags, so
        the same flags work on every platform including Windows."""
        from wynxo.cli import apply_flags, build_parser
        from wynxo.config import Config

        config = Config()
        config.chat_layout = True
        apply_flags(config, build_parser().parse_args(["--classic"]))
        assert config.chat_layout is False
        apply_flags(config, build_parser().parse_args(["--chat"]))
        assert config.chat_layout is True


class TestOnlyOneThingReadsTheKeyboard:
    """The bug: characters typed during a turn simply vanished.

    A turn started a KeyWatcher, which reads the tty in a thread of its own
    to catch Ctrl-O and to collect type-ahead. Under the chat layout
    prompt_toolkit is already reading that same terminal, so every byte went
    to whichever reader got there first -- one keystroke in eight or so
    disappeared out of the composer, and Ctrl-O worked or did not depending
    on the race. Typing "second message" during a turn produced "econd
    message".
    """

    def test_the_watcher_stays_out_of_the_layout(self):
        import ast
        import inspect

        from wynxo.cli import Repl

        tree = ast.parse(inspect.getsource(Repl.turn).strip())
        starts = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and ast.unparse(node.func) == "watcher.start"]
        assert starts, "the watcher is gone entirely; the classic REPL needs it"
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "self.chat is None" not in ast.unparse(node.test):
                continue
            if any(ast.unparse(inner.func) == "watcher.start"
                   for inner in ast.walk(node)
                   if isinstance(inner, ast.Call)):
                guarded = True
        assert guarded, (
            "the watcher reads the same terminal prompt_toolkit does; "
            "starting it under the chat layout steals keystrokes")

    def _press(self, chat, key: str):
        import types

        for binding in chat.app.key_bindings.bindings:
            names = tuple(getattr(k, "value", str(k)) for k in binding.keys)
            if names == (key,):
                binding.handler(types.SimpleNamespace(data="", app=chat.app))
                return True
        return False

    def test_the_layout_binds_the_keys_the_watcher_used_to_catch(self):
        """Otherwise turning the watcher off would take Ctrl-O with it."""
        pressed = []
        chat = ChatUI(status=lambda: "",
                      on_thinking=lambda: pressed.append("thinking"),
                      on_tools=lambda: pressed.append("tools"))
        assert self._press(chat, "c-o")
        assert self._press(chat, "c-t")
        assert pressed == ["thinking", "tools"]

    def test_those_keys_are_harmless_when_nothing_is_wired(self):
        chat = ChatUI(status=lambda: "")
        assert self._press(chat, "c-o")      # must not raise
        assert self._press(chat, "c-t")

    def test_the_repl_wires_them(self):
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl.use_chat_layout)
        assert "on_thinking" in source and "on_tools" in source


class TestStartingWithAPrompt:
    """`wynxo "add retries"` -- a prompt on the command line.

    It ran the turn before the application started, and use_chat_layout()
    has already pointed the console at the transcript by then. So the whole
    answer was written into a buffer nobody was showing: the file was
    edited, the screen stayed empty, and what came back was the classic
    prompt drawn without its box.
    """

    def _repl(self):
        import asyncio as _asyncio

        from wynxo.cli import Repl

        repl = Repl.__new__(Repl)
        repl.chat = ChatUI(status=lambda: "")
        repl.turn_calls = []

        async def _connect():
            return True

        async def _chat_loop():
            return 0

        async def turn(text):
            repl.turn_calls.append(text)

        repl._connect = _connect
        repl._chat_loop = _chat_loop
        repl.turn = turn
        return repl

    def test_the_prompt_is_answered_inside_the_layout(self):
        repl = self._repl()
        assert asyncio.run(repl.start_with("add retries")) == 0
        assert repl.chat.submissions.get_nowait() == "add retries"

    def test_the_turn_is_not_taken_before_the_screen_exists(self):
        repl = self._repl()
        asyncio.run(repl.start_with("add retries"))
        assert repl.turn_calls == [], (
            "the answer would go into a buffer nobody is showing")

    def test_the_classic_path_is_unchanged(self):
        import asyncio as _asyncio

        from wynxo.cli import Repl

        repl = Repl.__new__(Repl)
        repl.chat = None
        repl.turn_calls = []

        async def _connect():
            return True

        async def turn(text):
            repl.turn_calls.append(text)

        async def _loop():
            return 0

        repl._connect, repl.turn, repl._loop = _connect, turn, _loop
        assert _asyncio.run(repl.start_with("add retries")) == 0
        assert repl.turn_calls == ["add retries"]


class TestAWindowTooSmallForTheLayout:
    """Header, status row and composer are pinned, so a very short window
    has nothing left for the conversation -- and prompt_toolkit replaces the
    whole screen with "Window too small..." rather than drawing it. A five
    row pane got that and nothing else."""

    def _usable(self, monkeypatch, rows, tty=True):
        from wynxo import tui

        monkeypatch.setattr(tui, "_terminal_height", lambda default=24: rows)
        monkeypatch.setenv("TERM", "xterm-256color")

        class _TTY:
            def isatty(self):
                return tty

        # usable() imports sys itself, so the real module is what to patch.
        monkeypatch.setattr("sys.stdin", _TTY())
        monkeypatch.setattr("sys.stdout", _TTY())
        return tui.usable()

    def test_a_short_window_takes_the_scrolling_prompt(self, monkeypatch):
        from wynxo.tui import MIN_ROWS

        assert self._usable(monkeypatch, MIN_ROWS - 1) is False

    def test_a_tall_enough_window_keeps_the_layout(self, monkeypatch):
        from wynxo.tui import MIN_ROWS

        assert self._usable(monkeypatch, MIN_ROWS) is True

    def test_the_floor_leaves_room_to_read_something(self):
        from wynxo.tui import MIN_ROWS, ChatUI

        furniture = (ChatUI.HEADER_ROWS + ChatUI.COMPOSER_ROWS
                     + ChatUI.FOOTER_ROWS)
        assert MIN_ROWS - furniture >= 2


class TestTheCompletionMenuOnAShortWindow:
    """The scrolling prompt is where a small window ends up, so it has to
    work there. It reserved six rows for the suggestion list on top of its
    own prompt and toolbar, which a four-row pane cannot pay -- so that
    fallback was showing "Window too small..." too."""

    def _rows(self, monkeypatch, rows):
        from wynxo import cli

        monkeypatch.setattr(cli, "terminal_height", lambda: rows)
        return cli._menu_rows()

    def test_a_normal_window_keeps_the_menu(self, monkeypatch):
        assert self._rows(monkeypatch, 40) == 6

    def test_a_short_window_reserves_less(self, monkeypatch):
        assert 0 < self._rows(monkeypatch, 11) < 6

    def test_a_tiny_window_reserves_nothing(self, monkeypatch):
        assert self._rows(monkeypatch, 5) == 0

    def test_it_never_asks_for_more_than_the_window_has(self, monkeypatch):
        for rows in range(3, 40):
            assert self._rows(monkeypatch, rows) < rows

    def test_the_prompt_actually_uses_it(self):
        """A constant here is the bug; the point is that it varies."""
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl.__init__)
        assert "reserve_space_for_menu=_menu_rows()" in source


class TestThePinnedBlockGrowsForThePlan:
    """The pinned area is the activity bar and whatever sits above it -- the
    plan, the line being written. It was one row, and the renderer kept only
    the first line of what it was given.

    So with a plan up, the pinned row showed the top border of the plan
    panel and nothing else: no items, and no activity bar either -- no
    tokens, no elapsed time, no context figure -- for as long as the plan
    lived, which on a multi-step job is the whole job.
    """

    def _block(self, lines):
        from rich.text import Text

        return Text("\n".join(lines))

    def test_more_than_one_line_survives(self):
        from wynxo.tui import render_to_ansi

        rendered = render_to_ansi(self._block(["one", "two", "three"]),
                                  width=40, max_rows=6)
        assert rendered.count("\n") == 2

    def test_the_default_is_still_one_line(self):
        """The header is one line and always will be."""
        from wynxo.tui import render_to_ansi

        rendered = render_to_ansi(self._block(["one", "two"]), width=40)
        assert "\n" not in rendered

    def test_the_bar_is_what_survives_a_squeeze(self):
        """It is the last line of the block, and the one that must never be
        pushed off: it holds the token count and the context figure."""
        from wynxo.tui import render_to_ansi

        rendered = render_to_ansi(
            self._block(["plan top", "item", "item", "THE BAR"]),
            width=40, max_rows=2)
        assert "THE BAR" in rendered
        assert "plan top" not in rendered

    def test_status_is_always_one_row(self):
        chat = ChatUI(status=lambda: "a\nb\nc\nd")
        assert chat._status_fragments().value.count("\n") == 0
        assert chat.FOOTER_ROWS == 1
        before = chat.transcript_rows()
        chat._status = lambda: "x\ny\nz"
        assert chat.transcript_rows() == before

    def test_verbose_status_is_flattened_without_reflow(self):
        chat = ChatUI(status=lambda: "\n".join(["x"] * 400))
        assert chat.transcript_rows() >= 1
        assert "\n" not in chat._status_fragments().value

    def test_the_conversation_keeps_room_on_a_short_window(self):
        chat = ChatUI(status=lambda: "\n".join(["x"] * 40), width=80)
        chat.size = lambda: (80, 12)
        assert chat.transcript_rows() >= 1

    def test_status_changes_do_not_change_transcript_rows(self):
        chat = ChatUI(status=lambda: "one line")
        tall = chat.transcript_rows()
        chat._status = lambda: "a\nb\nc\nd\ne"
        assert chat.transcript_rows() == tall


class TestTheWindowChangingSize:
    """rich wraps to ui.width before anything reaches the pane, and the pane
    truncates rather than wraps. So a window made narrower mid-session cut
    the right-hand end off every line written after it: eight words of a
    sixteen-word answer, gone, until the session was restarted."""

    def test_a_new_width_is_announced_once(self):
        chat = ChatUI(status=lambda: "", width=100)
        seen = []
        chat.on_resize = seen.append
        chat._measured(80, 24)
        chat._measured(80, 24)
        chat._measured(60, 24)
        assert seen == [80, 60]

    def test_it_survives_having_nobody_listening(self):
        chat = ChatUI(status=lambda: "", width=100)
        assert chat._measured(80, 24) == (80, 24)

    def test_the_repl_keeps_the_ui_in_step(self):
        import inspect

        from wynxo.cli import Repl

        assert "on_resize" in inspect.getsource(Repl.use_chat_layout)
        assert "self.ui.width" in inspect.getsource(Repl._resized)


class TestTheStartupChecklist:
    """Status writes straight to stdout, and the application takes the
    alternate screen the moment it starts -- so the warnings collected on
    the way up were printed to the screen being left behind."""

    def test_it_is_written_where_the_conversation_is(self):
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._connect)
        assert "transcript.console.file" in source

    def test_nothing_in_it_prints_to_stdout_by_default(self):
        import ast
        import inspect

        from wynxo.cli import Repl

        tree = ast.parse(inspect.getsource(Repl._connect).strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "print":
                assert any(k.arg == "file" for k in node.keywords), (
                    "a bare print() lands on the screen the layout replaces")


class TestNothingOpensItsOwnPrompt:
    """The bug this class exists for, in its fourth guise.

    A second prompt_toolkit application cannot run inside the chat layout's
    one. It was fixed for the permission prompt, then for the settings
    pickers, then for /model and /resume -- and four more were still doing
    it: the keep-or-revert question after every turn that changed a file,
    the file-by-file walk behind it, /commit and its message editor, and the
    confirmation for widening the scope to the whole machine.

    The first of those runs by itself after any turn that edits anything,
    so the default layout was being torn apart in ordinary use.
    """

    ALLOWED = {"_question", "_type_in", "_loop", "_ask"}
    """_loop is the classic REPL's own prompt, which only runs when there is
    no chat layout; _ask has its own branch for the same reason. Everything
    else goes through the two helpers."""

    def test_every_question_goes_through_the_one_door(self):
        import ast
        import inspect

        from wynxo import cli as cli_module

        tree = ast.parse(inspect.getsource(cli_module))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in self.ALLOWED:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and \
                        "prompt_async" in ast.unparse(inner.func):
                    raise AssertionError(
                        f"{node.name} opens its own prompt; use _question or "
                        "_type_in, or it will tear the chat layout apart")

    def test_the_door_prefers_the_layout(self):
        import inspect

        from wynxo.cli import Repl

        for helper in (Repl._question, Repl._type_in):
            source = inspect.getsource(helper)
            assert "self.chat" in source
            assert source.index("self.chat") < source.index("prompt_async")


class TestReadingALineInsideTheApp:
    """Editing a commit message needs free text, not a single key."""

    @pytest.fixture
    def chat(self):
        return ChatUI(status=lambda: "")

    def test_the_default_is_there_to_be_edited(self, chat):
        """Put in the composer rather than described, so it is a starting
        point instead of something to retype."""
        async def go():
            pending = asyncio.ensure_future(
                chat.prompt("message:", "fix the parser"))
            await asyncio.sleep(0)
            text = chat.buffer.text
            chat.buffer.validate_and_handle()
            await asyncio.wait_for(pending, 1)
            return text

        assert asyncio.run(go()) == "fix the parser"

    def test_what_is_entered_comes_back(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.prompt("message:", "old"))
            await asyncio.sleep(0)
            chat.buffer.text = "a better message"
            chat.buffer.validate_and_handle()
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "a better message"

    def test_the_question_is_shown_where_the_caret_was(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.prompt("message:", ""))
            await asyncio.sleep(0)
            prefix = chat._composer_prefix()
            chat.buffer.validate_and_handle()
            await asyncio.wait_for(pending, 1)
            return prefix

        assert "message:" in asyncio.run(go())

    def test_the_caret_comes_back_afterwards(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.prompt("message:", "x"))
            await asyncio.sleep(0)
            chat.buffer.validate_and_handle()
            await asyncio.wait_for(pending, 1)

        asyncio.run(go())
        assert chat._composer_prefix() == "❯ "
        assert chat.buffer.text == ""

    def test_what_you_type_is_not_sent_as_a_message(self, chat):
        """It is an answer to the question on screen, not the next thing
        you said."""
        async def go():
            pending = asyncio.ensure_future(chat.prompt("message:", ""))
            await asyncio.sleep(0)
            chat.buffer.text = "a better message"
            chat.buffer.validate_and_handle()
            await asyncio.wait_for(pending, 1)

        asyncio.run(go())
        assert chat.submissions.empty()

    def test_ctrl_c_cancels_it(self, chat):
        import types

        async def go():
            pending = asyncio.ensure_future(chat.prompt("message:", "x"))
            await asyncio.sleep(0)
            for binding in chat.app.key_bindings.bindings:
                names = tuple(getattr(k, "value", str(k))
                              for k in binding.keys)
                if names == ("c-c",):
                    binding.handler(
                        types.SimpleNamespace(data="", app=chat.app))
                    break
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == ""


class TestToolCardsLiveInTheConversation:
    """In the chat layout a tool call is one compact line when it starts and
    one when it lands -- the coding-agent shape, inside the conversation."""

    def _callbacks(self):
        from unittest.mock import MagicMock

        from wynxo.cli import TerminalCallbacks
        from wynxo.ui import UI

        chat = ChatUI(status=lambda: "")
        ui = UI()
        ui.console = chat.transcript.console
        cb = TerminalCallbacks(ui)
        cb.chat = chat
        cb.bar = MagicMock()
        return cb, chat

    def test_a_tool_start_prints_one_compact_line(self):
        cb, chat = self._callbacks()
        asyncio.run(cb._on_tool_start_locked("read_file", "x.py", None))
        chat.transcript.drain()
        body = _plain(chat.transcript.lines)
        assert "→ read_file" in body
        assert "x.py" in body

    def test_a_tool_result_prints_one_compact_line(self):
        cb, chat = self._callbacks()
        asyncio.run(cb._on_tool_result_locked("read_file", True,
                                              "312 lines", "312", None))
        chat.transcript.drain()
        body = _plain(chat.transcript.lines)
        assert "✓ read_file" in body
        assert "312 lines" in body

    def test_a_failed_tool_result_prints_a_cross(self):
        cb, chat = self._callbacks()
        asyncio.run(cb._on_tool_result_locked("run_tests", False,
                                              "1 failed", "traceback", None))
        chat.transcript.drain()
        assert "✗ run_tests" in _plain(chat.transcript.lines)

    def test_todo_writes_do_not_spam_the_conversation(self):
        cb, chat = self._callbacks()
        asyncio.run(cb._on_tool_start_locked("todo_write", "[x] step", None))
        chat.transcript.drain()
        assert "todo_write" not in _plain(chat.transcript.lines)


class TestTheSpinnerWhereNothingCanRepaint:
    """rich's Status is a Live, and it was the one that got away.

    Fifteen commands wrap slow work in ui.status(), and under this layout
    every one of them wrote its cursor-hide, its redraw and its carriage
    return into a buffer of finished lines. /model printed two spinners'
    worth of "?25l" and "^M" into the conversation before its picker even
    opened.
    """

    def _ui(self, live_ok: bool):
        import io

        from rich.console import Console

        from wynxo.ui import UI

        ui = UI()
        ui.console = Console(file=io.StringIO(), force_terminal=True,
                             width=80)
        ui.live_ok = live_ok
        return ui

    def test_no_cursor_control_reaches_the_transcript(self):
        ui = self._ui(live_ok=False)
        with ui.status("asking the server what it has..."):
            pass
        written = ui.console.file.getvalue()
        assert "\x1b[?25l" not in written
        assert "\r" not in written

    def test_the_message_is_still_said(self):
        """Silence during a slow call reads as a hang."""
        ui = self._ui(live_ok=False)
        with ui.status("asking the server what it has..."):
            pass
        assert "asking the server" in ui.console.file.getvalue()

    def test_an_update_is_said_too(self):
        ui = self._ui(live_ok=False)
        with ui.status("checking this machine...") as status:
            status.update("checking 192.168.1.0/24...")
        assert "192.168.1.0/24" in ui.console.file.getvalue()

    def test_a_normal_terminal_still_gets_the_spinner(self):
        from rich.status import Status

        assert isinstance(self._ui(live_ok=True).status("working..."), Status)


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
        """Press one key, matched exactly.

        Exactly, because the answer keys are bound one character at a time
        now -- a substring match would fire whichever binding happened to
        contain the letter.
        """
        import types

        data = data or (key if len(key) == 1 else "")
        for binding in chat.app.key_bindings.bindings:
            names = tuple(getattr(k, "value", str(k)) for k in binding.keys)
            if names == (key,):
                binding.handler(types.SimpleNamespace(data=data, app=chat.app))
                return True
        return False

    def _type(self, chat, text: str) -> None:
        """Type a sentence the way the terminal delivers it.

        Only the answer keys are bound here; everything else -- the space in
        "hello again" among them -- reaches the composer's own insert
        binding, which is what this stands in for.
        """
        for character in text:
            if not self._press(chat, character):
                chat.buffer.insert_text(character)

    def _binds(self, chat, key: str) -> bool:
        """Whether the layout claims this key for itself."""
        for binding in chat.app.key_bindings.bindings:
            names = tuple(getattr(k, "value", str(k)) for k in binding.keys)
            if names == (key,):
                return True
        return False

    def test_a_single_key_answers(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[y] yes [n] no:", {"y": "yes", "n": "no"}))
            await asyncio.sleep(0)
            self._press(chat, "y")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "y"

    def test_the_question_replaces_the_caret(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[y] yes [n] no:", {"y": "yes", "n": "no"}))
            await asyncio.sleep(0)
            prefix = chat._composer_prefix()
            self._press(chat, "n")
            await asyncio.wait_for(pending, 1)
            return prefix

        assert "[y] yes" in asyncio.run(go())

    def test_the_caret_comes_back_afterwards(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.ask("?", {"y": "yes"}))
            await asyncio.sleep(0)
            self._press(chat, "y")
            await asyncio.wait_for(pending, 1)

        asyncio.run(go())
        assert chat._composer_prefix() == "❯ "

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
            self._press(chat, "z")
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
            self._type(chat, "hello again")
            assert not pending.done(), "a stray keystroke answered it"
            assert chat.buffer.text == "hello again"
            self._press(chat, "c-c")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "q"

    # "end" is missing on purpose: it is the layout's own key for following
    # the transcript again after scrolling back, and the status row says so.
    @pytest.mark.parametrize("key", [
        "backspace", "c-h", "delete", "left", "right",
        "home", "c-a", "c-e", "c-w", "c-u",
    ])
    def test_the_editing_keys_are_left_to_the_composer(self, chat, key):
        """The bug: the answer keys were bound as <any>, marked eager.

        <any> matches every key press there is, and eager made it win over
        everything else -- so with a question up, backspace inserted a
        literal "^?" instead of deleting, the arrows did nothing, and the
        prompt read "[y] yes  [a] always  [n] no  [q] stop: hello^?^?^C".
        A typo could not be corrected. Nothing here may claim these.
        """
        assert not self._binds(chat, key), (
            f"{key} is claimed by the layout; it belongs to the composer")

    def test_nothing_is_bound_to_every_key_at_once(self, chat):
        for binding in chat.app.key_bindings.bindings:
            names = [getattr(k, "value", str(k)) for k in binding.keys]
            assert "<any>" not in names, (
                "an <any> binding swallows backspace and Ctrl-C with it")

    def test_ctrl_c_gets_you_out_of_the_question(self, chat):
        """With the question swallowing Ctrl-C there was no way out of a
        permission prompt at all -- not the answer, not the key, not /quit.
        The process had to be killed."""
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[y] yes [n] no [q] stop:",
                         {"y": "yes", "n": "no", "q": "stop"}))
            await asyncio.sleep(0)
            self._press(chat, "c-c")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "q"

    def test_ctrl_c_gets_you_out_of_a_question_with_no_stop(self, chat):
        """An answer no branch matches, which every caller reads as abort."""
        async def go():
            pending = asyncio.ensure_future(chat.ask("?", {"y": "yes"}))
            await asyncio.sleep(0)
            self._press(chat, "c-c")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) not in ("y",)

    def test_ctrl_c_does_not_end_the_session(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.ask("?", {"y": "yes"}))
            await asyncio.sleep(0)
            self._press(chat, "c-c")
            await asyncio.wait_for(pending, 1)

        asyncio.run(go())
        assert not chat.app.is_done

    def test_enter_takes_the_default_where_there_is_one(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[k] keep [r] revert:", {"k": "keep", "r": "revert"},
                         default="k"))
            await asyncio.sleep(0)
            chat.buffer.validate_and_handle()
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "k"

    def test_enter_does_nothing_where_there_is_no_default(self, chat):
        """A permission prompt names none on purpose: a reflex press of
        enter must never be able to grant one."""
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("[y] yes [n] no [q] stop:",
                         {"y": "yes", "n": "no", "q": "stop"}))
            await asyncio.sleep(0)
            chat.buffer.validate_and_handle()
            await asyncio.sleep(0)
            done = pending.done()
            self._press(chat, "n")
            await asyncio.wait_for(pending, 1)
            return done

        assert asyncio.run(go()) is False

    def test_the_permission_prompt_names_no_default(self):
        """Checked at the call site, since that is where it would be added
        by someone tidying up."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.TerminalCallbacks._ask)
        question = source[source.index("chat.ask"):]
        assert "default" not in question[:400]

    def test_a_single_key_still_answers_from_an_empty_composer(self, chat):
        async def go():
            pending = asyncio.ensure_future(
                chat.ask("?", {"y": "yes", "a": "always"}))
            await asyncio.sleep(0)
            self._press(chat, "a")
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

    def test_every_picker_goes_through_the_one_door(self):
        """/model and /resume each opened a standalone picker of their own.

        A prompt_toolkit application cannot run inside another one, so in
        this layout /model drew its rows over the composer, took the bottom
        border with it and left the pinned header shredded behind it -- with
        the model list still on screen after it had gone. _pick is the only
        place allowed to reach for the standalone picker.
        """
        import ast
        import inspect

        from wynxo import cli as cli_module

        tree = ast.parse(inspect.getsource(cli_module))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name == "_pick":
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and \
                        ast.unparse(inner.func) == "choose":
                    raise AssertionError(
                        f"{node.name} opens its own picker; go through _pick, "
                        "or it will draw over the chat layout")

    def test_a_row_can_show_one_thing_and_return_another(self, chat):
        """/resume shows "2h ago" and has to hand back a session id."""
        async def go():
            pending = asyncio.ensure_future(chat.choose(
                "resume", [("just now", "2 msgs", "8c1bc3c0"),
                           ("2h ago", "9 msgs", "deadbeef")], "just now"))
            await asyncio.sleep(0)
            self._press(chat, "down")
            self._press(chat, "enter")
            return await asyncio.wait_for(pending, 1)

        assert asyncio.run(go()) == "deadbeef"

    def test_those_rows_show_the_label_not_the_value(self, chat):
        async def go():
            pending = asyncio.ensure_future(chat.choose(
                "resume", [("just now", "2 msgs", "8c1bc3c0")], "just now"))
            await asyncio.sleep(0)
            body = str(chat._transcript_fragments().value)
            self._press(chat, "escape")
            await asyncio.wait_for(pending, 1)
            return body

        import re

        # The highlighted row is coloured a character at a time, so the text
        # only reads back once the escapes are out of the way.
        body = re.sub(r"\x1b\[[0-9;]*m", "", asyncio.run(go()))
        assert "just now" in body and "2 msgs" in body
        assert "8c1bc3c0" not in body

    def test_a_rich_choice_survives_the_trip(self):
        """_pick takes the same rows the standalone picker does, so a command
        does not have to flatten its list to reach this layout."""
        import asyncio as _asyncio

        from wynxo.cli import Repl
        from wynxo.select import Choice

        seen = {}

        class _Chat:
            async def choose(self, title, options, current):
                seen["options"] = options
                seen["current"] = current
                return options[0][2]

        repl = Repl.__new__(Repl)
        repl.chat = _Chat()
        chosen = _asyncio.run(repl._pick(
            "model",
            [Choice(value="qwen:30b", label="qwen:30b", badge="tools",
                    hint="30B  256k ctx"),
             Choice(value="gemma:2b", label="gemma:2b", badge="no tools",
                    hint="2B")],
            "gemma:2b"))
        assert chosen == "qwen:30b"
        assert seen["options"][0] == ("qwen:30b", "tools  30B  256k ctx",
                                      "qwen:30b")
        assert seen["current"] == "gemma:2b"


class TestNoTestMayNeedAConsole:
    """A Windows runner has no console screen buffer, and prompt_toolkit
    raises from the constructor when it asks for one. Building a
    PromptSession or an Application in a test is therefore a Windows-only
    failure that nothing here can reproduce -- which is exactly how it got
    in twice.
    """

    def test_nothing_builds_a_prompt_session(self):
        import ast
        import pathlib

        for path in sorted(pathlib.Path("tests").glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and \
                        ast.unparse(node.func) == "PromptSession":
                    raise AssertionError(
                        f"{path.name}:{node.lineno} builds a PromptSession; "
                        "on a Windows runner that raises "
                        "NoConsoleScreenBufferError")


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


class TestCommandSuggestions:
    """A Buffer with a completer will happily compute suggestions and show
    none of them unless the layout contains a menu to float over it, which
    is why /mo… stopped offering /model when the composer moved here.
    """

    def test_the_layout_has_somewhere_to_show_them(self):
        from prompt_toolkit.layout.menus import CompletionsMenu

        chat = ChatUI()
        floats = chat.app.layout.container.floats
        assert any(isinstance(f.content, CompletionsMenu) for f in floats)

    def test_the_menu_follows_the_cursor(self):
        """Anchored to the caret rather than the corner, so it appears over
        the word being typed."""
        chat = ChatUI()
        menu = chat.app.layout.container.floats[0]
        assert menu.xcursor and menu.ycursor

    def test_the_completer_is_actually_attached(self):
        from wynxo.cli import CommandCompleter

        chat = ChatUI(completer=CommandCompleter(lambda: "."))
        assert chat.buffer.completer is not None

    def test_typing_a_prefix_offers_the_commands(self):
        from prompt_toolkit.document import Document

        from wynxo.cli import CommandCompleter

        completer = CommandCompleter(lambda: ".")
        found = [c.text for c in completer.get_completions(
            Document("/mo", len("/mo")), None)]
        assert "/model" in found and "/mode" in found


class TestThePickerIsAlive:
    def test_the_selected_row_is_coloured(self):
        chat = ChatUI()
        chat.picker = {"title": "effort",
                       "options": [("low", "quick"), ("max", "everything")],
                       "index": 1}
        rows = chat._picker_lines(80)
        assert "\x1b[38;2;" in rows[2], "the selection is not lit"

    def test_the_others_stay_dim(self):
        chat = ChatUI()
        chat.picker = {"title": "effort",
                       "options": [("low", "quick"), ("max", "everything")],
                       "index": 1}
        rows = chat._picker_lines(80)
        assert "\x1b[38;2;" not in rows[1]

    def test_the_colour_moves_over_time(self, monkeypatch):
        import time as _time

        chat = ChatUI()
        chat.picker = {"title": "t", "options": [("a", "")], "index": 0}
        monkeypatch.setattr(_time, "monotonic", lambda: 0.0)
        first = chat._picker_lines(40)[1]
        monkeypatch.setattr(_time, "monotonic", lambda: 5.0)
        assert chat._picker_lines(40)[1] != first
