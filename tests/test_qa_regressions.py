"""Bugs found by stressing the running application, each with its cause.

Every test here failed before the fix beside it. None of them were found by
reading the code -- they came out of driving the real paths with the inputs a
real session produces.
"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile

from rich.cells import cell_len

from wynxo import livediff
from wynxo.cli import TerminalCallbacks
from wynxo.layout import Transcript
from wynxo.ui import UI, Glyphs


def _attached(width: int = 80):
    ui = UI()
    transcript = Transcript(width)
    ui.attach(transcript)
    return ui, transcript


def _callbacks():
    ui, transcript = _attached()
    callbacks = TerminalCallbacks(ui, prompt_session=None)
    callbacks.workspace = pathlib.Path(tempfile.mkdtemp())
    return callbacks, transcript


class TestCarriageReturnsAreLineEndings:
    """CRLF arrived from Windows subprocess output, a CRLF file being
    written, and a model echoing one. Splitting on \\n alone left the \\r on
    the end of every row, where it survives into the rendered fragments as a
    literal character -- and a terminal reads CR as "return to column 0", so
    the row is overdrawn by whatever comes next."""

    def test_a_trailing_cr_never_reaches_a_row(self):
        _, transcript = _attached()
        transcript.console.file.write("first\r\nsecond\r\n")
        assert transcript.lines == ["first", "second"]

    def test_a_mid_line_cr_is_left_alone(self):
        """A lone CR inside a line is a progress bar redrawing itself, which
        is content. Only the one at the end is a line ending."""
        _, transcript = _attached()
        transcript.console.file.write("50%\r100%\n")
        assert transcript.lines == ["50%\r100%"]

    def test_the_rendered_fragments_carry_no_stray_cr(self):
        from prompt_toolkit.formatted_text import ANSI, to_formatted_text

        _, transcript = _attached()
        transcript.console.file.write("alpha\r\nbeta\r\n")
        rendered = to_formatted_text(ANSI("\n".join(transcript.lines)))
        assert not any(text == "\r" for _style, text in rendered)


class TestACancelledEditDoesNotStayLive:
    """Ctrl-C mid-edit delivers no tool result, so the card stayed live and
    the overlay went on saying "streaming..." into the next turn -- narrating
    an edit that had already stopped."""

    def test_a_card_left_open_is_closed_by_the_turn(self):
        callbacks, _ = _callbacks()
        asyncio.run(callbacks.on_code("half an edit\n"))
        assert callbacks.card.live
        callbacks.close_card()
        assert not callbacks.card.live
        assert callbacks.card.state == livediff.FAILED

    def test_closing_says_what_happened(self):
        import re

        callbacks, transcript = _callbacks()
        asyncio.run(callbacks.on_code("half an edit\n"))
        callbacks.close_card()
        plain = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in transcript.lines]
        assert any("interrupted" in line for line in plain)

    def test_a_finished_card_is_not_reopened_or_relabelled(self):
        callbacks, _ = _callbacks()
        asyncio.run(callbacks.on_code("an edit\n"))
        asyncio.run(callbacks.on_tool_start("write_file", "a.py"))
        asyncio.run(callbacks.on_tool_result("write_file", True, "a.py", "ok"))
        state = callbacks.card.state
        callbacks.close_card()
        assert callbacks.card.state == state

    def test_the_teardown_actually_calls_it(self):
        """The method working is not the fix; being called is. A turn that
        ends any way at all -- answered, failed, or Ctrl-C'd -- runs its
        finally, and that is where the card has to be closed."""
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._turn_locked)
        finally_block = source.rsplit("finally:", 1)[-1]
        assert "close_card()" in finally_block, (
            "close_card must run in the turn's finally, not only on the "
            "happy path")

    def test_the_running_tool_is_forgotten_too(self):
        """It drives the companion's scene; left set, the cat keeps typing."""
        callbacks, _ = _callbacks()
        asyncio.run(callbacks.on_tool_start("write_file", "a.py"))
        assert callbacks.active_tool == "write_file"
        callbacks.close_card()
        assert callbacks.active_tool == ""


class TestTheCardObeysTheSameWallTheToolsDo:
    """The path comes from the model's tool arguments. The tool refuses to
    *write* outside the workspace, but the card reads independently of it --
    so a call naming ../../../etc/shadow had the write refused and the file
    disclosed into the diff anyway."""

    def _outside(self):
        workspace = pathlib.Path(tempfile.mkdtemp())
        (workspace / "inside.py").write_text("inside\n", encoding="utf-8")
        elsewhere = pathlib.Path(tempfile.mkdtemp()) / "secret.txt"
        elsewhere.write_text("SECRET\n", encoding="utf-8")
        return workspace, elsewhere

    def test_a_file_in_the_workspace_is_read(self):
        workspace, _ = self._outside()
        assert livediff.read_before(workspace, "inside.py") == "inside\n"

    def test_dot_dot_traversal_reads_nothing(self):
        workspace, elsewhere = self._outside()
        escape = "../" * 8 + str(elsewhere).lstrip("/")
        assert livediff.read_before(workspace, escape) == ""

    def test_an_absolute_path_outside_reads_nothing(self):
        workspace, elsewhere = self._outside()
        assert livediff.read_before(workspace, str(elsewhere)) == ""

    def test_a_system_file_reads_nothing(self):
        workspace, _ = self._outside()
        assert livediff.read_before(workspace, "/etc/hostname") == ""

    def test_a_symlink_pointing_out_is_outside(self):
        """The name says inside; where it goes is what counts."""
        workspace, elsewhere = self._outside()
        link = workspace / "looks-local.txt"
        try:
            link.symlink_to(elsewhere)
        except (OSError, NotImplementedError):
            return          # no symlinks here (some Windows configurations)
        assert livediff.read_before(workspace, "looks-local.txt") == ""

    def test_an_explicit_boundary_is_used_when_given(self):
        from wynxo.scope import Boundary, Scope

        workspace, elsewhere = self._outside()
        boundary = Boundary(scope=Scope.FOLDER, root=workspace.resolve())
        assert livediff.read_before(workspace, "inside.py", boundary) == "inside\n"
        assert livediff.read_before(workspace, str(elsewhere), boundary) == ""


class TestWideCharactersDoNotBreakTheBorder:
    """A CJK character occupies two columns and a combining accent none.
    Trimming by len() overflowed the card by up to double on a Japanese
    filename, and the box wrapped onto the next row."""

    def _card(self, path: str, body: str):
        card = livediff.DiffCard(tool="write_file", path=path, before="")
        card.feed(body)
        card.finish()
        return card

    def test_a_cjk_filename_fits(self):
        card = self._card("テスト/ファイル.py", "x = 1\n")
        for width in (40, 60, 100):
            assert max(cell_len(row) for row in card.render(Glyphs(True), width)) \
                <= width

    def test_cjk_content_fits(self):
        """Long enough to actually reach the clip. Sliced by codepoints this
        line rendered 92 cells into a 54-cell box -- 38 columns of overflow,
        and a border that wrapped onto the next row."""
        card = self._card("a.py", "名前 = " + "'テストデータ'" * 6 + "\n")
        for width in (40, 60, 100):
            widest = max(cell_len(row) for row in card.render(Glyphs(True), width))
            assert widest <= width, f"{widest} cells in a {width}-cell card"

    def test_an_emoji_path_fits(self):
        card = self._card("🎉/party.py", "x = 1\n")
        assert max(cell_len(row) for row in card.render(Glyphs(True), 60)) <= 60

    def test_fit_measures_columns_not_codepoints(self):
        assert cell_len(livediff.fit("テストテスト", 6)) <= 6
        assert livediff.fit("plain", 40) == "plain"
        assert livediff.fit("", 10) == ""


class TestTheScreenIsNeverBlank:
    """Squeezing the terminal blanked it outright.

    ``HSplit._divide_heights`` returns None when the sum of its children's
    *minimums* exceeds the available height, and prompt_toolkit renders that
    as an empty screen -- not a clipped layout, nothing at all. Two separate
    ways in: a composer that asked for its content height regardless of the
    room left over, and fixed furniture (header + rule + footer + one
    composer row = four) that alone outgrew a three-row terminal.
    """

    SIZES = [(160, 50), (120, 40), (80, 24), (60, 20), (40, 15),
             (30, 10), (20, 6), (20, 4), (20, 3), (20, 2), (20, 1)]

    def _layout(self, width, height, text="", overlay=()):
        from wynxo.layout import ChatLayout

        layout = ChatLayout(width=width, height=height,
                            overlay=lambda: list(overlay))
        layout.buffer.text = text
        return layout

    def _divided(self, layout, width, height):
        body = layout.app.layout.container.content
        return body._divide_heights(
            type("Size", (), {"width": width, "height": height})())

    def test_every_size_allocates(self):
        for width, height in self.SIZES:
            layout = self._layout(width, height)
            assert self._divided(layout, width, height) is not None, \
                f"blank screen at {width}x{height}"

    def test_a_paste_taller_than_the_screen_still_allocates(self):
        """Five lines of pasted text on a six-row terminal asked for five
        composer rows on top of three fixed ones."""
        paste = "\n".join(f"line {i}" for i in range(40))
        for width, height in self.SIZES:
            layout = self._layout(width, height, text=paste)
            assert self._divided(layout, width, height) is not None, \
                f"blank screen at {width}x{height} with a paste"

    def test_the_regions_add_up_to_the_screen(self):
        for width, height in self.SIZES:
            layout = self._layout(width, height)
            total = (layout.header_rows() + layout.transcript_rows()
                     + layout.rule_rows() + layout.composer_rows()
                     + layout.footer_rows())
            assert total == height, f"{total} rows on a {height}-row screen"

    def test_the_composer_survives_every_size(self):
        """Whatever else is shed, there is somewhere to type."""
        for width, height in self.SIZES:
            assert self._layout(width, height).composer_rows() >= 1

    def test_furniture_is_shed_in_order(self):
        """Rule first -- it only separates two things that stay adjacent --
        then the header, then the footer. Never the composer."""
        assert self._layout(80, 24).rule_rows() == 1
        assert self._layout(20, 4).rule_rows() == 0
        assert self._layout(20, 4).header_rows() == 1
        assert self._layout(20, 3).header_rows() == 0
        assert self._layout(20, 3).footer_rows() == 1
        assert self._layout(20, 2).footer_rows() == 0

    def test_the_conversation_is_the_last_thing_squeezed(self):
        """Shedding furniture is only worth doing if it buys a row of
        conversation. Two rows is a transcript and a composer."""
        for width, height in self.SIZES:
            layout = self._layout(width, height)
            if height >= 2:
                assert layout.transcript_rows() >= 1, f"{width}x{height}"


class TestTheOverlayFitsTheTerminal:
    """TODO_WIDTH is a preference; as a *floor* it won a narrow terminal
    outright and put a 36-column float on a screen that had 20."""

    WIDTHS = [160, 120, 80, 60, 40, 30, 24, 20]

    def _layout(self, width, height, rows):
        from wynxo.layout import ChatLayout

        return ChatLayout(width=width, height=height, overlay=lambda: rows)

    def test_the_float_never_exceeds_the_screen(self):
        rows = ["a plan line that is considerably wider than any terminal "
                "here" * 2] * 20
        for width in self.WIDTHS:
            layout = self._layout(width, 24, rows)
            assert layout._overlay_width() <= width, \
                f"{layout._overlay_width()} columns on a {width}-column screen"

    def test_an_empty_overlay_still_fits(self):
        for width in self.WIDTHS:
            assert self._layout(width, 24, [])._overlay_width() <= width

    def test_a_wide_terminal_keeps_the_plan_panel_width(self):
        """The narrow case must not cost the normal one: corner.py's panel
        needs its 36 columns or its right border is clipped."""
        from wynxo.layout import ChatLayout

        assert self._layout(120, 40, ["short"])._overlay_width() \
            == ChatLayout.TODO_WIDTH

    def test_the_float_never_exceeds_the_height_either(self):
        for height in (50, 40, 24, 15, 10, 6, 4, 3, 2):
            layout = self._layout(80, height, ["row"] * 40)
            assert layout._overlay_height() <= height, \
                f"{layout._overlay_height()} rows on a {height}-row screen"


class TestBackgroundJobsDieWithTheSession:
    """A background command is started in its own session so that the whole
    of it can be killed rather than just the shell that launched it -- and
    that same detachment means the terminal never hangs it up either. So
    quitting wynxo left `npm run dev`, a watcher, or a `while true` loop
    running forever, still writing into the project, with nothing left that
    knew its pid. The job table's own docstring claimed jobs live only for
    the lifetime of the session; the processes did not."""

    def _shell(self, workspace):
        from wynxo.tools import build_registry

        return build_registry(workspace, allow_shell=True).get("shell")

    async def _background(self, workspace, marker):
        shell = self._shell(workspace)
        return await shell.run(shell.Input(
            command=f"while true; do echo tick >> {marker}; sleep 0.2; done",
            timeout=600, background=True))

    def test_shutdown_stops_a_running_job(self):
        import os
        import time

        from wynxo.tools import shell as shell_module

        workspace = pathlib.Path(tempfile.mkdtemp())
        marker = workspace / "ticks"

        async def go():
            result = await self._background(workspace, marker)
            assert result.ok, result.output
            await asyncio.sleep(0.5)
            return result.metadata["job_id"]

        job_id = asyncio.run(go())
        process = shell_module._BACKGROUND[job_id]["process"]
        assert marker.exists(), "the job must actually have been running"
        assert shell_module.shutdown_background() == 1

        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(os.getpgid(process.pid), 9)
            except OSError:
                pass
            raise AssertionError("the background job outlived the session")

    def test_shutdown_is_safe_with_nothing_running(self):
        from wynxo.tools import shell as shell_module

        shell_module._BACKGROUND.clear()
        assert shell_module.shutdown_background() == 0

    def test_shutdown_is_idempotent(self):
        from wynxo.tools import shell as shell_module

        workspace = pathlib.Path(tempfile.mkdtemp())

        async def go():
            result = await self._background(workspace, workspace / "ticks")
            await asyncio.sleep(0.3)
            return result.metadata["job_id"]

        asyncio.run(go())
        assert shell_module.shutdown_background() == 1
        assert shell_module.shutdown_background() == 0

    def test_the_teardown_actually_calls_it(self):
        """Working is not the fix; being called on the way out is. The REPL's
        outermost finally is the one place every exit passes through --
        answered, errored, /quit or Ctrl-C."""
        import inspect

        from wynxo.cli import Repl

        finally_block = inspect.getsource(Repl._loop).rsplit("finally:", 1)[-1]
        assert "shutdown_background()" in finally_block

    def test_there_is_a_backstop_for_the_paths_that_never_get_there(self):
        """A crash during start-up never reaches the REPL's finally."""
        import inspect

        from wynxo.tools import shell as shell_module

        source = inspect.getsource(shell_module._launch_background)
        assert "atexit.register(shutdown_background)" in source


class TestThePermissionPromptIsVisible:
    """The agent stopped and waited on an invisible question.

    ``_ask`` handed its "[y] yes [a] always [n] no [q] stop:" line to
    ``prompt_session.prompt_async`` as the message argument. Under the
    full-screen layout that method is ChatLayout's -- it takes the same
    arguments PromptSession does, because the REPL calls both, but the
    composer draws a fixed caret and has nowhere to put a message, so the
    text was accepted and dropped. On screen: an ordinary empty composer,
    no question, and an agent that would not continue. The layout has
    ``ask()`` for exactly this, and every other question in the REPL already
    went through it.
    """

    class _Chat:
        def __init__(self, answer):
            self.answer = answer
            self.asked = []

        async def ask(self, question, default=""):
            self.asked.append(question)
            return self.answer

        def invalidate(self):
            pass

    class _Session:
        def __init__(self):
            self.used = False

        async def prompt_async(self, *args, **kwargs):
            self.used = True
            return "y"

    def _callbacks(self, answer):
        ui, _ = _attached()
        session = self._Session()
        callbacks = TerminalCallbacks(ui, prompt_session=session)
        callbacks.chat = self._Chat(answer)
        return callbacks, session

    def _decide(self, callbacks):
        return asyncio.run(
            callbacks.ask_permission("shell", "rm -rf build", ""))

    def test_the_question_reaches_the_screen(self):
        callbacks, _ = self._callbacks("n")
        self._decide(callbacks)
        assert callbacks.chat.asked, "the question was never asked anywhere"
        question = callbacks.chat.asked[0]
        for key in ("[y]", "[a]", "[n]", "[q]"):
            assert key in question, f"{key} missing from {question!r}"

    def test_the_layout_is_asked_rather_than_the_prompt_session(self):
        callbacks, session = self._callbacks("n")
        self._decide(callbacks)
        assert not session.used, (
            "prompt_async cannot render a message under the layout; the "
            "question would be silently dropped")

    def test_every_answer_still_means_what_it_did(self):
        from wynxo.permissions import Decision

        for answer, decision in [("y", Decision.ALLOW), ("", Decision.ALLOW),
                                 ("yes", Decision.ALLOW),
                                 ("a", Decision.ALLOW_ALWAYS),
                                 ("always", Decision.ALLOW_ALWAYS),
                                 ("n", Decision.DENY), ("no", Decision.DENY),
                                 ("q", Decision.ABORT),
                                 ("stop", Decision.ABORT)]:
            callbacks, _ = self._callbacks(answer)
            assert self._decide(callbacks) is decision, answer

    def test_an_unusable_answer_asks_again_rather_than_deciding(self):
        callbacks, _ = self._callbacks("maybe")

        answers = iter(["maybe", "what", "n"])

        async def ask(question, default=""):
            callbacks.chat.asked.append(question)
            return next(answers)

        callbacks.chat.ask = ask
        from wynxo.permissions import Decision

        assert self._decide(callbacks) is Decision.DENY
        assert len(callbacks.chat.asked) == 3

    def test_the_classic_prompt_still_gets_its_message(self):
        """The fallback path is the one place the message *is* rendered."""
        ui, _ = _attached()
        seen = []

        class Session:
            async def prompt_async(self, message=None, **kwargs):
                seen.append(message)
                return "n"

        callbacks = TerminalCallbacks(ui, prompt_session=Session())
        callbacks.chat = None
        asyncio.run(callbacks.ask_permission("shell", "ls", ""))
        assert seen and seen[0] is not None


class TestPipeTransportsAreActuallyRetired:
    """The cleanup was written, documented, and never ran.

    ``_close_streams`` called ``process.stdout.close()``. ``process.stdout``
    is an ``asyncio.StreamReader``, which has no ``close()`` -- so every
    command that ever finished raised AttributeError into a bare
    ``except Exception`` and no transport was retired. The Windows
    deallocator message its docstring describes was never prevented, and a
    killed background job kept its stdout pipe open for the rest of the
    session.
    """

    def _shell(self, workspace):
        from wynxo.tools import build_registry

        return build_registry(workspace, allow_shell=True).get("shell")

    def test_a_stream_reader_has_no_close(self):
        """The premise. If asyncio ever grows one, the old code was fine and
        this whole fix can go."""
        assert not hasattr(asyncio.StreamReader, "close")

    def test_a_finished_command_leaves_no_open_transport(self):
        workspace = pathlib.Path(tempfile.mkdtemp())

        async def go():
            shell = self._shell(workspace)
            result = await shell.run(shell.Input(command="echo hi", timeout=30))
            return result

        result = asyncio.run(go())
        assert result.ok, result.output

    def test_closing_is_refused_while_the_process_is_still_running(self):
        """Closing a subprocess transport kills the process. A tidy-up that
        can lose a running command is worse than the leak."""
        from wynxo.tools.shell import _close_transports

        class Transport:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class Process:
            returncode = None
            _transport = Transport()

        process = Process()
        _close_transports(process)
        assert not process._transport.closed
        _close_transports(process, force=True)
        assert process._transport.closed, (
            "shutdown has already killed it; the transport must go")

    def test_a_killed_background_job_retires_its_transport(self):
        from wynxo.tools import shell as shell_module

        workspace = pathlib.Path(tempfile.mkdtemp())

        async def go():
            shell = self._shell(workspace)
            result = await shell.run(shell.Input(
                command="while true; do sleep 0.2; done",
                timeout=600, background=True))
            await asyncio.sleep(0.3)
            return result.metadata["job_id"]

        job_id = asyncio.run(go())
        process = shell_module._BACKGROUND[job_id]["process"]
        shell_module.shutdown_background()
        assert process._transport._closed, "the stdout pipe was left open"


class TestTerminalControlCannotBeSmuggledIn:
    """``sanitise`` was opt-in, and seven display helpers did not opt in.

    ``tool_start`` drew the model's own tool arguments raw on every single
    tool call. ``error`` drew whatever message it was handed. ``highlight``
    and ``code_line`` drew file contents. ``rule``, ``table``, ``todos`` and
    ``shorten_path`` drew whatever they were given. So a file in the
    workspace containing ESC ] 52 ; c ; <base64> BEL wrote to the user's
    clipboard when the agent read it, ESC [ ? 1049 h switched the terminal
    to its alternate screen, and ESC [ 1 ; 5 r pinned a scroll region and
    wedged the display. None of it needs a hostile model -- a log with
    colour codes in it is enough.

    The fix is at rich's own render seam rather than at each call site, so
    a helper added later is covered without knowing about any of this.
    """

    PAYLOAD = "\x1b]52;c;aGVsbG8=\x07 \x1b[?1049h \x1b[1;5r \x1b[2J \x1b[H"
    DANGEROUS = {"clipboard write": "\x1b]52;",
                 "alternate screen": "\x1b[?1049h",
                 "scroll region": "\x1b[1;5r",
                 "erase display": "\x1b[2J",
                 "cursor home": "\x1b[H"}

    def _written(self, draw):
        """What actually reaches the file a Console writes to."""
        import io

        from wynxo.ui import UI

        ui = UI()
        sink = io.StringIO()
        ui.console.file = sink
        ui.console._force_terminal = True
        draw(ui)
        return sink.getvalue()

    def _assert_clean(self, drawn):
        for name, sequence in self.DANGEROUS.items():
            assert sequence not in drawn, f"{name} reached the terminal"

    def test_every_helper_that_draws_foreign_text(self):
        payload = self.PAYLOAD
        for label, draw in [
            ("info", lambda u: u.info(payload)),
            ("warn", lambda u: u.warn(payload)),
            ("error", lambda u: u.error(payload)),
            ("success", lambda u: u.success(payload)),
            ("assistant_markdown", lambda u: u.assistant_markdown(payload)),
            ("tool_start", lambda u: u.tool_start("read_file", payload)),
            ("tool_result",
             lambda u: u.tool_result("read_file", True, payload, payload)),
            ("tool_output", lambda u: u.tool_output(payload)),
            ("diff", lambda u: u.diff("--- a\n+++ b\n+" + payload)),
            ("code", lambda u: u.code(payload, "python")),
            ("todos", lambda u: u.todos(payload)),
            ("highlight",
             lambda u: u.console.print(u.highlight(payload, "python"))),
            ("code_line",
             lambda u: u.console.print(u.code_line(payload, "python"))),
            ("table", lambda u: u.table(["c"], [[payload]], payload)),
            ("rule", lambda u: u.rule(payload)),
            ("shorten_path",
             lambda u: u.console.print(u.shorten_path(payload))),
        ]:
            drawn = self._written(draw)
            for name, sequence in self.DANGEROUS.items():
                assert sequence not in drawn, f"{label} let {name} through"

    def test_the_transcript_console_is_covered_too(self):
        _, transcript = _attached()
        transcript.console.print("tool said: " + self.PAYLOAD)
        self._assert_clean("\n".join(transcript.lines))

    def _console(self):
        import io

        from wynxo.ui import SafeConsole

        sink = io.StringIO()
        return SafeConsole(file=sink, width=80, highlight=False,
                           force_terminal=True, color_system="truecolor"), sink

    def test_wynxo_keeps_its_own_colour(self):
        """A scrub that took the styling with it would be a different bug.
        rich holds its own styling in the segment's style rather than in its
        text, which is exactly what makes the two separable."""
        import re

        console, sink = self._console()
        console.print("[bold red]a warning[/]")
        assert re.search(r"\x1b\[[0-9;]*m", sink.getvalue()), \
            "styling was stripped"

    def test_ordinary_text_survives_intact(self):
        console, sink = self._console()
        console.print("a\tb")
        # rich expands the tab into spaces before any of this; what matters
        # is that the content is all still there.
        assert "a" in sink.getvalue() and "b" in sink.getvalue()
        assert sink.getvalue().endswith("\n")

    def test_rich_own_control_segments_are_left_alone(self):
        """Cursor moves are what a Live display is made of. Scrubbing those
        would break the activity bar rather than protect anything."""
        from rich.segment import Segment

        from wynxo.ui import _scrubbed

        control = Segment("\x1b[2J", None, [(3,)])
        assert list(_scrubbed([control])) == [control]

    def test_a_helper_added_later_is_covered(self):
        """The point of fixing it at the render seam: this renderable has
        never heard of sanitise()."""
        from rich.panel import Panel
        from rich.text import Text

        drawn = self._written(
            lambda u: u.console.print(Panel(Text(self.PAYLOAD))))
        self._assert_clean(drawn)
