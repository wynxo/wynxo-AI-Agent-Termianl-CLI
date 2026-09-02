"""What the terminal actually shows.

These exist because two rendering bugs shipped and neither was catchable by
the other tests: one only happened on a real terminal, and one turned every
colour code into visible garbage without failing anything.
"""

import inspect

import pytest

from wynxo.ui import UI, ActivityBar, CodeStreamer


class TestAnsiIsNotMangled:
    """prompt_toolkit's patch_stdout routes output through Vt100_Output.write,
    which replaces every ESC byte with "?" as an escape-injection guard. Under
    it, every colour code from rich and from the status lines rendered as
    literal "?[1;32m" on screen. raw=True uses write_raw instead."""

    def test_patch_stdout_is_used_in_raw_mode(self):
        from wynxo import cli

        source = inspect.getsource(cli.amain)
        assert "patch_stdout(raw=True)" in source, (
            "patch_stdout() without raw=True turns every escape code into "
            'literal "?[...m" text'
        )

    def test_prompt_toolkit_still_escapes_without_raw(self):
        """Pin the upstream behaviour this guards against, so the day it
        changes, this test says so rather than the fix silently being moot."""
        from prompt_toolkit.output.vt100 import Vt100_Output

        source = inspect.getsource(Vt100_Output.write)
        assert '"?"' in source or "'?'" in source


class TestCodeStreaming:
    """The earlier implementation printed a dim preview, then rewound the
    cursor to overwrite it. That crashed on rich 15 (no Control.clear_lines)
    and only on a real terminal, so every non-tty test passed."""

    def _render(self, chunks):
        ui = UI()
        streamer = CodeStreamer(ui)
        for chunk in chunks:
            streamer.feed(chunk)
        streamer.finish()

    def test_no_cursor_control_is_used(self):
        """Checked as code, not text -- the docstring explaining why this is
        avoided obviously mentions it."""
        import ast

        from wynxo import ui as ui_module

        tree = ast.parse(inspect.getsource(ui_module))
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        for forbidden in ("clear_lines", "console.control", "Control.move"):
            assert not any(forbidden in call for call in calls), forbidden
        # Control may be named, but only to recognise rich's own repaints:
        # a Live refresh reaches the console as print(Control(...)), and the
        # transcript has to know that such a print leaves nothing behind.
        # Recognising one is not the same as emitting one, so what is banned
        # is constructing it -- which is the act that moves the cursor.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "Control":
                raise AssertionError("ui.py constructs a Control")
        used = [node for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id == "Control"]
        checks = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and ast.unparse(node.func) == "isinstance"
                  and "Control" in ast.unparse(node)]
        assert len(used) <= len(checks) + 1, (
            "Control is referenced outside an isinstance check")

    def test_code_renders_once(self, capsys):
        self._render(["```python\n", "x = 1\n", "y = 2\n", "```\n"])
        out = capsys.readouterr().out
        assert out.count("x = 1") == 1
        assert out.count("y = 2") == 1
        assert "```" not in out

    def test_fences_never_reach_the_screen(self, capsys):
        self._render(["a\n", "```js\n", "let x = 1\n", "```\n", "b\n"])
        assert "```" not in capsys.readouterr().out

    def test_unclosed_block_still_prints(self, capsys):
        self._render(["```python\n", "z = 3\n"])
        assert "z = 3" in capsys.readouterr().out

    def test_language_aliases_normalise(self):
        from wynxo.ui import _language

        assert _language("py") == "python"
        assert _language("sh") == "bash"
        assert _language("") == "text"
        assert _language("rust") == "rust"

    def test_an_unknown_language_does_not_raise(self, capsys):
        self._render(["```nonsense-lang\n", "some text\n", "```\n"])
        assert "some text" in capsys.readouterr().out

    def test_code_survives_chunks_split_mid_token(self, capsys):
        self._render(["```py", "thon\n", "def f", "oo():\n", "    pa", "ss\n", "``", "`\n"])
        out = capsys.readouterr().out
        assert "def foo():" in out and "pass" in out
        assert "```" not in out

    def test_a_fence_split_one_backtick_at_a_time(self, capsys):
        """The worst case a real stream produces: a token boundary between
        every character of the marker."""
        self._render(list("`" "`" "`" "python\nvalue = 1\n" "`" "`" "`" "\n"))
        out = capsys.readouterr().out
        assert "value = 1" in out
        assert "`" not in out

    def test_an_inline_backtick_is_not_held_forever(self, capsys):
        """A line starting with a backtick waits only until it is clear it is
        not a fence -- otherwise the reply would stall on `like this`."""
        self._render(["`inline` at the start of a line\n"])
        assert "inline" in capsys.readouterr().out

    def test_backticks_mid_line_are_not_a_fence(self, capsys):
        """A chunk boundary can fall anywhere, so a segment may begin with
        three backticks without its line doing so."""
        self._render(["see ", "```", " in the docs\n"])
        out = capsys.readouterr().out
        assert "in the docs" in out


class TestInlineCodeInTheModelsProse:
    """`like this` reads as code in every chat window there is, and a local
    model writes it constantly. It was arriving as literal backticks.

    Toggled per character rather than matched per finished line, because a
    line is not finished when it is drawn -- and by the time it is, its
    characters have already gone to the terminal and cannot be restyled.
    """

    STRIP = r"\x1b\[[0-9;]*m"

    def _stream(self, text, with_bar=True, **kwargs):
        import io

        from rich.console import Console

        from wynxo.ui import ActivityBar, CodeStreamer

        ui = UI()
        ui.console = Console(file=io.StringIO(), force_terminal=True, width=70)
        if with_bar:
            ui.live_ok = False
            ui.bar = ActivityBar(ui, "low")
            ui.bar.set_lead = lambda line: None
        streamer = CodeStreamer(ui, **kwargs)
        for character in text:
            streamer.feed(character)
        streamer.finish()
        return ui.console.file.getvalue()

    def _plain(self, written):
        import re

        return re.sub(self.STRIP, "", written)

    @pytest.mark.parametrize("with_bar", [True, False])
    def test_the_backticks_are_not_shown(self, with_bar):
        written = self._plain(
            self._stream("call `fetch(url)` first\n", with_bar))
        assert "fetch(url)" in written
        assert "`" not in written

    @pytest.mark.parametrize("with_bar", [True, False])
    def test_the_span_is_coloured(self, with_bar):
        from wynxo.ui import CODE_SPAN

        written = self._stream("call `fetch(url)` first\n", with_bar)
        assert "\x1b[" in written, "no styling at all"
        # The colour is a real one, not whatever rich felt like.
        assert CODE_SPAN.lstrip("#")

    def test_it_costs_one_span_not_one_per_letter(self):
        """A span per character is what made a coloured line cost an escape
        pair per letter."""
        short = self._stream("a `xy` b\n").count("\x1b[")
        long = self._stream("a `xyxyxyxyxyxyxyxyxyxy` b\n").count("\x1b[")
        assert short == long

    def test_an_unpaired_backtick_stops_at_the_end_of_the_line(self):
        """Otherwise one stray mark colours the rest of the answer."""
        written = self._stream("a lone ` here\nand a plain line\n")
        tail = written.split("and a plain line")[-1]
        assert "\x1b[" not in tail.rstrip()[:-1] or tail.count("\x1b[") <= 1

    def test_a_file_being_written_keeps_its_backticks(self):
        """Inside a file's contents a backtick is just a character -- a
        shell script full of them must arrive intact."""
        written = self._plain(
            self._stream("echo `date`\n", literal=True, code=False))
        assert "`date`" in written

    def test_a_fenced_block_is_untouched(self):
        written = self._plain(
            self._stream("```sh\necho `date`\n```\n"))
        assert "`date`" in written

    def test_the_words_either_side_are_unharmed(self):
        written = self._plain(self._stream("use `x` and `y` now\n"))
        assert "use x and y now" in written

    @pytest.mark.parametrize("with_bar", [True, False])
    def test_bold_loses_its_asterisks(self, with_bar):
        written = self._plain(
            self._stream("this is **much** safer\n", with_bar))
        assert "this is much safer" in written

    @pytest.mark.parametrize("text", [
        "maths: 2 * 3 * 4 = 24",
        "a glob like *.py stays",
        "one lone * on its own",
        "trailing star at the end *",
    ])
    def test_a_single_asterisk_is_meant_literally(self, text):
        """Held for exactly one character to see whether a second follows,
        then written if none does."""
        written = self._plain(self._stream(text + "\n"))
        assert text in written

    def test_a_heading_loses_its_hashes(self):
        written = self._plain(self._stream("## What changed\nbody\n"))
        assert "What changed" in written
        assert "#" not in written

    @pytest.mark.parametrize("text", [
        "#notaheading stays",
        "issue #42 is fixed",
        "a # in the middle is fine",
        "####### seven is too many",
    ])
    def test_a_hash_that_is_not_a_heading_survives(self, text):
        written = self._plain(self._stream(text + "\n"))
        assert text in written

    def test_a_heading_is_styled(self):
        written = self._stream("# Summary\n")
        assert "\x1b[" in written

    def test_emphasis_does_not_leak_into_the_next_line(self):
        written = self._plain(
            self._stream("unclosed **bold here\nplain line follows\n"))
        assert "plain line follows" in written

    def test_a_file_being_written_keeps_its_asterisks_and_hashes(self):
        source = "# a comment\nx = 2 * 3\n"
        written = self._plain(
            self._stream(source, literal=True, code=False))
        assert "# a comment" in written
        assert "2 * 3" in written


class TestSomebodyElsesTextCannotDriveTheTerminal:
    """A terminal acts on escape sequences in what it is shown.

    ESC[2J clears the screen and takes the scrollback with it; ESC]0;
    renames the window. The model's answer, the contents of a file, the
    output of a command -- all of it reaches the screen, and none of it is
    text wynxo wrote. It does not take a hostile model: a log with colour
    codes in it, or a terminal recording, echoed back, is enough.

    rich neutralises this when handed a plain string and does not when
    handed a Text, a Syntax or a Markdown, which is most of what ui.py
    builds. Driving the real thing in a pty, the model's ESC[2J wiped the
    session and its ESC]0; renamed the window.
    """

    PAYLOAD = ("before \x1b[2J\x1b[H middle \x1b]0;PWNED\x07 after "
               "\x1b[31mred\x1b[0m end")

    def _console(self):
        """A console that scrubs nothing, so each helper is tested alone.

        SafeConsole strips control sequences out of everything rich renders,
        which would make this pass whether or not a single helper called
        sanitise -- and defence in depth is only defence if both layers
        work. It cannot be a bare rich Console either: the transcript's
        spacing lives on SafeConsole, so a plain one is missing methods the
        UI calls. Subclassing and dropping only the scrub leaves exactly the
        thing under test.
        """
        import io

        from rich.console import Console

        from wynxo.ui import SafeConsole

        class _NoScrub(SafeConsole):
            def _render_buffer(self, buffer):
                return Console._render_buffer(self, buffer)

        ui = UI()
        ui.console = _NoScrub(file=io.StringIO(), force_terminal=True,
                              width=80)
        return ui

    def _shown(self, ui):
        """What the terminal would receive, minus the colours rich itself
        chose. Anything left is something wynxo did not put there."""
        import re

        written = ui.console.file.getvalue()
        return re.sub(r"\x1b\[[0-9;]*m", "", written)

    @pytest.mark.parametrize("call", [
        lambda ui, text: ui.assistant_markdown(text),
        lambda ui, text: ui.tool_result("shell", True, "", text),
        lambda ui, text: ui.tool_output(text),
        lambda ui, text: ui.code(text),
        lambda ui, text: ui.diff("--- a/x\n+++ b/x\n+" + text),
    ])
    def test_no_escape_survives(self, call):
        ui = self._console()
        ui.show_thinking = True
        call(ui, self.PAYLOAD)
        assert "\x1b" not in self._shown(ui)

    def test_streamed_text_is_cleaned_too(self):
        from wynxo.ui import CodeStreamer

        ui = self._console()
        streamer = CodeStreamer(ui)
        for character in self.PAYLOAD:
            streamer.feed(character)
        streamer.finish()
        assert "\x1b" not in self._shown(ui)

    def test_the_words_are_still_there(self):
        """Stripping is not censoring: what the model actually said stays."""
        from wynxo.ui import CodeStreamer

        ui = self._console()
        streamer = CodeStreamer(ui)
        streamer.feed(self.PAYLOAD + "\n")
        streamer.finish()
        shown = self._shown(ui)
        for word in ("before", "middle", "after", "end"):
            assert word in shown

    def test_newlines_and_tabs_are_not_control_characters(self):
        from wynxo.ui import sanitise

        assert sanitise("a\nb\tc") == "a\nb\tc"

    @pytest.mark.parametrize("char", ["\x00", "\x07", "\x08", "\x1b",
                                      "\r", "\x7f"])
    def test_everything_else_in_the_range_goes(self, char):
        from wynxo.ui import sanitise

        assert sanitise(f"a{char}b") == "ab"


class TestOneStylePerLineNotPerLetter:
    """Streaming a character at a time made the styling per character too.

    Appending with a style creates a span per call, and there is now a call
    for every letter -- so a styled line went out as one escape pair per
    character, ten bytes of colour for each byte of text. All of it kept in
    the transcript, and re-rendered on every repaint.
    """

    def _stream(self, text, style):
        import io

        from rich.console import Console

        from wynxo.ui import ActivityBar, CodeStreamer

        ui = UI()
        ui.console = Console(file=io.StringIO(), force_terminal=True, width=80)
        ui.live_ok = False
        ui.bar = ActivityBar(ui, "low")
        ui.bar.set_lead = lambda line: None
        streamer = CodeStreamer(ui, style=style, code=False, literal=True)
        for character in text:
            streamer.feed(character)
        streamer.finish()
        return ui.console.file.getvalue()

    def test_a_styled_line_is_wrapped_once(self):
        written = self._stream("return get(url)\n", "#8a8a8a")
        assert written.count("\x1b[") <= 4

    def test_the_text_survives_intact(self):
        import re

        written = self._stream("return get(url)\n", "#8a8a8a")
        assert "return get(url)" in re.sub(r"\x1b\[[0-9;]*m", "", written)

    def test_an_unstyled_line_carries_no_escapes_at_all(self):
        assert "\x1b[" not in self._stream("plain words here\n", "")

    def test_it_does_not_grow_with_the_line(self):
        """The count is per line, not per character."""
        short = self._stream("ab\n", "#8a8a8a").count("\x1b[")
        long = self._stream("a" * 60 + "\n", "#8a8a8a").count("\x1b[")
        assert short == long


class TestPinnedBar:
    def test_bar_is_exactly_one_line(self):
        ui = UI()
        ui.width = 80
        bar = ActivityBar(ui, "medium")
        bar.update(tokens=42)
        assert "\n" not in bar._render().plain

    def test_bar_fills_the_terminal_width(self):
        for width in (40, 60, 80, 120, 200):
            ui = UI()
            ui.width = width
            bar = ActivityBar(ui, "medium")
            bar.update(activity="writing", tokens=1234)
            assert bar._render().cell_len == width, f"at {width}"

    def test_tokens_are_always_shown(self):
        """The live counter is the point; it must never be the thing dropped."""
        for width in (40, 50, 72, 100, 160):
            ui = UI()
            ui.width = width
            bar = ActivityBar(ui, "medium", "^O thinking  ^T detail")
            bar.update(activity="editing", detail="a/long/path/name.py", tokens=777)
            assert "777 tok" in bar._render().plain, f"dropped at {width}"

    def test_counter_advances_per_chunk(self):
        bar = ActivityBar(UI(), "low")
        for _ in range(5):
            bar.add_token()
        assert bar.tokens == 5

    @pytest.mark.parametrize("width", [30, 45, 80, 200])
    def test_never_wraps(self, width):
        ui = UI()
        ui.width = width
        bar = ActivityBar(ui, "ultra", "^O thinking  ^T detail")
        bar.update(activity="verifying", detail="round 2/4", tokens=99999)
        assert bar._render().cell_len <= width


class TestHeader:
    def test_the_header_is_a_name_over_a_detail_line(self, capsys):
        """Five facts joined by dots is a dashboard row, not a header: there
        is nothing in it to read first. The name carries the weight and the
        settings sit under it.

        No rule under it any more. A full-width line of ─ is the loudest
        thing a terminal can draw and it was being spent on a separator
        between the header and a blank line -- the whitespace was already
        doing the separating, and doing it more quietly.
        """
        UI().banner("qwen3-coder:30b", "http://127.0.0.1:11434", "medium",
                    "/tmp/p")
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == 1, lines
        assert lines[0].startswith("wynxo")
        assert "qwen3-coder:30b" in lines[0] and "/tmp/p" in lines[0]
        assert "─" not in lines[0]

    def test_the_companion_is_not_on_the_identity_line(self, capsys):
        """The character belongs to the work, not to the title.

        It stood here for a pass -- three rows, the cat beside a stacked
        identity -- which meant you were looking at a mascot during every
        minute the agent was idle, which is most of them. It appears in the
        live region while a task runs and goes when the task does.
        """
        from wynxo.pet import Pet

        UI().banner("m", "http://127.0.0.1:11434", "medium", "/tmp/p",
                    pet=Pet(), greeting="hello")
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == 1, lines
        assert "hello" not in lines[0]
        assert set(lines[0]) & set("▀▄█") == set()

    def test_narrow_header_drops_parts_rather_than_truncating(self, capsys):
        ui = UI()
        ui.width = 44
        ui.banner("qwen3-coder:30b", "http://192.168.1.50:11434", "medium",
                  "/home/u/code/project")
        out = capsys.readouterr().out
        assert "qwen3-coder:30b" in out
        assert "192.168.1.50" not in out, "the server should go before the model does"

    def test_no_stray_escape_codes_in_plain_mode(self, capsys):
        UI().banner("m", "http://127.0.0.1:11434", "low", "/tmp/p")
        assert "?[" not in capsys.readouterr().out

    def test_the_header_carries_no_settings_at_all(self, capsys):
        """Neither the effort level nor the server.

        "███   ultra" spent four cells of solid block on a word that was
        already beside it. Then the word went too: a header is read once,
        and a setting you can change at any moment is not something you
        need told once -- it is on the status line under the prompt, which
        is where the things that change live. What is left is the pair of
        facts you cannot get at a glance anywhere else."""
        from wynxo.ui import Glyphs

        ui = UI()
        ui.g = Glyphs(True)
        ui.banner("m", "http://127.0.0.1:11434", "ultra", "/tmp/p")
        out = capsys.readouterr().out
        assert "ultra" not in out
        assert "127.0.0.1" not in out
        assert "█" not in out
        assert "m" in out and "/tmp/p" in out


class TestAMessageStaysInItsOwnColumn:
    """rich wraps a Text at the console edge and starts the next line at
    column zero, so any message longer than the terminal is wide fell out
    from under its own marker and ran into the left edge. The lines this
    affects carry the warnings and errors -- the ones a person actually
    stops to read."""

    LONG = ("The model sent back an empty answer. Usually that means its "
            "chat template does not fit the prompt wynxo builds, or the "
            "conversation has outgrown the context window.")

    def _lines(self, method, message, width=70):
        import io

        from rich.console import Console

        ui = UI()
        ui.width = width
        ui.console = Console(file=io.StringIO(), force_terminal=False, width=width)
        getattr(ui, method)(message)
        return [line for line in ui.console.file.getvalue().split("\n") if line.strip()]

    def test_a_long_warning_wraps_onto_more_than_one_line(self):
        assert len(self._lines("warn", self.LONG)) > 1

    def test_every_line_after_the_first_is_indented(self):
        lines = self._lines("warn", self.LONG)
        assert lines[0].startswith("! ")
        for line in lines[1:]:
            assert line.startswith("  "), line
            assert not line.lstrip().startswith("!"), \
                "the marker belongs on the first line only"

    def test_the_continuation_sits_under_the_first_word(self):
        lines = self._lines("warn", self.LONG)
        head = len(lines[0]) - len(lines[0].lstrip())
        assert lines[0][head:head + 2] == "! "
        for line in lines[1:]:
            assert len(line) - len(line.lstrip()) == head + 2, line

    def test_nothing_overflows_the_terminal(self):
        for width in (40, 60, 70, 100):
            for line in self._lines("warn", self.LONG, width=width):
                assert len(line) <= width, (width, line)

    def test_every_kind_of_message_lines_up_the_same_way(self):
        """The invariant, stated once for all three: a continuation line
        begins in the column the first line's *text* begins in. For a
        message with a marker that leaves the marker alone in its column;
        for one without, the block simply stays square."""
        for method in ("warn", "info", "success"):
            lines = self._lines(method, self.LONG)
            assert len(lines) > 1, method
            column = lines[0].index("The model sent back")
            for line in lines[1:]:
                assert len(line) - len(line.lstrip()) == column, (method, line)

    def test_a_short_message_is_left_alone(self):
        assert self._lines("warn", "not a git checkout") == ["! not a git checkout"]

    def test_the_marker_gets_the_bold(self):
        """The marker carries the emphasis; the words keep the status
        colour. A bold ! reads as a marker, plain text as the message."""
        import io

        from rich.console import Console

        ui = UI()
        ui.width = 70
        ui.console = Console(file=io.StringIO(), force_terminal=True, width=70)
        ui.warn("not a git checkout")
        out = ui.console.file.getvalue()
        marker = out.index("!")
        assert "\x1b[1" in out[:marker], "the bold escape must precede the marker"
        import re

        plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert "! not a git checkout" in plain, "the marker keeps its column"

    def test_a_message_keeps_its_own_line_breaks(self):
        lines = self._lines("info", "first line\nsecond line")
        assert len(lines) == 2
        assert lines[0].strip() == "first line"
        assert lines[1].strip() == "second line"

    def test_a_very_narrow_terminal_still_produces_something(self):
        # No crash, no zero-width wrap loop.
        assert self._lines("warn", self.LONG, width=12)
