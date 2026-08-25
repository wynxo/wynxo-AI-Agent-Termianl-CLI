"""Watching a file being written, and text arriving a character at a time.

Two separate complaints, one theme: the screen should show work happening.
A tool call used to show nothing at all between "the model started writing"
and "the file exists", which on a slow local model is long enough to wonder
whether anything is running.
"""

from __future__ import annotations

import re

import pytest

from wynxo.parsing import LiveContentFilter, partial_string_value


class TestReadingHalfWrittenJson:
    """A streaming tool call never parses until it is finished, which is
    exactly when it stops being interesting."""

    def test_the_value_so_far_is_returned(self):
        assert partial_string_value(
            '{"name":"write_file","arguments":{"path":"a.py",'
            '"content":"def add(a, b):') == "def add(a, b):"

    def test_escapes_are_decoded(self):
        assert partial_string_value(
            '{"content":"one\\ntwo\\t3') == "one\ntwo\t3"

    def test_a_finished_value_stops_at_the_quote(self):
        assert partial_string_value('{"content":"done"}') == "done"

    def test_nothing_until_the_value_starts(self):
        """Showing something on the strength of a key name alone would put
        the previous call's tail on screen."""
        assert partial_string_value('{"content":') == ""
        assert partial_string_value('{"name":"write_file","arguments":{') == ""

    def test_an_escape_cut_in_half_waits(self):
        """Half of \\n is a backslash, and printing it is wrong."""
        assert partial_string_value('{"content":"line\\') == "line"

    def test_a_unicode_escape_cut_in_half_waits(self):
        assert partial_string_value('{"content":"x\\u26') == "x"

    def test_a_complete_unicode_escape_decodes(self):
        assert partial_string_value('{"content":"x\\u0041') == "xA"

    def test_the_edit_argument_is_watched_too(self):
        assert partial_string_value('{"new_text":"edited"}') == "edited"

    def test_a_non_string_value_is_ignored(self):
        assert partial_string_value('{"content":42}') == ""


class TestTheCodeArrivesWhileItIsWritten:
    def _stream(self, chunks):
        live = LiveContentFilter()
        visible, code = [], []
        for chunk in chunks:
            visible.append(live.feed(chunk))
            code.append(live.code_delta())
        return "".join(visible), "".join(code)

    CALL = ['Adding it. ', '<tool_call>{"name":"write_file",',
            '"arguments":{"path":"a.py","content":"def add(a, b):',
            '\\n    return a + b\\n', '"}}', '</tool_call>', ' Done.']

    def test_the_code_comes_out_as_it_arrives(self):
        _visible, code = self._stream(self.CALL)
        assert code == "def add(a, b):\n    return a + b\n"

    def test_the_prose_around_it_is_untouched(self):
        visible, _code = self._stream(self.CALL)
        assert visible == "Adding it.  Done."

    def test_no_protocol_leaks_into_the_prose(self):
        visible, _code = self._stream(self.CALL)
        assert "tool_call" not in visible and "write_file" not in visible

    def test_nothing_is_reported_twice(self):
        """The delta is what is new, not the whole value each time."""
        live = LiveContentFilter()
        live.feed('<tool_call>{"content":"abc')
        first = live.code_delta()
        second = live.code_delta()
        assert first == "abc" and second == ""

    def test_a_second_call_starts_clean(self):
        live = LiveContentFilter()
        live.feed('<tool_call>{"content":"first"}</tool_call>')
        live.code_delta()
        live.feed('<tool_call>{"content":"second')
        assert live.code_delta() == "second"

    def test_outside_a_call_there_is_no_code(self):
        live = LiveContentFilter()
        live.feed('here is some {"content":"not a tool call"} prose')
        assert live.code_delta() == ""


class TestTextArrivesCharacterByCharacter:
    def _render(self, text, width=46, bar=False):
        from wynxo.pet import Pet
        from wynxo.tui import Transcript
        from wynxo.ui import ActivityBar, CodeStreamer, UI

        page = Transcript(width=width)
        ui = UI()
        ui.console = page.console
        ui.width = width
        ui.live_ok = False
        if bar:
            ui.bar = ActivityBar(ui, "medium", pet=Pet(enabled=False))
        streamer = CodeStreamer(ui, indent="  ")
        for char in text:
            streamer.feed(char)
        streamer.finish()
        page.drain()
        return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(page.lines))

    def test_a_single_character_shows_immediately(self):
        """Holding the partial word is what made the answer arrive in
        jumps, and a model pausing mid-word look like it had stopped."""
        from wynxo.tui import Transcript
        from wynxo.ui import CodeStreamer, UI

        page = Transcript(width=40)
        ui = UI()
        ui.console = page.console
        ui.width = 40
        streamer = CodeStreamer(ui, indent="  ")
        for char in "hel":
            streamer.feed(char)
        page.drain()
        assert "hel" in "".join(page.lines)

    def test_every_word_survives_the_wrap(self):
        text = ("The retry helper in upload.py is linear and should back "
                "off exponentially instead of waiting the same amount.")
        assert self._render(text, bar=True).split() == text.split()

    def test_no_line_is_wider_than_the_terminal(self):
        text = "word " * 60
        for line in self._render(text, width=40, bar=True).splitlines():
            assert len(line) <= 40


class TestCodeKeepsItsIndentation:
    def _render(self, text, width=60):
        from wynxo.tui import Transcript
        from wynxo.ui import CodeStreamer, UI

        page = Transcript(width=width)
        ui = UI()
        ui.console = page.console
        ui.width = width
        streamer = CodeStreamer(ui, indent="  ", code=False, literal=True)
        for char in text:
            streamer.feed(char)
        streamer.finish()
        page.drain()
        return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(page.lines))

    def test_leading_whitespace_is_kept(self):
        """Prose drops whitespace at the start of a line, which is right for
        a sentence and destroys the indentation of every line of Python."""
        body = self._render("def f():\n    return 1\n")
        assert any(line.endswith("    return 1") for line in body.splitlines())

    def test_blank_lines_are_kept(self):
        body = self._render("a = 1\n\nb = 2\n")
        assert body.count("\n") >= 3

    def test_a_long_line_breaks_rather_than_reflows(self):
        """Rearranging code into something that no longer parses is worse
        than breaking it at the edge."""
        body = self._render("x = " + "1234567890" * 12, width=40)
        assert all(len(line) <= 40 for line in body.splitlines())
