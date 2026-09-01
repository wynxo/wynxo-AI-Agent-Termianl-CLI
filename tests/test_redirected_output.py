"""`wynxo > notes.md` must produce a file, not a screen recording.

rich already knows the rule -- colour is for terminals, and redirected
output gets none. The streamer is the one place that goes round rich: with
no activity bar to hold the half-written line, it writes the line straight
to ``console.file`` so a word can appear before its line is finished, and
that path never passed under rich's check. So every inline-code span in an
answer arrived as a truecolor escape pair in the file, while every other
line in the same file came out clean.
"""

from __future__ import annotations

import io

from wynxo.ui import CodeStreamer, SafeConsole, UI


def _ui(*, terminal: bool):
    ui = UI()
    ui.live_ok = False
    ui.bar = None            # no live region: the direct-write path
    ui.console = SafeConsole(
        file=io.StringIO(), width=80, highlight=False, soft_wrap=False,
        force_terminal=True if terminal else False,
        color_system="truecolor" if terminal else None)
    ui.width = 80
    return ui


def _stream(ui, text: str) -> str:
    streamer = CodeStreamer(ui, indent="  ")
    for chunk in text:
        streamer.feed(chunk)
    streamer.finish()
    return ui.console.file.getvalue()


ANSWER = "Call `todo_write` and then **stop**.\nSecond line.\n"


class TestRedirectedOutputIsPlain:
    def test_no_escapes_reach_a_file(self):
        written = _stream(_ui(terminal=False), ANSWER)
        assert "\x1b" not in written, repr(written)

    def test_the_words_are_all_there(self):
        """Dropping the colour must not drop the text with it."""
        written = _stream(_ui(terminal=False), ANSWER)
        for word in ("Call", "todo_write", "stop", "Second line."):
            assert word in written, written

    def test_a_reset_is_not_written_either(self):
        """The reset at end-of-line is written only when a pen was shown.
        Suppressing the pen has to suppress its reset too, or the escape
        comes back through the other door."""
        written = _stream(_ui(terminal=False), ANSWER)
        assert "[0m" not in written


class TestATerminalStillGetsItsColour:
    def test_escapes_are_written_to_a_terminal(self):
        written = _stream(_ui(terminal=True), ANSWER)
        assert "\x1b[" in written, "the colour was dropped everywhere"

    def test_the_words_survive_there_too(self):
        written = _stream(_ui(terminal=True), ANSWER)
        for word in ("Call", "todo_write", "stop"):
            assert word in written
