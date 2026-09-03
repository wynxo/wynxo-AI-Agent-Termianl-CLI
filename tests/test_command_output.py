"""What a command printed, on the screen.

`which firefox` answers in one line, and that line is the whole point of
running it. It appeared nowhere: not in the tool block, and not under it
with the detail toggle on either.

Three separate faults stacked up to hide it, which is why nothing looked
broken -- a block with a `$` in it and a gutter of empty rows both look
like a command that printed nothing.
"""

from __future__ import annotations

import io

import pytest

from wynxo.cli import TerminalCallbacks, _LANGUAGE, _is_a_sentence
from wynxo.ui import UI


def _ui(width: int = 90) -> UI:
    made = UI()
    made.console.file = io.StringIO()
    made.console.width = made.width = width
    return made


async def run(command: str, output: str, *, verbose=False, ok=True,
              name="shell", display=None):
    ui = _ui()
    cb = TerminalCallbacks(ui)
    cb.verbose_tools = verbose
    await cb.on_tool_start(name, command)
    await cb.on_tool_result(
        name, ok, "$ " + command if display is None else display, output)
    return ui.console.file.getvalue()


class TestTheOutputIsOnTheScreen:
    async def test_a_one_line_answer_is_shown(self):
        """`display or output` preferred the tool's own summary, and
        shell's summary is the command echoed back -- so what the command
        printed was never even looked at."""
        assert "/usr/bin/firefox" in await run(
            "which firefox", "/usr/bin/firefox")

    async def test_it_is_shown_without_the_detail_toggle(self):
        """One line is the block's business. The toggle is for the rest."""
        said = await run("echo one", "one", verbose=False)
        assert "one" in said

    async def test_the_command_stays_on_the_head_line(self):
        """`ls wynxo | head -4` is five words with no slash in it, so it
        was judged prose and moved off the head -- into the row the
        output should have been in. The block then said what was run
        twice and what came of it not at all."""
        said = await run("ls wynxo | head -4",
                         "__init__.py\\n__main__.py\\n__pycache__")
        head = [ln for ln in said.splitlines() if ln.strip()][0]
        assert "ls wynxo | head -4" in head, said
        assert "__init__.py" in said

    def test_a_shell_command_is_never_prose(self):
        assert not _is_a_sentence("ls wynxo | head -4", "shell")
        assert not _is_a_sentence("git commit -m done", "shell")

    def test_a_tool_describing_itself_still_is(self):
        """run_tests reports "syntax check passed (compileall)", and prose
        on the head line reads as a filename that got out of hand."""
        assert _is_a_sentence("syntax check passed", "run_tests")


class TestTheDetailToggleExpandsEverything:
    async def test_the_rest_of_a_long_output_is_shown(self):
        said = await run("ls", "a.py\\nb.py\\nc.py\\nd.py", verbose=True)
        for name in ("a.py", "b.py", "c.py", "d.py"):
            assert name in said, name

    async def test_nothing_is_said_twice(self):
        """The block already carries a one-line answer. Quoting it again
        underneath is the same sentence twice, which is not what asking
        for detail meant."""
        said = await run("which firefox", "/usr/bin/firefox", verbose=True)
        assert said.count("/usr/bin/firefox") == 1, said

    async def test_a_one_line_failure_is_not_said_twice(self):
        """The detail has "ERROR:" stripped off it and the output does
        not, so comparing them raw finds a difference that is not one."""
        said = await run("cat nope", "ERROR: no such file", verbose=True,
                         ok=False)
        assert said.count("no such file") == 1, said


class TestTheHighlighterCannotEatTheText:
    def test_a_lexer_that_returns_nothing_does_not_blank_the_line(self):
        """pygments' BashSessionLexer -- which is what "console" resolves
        to -- returns no tokens at all for a line with no shell prompt on
        it. Every line of every command's output rendered as an empty
        gutter, and nothing said so, because a blank line is what an empty
        result looks like."""
        ui = _ui(70)
        ui.code("__init__.py\\n__main__.py", "console")
        assert "__init__.py" in ui.console.file.getvalue()

    @pytest.mark.parametrize("language", ["text", "python", "bash", "json",
                                          "console", "yaml", "diff", "rust"])
    def test_no_language_loses_a_character(self, language):
        body = "alpha bravo\\ncharlie delta"
        ui = _ui(70)
        ui.code(body, language)
        drawn = ui.console.file.getvalue()
        for word in ("alpha", "bravo", "charlie", "delta"):
            assert word in drawn, (language, drawn)

    def test_shell_output_is_not_lexed_as_a_session_transcript(self):
        """A command prints whatever it prints. "text" is the honest answer
        to what language that is."""
        assert _LANGUAGE.get("shell") != "console"

    def test_colour_still_happens_where_it_can(self):
        """The guard must fall back, not give up on highlighting."""
        ui = _ui(70)
        assert ui.highlight("def go(): pass", "python").spans
