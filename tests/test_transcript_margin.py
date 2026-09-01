"""Everything the conversation contains sits at the same left margin.

The user's line, the tool lines, the answer, the diffs and the companion's
greeting all start a couple of columns in. A block that starts hard against
column zero reads as something that has escaped the conversation rather than
as part of it -- and a Syntax block, which draws a filled background band,
reads that way most of all.

Verbose tool output was the one kind of content in the whole transcript with
no margin. The same call also asked for its tick-or-cross line with empty
arguments, and that line returns early on empty text -- so verbose mode
printed a wall of output with no indication of whether the tool had
succeeded.
"""

from __future__ import annotations

import asyncio
import io
import re

from wynxo.cli import TerminalCallbacks
from wynxo.ui import SafeConsole, UI


def _ui(width: int = 90):
    ui = UI()
    ui.live_ok = False
    ui.console = SafeConsole(file=io.StringIO(), force_terminal=True,
                             color_system="truecolor", highlight=False,
                             soft_wrap=False, width=width, height=10_000)
    ui.width = width
    return ui


def _lines(ui) -> list[str]:
    plain = re.sub(r"\x1b\[[0-9;]*m", "", ui.console.file.getvalue())
    return [line for line in plain.split("\n") if line.strip()]


class TestCodeBlocksAreIndented:
    def test_a_block_starts_past_column_zero(self):
        ui = _ui()
        ui.code("x = 1\ny = 2\n", "python")
        assert _lines(ui), "nothing was drawn"
        for line in _lines(ui):
            assert line.startswith("    "), repr(line)

    def test_the_content_is_still_there(self):
        ui = _ui()
        ui.code("alpha = 1\n", "python")
        assert any("alpha" in line for line in _lines(ui))

    def test_it_matches_the_answer_margin(self):
        """Not a number invented here: the same margin the answer uses."""
        answer = _ui()
        answer.assistant_markdown("a sentence")
        block = _ui()
        block.code("x = 1\n", "python")
        indent = min(len(line) - len(line.lstrip())
                     for line in _lines(answer))
        assert all(len(line) - len(line.lstrip()) >= indent
                   for line in _lines(block))


class TestVerboseToolOutputSaysWhetherItWorked:
    def _run(self, ok: bool):
        ui = _ui()
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.verbose_tools = True
        asyncio.run(callbacks.on_tool_result(
            "read_file", ok, "$ read_file a.py",
            "line one\nline two\nline three"))
        return ui

    def test_a_failure_is_marked(self):
        ui = self._run(ok=False)
        glyphs = ui.g
        assert any(glyphs.cross in line for line in _lines(ui)), _lines(ui)

    def test_a_success_is_marked(self):
        ui = self._run(ok=True)
        assert any(ui.g.tick in line for line in _lines(ui)), _lines(ui)

    def test_the_whole_output_is_still_shown(self):
        ui = self._run(ok=True)
        body = "\n".join(_lines(ui))
        for word in ("line one", "line two", "line three"):
            assert word in body

    def test_a_one_line_result_is_not_said_twice(self):
        """The marker line already carries it. Printing it again as a
        syntax-highlighted block is the same sentence twice, which is not
        what asking for detail meant."""
        ui = _ui()
        callbacks = TerminalCallbacks(ui, prompt_session=None)
        callbacks.verbose_tools = True
        asyncio.run(callbacks.on_tool_result(
            "read_file", False, "", "ERROR: a.py does not exist."))
        assert sum("does not exist" in line for line in _lines(ui)) == 1, \
            _lines(ui)

    def test_nothing_in_it_starts_at_column_zero(self):
        for ok in (True, False):
            for line in _lines(self._run(ok)):
                assert line.startswith(" "), repr(line)


class TestTheCompletionReportKeepsTheMargin:
    """It was the last block still starting hard against column zero, which
    reads as output that escaped the transcript rather than as the turn's
    own summing-up."""

    def test_the_report_is_indented(self):
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._turn_locked)
        block = source.split("completion_report()", 1)[1].split("# No stats")[0]
        assert 'f"  {line}"' in block, block[:400]

    def test_every_line_of_a_real_report_is_indented(self):
        from wynxo.task_state import TaskState, TaskStateMachine

        machine = TaskStateMachine()
        machine.begin("fix it")
        machine.transition(TaskState.EXECUTING)
        machine.add_file("demo.py", changed=True)
        machine.record_failure("tests failed: 2 of 9")
        report = machine.completion_report()
        assert report
        indented = "\n".join(f"  {line}" for line in report.splitlines())
        for line in indented.splitlines():
            assert line.startswith("  "), repr(line)
