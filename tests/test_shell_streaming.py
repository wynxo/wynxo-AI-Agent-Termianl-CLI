"""Showing a command's output while it is still running.

communicate() waits for the process to exit, so a five-minute test run or an
npm install showed nothing at all until it finished -- and if it hit the
timeout, the output that would have explained why was thrown away with it.
Both are worst exactly when something is going wrong.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from wynxo.tools.shell import Shell, ShellInput

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="uses POSIX shell syntax")


def run(tool: Shell, command: str, timeout: int = 30):
    return asyncio.run(tool.run(ShellInput(command=command, timeout=timeout)))


@pytest.fixture
def shell(tmp_path):
    return Shell(workspace=tmp_path)


class TestOutputArrivesWhileItRuns:
    def test_lines_are_delivered_before_the_command_finishes(self, shell):
        seen = []
        start = time.monotonic()

        async def sink(line):
            seen.append((time.monotonic() - start, line))

        shell.on_output = sink
        run(shell, 'for i in 1 2 3; do echo "step $i"; sleep 0.25; done')

        assert [line for _, line in seen] == ["step 1", "step 2", "step 3"]
        # The point of the whole change: the first line must not have waited
        # for the last one.
        first, last = seen[0][0], seen[-1][0]
        assert last - first > 0.2, "output was delivered in one batch"

    def test_the_model_still_gets_the_whole_output(self, shell):
        result = run(shell, 'echo one; echo two; echo three')
        assert result.output == "one\ntwo\nthree"

    def test_it_works_with_no_listener_attached(self, shell):
        assert run(shell, "echo hello").output == "hello"

    def test_a_listener_that_raises_does_not_kill_the_command(self, shell):
        """A progress display is not worth failing a build over."""
        async def broken(line):
            raise RuntimeError("UI fell over")

        shell.on_output = broken
        result = run(shell, "echo still fine")
        assert result.ok and result.output == "still fine"

    def test_exit_codes_still_come_through(self, shell):
        result = run(shell, "echo bad output; exit 3")
        assert result.ok is False
        assert "exit code 3" in result.output
        assert "bad output" in result.output


class TestTimeoutKeepsWhatItSaw:
    def test_output_before_the_kill_is_reported(self, shell):
        """A hung command is when its last lines matter most: they say which
        test wedged. Discarding them leaves the model only the word
        'timeout' to act on."""
        result = run(shell, "echo starting; echo compiling; sleep 30",
                     timeout=2)
        assert result.ok is False
        assert "timed out" in result.output
        assert "starting" in result.output and "compiling" in result.output

    def test_it_still_times_out_promptly(self, shell):
        start = time.monotonic()
        run(shell, "sleep 30", timeout=2)
        assert time.monotonic() - start < 15

    def test_a_silent_hang_still_times_out(self, shell):
        """Nothing is ever written, so the read never returns on its own."""
        result = run(shell, "sleep 30", timeout=2)
        assert result.ok is False and "timed out" in result.output


class TestAwkwardOutput:
    def test_a_progress_bar_reads_as_its_final_state(self, shell):
        r"""\r rewrites one line. Kept verbatim it would be every frame at
        once; the last state is what a person would have seen."""
        result = run(shell, r"printf '10%%\r50%%\r100%%\n'")
        assert result.output == "100%"

    def test_a_line_with_no_trailing_newline_is_not_lost(self, shell):
        assert run(shell, "printf 'no newline'").output == "no newline"

    def test_an_enormous_single_line_does_not_hang(self, shell):
        """asyncio's readline() raises past its buffer limit, which is why
        this reads in chunks instead."""
        result = run(shell, "python3 -c \"print('x' * 200000)\"", timeout=20)
        assert result.ok and "x" in result.output

    def test_a_very_chatty_command_keeps_head_and_tail(self, shell):
        result = run(shell, "python3 -c \"[print(i) for i in range(20000)]\"",
                     timeout=30)
        assert result.ok
        assert result.output.startswith("0\n1\n")
        assert result.output.rstrip().endswith("19999")

    def test_binary_bytes_do_not_crash_the_decoder(self, shell):
        result = run(shell, r"printf '\xff\xfe bad bytes\n'")
        assert result.ok

    def test_stderr_is_included(self, shell):
        assert "to stderr" in run(shell, "echo to stderr >&2").output


class TestTheHookIsClearedAfterwards:
    def test_the_agent_unsets_it_even_when_the_tool_fails(self):
        """A tool object outlives one call, so a stale hook would write into
        a line that has already been closed."""
        import inspect

        from wynxo.agent import Agent

        source = inspect.getsource(Agent._run_tool_calls)
        assert "finally:" in source
        assert "tool.on_output = None" in source
