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
    def test_the_agent_unsets_it_even_when_the_tool_fails(self, tmp_path):
        """A tool object outlives one call, so a stale hook would write into
        a line that has already been closed."""
        from unittest.mock import MagicMock

        from wynxo.agent import Agent
        from wynxo.config import Config
        from wynxo.effort import resolve
        from wynxo.parsing import ToolCall

        agent = Agent(client=MagicMock(), config=Config(),
                      policy=resolve("medium"), workspace=tmp_path)
        tool = agent.tools.get("read_file")

        async def explode(_args):
            raise RuntimeError("tool fell over")

        # Replacing invoke rather than run, so the failure escapes the
        # layer that normally turns it into a result -- the worst case for
        # leaving state behind.
        tool.invoke = explode
        with pytest.raises(RuntimeError):
            asyncio.run(agent._run_one(
                ToolCall(name="read_file", arguments={"path": "x"}, call_id="1")))
        assert tool.on_output is None
        assert tool.context_left == 0


class TestStoppingMeansStopping:
    """Killing the shell is not killing the command.

    `make -j8` and `npm install` spawn workers, so signalling only the
    process wynxo launched leaves those workers running -- eating the machine
    long after the user believes they stopped it. And an abandoned await
    stops nothing at all: the command carries on writing to the project.
    """

    @pytest.fixture
    def probe(self, tmp_path):
        """A script we can look for by name, so the search cannot match the
        test runner's own command line."""
        script = tmp_path / "probe.sh"
        script.write_text("#!/bin/sh\nsleep 60\n")
        script.chmod(0o755)
        return script

    def survivors(self, probe) -> list[str]:
        import os
        import subprocess

        found = subprocess.run(["pgrep", "-f", str(probe)],
                               capture_output=True, text=True).stdout.split()
        return [p for p in found if p.isdigit() and int(p) != os.getpid()]

    def reap(self, pids) -> None:
        import os
        import signal as _signal

        for pid in pids:
            try:
                os.kill(int(pid), _signal.SIGKILL)
            except OSError:
                pass

    def test_a_timeout_takes_the_workers_with_it(self, shell, probe):
        result = run(shell, f"{probe} & {probe} & wait", timeout=2)
        assert result.ok is False
        time.sleep(0.5)
        left = self.survivors(probe)
        self.reap(left)
        assert left == [], f"{len(left)} workers outlived the timeout"

    def test_cancelling_takes_the_command_with_it(self, shell, probe):
        """Ctrl-C during a build. The await used to be abandoned while the
        command carried on in the background."""
        async def cancel_midway():
            task = asyncio.create_task(
                shell.run(ShellInput(command=f"{probe} & {probe} & wait",
                                     timeout=60)))
            await asyncio.sleep(0.6)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_midway())
        time.sleep(0.5)
        left = self.survivors(probe)
        self.reap(left)
        assert left == [], f"{len(left)} processes survived the interrupt"

    def test_cancelling_does_not_swallow_the_interrupt(self, shell):
        """The REPL relies on CancelledError reaching it to end the turn."""
        async def cancel_midway():
            task = asyncio.create_task(
                shell.run(ShellInput(command="sleep 30", timeout=60)))
            await asyncio.sleep(0.3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_midway())

    def test_a_command_that_already_finished_is_not_signalled(self, shell):
        """Signalling a reaped pid can hit whatever reused the number."""
        async def check():
            result = await shell.run(ShellInput(command="echo quick"))
            assert result.ok
            # A second teardown must be a no-op rather than an error.
            await shell._terminate(
                type("Done", (), {"returncode": 0, "pid": 999999})())

        asyncio.run(check())


def test_max_output_config_changes_what_is_kept(tmp_path):
    """config.max_command_output_chars must actually reach the shell: the
    hardcoded constant used to make the setting a lie."""
    import sys
    py = sys.executable
    big = f"& '{py}' -c \"import sys; [print('x'*80) for _ in range(600)]\""
    full = run(Shell(workspace=tmp_path), big)
    capped = run(Shell(workspace=tmp_path, max_output=2000), big)
    assert full.ok, full.error
    assert capped.ok, capped.error
    assert len(capped.output) < len(full.output), (
        "a smaller max_output must retain less output")
    # And the smaller cap is still useful: the tail is what a failing build
    # shows, so the end of the output must survive even the small cap.
    assert capped.output.strip().endswith("x" * 80)
