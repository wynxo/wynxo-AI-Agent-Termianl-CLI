"""Reading piped stdin must never block on a writer that is not coming.

The failure this guards against is silent: wynxo sits with no output and no
error, indistinguishable from a crash, whenever it is started with stdin
attached to an idle pipe -- CI, supervisors, editor terminals.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args, stdin, timeout=30):
    """Run the real CLI as a subprocess, since this is about process stdin."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "WYNXO_ENDPOINT": "http://127.0.0.1:1"}
    return subprocess.run(
        [sys.executable, "-m", "wynxo", *args],
        stdin=stdin, capture_output=True, text=True, timeout=timeout, env=env,
    )


class TestPipedStdin:
    def test_idle_pipe_does_not_hang(self, tmp_path):
        """An open pipe nobody writes to must be given up on, quickly."""
        read_fd, write_fd = os.pipe()
        started = time.monotonic()
        try:
            with os.fdopen(read_fd) as reader:
                result = run_cli(["-p", "hello", "-C", str(tmp_path)], reader, timeout=25)
        finally:
            os.close(write_fd)
        elapsed = time.monotonic() - started
        # It should fail on the unreachable endpoint, not sit on stdin.
        assert elapsed < 20, f"took {elapsed:.1f}s -- it blocked on stdin"
        assert "Cannot reach an Ollama server" in (result.stdout + result.stderr)

    def test_file_redirect_is_read(self, tmp_path):
        source = tmp_path / "input.txt"
        source.write_text("UNIQUE_MARKER_TEXT\n")
        with source.open() as handle:
            result = run_cli(["-C", str(tmp_path)], handle)
        # It reached the provider (and failed there), meaning stdin was consumed
        # and it did not fall through to the interactive path.
        assert "Cannot reach an Ollama server" in (result.stdout + result.stderr)

    def test_closed_stdin_is_fine(self, tmp_path):
        result = run_cli(["-p", "hello", "-C", str(tmp_path)], subprocess.DEVNULL)
        assert "Cannot reach an Ollama server" in (result.stdout + result.stderr)


class TestReadPipedStdinUnit:
    def test_tty_returns_empty(self, monkeypatch):
        from wynxo.cli import read_piped_stdin

        class FakeStdin:
            def isatty(self):
                return True

        monkeypatch.setattr(sys, "stdin", FakeStdin())
        assert read_piped_stdin() == ""

    def test_reads_a_regular_file(self, tmp_path, monkeypatch):
        from wynxo.cli import read_piped_stdin

        source = tmp_path / "in.txt"
        source.write_text("  piped content  \n")
        with source.open() as handle:
            monkeypatch.setattr(sys, "stdin", handle)
            assert read_piped_stdin() == "piped content"

    def test_idle_pipe_returns_empty_within_the_grace_window(self, monkeypatch):
        from wynxo.cli import read_piped_stdin

        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd) as reader:
                monkeypatch.setattr(sys, "stdin", reader)
                started = time.monotonic()
                assert read_piped_stdin(grace=0.2) == ""
                assert time.monotonic() - started < 2.0
        finally:
            os.close(write_fd)

    def test_pipe_with_data_is_read(self, monkeypatch):
        from wynxo.cli import read_piped_stdin

        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"hello from the pipe\n")
        os.close(write_fd)   # EOF, so the read terminates
        with os.fdopen(read_fd) as reader:
            monkeypatch.setattr(sys, "stdin", reader)
            assert read_piped_stdin() == "hello from the pipe"


class TestWindowsCanPipeToo:
    """`git diff | wynxo -p "review"` silently reviewed nothing on Windows.

    The branch guarding the fallback asked `hasattr(select, "select")` -- and
    Windows *has* select.select, it just cannot use it on a pipe. So Windows
    took the select path, the call raised, and the piped input was dropped.
    """

    def test_the_branch_is_chosen_by_platform_not_by_attribute(self):
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.read_piped_stdin)
        assert 'os.name != "nt"' in source
        assert 'hasattr(select, "select")' not in source

    def test_windows_asks_the_pipe_rather_than_blocking_on_it(self):
        """Reading in a thread and abandoning it leaves that thread blocked
        on stdin for the life of the process, holding the handle. Peeking
        answers immediately and leaves nothing behind."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli._windows_pipe_text)
        assert "PeekNamedPipe" in source
        assert "Thread" not in inspect.getsource(cli.read_piped_stdin)

    def test_no_thread_is_left_behind_anywhere(self):
        """The whole module: a daemon thread parked in a blocking read is
        the thing this replaced."""
        import inspect

        from wynxo import cli

        assert "threading" not in inspect.getsource(cli)

    def test_peeking_is_harmless_off_windows(self):
        """It is only ever called on Windows, but it must not explode if it
        is reached anywhere else."""
        from wynxo import cli

        assert cli._windows_pipe_text(0.0) == ""
