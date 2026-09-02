"""The window changed size. Everything that wraps has to hear about it.

Two independent faults, both of which left every wrap in the session
computed against the width the terminal had at launch. On a window made
narrower that is not cosmetic: every streamed line runs off the edge and
the terminal wraps it a second time, so the answer arrives in ragged
half-lines.

  1. ``shutil.get_terminal_size`` consults ``COLUMNS``/``LINES`` first and
     the ioctl second. Those variables are set once, by whatever started
     the process, and nothing updates them on a resize.

  2. prompt_toolkit's Application takes SIGWINCH for the length of every
     read. It restores the previous handler afterwards, which is correct of
     it -- but for the whole time somebody is sitting at the prompt the
     signal is its, and a handler is the only thing that was updating the
     cached width. Resize, then type, is the ordinary case.
"""

from __future__ import annotations

import inspect
import os
import struct

import pytest

from wynxo import platforms
from wynxo.ui import UI


class TestTheSizeComesFromTheTerminal:
    """Not from an environment variable that nothing updates."""

    def _pty(self, columns: int, lines: int = 24):
        """A real pty sized exactly as asked. Returns the slave file."""
        pty = pytest.importorskip("pty")
        fcntl = pytest.importorskip("fcntl")
        termios = pytest.importorskip("termios")

        primary, secondary = pty.openpty()
        fcntl.ioctl(primary, termios.TIOCSWINSZ,
                    struct.pack("HHHH", lines, columns, 0, 0))
        return primary, secondary

    def test_a_stale_columns_does_not_win(self, monkeypatch):
        primary, secondary = self._pty(120, 40)
        try:
            monkeypatch.setenv("COLUMNS", "40")
            monkeypatch.setenv("LINES", "10")
            monkeypatch.setattr("sys.stdout", os.fdopen(secondary, "w"))
            assert platforms.terminal_width() == 120
            assert platforms.terminal_height() == 40
        finally:
            os.close(primary)

    def test_a_later_resize_is_seen(self, monkeypatch):
        fcntl = pytest.importorskip("fcntl")
        termios = pytest.importorskip("termios")

        primary, secondary = self._pty(120, 40)
        try:
            monkeypatch.setattr("sys.stdout", os.fdopen(secondary, "w"))
            assert platforms.terminal_width() == 120
            fcntl.ioctl(primary, termios.TIOCSWINSZ,
                        struct.pack("HHHH", 20, 60, 0, 0))
            assert platforms.terminal_width() == 60, (
                "the size was read once and cached somewhere")
            assert platforms.terminal_height() == 20
        finally:
            os.close(primary)

    def test_without_a_terminal_the_environment_is_the_answer(self, monkeypatch):
        """A pipe has no size of its own, so COLUMNS is the best available
        statement of intent and must still be honoured."""
        import io

        monkeypatch.setenv("COLUMNS", "133")
        monkeypatch.setattr("sys.stdout", io.StringIO())
        monkeypatch.setattr("sys.stderr", io.StringIO())
        monkeypatch.setattr("sys.stdin", io.StringIO())
        assert platforms.terminal_width() == 133


class TestEveryDrawMeasures:
    """A signal another component owns half the time cannot be the
    mechanism. It is a nudge; the re-measure happens where a draw begins."""

    def test_the_prompt_loop_measures_before_it_draws(self):
        from wynxo.cli import Repl

        source = inspect.getsource(Repl._prompt_loop)
        head = source.split("try:", 1)[0]
        assert "refresh_size()" in head, (
            "a resize that happened while the prompt was waiting is never "
            "noticed")

    def test_the_activity_bar_measures_on_every_frame(self):
        """``__rich_console__`` is the per-frame entry point, and the only
        one. Measuring in a part instead would leave whichever parts were
        drawn first using the old width."""
        from wynxo.ui import ActivityBar

        source = inspect.getsource(ActivityBar.__rich_console__)
        assert "refresh_size()" in source

    def test_a_frame_actually_picks_up_the_new_width(self, monkeypatch):
        import io

        from rich.console import Console

        from wynxo.ui import ActivityBar

        ui = UI()
        ui.live_ok = False
        bar = ActivityBar(ui, "medium")

        def widths(columns):
            monkeypatch.setattr("wynxo.ui.terminal_width", lambda: columns)
            console = Console(file=io.StringIO(), width=columns,
                              force_terminal=False)
            console.print(bar)
            return {len(line) for line
                    in console.file.getvalue().rstrip("\n").split("\n")}

        # No row may exceed the terminal; a row shorter than it is fine.
        # The strip is padded to the full width because it is a band with a
        # background; the scene rows above it are transparent, so padding
        # them would paint over the conversation showing through.
        assert max(widths(120)) <= 120
        assert 120 in widths(120), "the strip is no longer a full-width band"
        assert max(widths(48)) <= 48, (
            "the strip is still padded to the old width, so it wraps")

    def test_a_streamer_follows_the_new_width(self, monkeypatch):
        """The streamer holds no width of its own for exactly this reason:
        it lives for a whole turn."""
        from wynxo.ui import CodeStreamer

        ui = UI()
        ui.live_ok = False
        monkeypatch.setattr("wynxo.ui.terminal_width", lambda: 120)
        ui.refresh_size()
        streamer = CodeStreamer(ui, indent="  ")
        wide = streamer.width
        monkeypatch.setattr("wynxo.ui.terminal_width", lambda: 48)
        ui.refresh_size()
        assert streamer.width < wide
