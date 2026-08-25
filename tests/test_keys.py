"""Mid-turn keystroke handling.

The terminal-restore path matters most: leaving a shell in cbreak mode with
echo off is a genuinely bad outcome, so it must survive handler exceptions,
double stops, and a watcher that never started.
"""

import os
import sys
import time

import pytest

# pty and termios do not exist on Windows, and importing them at module
# level would fail collection for this whole file -- hiding every other
# result on that platform rather than reporting one skip.
pty = pytest.importorskip("pty", reason="POSIX terminals only")
termios = pytest.importorskip("termios", reason="POSIX terminals only")

from wynxo.keys import CTRL, KeyWatcher, describe_bindings, key_name


class TestKeyNames:
    def test_control_characters(self):
        assert key_name("\x0f") == "ctrl+o"
        assert key_name("\x14") == "ctrl+t"
        assert key_name("\x05") == "ctrl+e"
        assert key_name("\x02") == "ctrl+b"

    def test_printable_passes_through(self):
        assert key_name("a") == "a"
        assert key_name("?") == "?"

    def test_table_covers_the_alphabet(self):
        assert len(CTRL) == 26
        assert CTRL["\x01"] == "ctrl+a"
        assert CTRL["\x1a"] == "ctrl+z"

    def test_hint_rendering(self):
        assert describe_bindings({"ctrl+o": "thinking"}) == "^O thinking"
        assert "^T detail" in describe_bindings({"ctrl+o": "a", "ctrl+t": "detail"})


class TestSafety:
    def test_start_is_a_noop_without_a_tty(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", open(os.devnull))
        watcher = KeyWatcher({"ctrl+o": lambda: None})
        assert not watcher.available
        watcher.start()
        watcher.stop()          # must not raise

    def test_stop_twice_is_safe(self):
        watcher = KeyWatcher({})
        watcher.stop()
        watcher.stop()

    def test_a_raising_handler_does_not_kill_the_watcher(self):
        calls = []
        watcher = KeyWatcher({
            "ctrl+o": lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            "ctrl+t": lambda: calls.append("t"),
        })
        watcher._dispatch("\x0f")      # raises inside, swallowed
        watcher._dispatch("\x14")
        assert calls == ["t"]

    def test_unknown_key_is_ignored(self):
        calls = []
        watcher = KeyWatcher({"ctrl+o": lambda: calls.append("o")})
        watcher._dispatch("z")
        assert calls == []


@pytest.mark.skipif(sys.platform == "win32", reason="posix termios")
class TestTerminalRestore:
    def test_terminal_settings_are_restored(self, monkeypatch):
        """The important one: the shell must be usable afterwards."""
        master, slave = pty.openpty()
        try:
            reader = os.fdopen(slave, "r")
            monkeypatch.setattr(sys, "stdin", reader)
            before = termios.tcgetattr(slave)

            watcher = KeyWatcher({"ctrl+o": lambda: None})
            watcher.start()
            assert watcher._saved is not None, "cbreak was never entered"
            during = termios.tcgetattr(slave)
            assert not (during[3] & termios.ICANON), "cbreak not actually applied"

            watcher.stop()
            assert termios.tcgetattr(slave) == before
        finally:
            os.close(master)

    def test_keypresses_reach_handlers(self, monkeypatch):
        master, slave = pty.openpty()
        try:
            reader = os.fdopen(slave, "r")
            monkeypatch.setattr(sys, "stdin", reader)
            seen = []
            watcher = KeyWatcher({"ctrl+o": lambda: seen.append("o"),
                                  "ctrl+t": lambda: seen.append("t")})
            watcher.start()
            os.write(master, b"\x0f\x14\x0f")
            deadline = time.monotonic() + 3
            while len(seen) < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            watcher.stop()
            assert seen == ["o", "t", "o"]
        finally:
            os.close(master)

    def test_sigint_still_works(self, monkeypatch):
        """cbreak, not raw: Ctrl-C must keep generating SIGINT so the
        existing interrupt path is unaffected."""
        master, slave = pty.openpty()
        try:
            reader = os.fdopen(slave, "r")
            monkeypatch.setattr(sys, "stdin", reader)
            watcher = KeyWatcher({})
            watcher.start()
            attrs = termios.tcgetattr(slave)
            assert attrs[3] & termios.ISIG, "ISIG cleared; Ctrl-C would be swallowed"
            watcher.stop()
        finally:
            os.close(master)
