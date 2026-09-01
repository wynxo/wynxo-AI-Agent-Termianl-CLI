"""The terminal has to be usable after wynxo has gone. Every way it can go.

The key watcher puts the terminal in cbreak mode for the length of a turn:
ICANON and ECHO off, so single keypresses arrive and are not echoed twice.
Handing that back is a ``finally``, which covers a return, an exception and a
cancellation -- and not SIGTERM or SIGHUP, which terminate the process with
their default disposition before any Python runs.

`kill`, a session manager, `timeout`, and closing the window mid-turn are all
ordinary ways to end a session, and each of them used to hand back a shell
with echo off: you type and nothing appears. The restore is therefore also an
atexit hook and a signal handler, and the handler re-raises so the exit status
still says what killed the process.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

pty = pytest.importorskip("pty")
termios = pytest.importorskip("termios")

CHILD = r"""
import os, signal, sys, time
sys.path.insert(0, {root!r})
from wynxo.keys import KeyWatcher

watcher = KeyWatcher({{}})
watcher.start()
# Prove the mode really changed before the parent signals: a test that
# passes because cbreak was never entered would prove nothing.
import termios
flags = termios.tcgetattr(sys.stdin.fileno())[3]
sys.stdout.write("RAW %d %d\n" % (bool(flags & termios.ECHO),
                                  bool(flags & termios.ICANON)))
sys.stdout.flush()
time.sleep(30)
"""


def _child_holding_the_terminal(root: str):
    """Fork a child into a pty, wait until it has the terminal in cbreak."""
    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.execv(sys.executable, [sys.executable, "-c",
                                      CHILD.format(root=root)])
        finally:
            os._exit(127)

    banner = b""
    deadline = time.time() + 10
    while b"\n" not in banner and time.time() < deadline:
        try:
            banner += os.read(fd, 1024)
        except OSError:
            break
    return pid, fd, banner.decode("utf-8", "replace")


def _reap(pid, fd):
    for _ in range(100):
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return status
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    return None


@pytest.fixture
def root():
    import wynxo

    return str(os.path.dirname(os.path.dirname(os.path.abspath(wynxo.__file__))))


class TestAFatalSignalStillGivesTheTerminalBack:
    @pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP", "SIGQUIT"])
    def test_the_shell_can_still_echo(self, root, signame):
        number = getattr(signal, signame)
        pid, fd, banner = _child_holding_the_terminal(root)
        try:
            assert banner.startswith("RAW 0 0"), (
                f"the child never entered cbreak: {banner!r}")
            os.kill(pid, number)
            status = _reap(pid, fd)
            assert status is not None, f"{signame} did not end the process"
            flags = termios.tcgetattr(fd)[3]
            assert flags & termios.ECHO, (
                f"{signame} left the terminal with echo off")
            assert flags & termios.ICANON, (
                f"{signame} left the terminal in cbreak mode")
        finally:
            os.close(fd)

    def test_the_exit_status_still_names_the_signal(self, root):
        """Restoring must not turn "killed by SIGTERM" into a tidy exit --
        a supervisor reads that status."""
        pid, fd, banner = _child_holding_the_terminal(root)
        try:
            assert banner.startswith("RAW 0 0")
            os.kill(pid, signal.SIGTERM)
            status = _reap(pid, fd)
            assert os.WIFSIGNALED(status), f"raw status {status}"
            assert os.WTERMSIG(status) == signal.SIGTERM
        finally:
            os.close(fd)


class TestTheGuardIsPolite:
    def test_it_does_not_override_an_ignored_signal(self, monkeypatch):
        """SIG_IGN is a deliberate decision by whoever started the process
        -- nohup, a daemon supervisor -- and not ours to undo."""
        import wynxo.keys as keys

        monkeypatch.setattr(keys, "_guard_armed", False)
        monkeypatch.setattr(keys, "_previous_handlers", {})
        installed = []

        monkeypatch.setattr(keys.signal, "getsignal",
                            lambda _sig: keys.signal.SIG_IGN)
        monkeypatch.setattr(keys.signal, "signal",
                            lambda sig, handler: installed.append(sig))
        monkeypatch.setattr(keys.atexit, "register", lambda _fn: None)
        keys._arm_guard()
        assert installed == []

    def test_arming_twice_installs_once(self, monkeypatch):
        import wynxo.keys as keys

        monkeypatch.setattr(keys, "_guard_armed", False)
        monkeypatch.setattr(keys, "_previous_handlers", {})
        installed = []

        monkeypatch.setattr(keys.signal, "getsignal",
                            lambda _sig: keys.signal.SIG_DFL)
        monkeypatch.setattr(keys.signal, "signal",
                            lambda sig, handler: installed.append(sig))
        monkeypatch.setattr(keys.atexit, "register", lambda _fn: None)
        keys._arm_guard()
        first = list(installed)
        keys._arm_guard()
        assert installed == first and first, first

    def test_a_stopped_watcher_is_not_still_registered(self):
        """The guard walks the holders. One that has already given the
        terminal back must not be in the set, or a later signal restores
        settings that are no longer current."""
        import wynxo.keys as keys

        watcher = keys.KeyWatcher({})
        keys._HOLDERS.add(watcher)
        watcher._restore()
        assert watcher not in keys._HOLDERS
