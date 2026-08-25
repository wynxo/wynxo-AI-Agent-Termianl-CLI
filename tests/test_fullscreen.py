"""The alternate screen, and above all: coming back off it.

The failure that matters here is not "fullscreen did not turn on". It is a
process that dies while still on the alternate screen, leaving the user in a
terminal that appears to have eaten their scrollback, fixable only by
knowing to type `reset`. So most of this file is about exits.
"""

from __future__ import annotations

import io
import os
import select
import signal
import sys
import time

import pytest

from wynxo.fullscreen import ENTER, LEAVE, Screen, note, supported


class FakeTTY(io.StringIO):
    def __init__(self, tty: bool = True):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class TestWhetherToEvenTry:
    def test_a_pipe_has_no_screen_to_switch(self):
        assert supported(FakeTTY(tty=False)) is False

    def test_a_dumb_terminal_is_refused(self, monkeypatch):
        monkeypatch.setenv("TERM", "dumb")
        assert supported(FakeTTY()) is False

    def test_a_real_terminal_is_accepted(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        assert supported(FakeTTY()) is True

    def test_a_stream_that_raises_on_isatty_is_refused(self):
        class Hostile:
            def isatty(self):
                raise ValueError("closed")

        assert supported(Hostile()) is False

    def test_disabled_means_nothing_is_written(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        stream = FakeTTY()
        with Screen(enabled=False, stream=stream) as screen:
            assert screen.active is False
        assert stream.getvalue() == ""


class TestTheTwoWrites:
    @pytest.fixture(autouse=True)
    def _terminal(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")

    def test_it_enters_and_leaves_exactly_once_each(self):
        stream = FakeTTY()
        with Screen(enabled=True, stream=stream):
            pass
        assert stream.getvalue() == ENTER + LEAVE

    def test_entering_twice_does_not_write_twice(self):
        """Two ENTERs mean the first screen's content is what LEAVE restores
        to -- the user lands back inside wynxo instead of their shell."""
        stream = FakeTTY()
        screen = Screen(enabled=True, stream=stream)
        screen.enter()
        assert screen.enter() is False
        screen.leave()
        assert stream.getvalue() == ENTER + LEAVE

    def test_leaving_twice_is_harmless(self):
        stream = FakeTTY()
        screen = Screen(enabled=True, stream=stream)
        screen.enter()
        screen.leave()
        assert screen.leave() is False
        assert stream.getvalue().count(LEAVE) == 1

    def test_leaving_without_entering_writes_nothing(self):
        stream = FakeTTY()
        assert Screen(enabled=True, stream=stream).leave() is False
        assert stream.getvalue() == ""

    def test_a_terminal_that_vanishes_does_not_raise(self):
        """Losing the terminal mid-session is cosmetic. Raising here would
        turn it into a crash on the way out of a working session."""
        class Gone(FakeTTY):
            def write(self, _):
                raise OSError("broken pipe")

        screen = Screen(enabled=True, stream=Gone())
        assert screen.enter() is False
        assert screen.active is False

    def test_a_write_failure_on_the_way_out_still_marks_it_inactive(self):
        stream = FakeTTY()
        screen = Screen(enabled=True, stream=stream)
        screen.enter()

        def refuse(_):
            raise OSError("gone")

        stream.write = refuse
        screen.leave()
        assert screen.active is False, "a stuck flag would retry forever"

    def test_an_exception_inside_still_leaves_the_screen(self):
        stream = FakeTTY()
        with pytest.raises(RuntimeError):
            with Screen(enabled=True, stream=stream):
                raise RuntimeError("boom")
        assert stream.getvalue().endswith(LEAVE)

    def test_the_context_manager_does_not_swallow_the_exception(self):
        with pytest.raises(RuntimeError):
            with Screen(enabled=True, stream=FakeTTY()):
                raise RuntimeError("boom")


class TestSignalHandling:
    @pytest.fixture(autouse=True)
    def _terminal(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")

    def test_the_previous_handler_is_put_back_on_leaving(self):
        """wynxo is not the only thing that wants SIGTERM. Keeping the hook
        after the screen is gone would quietly change what kill does."""
        if not hasattr(signal, "SIGTERM"):
            pytest.skip("no SIGTERM here")

        original = signal.getsignal(signal.SIGTERM)
        try:
            screen = Screen(enabled=True, stream=FakeTTY())
            screen.enter()
            assert signal.getsignal(signal.SIGTERM) is not original
            screen.leave()
            assert signal.getsignal(signal.SIGTERM) is original
        finally:
            signal.signal(signal.SIGTERM, original)

    def test_an_existing_handler_is_called_not_replaced(self):
        if not hasattr(signal, "SIGTERM"):
            pytest.skip("no SIGTERM here")

        called = []
        original = signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGTERM, lambda *a: called.append(a))
            stream = FakeTTY()
            screen = Screen(enabled=True, stream=stream)
            screen.enter()
            signal.raise_signal(signal.SIGTERM)
            assert called, "the previous handler was dropped"
            assert stream.getvalue().endswith(LEAVE)
        finally:
            signal.signal(signal.SIGTERM, original)


class TestTheStartupNote:
    def test_it_says_nothing_when_off(self):
        assert note(False) == ""

    def test_it_warns_about_the_missing_scrollback(self):
        assert "scroll" in note(True).lower()

    def test_ascii_terminals_get_no_unicode(self):
        assert note(True, unicode=False).isascii()


# -- the part that actually proves it ------------------------------------

def _run_under_pty(code: str, kill_with: int | None = None) -> bytes:
    """Run code on a real pty and return every byte it wrote."""
    import pty

    pid, fd = pty.fork()
    if pid == 0:                                  # pragma: no cover - child
        os.environ["TERM"] = "xterm-256color"
        os.execv(sys.executable, [sys.executable, "-c", code])

    out, sent, deadline = b"", False, time.time() + 10
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        if kill_with and not sent and b"READY" in out:
            os.kill(pid, kill_with)
            sent = True
    try:
        os.waitpid(pid, 0)
    except (ChildProcessError, OSError):
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    return out


PREAMBLE = (
    "import sys, os, time\n"
    f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})\n"
    "from wynxo.fullscreen import Screen\n"
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs a pty")
class TestOnARealTerminal:
    """Escape sequences are only real once a terminal has seen them."""

    def test_the_sequences_reach_the_wire_in_order(self):
        out = _run_under_pty(PREAMBLE + (
            "with Screen(True):\n"
            "    print('inside')\n"
        ))
        assert b"\x1b[?1049h" in out and b"\x1b[?1049l" in out
        assert out.find(b"\x1b[?1049h") < out.find(b"\x1b[?1049l")

    def test_an_uncaught_exception_still_gives_the_terminal_back(self):
        out = _run_under_pty(PREAMBLE + (
            "with Screen(True):\n"
            "    raise RuntimeError('boom')\n"
        ))
        assert b"\x1b[?1049l" in out, "the user is left on the alternate screen"

    def test_sys_exit_from_inside_gives_the_terminal_back(self):
        out = _run_under_pty(PREAMBLE + (
            "s = Screen(True); s.enter()\n"
            "sys.exit(3)\n"
        ))
        assert b"\x1b[?1049l" in out

    @pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="no SIGTERM")
    def test_being_killed_gives_the_terminal_back(self):
        out = _run_under_pty(PREAMBLE + (
            "s = Screen(True); s.enter()\n"
            "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        ), kill_with=signal.SIGTERM)
        assert b"\x1b[?1049l" in out

    @pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP")
    def test_closing_the_terminal_gives_it_back(self):
        """SIGHUP is the window being closed -- the case where a stuck
        screen would outlive the session and greet the next one."""
        out = _run_under_pty(PREAMBLE + (
            "s = Screen(True); s.enter()\n"
            "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        ), kill_with=signal.SIGHUP)
        assert b"\x1b[?1049l" in out

    def test_wynxo_never_calls_os_exit(self):
        """os._exit skips atexit by design, so it is the one exit nothing
        here can cover. It is only safe because wynxo does not use it."""
        import pathlib

        package = pathlib.Path(__file__).parent.parent / "wynxo"
        offenders = [p.name for p in package.rglob("*.py")
                     if "os._exit" in p.read_text(encoding="utf-8")]
        assert offenders == []


class TestTheSettingItself:
    def test_it_is_off_unless_asked_for(self):
        """'support fullscreen and not fully screen': available, not imposed."""
        from wynxo.config import Config

        assert Config().fullscreen is False

    def test_both_flags_exist_and_disagree(self):
        from wynxo.cli import build_parser

        parser = build_parser()
        assert parser.parse_args(["--fullscreen"]).fullscreen is True
        assert parser.parse_args(["--no-fullscreen"]).no_fullscreen is True

    def test_it_is_listed_in_help(self):
        from wynxo.cli import COMMANDS

        assert "/fullscreen" in COMMANDS


class TestTheChatLayoutOwnsTheScreen:
    """The bug this class exists for: two owners of the same escape.

    prompt_toolkit builds the chat layout's Application with full_screen=True,
    so it enters and leaves the alternate screen itself. wynxo used to wrap
    that in a Screen of its own, which made /fullscreen off write the leave
    sequence while the application was still running -- switching the
    terminal back to the primary screen with a full-screen interface still
    painting on it. The result was wynxo drawn over the user's shell, and
    left there after it exited.
    """

    def test_nothing_asks_for_a_screen_that_is_always_on(self):
        """Checked as code: the wrapper is gone from every call site.

        A behavioural test cannot reach this -- the call is inside the
        function that runs the whole session.
        """
        import ast
        import inspect

        from wynxo import cli as cli_module

        tree = ast.parse(inspect.getsource(cli_module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if ast.unparse(node.func) != "fullscreen.Screen":
                continue
            for argument in list(node.args) + [k.value for k in node.keywords]:
                assert ast.unparse(argument) != "True", (
                    "the chat layout's Application already owns the alternate "
                    "screen; a second owner is what broke /fullscreen")

    def _repl(self):
        """A Repl with just enough on it for cmd_fullscreen."""
        from wynxo.cli import Repl

        repl = Repl.__new__(Repl)
        repl.chat = object()          # any chat layout at all
        repl.screen = Screen(enabled=True, stream=FakeTTY())
        repl.screen.active = True     # as it would be, mid-session
        repl.config = _StubConfig()
        repl.ui = _StubUI()
        return repl

    def test_turning_it_off_does_not_switch_the_screen_underneath(self):
        import asyncio

        repl = self._repl()
        asyncio.run(repl.cmd_fullscreen(["off"]))
        assert repl.screen.active is True, (
            "the running application is still drawing on that screen")
        assert LEAVE not in repl.screen.stream.getvalue()

    def test_turning_it_off_changes_something_real(self):
        """A setting that says it changed while nothing did is worse than
        no setting: it is the layout that has to go."""
        import asyncio

        repl = self._repl()
        asyncio.run(repl.cmd_fullscreen(["off"]))
        assert repl.config.chat_layout is False
        assert repl.config.saved is True

    def test_changing_your_mind_puts_the_layout_back(self):
        import asyncio

        repl = self._repl()
        asyncio.run(repl.cmd_fullscreen(["off"]))
        asyncio.run(repl.cmd_fullscreen(["on"]))
        assert repl.config.chat_layout is True

    def test_turning_it_on_never_writes_the_enter_sequence_twice(self):
        import asyncio

        repl = self._repl()
        asyncio.run(repl.cmd_fullscreen(["on"]))
        assert ENTER not in repl.screen.stream.getvalue()


class TestTheFlagsAgree:
    def _config(self, *argv):
        from wynxo.cli import apply_flags, build_parser
        from wynxo.config import Config

        config = Config()
        apply_flags(config, build_parser().parse_args(list(argv)))
        return config

    def test_no_fullscreen_gives_the_scrolling_terminal(self):
        """Under the chat layout the flag had nothing to turn off, so it
        silently did nothing -- and the terminal was taken over anyway."""
        config = self._config("--no-fullscreen")
        assert config.fullscreen is False
        assert config.chat_layout is False

    def test_asking_for_both_keeps_the_chat_layout(self):
        """--chat is the specific request; it wins."""
        assert self._config("--no-fullscreen", "--chat").chat_layout is True

    def test_plain_start_is_unchanged(self):
        config = self._config()
        assert config.chat_layout is True
        assert config.fullscreen is False


class _StubConfig:
    chat_layout = True
    fullscreen = True
    saved = False

    def save(self):
        self.saved = True


class _StubUI:
    class _G:
        dot = "-"
        unicode = False

    g = _G()

    def info(self, *_a, **_k):
        pass

    def success(self, *_a, **_k):
        pass

    def warn(self, *_a, **_k):
        pass
