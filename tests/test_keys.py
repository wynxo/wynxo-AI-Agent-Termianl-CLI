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
        with open(os.devnull) as not_a_terminal:
            monkeypatch.setattr(sys, "stdin", not_a_terminal)
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

    def test_a_handler_runs_on_the_loop_not_the_reader_thread(self):
        """Handlers print into the console the answer is streaming to, and
        rich cannot be written from two threads at once. The watcher reads
        on a thread of its own, so what it catches is handed back."""
        import asyncio
        import threading

        where = {}

        async def go():
            watcher = KeyWatcher({"ctrl+o": lambda: where.setdefault(
                "thread", threading.current_thread().name)})
            watcher._loop = asyncio.get_running_loop()
            main = threading.current_thread().name

            done = threading.Event()

            def reader():
                watcher._dispatch("\x0f")     # as the watcher's thread does
                done.set()

            threading.Thread(target=reader).start()
            done.wait(1)
            await asyncio.sleep(0.05)          # let the loop run it
            return main

        main = asyncio.run(go())
        assert where.get("thread") == main

    def test_it_still_works_with_no_loop_to_hand_back_to(self):
        calls = []
        watcher = KeyWatcher({"ctrl+o": lambda: calls.append("o")})
        watcher._dispatch("\x0f")
        assert calls == ["o"]

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
            reader.close()          # takes the slave fd with it
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
            reader.close()          # takes the slave fd with it
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
            reader.close()          # takes the slave fd with it
            os.close(master)


class TestEscapeSequences:
    """Arrow keys, function keys and mouse reports arrive as "ESC [ ...".

    Nothing here binds them, and the watcher hands whatever no binding
    claimed to type-ahead -- so dispatching them character by character
    appended their tail to the message waiting to be sent. Pressing Up
    while the agent worked queued "[A".
    """

    def watcher(self):
        typed = []
        hit = []
        return KeyWatcher({"ctrl+o": lambda: hit.append("o")},
                          on_key=typed.append), typed, hit

    def test_an_arrow_key_types_nothing(self):
        watcher, typed, _ = self.watcher()
        watcher.feed("\x1b[A")
        assert typed == []

    def test_a_modified_arrow_types_nothing(self):
        """Ctrl-Left is "ESC [ 1 ; 5 D" -- parameters before the final."""
        watcher, typed, _ = self.watcher()
        watcher.feed("\x1b[1;5D")
        assert typed == []

    def test_a_function_key_types_nothing(self):
        """F1 is SS3: "ESC O P", two characters rather than a CSI."""
        watcher, typed, _ = self.watcher()
        watcher.feed("\x1bOP")
        assert typed == []

    def test_a_mouse_report_types_nothing(self):
        watcher, typed, _ = self.watcher()
        watcher.feed("\x1b[<0;12;4M")
        assert typed == []

    def test_an_x10_mouse_report_types_nothing(self):
        """"M" is a valid CSI final, but three raw coordinate bytes follow
        it -- and they are printable, so ending at the "M" queued them."""
        watcher, typed, _ = self.watcher()
        watcher.feed("\x1b[M !\"")
        assert typed == []

    def test_real_keys_around_a_sequence_still_land(self):
        watcher, typed, hit = self.watcher()
        watcher.feed("a\x1b[Ab\x0fc")
        assert typed == ["a", "b", "c"]
        assert hit == ["o"]

    def test_a_sequence_split_across_reads_is_still_one_key(self):
        """A 1024-byte read is not a keypress boundary: the terminal may
        hand over "ESC [" and "A" separately."""
        watcher, typed, _ = self.watcher()
        watcher.feed("\x1b[")
        watcher.feed("A")
        watcher.feed("z")
        assert typed == ["z"]

    def test_a_bare_escape_is_released_rather_than_eating_the_next_key(self):
        """Esc pressed alone opens a sequence that never finishes. The read
        loop flushes it when nothing follows; without that, the next
        keystroke -- and only that one -- would vanish."""
        watcher, typed, _ = self.watcher()
        watcher.feed("\x1b")
        watcher._flush_escape()
        watcher.feed("a")
        assert typed == ["a"]

    def test_a_runaway_sequence_cannot_eat_the_session(self):
        """A terminal that opens a sequence and never closes it would
        otherwise swallow every keystroke for the rest of the turn. The cap
        gives up on it; what matters is that the state recovers."""
        watcher, typed, hit = self.watcher()
        watcher.feed("\x1b[" + "1;" * 60)
        assert watcher._escape is None, "still inside the sequence"
        watcher.feed("a\x0f")
        assert typed[-1] == "a"
        assert hit == ["o"], "bindings stopped firing"


class TestMultiByteCharacters:
    """One byte is not one keypress. Decoding per byte turned every
    non-ASCII character into as many U+FFFD as it had bytes -- and U+FFFD
    is printable, so the garbage went into the queued message."""

    def test_a_two_byte_character_arrives_whole(self):
        typed = []
        watcher = KeyWatcher({}, on_key=typed.append)
        for byte in "é".encode():
            watcher.feed(watcher._decoder.decode(bytes([byte])))
        assert typed == ["é"]

    def test_an_emoji_arrives_whole(self):
        typed = []
        watcher = KeyWatcher({}, on_key=typed.append)
        watcher.feed(watcher._decoder.decode("🙂".encode()))
        assert typed == ["🙂"]

    def test_a_chunk_read_dispatches_every_character(self):
        typed = []
        watcher = KeyWatcher({}, on_key=typed.append)
        watcher.feed(watcher._decoder.decode(b"hello"))
        assert typed == list("hello")
