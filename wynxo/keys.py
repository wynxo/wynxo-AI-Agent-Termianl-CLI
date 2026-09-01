"""Single-keystroke handling while the agent is working.

prompt_toolkit owns the keyboard at the prompt, but during a turn it is idle
and nothing is reading the terminal -- so a binding registered there cannot
fire mid-generation. This reads raw keypresses in a background thread for
exactly as long as a turn is running, then puts the terminal back.

Restoring the terminal is the part that has to be right. A shell left in
cbreak mode with echo off is a genuinely bad outcome -- you type and see
nothing -- so the restore runs from a ``finally``, is safe to call twice,
and, because a ``finally`` only runs for exits Python gets to see, is also
wired to ``atexit`` and to the fatal signals below.
"""

from __future__ import annotations

import asyncio
import atexit
import codecs
import contextlib
import os
import signal
import sys
import threading
from typing import Callable

# -- getting the terminal back, whatever happens ---------------------------
#
# A ``finally`` covers a return, an exception and a cancellation. It does not
# cover SIGTERM or SIGHUP: those terminate the process with their default
# disposition, so no Python code runs at all and the terminal keeps whatever
# mode it was left in. `kill`, a session manager, `timeout`, and closing the
# window while a turn is running are all ordinary ways to end a session, and
# every one of them handed back a shell with echo off.
#
# So the restore is also armed as a signal handler and an atexit hook, while
# any watcher holds the terminal. The handler puts the terminal back and then
# dies of the original signal with the default disposition, so the exit
# status a caller sees is unchanged.

_HOLDERS: "set[KeyWatcher]" = set()
"""Watchers currently holding a terminal in cbreak mode. Usually one."""

_FATAL = ("SIGTERM", "SIGHUP", "SIGQUIT")
_previous_handlers: dict = {}
_guard_armed = False


def _restore_everything() -> None:
    for watcher in list(_HOLDERS):
        watcher._restore()


def _die_after_restoring(signum, _frame) -> None:
    _restore_everything()
    previous = _previous_handlers.get(signum)
    with contextlib.suppress(Exception):
        signal.signal(signum, previous if callable(previous) else signal.SIG_DFL)
    # Re-raise rather than exit: the caller's `$?` should say the process died
    # of this signal, which is what it would have done without the handler.
    with contextlib.suppress(Exception):
        os.kill(os.getpid(), signum)


def _arm_guard() -> None:
    """Install the hooks, once, the first time a terminal is held."""
    global _guard_armed
    if _guard_armed:
        return
    _guard_armed = True
    atexit.register(_restore_everything)
    for name in _FATAL:
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            # Only from the main thread, and never over a handler somebody
            # else installed deliberately -- SIG_IGN means "this process has
            # decided to ignore it" and is not ours to override.
            existing = signal.getsignal(number)
            if existing is signal.SIG_IGN:
                continue
            _previous_handlers[number] = existing
            signal.signal(number, _die_after_restoring)
        except (ValueError, OSError, RuntimeError):
            # Not the main thread, or a platform without the signal. The
            # finally-based restore still covers every ordinary exit.
            continue


# Control characters, by the letter people actually press.
CTRL = {chr(i + 1): f"ctrl+{chr(ord('a') + i)}" for i in range(26)}

ESC = "\x1b"

# The last byte of a CSI sequence ("ESC [ ... final"), by the range the
# standard reserves for it. Everything before it is a parameter or an
# intermediate byte and belongs to the same keypress.
CSI_FINAL = tuple(chr(i) for i in range(0x40, 0x7F))

MAX_ESCAPE = 64
"""A cap on how much a half-read escape sequence may swallow. Nothing a
keyboard or a mouse report sends comes close; a terminal that starts a
sequence and never finishes it would otherwise eat every later keystroke."""


def key_name(char: str) -> str:
    """'\\x0f' -> 'ctrl+o'. Printable characters come back unchanged."""
    if char in CTRL:
        return CTRL[char]
    return char


class KeyWatcher:
    """Dispatches single keypresses to handlers while running.

    Handlers are called on the watcher's thread, so they must be cheap and
    must not touch the agent's state directly -- setting a flag is the
    intended use.
    """

    def __init__(self, handlers: dict[str, Callable[[], None]],
                 on_key: Callable[[str], None] | None = None):
        self.handlers = handlers
        self.on_key = on_key
        """Everything no binding claimed. This is how type-ahead gets its
        characters without a second reader competing for stdin."""
        self._thread: threading.Thread | None = None
        self._loop = None
        """The event loop to hand keypresses back to, so handlers run where
        everything else does rather than on this thread."""
        self._stop = threading.Event()
        self._saved = None
        self._fd: int | None = None
        self._restoring = threading.Lock()
        """The reader thread restores from its ``finally`` and ``stop()``
        restores after joining it; when the join times out both run. The
        lock is what makes "safe to call twice" true across threads rather
        than only in sequence."""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        """Bytes arrive in whatever sizes the terminal hands over, and a
        keypress is not a byte: reading and decoding one at a time turned
        every non-ASCII character into as many replacement characters as it
        had bytes, and typing "e" with an accent mid-turn queued garbage."""
        self._escape: str | None = None
        """The escape sequence being read, without its leading ESC. None
        when we are not inside one."""

    # -- lifecycle ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """Only worth starting when a real terminal is attached."""
        try:
            return sys.stdin.isatty()
        except (AttributeError, ValueError):
            return False

    def start(self) -> None:
        if self._thread is not None or not self.available:
            return
        self._stop.clear()
        # Captured here because start() is called from the turn, which is
        # async. Handlers are then run on the loop instead of on the reader
        # thread: they print into the same console the answer is streaming
        # to, and rich is not safe to write from two threads at once -- a
        # Ctrl-O landing mid-stream could interleave the backlog with the
        # line being written.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        if sys.platform == "win32":
            target = self._loop_windows
        else:
            if not self._enter_cbreak():
                return
            target = self._loop_posix
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=0.5)
        self._restore()

    def __enter__(self) -> "KeyWatcher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- posix -------------------------------------------------------------

    def _enter_cbreak(self) -> bool:
        try:
            import termios
            import tty
        except ImportError:
            return False
        try:
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            # cbreak, not raw: ISIG stays enabled so Ctrl-C still raises
            # SIGINT and the existing interrupt path keeps working.
            tty.setcbreak(self._fd)
            self._free_the_bound_keys(termios)
            _HOLDERS.add(self)
            _arm_guard()
            return True
        except Exception:
            self._saved = None
            self._fd = None
            return False

    def _free_the_bound_keys(self, termios) -> None:
        """Stop the terminal driver from eating keys we bind.

        On BSD and macOS, Ctrl-O is VDISCARD and Ctrl-V is VLNEXT: the driver
        acts on them itself and the byte never reaches us. Linux documents
        VDISCARD as not implemented, which is why ^O worked there and
        silently did nothing on a Mac.

        IEXTEN is what enables both, and it is separate from ISIG -- so
        clearing it frees the keys without touching Ctrl-C.
        """
        try:
            attrs = termios.tcgetattr(self._fd)
            attrs[3] &= ~termios.IEXTEN          # lflag
            # Belt and braces: disable the characters individually too, for
            # drivers that honour them regardless of IEXTEN.
            disable = getattr(termios, "_POSIX_VDISABLE", 0)
            for name in ("VDISCARD", "VLNEXT", "VREPRINT", "VSTATUS"):
                if (index := getattr(termios, name, None)) is not None:
                    try:
                        attrs[6][index] = disable
                    except (IndexError, TypeError):
                        pass
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        except Exception:
            # The keys stay bound to the driver; the agent still works.
            pass

    def _loop_posix(self) -> None:
        import select

        try:
            while not self._stop.is_set():
                try:
                    ready, _, _ = select.select([self._fd], [], [], 0.15)
                except (OSError, ValueError):
                    return
                if not ready:
                    # Nothing for 150ms. Any escape sequence still open was
                    # a bare Esc keypress, not the start of an arrow key --
                    # release it rather than let it swallow the next key.
                    self._flush_escape()
                    continue
                try:
                    # A chunk, not a byte: a paste, a mouse report and a
                    # multi-byte character all arrive as several bytes, and
                    # a read per byte costs a select syscall each.
                    data = os.read(self._fd, 1024)
                except (OSError, ValueError):
                    return
                if not data:
                    return
                self.feed(self._decoder.decode(data))
        finally:
            self._restore()

    # -- windows -----------------------------------------------------------

    def _loop_windows(self) -> None:
        try:
            import msvcrt
        except ImportError:
            return
        while not self._stop.is_set():
            if not msvcrt.kbhit():
                # Same as the posix loop: an escape sequence that stopped
                # arriving was a bare Esc, and must not swallow the next key.
                self._flush_escape()
                self._stop.wait(0.05)
                continue
            try:
                char = msvcrt.getwch()
            except Exception:
                return
            # The console reports arrow keys, function keys and the numeric
            # keypad as a two-character sequence led by NUL or 0xE0. Read
            # the scan code and drop the pair: without this, pressing Up
            # mid-turn typed "a" with a grave accent into the queued line.
            if char in ("\x00", "\xe0"):
                with contextlib.suppress(Exception):
                    msvcrt.getwch()
                continue
            self.feed(char)

    # -- shared ------------------------------------------------------------

    def feed(self, text: str) -> None:
        """Push decoded input through the escape filter, then dispatch.

        Arrow keys, Home/End, function keys and mouse reports all arrive as
        "ESC [ ...". Dispatching those characters one by one meant no
        binding claimed them and the tail fell through to type-ahead, so
        pressing Up while the agent worked appended "[A" to the message
        waiting to be sent. They are keypresses nothing here binds, so they
        are read to their end and dropped.
        """
        for char in text:
            if self._escape is not None:
                self._escape += char
                if self._escape_ended() or len(self._escape) >= MAX_ESCAPE:
                    self._escape = None
                continue
            if char == ESC:
                self._escape = ""
                continue
            self._dispatch(char)

    def _escape_ended(self) -> bool:
        """Whether the sequence read so far is complete."""
        seq = self._escape or ""
        if not seq:
            return False
        if seq.startswith("[M"):
            # X10 mouse reporting: "M" is a valid CSI final, but three raw
            # coordinate bytes follow it and any of them can be printable.
            # Ending at the "M" would type the coordinates into the queue.
            return len(seq) >= 5
        if seq[0] == "[":                     # CSI: parameters, then a final
            return len(seq) > 1 and seq[-1] in CSI_FINAL
        if seq[0] == "O":                     # SS3: exactly one more
            return len(seq) > 1
        # Alt-<key> and anything else: one character and done.
        return True

    def _flush_escape(self) -> None:
        """Give up on a sequence that stopped arriving -- a bare Esc."""
        self._escape = None

    def _dispatch(self, char: str) -> None:
        handler = self.handlers.get(key_name(char))
        if handler is not None:
            self._hand_over(handler)
            return
        if self.on_key is not None:
            self._hand_over(self.on_key, char)

    def _hand_over(self, handler, *args) -> None:
        """Run a handler where the rest of the session runs."""
        loop = self._loop
        if loop is None or loop.is_closed():
            with contextlib.suppress(Exception):
                handler(*args)
            return

        def call() -> None:
            with contextlib.suppress(Exception):
                handler(*args)

        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(call)

    def _restore(self) -> None:
        with self._restoring:
            _HOLDERS.discard(self)
            if self._saved is None or self._fd is None:
                return
            saved, fd = self._saved, self._fd
            self._saved, self._fd = None, None
        with contextlib.suppress(Exception):
            import termios

            # TCSADRAIN, not TCSANOW: pending output is flushed to the screen
            # under the *old* settings first, so a half-written line is not
            # left behind by the mode change.
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
