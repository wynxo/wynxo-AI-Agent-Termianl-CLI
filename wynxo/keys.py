"""Single-keystroke handling while the agent is working.

prompt_toolkit owns the keyboard at the prompt, but during a turn it is idle
and nothing is reading the terminal -- so a binding registered there cannot
fire mid-generation. This reads raw keypresses in a background thread for
exactly as long as a turn is running, then puts the terminal back.

Restoring the terminal is the part that has to be right. A crash that leaves
a shell in cbreak mode with echo off is a genuinely bad outcome, so the
restore runs from a ``finally`` and is safe to call twice.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Callable

# Control characters, by the letter people actually press.
CTRL = {chr(i + 1): f"ctrl+{chr(ord('a') + i)}" for i in range(26)}


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
        self._stop = threading.Event()
        self._saved = None
        self._fd: int | None = None

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
                    continue
                try:
                    data = os.read(self._fd, 1)
                except (OSError, ValueError):
                    return
                if not data:
                    return
                self._dispatch(data.decode("utf-8", "replace"))
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
                self._stop.wait(0.05)
                continue
            try:
                char = msvcrt.getwch()
            except Exception:
                return
            self._dispatch(char)

    # -- shared ------------------------------------------------------------

    def _dispatch(self, char: str) -> None:
        handler = self.handlers.get(key_name(char))
        if handler is not None:
            with contextlib.suppress(Exception):
                handler()
            return
        if self.on_key is not None:
            with contextlib.suppress(Exception):
                self.on_key(char)

    def _restore(self) -> None:
        if self._saved is None or self._fd is None:
            return
        saved, fd = self._saved, self._fd
        self._saved, self._fd = None, None
        with contextlib.suppress(Exception):
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def describe_bindings(bindings: dict[str, str]) -> str:
    """A compact hint line: '^O thinking  ^T detail'."""
    parts = []
    for key, label in bindings.items():
        pretty = key.replace("ctrl+", "^").upper() if key.startswith("ctrl+") else key
        parts.append(f"{pretty} {label}")
    return "  ".join(parts)
