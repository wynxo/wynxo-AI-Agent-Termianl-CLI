"""Running on the terminal's alternate screen.

Fullscreen here means what it means for vim, less and htop: wynxo draws on a
second screen buffer, and when it exits the terminal is put back exactly as
it was found -- the scrollback from before the session is still there, and
nothing wynxo printed is left smeared above the shell prompt. That is the
same instinct as the uninstaller: leave no marks.

It is off by default, and deliberately so. The alternate screen has no
scrollback in most terminals, so a long session's output scrolls away for
good rather than being there to page back through. That is a real cost, and
which way it falls depends on the session -- so it is a choice, not a
default.

The whole risk of this file is in one place: failing to switch back. A
process that dies on the alternate screen leaves the user staring at a
terminal that appears to have eaten their history, and the fix -- `reset`,
or `printf '\\033[?1049l'` -- is not something anyone should have to know.
So the restore is wired to every exit that can be caught: the context
manager's finally, an atexit hook, and the signals that would otherwise
terminate the process without running either.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys

ENTER = "\x1b[?1049h"
"""Switch to the alternate screen buffer, saving the cursor."""

LEAVE = "\x1b[?1049l"
"""Switch back, restoring the screen and cursor exactly."""

# Signals that end the process without unwinding the stack, so neither the
# `finally` nor atexit would fire. SIGHUP is the terminal being closed --
# the case where a stuck alternate screen would outlive the session and
# greet the user in their next one.
_FATAL_SIGNALS = ("SIGTERM", "SIGHUP", "SIGQUIT")


def supported(stream=None) -> bool:
    """Whether this terminal can be asked to switch screens.

    A pipe or a file has no screen to switch, and TERM=dumb promises no
    escape handling at all -- writing the sequence there would put a literal
    `[?1049h` into the output.
    """
    stream = stream or sys.stdout
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError, OSError):
        return False
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", "unknown"):
        # Set deliberately, on any platform, and it means what it says.
        return False
    if not term:
        # An unset TERM on Windows is normal: the modern console handles
        # these sequences and simply does not advertise itself this way.
        return sys.platform == "win32"
    return True


class Screen:
    """The alternate screen, entered and left at most once each.

    Used as a context manager. Entering when unsupported is a no-op rather
    than an error, so callers do not have to branch: ``with Screen(on):``
    reads the same either way.
    """

    def __init__(self, enabled: bool = True, stream=None):
        self.stream = stream or sys.stdout
        self.enabled = bool(enabled) and supported(self.stream)
        self.active = False
        self._previous: dict[int, object] = {}

    # -- the two writes ----------------------------------------------------

    def _write(self, sequence: str) -> bool:
        try:
            self.stream.write(sequence)
            self.stream.flush()
            return True
        except (OSError, ValueError, AttributeError):
            # The terminal went away mid-session. Nothing to restore to, and
            # raising here would replace a cosmetic problem with a crash.
            return False

    def enter(self) -> bool:
        if self.active or not self.enabled:
            return False
        if not self._write(ENTER):
            self.enabled = False
            return False
        self.active = True
        atexit.register(self.leave)
        self._catch_fatal_signals()
        return True

    def leave(self) -> bool:
        """Switch back. Safe to call repeatedly, and from an atexit hook."""
        if not self.active:
            return False
        self.active = False        # set first: a failed write must not retry
        self._restore_signal_handlers()
        try:
            atexit.unregister(self.leave)
        except Exception:
            pass
        return self._write(LEAVE)

    # -- surviving a signal ------------------------------------------------

    def _catch_fatal_signals(self) -> None:
        """Leave the alternate screen before the process is torn down.

        The previous handler is called afterwards rather than replaced: the
        job here is to restore the screen, not to change what the signal
        means.
        """
        for name in _FATAL_SIGNALS:
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                previous = signal.getsignal(number)
                signal.signal(number, self._on_fatal_signal)
                self._previous[number] = previous
            except (ValueError, OSError, RuntimeError):
                # Not the main thread, or a signal this platform will not
                # let us take. The atexit hook still covers most exits.
                continue

    def _restore_signal_handlers(self) -> None:
        for number, previous in list(self._previous.items()):
            try:
                if callable(previous) or previous in (signal.SIG_DFL, signal.SIG_IGN):
                    signal.signal(number, previous)
            except (ValueError, OSError, RuntimeError, TypeError):
                pass
        self._previous.clear()

    def _on_fatal_signal(self, number, frame):
        previous = self._previous.get(number, signal.SIG_DFL)
        self.leave()
        if callable(previous):
            return previous(number, frame)
        # Re-raise with the original disposition so the exit status is the
        # one the sender asked for, rather than a clean 0 that lies.
        try:
            signal.signal(number, signal.SIG_DFL)
            os.kill(os.getpid(), number)
        except (ValueError, OSError, RuntimeError):
            raise SystemExit(128 + int(number)) from None

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "Screen":
        self.enter()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.leave()
        return False


def note(enabled: bool, unicode: bool = True) -> str:
    """The one line shown at start-up about which screen this is."""
    if not enabled:
        return ""
    dot = "·" if unicode else "-"
    return (f"fullscreen  {dot}  your terminal is restored on exit, "
            f"but this screen does not scroll back")
