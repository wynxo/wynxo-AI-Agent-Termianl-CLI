"""systemd-style status lines.

Deliberately dependency-free so ``install.py`` can import it before anything
is installed, and so it works on a terminal with no colour support.

    [  OK  ] Python 3.12.3
    [ WARN ] context window is small
    [FAILED] cannot reach Ollama
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time

OK = "OK"
WARN = "WARN"
FAILED = "FAILED"
BUSY = "...."
SKIP = "SKIP"

_COLOURS = {OK: "32", WARN: "33", FAILED: "31", BUSY: "36", SKIP: "90"}
_WIDTH = 6   # the widest label, "FAILED"


def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # Ask the console to interpret VT sequences; harmless if already on.
        os.system("")
    return os.environ.get("TERM") != "dumb"


class Status:
    """Prints aligned status lines, optionally with a live spinner."""

    SPINNER = "|/-\\"

    def __init__(self, colour: bool | None = None, stream=None):
        self.stream = stream or sys.stdout
        self.colour = _supports_colour() if colour is None else colour
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None
        self._spinner_text = ""

    # -- primitives --------------------------------------------------------

    def _tag(self, state: str) -> str:
        label = state.center(_WIDTH)
        if not self.colour:
            return f"[{label}]"
        return f"[\033[1;{_COLOURS.get(state, '37')}m{label}\033[0m]"

    def _dim(self, text: str) -> str:
        return f"\033[2m{text}\033[0m" if self.colour else text

    def _bold(self, text: str) -> str:
        return f"\033[1m{text}\033[0m" if self.colour else text

    def line(self, state: str, message: str, detail: str = "") -> None:
        self._stop_spinner()
        text = f"{self._tag(state)} {message}"
        if detail:
            text += f" {self._dim(detail)}"
        print(text, file=self.stream, flush=True)

    # -- the usual four ----------------------------------------------------

    def ok(self, message: str, detail: str = "") -> None:
        self.line(OK, message, detail)

    def warn(self, message: str, detail: str = "") -> None:
        self.line(WARN, message, detail)

    def fail(self, message: str, detail: str = "") -> None:
        self.line(FAILED, message, detail)

    def skip(self, message: str, detail: str = "") -> None:
        self.line(SKIP, message, detail)

    def note(self, message: str) -> None:
        """An indented continuation line, aligned under the message column."""
        self._stop_spinner()
        print(f"{' ' * (_WIDTH + 2)} {self._dim(message)}", file=self.stream, flush=True)

    def header(self, message: str) -> None:
        self._stop_spinner()
        print(f"\n{self._bold(message)}", file=self.stream, flush=True)

    # -- in-progress -------------------------------------------------------

    def busy(self, message: str) -> None:
        """Show a spinning line that a later ok()/fail() replaces."""
        self._stop_spinner()
        if not self.colour or not self.stream.isatty():
            # No cursor control: print a plain line and leave it.
            self.line(BUSY, message)
            return
        self._spinner_text = message
        self._spinner_stop = threading.Event()
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()

    def update(self, message: str) -> None:
        """Change the text of the running spinner."""
        if self._spinner_thread is not None:
            self._spinner_text = message
        else:
            self.line(BUSY, message)

    def _spin(self) -> None:
        i = 0
        stop = self._spinner_stop
        assert stop is not None
        while not stop.is_set():
            frame = self.SPINNER[i % len(self.SPINNER)]
            tag = f"[\033[1;36m{frame.center(_WIDTH)}\033[0m]"
            width = shutil.get_terminal_size((80, 24)).columns
            text = f"\r{tag} {self._spinner_text}"[: width - 1]
            self.stream.write(text + "\033[K")
            self.stream.flush()
            i += 1
            stop.wait(0.12)

    def _stop_spinner(self) -> None:
        stop, thread = self._spinner_stop, self._spinner_thread
        self._spinner_stop = self._spinner_thread = None
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=0.3)
            self.stream.write("\r\033[K")
            self.stream.flush()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._stop_spinner()

    def __enter__(self) -> "Status":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Timer:
    """Times a step so it can be reported as a detail."""

    def __init__(self) -> None:
        self.started = time.monotonic()

    def elapsed(self) -> str:
        seconds = time.monotonic() - self.started
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{seconds // 60:.0f}m{seconds % 60:.0f}s"
