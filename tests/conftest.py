"""Shared test setup, and a watchdog for the thing tests do worst: hang.

A hung test on a CI runner is not a failure, it is a job that sits there
until the six-hour limit and then tells you nothing. That happened on
Windows: Linux finished the suite in twenty-three seconds while both Windows
jobs stopped dead with no output at all.

faulthandler turns that into a fast failure that names the exact frame --
every thread's stack, dumped to stderr, then exit. It is in the standard
library, so this costs nothing and works the same on both platforms.
"""

import faulthandler
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Generous next to the ~30s the suite actually takes, and far below any CI
# job limit. Overridable for a slow machine or a debugger.
HANG_TIMEOUT = float(os.environ.get("WYNXO_TEST_TIMEOUT", "300"))


HANG_REPORT = Path(__file__).resolve().parents[1] / "hang-traceback.txt"

_handle = None


def pytest_configure(config):
    """Arm the watchdog.

    The dump goes to a file rather than to stderr. pytest redirects the
    stderr *file descriptor*, so a dump written there lands in a capture
    buffer -- and it is immediately followed by _exit(), which discards it.
    The log then shows a hang with no explanation, which is the whole
    problem. A real file on disk survives, and CI prints it.
    """
    global _handle
    if HANG_TIMEOUT <= 0:
        return
    try:
        HANG_REPORT.unlink()
    except OSError:
        pass
    try:
        _handle = HANG_REPORT.open("w", encoding="utf-8")
    except OSError:
        return               # nowhere to write: better no watchdog than none
    faulthandler.enable(file=_handle)
    faulthandler.dump_traceback_later(HANG_TIMEOUT, exit=True, file=_handle)


def pytest_unconfigure(config):
    global _handle
    faulthandler.cancel_dump_traceback_later()
    if _handle is not None:
        _handle.close()
        _handle = None
    # Nothing hung, so the empty report is noise.
    try:
        if HANG_REPORT.exists() and HANG_REPORT.stat().st_size == 0:
            HANG_REPORT.unlink()
    except OSError:
        pass
