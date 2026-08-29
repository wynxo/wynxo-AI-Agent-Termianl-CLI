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

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Generous next to the ~35s the suite actually takes, including on the
# slower Windows runners, and short enough that the dump lands before
# anything else ends the job -- the first attempt at this was set to 300s
# and the run was superseded at 269, losing the diagnostic entirely.
# Overridable for a slow machine or a debugger.
HANG_TIMEOUT = float(os.environ.get("WYNXO_TEST_TIMEOUT", "150"))


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


# -- global interpreter state -------------------------------------------------

_PRISTINE_OS_NAME = os.name


def pytest_runtest_teardown(item, nextitem):
    """Fail the test that leaked process-wide state, not its innocent
    successors.

    ``os.name`` is the one that hurt. Tests reached for
    ``monkeypatch.setattr("wynxo.testing.os.name", "nt")`` to exercise a
    Windows branch, but ``wynxo.testing.os`` *is* the interpreter's one
    ``os`` module. pathlib reads ``os.name`` to choose its flavour, so from
    that moment every ``Path()`` in the process returned a ``WindowsPath``
    and raised ``NotImplementedError`` on a POSIX box -- including the
    ``Path()`` pytest itself uses to format a traceback. The result was an
    INTERNALERROR that killed the session partway through, so a third of
    the suite silently never ran and CI had been red for weeks without
    anyone being able to see why.

    Patch ``wynxo.testing._is_windows`` instead; it exists for this.
    """
    if os.name != _PRISTINE_OS_NAME:
        leaked, os.name = os.name, _PRISTINE_OS_NAME   # repair, then report
        raise pytest.UsageError(
            f"{item.nodeid} left os.name as {leaked!r} (was "
            f"{_PRISTINE_OS_NAME!r}). Patching os.name breaks pathlib for "
            f"the whole process; patch wynxo.testing._is_windows instead."
        )
