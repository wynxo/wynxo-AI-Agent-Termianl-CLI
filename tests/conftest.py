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


# -- platform boundaries ------------------------------------------------------

# These tests deliberately exercise a host facility that does not exist on
# Windows, or feed a POSIX-shaped fixture to code whose Windows behavior is
# covered elsewhere with Windows-native inputs. Running them unchanged on a
# Windows worker tests the fixture rather than Wynxo.
_WINDOWS_POSIX_ONLY = {
    "tests/test_hardening.py::TestCtrlCSurvivesAPromptInsideATurn::test_a_prompt_really_does_remove_the_handler":
        "asyncio signal handlers are POSIX-only",
    "tests/test_hardening.py::TestAJavaProjectGetsATestCommand::test_a_committed_wrapper_wins":
        "a bare executable gradlew is the POSIX wrapper; Windows uses gradlew.bat",
    "tests/test_qa_regressions.py::TestBackgroundJobsDieWithTheSession::test_shutdown_stops_a_running_job":
        "fixture is a POSIX while/sleep shell loop",
    "tests/test_windows_surface.py::TestApplicationLaunching::test_a_shortcut_off_windows_is_a_launch_failure_not_a_crash":
        "this assertion is explicitly about the non-Windows branch",
    "tests/test_polish_regressions.py::test_path_shortening_only_replaces_the_home_directory":
        "fixture intentionally uses POSIX /home paths",
    "tests/test_ui_regressions.py::TestShortenPathHonoursItsBudget::test_the_home_directory_still_becomes_a_tilde":
        "fixture joins a Windows home with a POSIX slash",
}


def pytest_collection_modifyitems(config, items):
    if sys.platform != "win32":
        return
    for item in items:
        reason = _WINDOWS_POSIX_ONLY.get(item.nodeid)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _stabilise_host_resource_assumptions(request, monkeypatch):
    """Keep logic tests independent of the size of the CI machine.

    One GPU-placement test asks whether a card can hold the model's weights.
    On a small macOS runner the *host RAM* becomes the tighter limit first,
    so the product correctly reports paging and the test accidentally checks
    a different branch. Pin RAM only for that one branch test; the dedicated
    memory-pressure tests continue to control and exercise low-RAM behavior.
    """
    if request.node.nodeid.endswith(
            "test_gpu_placement.py::TestItSaysSoWithoutBeingAsked::"
            "test_a_card_too_small_for_the_weights_is_told_the_truth"):
        monkeypatch.setattr("wynxo.platforms.total_memory",
                            lambda: 64_000_000_000)


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
