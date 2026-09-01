"""Every place the code branches on Windows, driven on this machine.

This is not a substitute for running wynxo on Windows -- see the note at the
bottom for what still needs the real thing. It is the strongest determinis-
tic coverage available from here: each branch is entered deliberately and
its output checked, so a change that breaks Windows fails here rather than
on somebody's desktop.

The platform is faked through each module's own indirection, never by
setting ``os.name``. Patching that reaches the one shared os module and
makes ``pathlib.Path()`` return WindowsPath, which cannot be instantiated on
POSIX -- a global monkeypatch of exactly that kind once took the whole suite
down at 55%, which is why testing._is_windows() exists at all.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest


class TestPowerShellQuoting:
    """PowerShell treats a quoted string as a string unless the call
    operator is in front of it, and backslashes are shlex escapes on POSIX
    -- so a Windows path must never be shlex'd."""

    def test_a_path_without_spaces_is_left_bare(self):
        from wynxo import testing

        with patch.object(testing, "_is_windows", lambda: True):
            assert testing._runnable(
                pathlib.PurePosixPath("C:/py/python.exe")) == "C:/py/python.exe"

    def test_a_path_with_spaces_gets_the_call_operator(self):
        from wynxo import testing

        with patch.object(testing, "_is_windows", lambda: True):
            assert testing._runnable(
                pathlib.PurePosixPath("C:/Program Files/py.exe")) \
                == '& "C:/Program Files/py.exe"'

    def test_an_argument_with_spaces_is_double_quoted(self):
        from wynxo import testing

        with patch.object(testing, "_is_windows", lambda: True):
            assert testing.quote_arg("a b") == '"a b"'

    def test_backslashes_survive_quoting(self):
        r"""shlex on POSIX would eat the separators in .venv\Scripts\."""
        from wynxo import testing

        with patch.object(testing, "_is_windows", lambda: True):
            assert testing.quote_arg(r".venv\Scripts\python.exe") \
                == r".venv\Scripts\python.exe"

    def test_the_call_operator_form_parses_back_to_one_argument(self):
        from wynxo import testing

        assert testing._interpreter_argv(r'& "C:\Program Files\py.exe"') \
            == [r"C:\Program Files\py.exe"]

    def test_a_backslash_path_parses_back_whole(self):
        from wynxo import testing

        assert testing._interpreter_argv(r".venv\Scripts\python.exe") \
            == [r".venv\Scripts\python.exe"]


class TestProcessesAndSignals:
    """Windows has no process groups in the POSIX sense; taskkill /T is the
    equivalent, and /F the only reliable form of it."""

    def test_a_command_gets_its_own_process_group(self):
        from wynxo.tools import shell

        with patch.object(shell.os, "name", "nt"):
            flags = shell._new_process_group()
        assert "start_new_session" not in flags

    def test_stopping_a_command_uses_taskkill(self):
        from wynxo.tools import shell

        process = MagicMock()
        process.pid = 4321
        with patch.object(shell.os, "name", "nt"), \
                patch.object(shell.subprocess, "run") as run:
            shell._signal_group(process, terminate=True)
        assert run.call_args[0][0] == ["taskkill", "/F", "/T", "/PID", "4321"]
        assert not process.terminate.called, "taskkill already did it"

    def test_taskkill_failing_falls_back_to_the_single_process(self):
        from wynxo.tools import shell

        process = MagicMock()
        process.pid = 4321
        with patch.object(shell.os, "name", "nt"), \
                patch.object(shell.subprocess, "run", side_effect=OSError):
            shell._signal_group(process, terminate=True)
        assert process.terminate.called

    def test_liveness_is_not_probed_with_signal_zero(self):
        from wynxo.tools import shell

        class Running:
            pid = 1
            returncode = None

        with patch.object(shell.os, "name", "nt"), \
                patch.object(shell.os, "kill",
                             side_effect=AssertionError("os.kill on Windows")):
            assert shell._gone(Running()) is False


class TestApplicationLaunching:
    def test_a_shortcut_is_started_through_cmd(self):
        """cmd /c start keeps ShellExecute semantics while handing the child
        null handles, so nothing it prints reaches the screen."""
        from wynxo.tools import apps

        fake_os = MagicMock()
        fake_os.name = "nt"
        with patch.object(apps.subprocess, "Popen") as popen, \
                patch.object(apps, "os", fake_os):
            apps._startfile(r"C:\app.lnk", "file.txt")
        argv = popen.call_args[0][0]
        assert argv[:4] == ["cmd", "/c", "start", ""]
        assert argv[4] == r"C:\app.lnk"
        assert argv[5] == "file.txt", "the file to open is passed through"

    def test_a_shortcut_off_windows_is_a_launch_failure_not_a_crash(self):
        """A .lnk reaches this only from a synced profile or a mounted
        Windows drive. OSError so the caller reports it as a failed launch."""
        from wynxo.tools import apps

        with pytest.raises(OSError):
            apps._startfile("x.lnk")


class TestTheApplicationCatalogOnWindows:
    def test_it_reads_the_app_paths_registry(self):
        from wynxo.tools import appcatalog

        with patch.object(appcatalog.sys, "platform", "win32"):
            assert appcatalog.Sources.for_platform().use_app_paths

    def test_it_looks_in_the_start_menu(self):
        from wynxo.tools import appcatalog

        environ = {"APPDATA": r"C:\Users\x\AppData\Roaming",
                   "PROGRAMDATA": r"C:\ProgramData"}
        with patch.object(appcatalog.sys, "platform", "win32"), \
                patch.dict(appcatalog.os.environ, environ, clear=False):
            dirs = [str(d) for d in appcatalog.Sources.for_platform().shortcut_dirs]
        assert len(dirs) >= 2
        assert all("Start Menu" in d and "Programs" in d for d in dirs), dirs

    def test_it_recognises_windows_executables(self):
        from wynxo.tools import appcatalog

        source = __import__("inspect").getsource(
            appcatalog.ApplicationCatalog._scan_path_dirs)
        for suffix in (".exe", ".cmd", ".bat"):
            assert suffix in source, suffix

    def test_scanning_the_path_does_not_raise(self):
        from wynxo.tools import appcatalog

        with patch.object(appcatalog.sys, "platform", "win32"):
            catalog = appcatalog.ApplicationCatalog(
                appcatalog.Sources.for_platform())
            catalog._scan_path_dirs(lambda *a, **k: None)


class TestTerminalAndShell:
    def test_the_shell_is_cmd(self):
        from wynxo import platforms

        with patch.object(platforms.sys, "platform", "win32"):
            assert platforms.default_shell() == ("cmd.exe", ["/c"])
            assert platforms.is_windows()

    def test_rich_is_told_to_use_modern_vt_output(self):
        """legacy_windows=False is what makes colour work in Windows
        Terminal rather than falling back to the old console API."""
        from wynxo import ui

        with patch.object(ui.sys, "platform", "win32"):
            assert ui.UI().console.legacy_windows is False

    def test_the_key_watcher_does_not_reach_for_termios(self):
        from wynxo import keys

        with patch.object(keys.sys, "platform", "win32"):
            watcher = keys.KeyWatcher({})
            watcher.start()
            watcher.stop()

    def test_colour_support_is_decided_without_raising(self):
        from wynxo import status

        with patch.object(status.sys, "platform", "win32"):
            assert isinstance(status._supports_colour(), bool)


class TestCarriageReturnsFromWindowsChildren:
    """A Windows child ends lines with CRLF, and a pipe can add more (
    PowerShell turns \\n into \\r\\n). Those are line endings, not
    progress-bar frames."""

    def test_crlf_becomes_a_plain_line_ending(self):
        from wynxo.tools.shell import _clean

        assert _clean(b"one\r\ntwo\r\n") == "one\ntwo"

    def test_a_doubled_carriage_return_does_not_empty_the_line(self):
        assert __import__("wynxo.tools.shell", fromlist=["_clean"])._clean(
            b"one\r\r\ntwo\r\r\n") == "one\ntwo"

    def test_a_real_progress_bar_still_collapses_to_its_last_frame(self):
        from wynxo.tools.shell import _clean

        assert _clean(b"10%\r50%\r100%\n") == "100%"

    def test_a_trailing_cr_never_reaches_the_terminal(self):
        """A CR that survives to the screen is read as "return to column 0",
        so the row is overdrawn by whatever comes next."""
        import io

        from wynxo.ui import SafeConsole, UI

        ui = UI()
        ui.console = SafeConsole(file=io.StringIO(), force_terminal=True,
                                 width=80, highlight=False, soft_wrap=False)
        ui.tool_output("first\rsecond")
        assert "\r" not in ui.console.file.getvalue()


# What still needs a real Windows machine, and cannot be faked from here:
#
#   * Windows Terminal, PowerShell and cmd as hosts -- rendering, colour,
#     and whether the alternate screen restores cleanly on exit.
#   * The mouse wheel, drag selection and clipboard copy. wynxo never
#     enables mouse reporting and never takes the alternate screen, so all
#     three are the terminal's own behaviour -- but only a real console can
#     confirm it.
#   * Ctrl-C delivery: the console sends CTRL_C_EVENT to the process group
#     rather than raising SIGINT the way a POSIX tty does.
#   * A real resize of the console window.
#   * cp1252 consoles, and what a non-UTF-8 code page does to the box glyphs.
#   * Actually killing a real process tree with taskkill.
#   * The App Paths registry, and real Start Menu shortcuts.
