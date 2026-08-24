"""The installer runs on machines we cannot test, so its decision logic is
tested directly. Getting the model choice wrong turns a successful install
into a confusing failure at the last step."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wynxo_install", ROOT / "install.py")
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)


class TestPrompts:
    def test_assume_yes_never_blocks(self, capsys):
        assert install.ask("anything?", default=False, assume_yes=True) is True

    def test_non_tty_takes_the_default(self, monkeypatch):
        monkeypatch.setattr(install.sys.stdin, "isatty", lambda: False)
        assert install.ask("q", default=True) is True
        assert install.ask("q", default=False) is False


class TestPlatform:
    def test_termux_detected_from_env(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
        assert install.is_termux()
        assert install.platform_name() == "Termux (Android)"

    def test_venv_python_path_per_platform(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install.sys, "platform", "win32")
        assert install.venv_python(tmp_path).name == "python.exe"
        monkeypatch.setattr(install.sys, "platform", "linux")
        assert install.venv_python(tmp_path) == tmp_path / "bin" / "python"


class TestScriptWrappers:
    def test_shell_wrapper_is_posix_and_executable(self):
        script = ROOT / "install.sh"
        assert script.exists()
        text = script.read_text()
        assert text.startswith("#!/usr/bin/env sh")
        assert "install.py" in text
        # bash-isms would break on Termux's default sh and on dash.
        assert "[[" not in text

    def test_powershell_wrapper_exists(self):
        text = (ROOT / "install.ps1").read_text()
        assert "install.py" in text
        assert "py -3" in text


class TestEntryPointScript:
    """python -m wynxo puts the caller's CWD on sys.path[0]; a real
    entry-point script puts its own directory there instead. Finding that
    script correctly is what lets the launcher avoid the shadowing bug."""

    def test_finds_the_running_interpreters_own_scripts_dir(self):
        import sys

        # This interpreter has no `wynxo` console script of its own (it is
        # the test runner, not an install), so nothing should be found --
        # but the lookup itself must not raise or time out.
        result = install.entry_point_script(Path(sys.executable))
        assert result is None or result.name in ("wynxo", "wynxo.exe")

    def test_missing_interpreter_returns_none_rather_than_raising(self, tmp_path):
        assert install.entry_point_script(tmp_path / "no" / "such" / "python") is None

    def test_a_script_that_exists_is_returned(self, monkeypatch, tmp_path):
        import sys

        scripts_dir = tmp_path
        (scripts_dir / "wynxo").write_text("#!/bin/sh\n")
        monkeypatch.setattr(
            install.subprocess, "run",
            lambda *a, **k: type("R", (), {"returncode": 0,
                                           "stdout": str(scripts_dir) + "\n"})())
        assert install.entry_point_script(Path(sys.executable)) == scripts_dir / "wynxo"

    def test_a_missing_script_is_not_returned(self, monkeypatch, tmp_path):
        import sys

        monkeypatch.setattr(
            install.subprocess, "run",
            lambda *a, **k: type("R", (), {"returncode": 0,
                                           "stdout": str(tmp_path) + "\n"})())
        assert install.entry_point_script(Path(sys.executable)) is None


class TestLauncher:
    def test_bin_dir_per_platform(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install.sys, "platform", "linux")
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "")
        assert install.user_bin_dir() == Path.home() / ".local" / "bin"

        monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
        monkeypatch.setenv("PREFIX", str(tmp_path))
        assert install.user_bin_dir() == tmp_path / "bin"

    def test_windows_bin_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install.sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert install.user_bin_dir() == tmp_path / "Programs" / "wynxo"

    def test_on_path_detection(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}{tmp_path}")
        assert install.on_path(tmp_path)
        assert not install.on_path(tmp_path / "elsewhere")

    def test_launcher_falls_back_to_pinning_the_interpreter(self, monkeypatch, tmp_path):
        """No real entry-point script to find (this python was never
        installed) -- the shim falls back to `-m wynxo`, still pinned to a
        specific interpreter rather than a symlink so it works whatever venv
        is active."""
        monkeypatch.setattr(install.sys, "platform", "linux")
        monkeypatch.setattr(install, "user_bin_dir", lambda: tmp_path / "bin")
        monkeypatch.setattr(install, "ask", lambda *a, **k: True)
        python = tmp_path / "venv" / "bin" / "python"
        launcher, _ = install.link_command(python, tmp_path / "venv", assume_yes=True)
        assert launcher is not None
        body = launcher.read_text()
        assert str(python) in body and "-m wynxo" in body
        assert launcher.stat().st_mode & 0o111, "must be executable"

    def test_declining_the_link_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install, "user_bin_dir", lambda: tmp_path / "bin")
        monkeypatch.setattr(install, "ask", lambda *a, **k: False)
        assert install.link_command(Path("/x/python"), tmp_path, False) == (None, False)

    def test_launcher_prefers_pips_own_entry_point(self, monkeypatch, tmp_path):
        """`python -m wynxo` puts the caller's working directory on
        sys.path[0] -- so a shell sitting inside wynxo's own source tree (it
        has files named select.py, queue.py, ... shadowing the standard
        library) breaks imports in a baffling way. Pip's own entry-point
        script does not have that problem, so the launcher should delegate to
        it whenever one exists, rather than reinventing `-m wynxo`."""
        monkeypatch.setattr(install.sys, "platform", "linux")
        monkeypatch.setattr(install, "user_bin_dir", lambda: tmp_path / "bin")
        monkeypatch.setattr(install, "ask", lambda *a, **k: True)

        scripts_dir = tmp_path / "venv" / "bin"
        scripts_dir.mkdir(parents=True)
        real_entry = scripts_dir / "wynxo"
        real_entry.write_text("#!/bin/sh\necho real\n")
        monkeypatch.setattr(install, "entry_point_script", lambda python: real_entry)

        launcher, _ = install.link_command(
            scripts_dir / "python", tmp_path / "venv", assume_yes=True)
        body = launcher.read_text()
        assert str(real_entry) in body
        assert "-m wynxo" not in body

    def test_launcher_does_not_overwrite_a_user_install_with_itself(
            self, monkeypatch, tmp_path):
        """`pip install --user` can put its own entry-point script exactly
        where the launcher would go (~/.local/bin on most of Linux). Writing
        a wrapper there that execs its own path would loop forever, so that
        case has to be left alone."""
        monkeypatch.setattr(install.sys, "platform", "linux")
        bin_dir = tmp_path / "bin"
        monkeypatch.setattr(install, "user_bin_dir", lambda: bin_dir)
        monkeypatch.setattr(install, "ask", lambda *a, **k: True)
        bin_dir.mkdir(parents=True)
        existing = bin_dir / "wynxo"
        existing.write_text("#!/bin/sh\necho already here\n")
        before = existing.read_text()

        monkeypatch.setattr(install, "entry_point_script", lambda python: existing)
        launcher, _ = install.link_command(Path("/x/python"), tmp_path, assume_yes=True)
        assert launcher == existing
        assert existing.read_text() == before

    def test_shell_rc_hint_matches_the_shell(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SHELL", "/usr/bin/zsh")
        assert ".zshrc" in install.shell_rc_hint(tmp_path)
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        assert "fish_add_path" in install.shell_rc_hint(tmp_path)


class TestVerifyInstall:
    """pip can report success while the result is unusable. The check has to
    be able to tell an install from a source checkout, which is exactly what
    it got wrong first: run from the repo directory, `import wynxo` finds the
    source folder whether or not anything was installed."""

    def test_passes_when_the_module_imports(self, capsys, monkeypatch):
        monkeypatch.setattr(
            install.subprocess, "run",
            lambda *a, **k: type("R", (), {
                "returncode": 0, "stdout": "0.1.0\n/venv/bin/python\n",
                "stderr": ""})())
        install.verify_install(Path("/venv/bin/python"))
        assert "verified: wynxo 0.1.0" in capsys.readouterr().out

    def test_runs_from_outside_the_source_tree(self, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            return type("R", (), {"returncode": 0, "stdout": "0.1.0\n/x\n", "stderr": ""})()

        monkeypatch.setattr(install.subprocess, "run", fake_run)
        install.verify_install(Path("/x/python"))
        cwd = Path(seen["cwd"]).resolve()
        assert cwd != ROOT, "checking from the repo proves nothing"

    def test_fails_when_the_module_is_missing(self, monkeypatch):
        monkeypatch.setattr(
            install.subprocess, "run",
            lambda *a, **k: type("R", (), {
                "returncode": 1, "stdout": "",
                "stderr": "ModuleNotFoundError: No module named 'wynxo'"})())
        with pytest.raises(SystemExit) as exc:
            install.verify_install(Path("/x/python"))
        assert exc.value.code == 1

    def test_reports_a_launch_failure(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("no such interpreter")
        monkeypatch.setattr(install.subprocess, "run", boom)
        with pytest.raises(SystemExit):
            install.verify_install(Path("/nope/python"))


class TestWindowsEntryPoints:
    """PowerShell's default Restricted policy blocks .ps1, which is the most
    common reason a Windows setup appears to do nothing."""

    def test_bat_wrapper_exists_and_is_not_powershell(self):
        script = ROOT / "install.bat"
        assert script.exists()
        text = script.read_text()
        assert text.lstrip().startswith("@echo off")
        assert "install.py" in text
        assert "py -3" in text

    def test_bat_wrapper_handles_a_missing_python(self):
        text = (ROOT / "install.bat").read_text()
        assert "python.org/downloads" in text

    def test_ps1_documents_the_policy_workaround(self):
        text = (ROOT / "install.ps1").read_text()
        assert "ExecutionPolicy Bypass" in text
        assert "install.bat" in text

    def test_readme_does_not_tell_windows_users_to_activate(self):
        readme = (ROOT / "README.md").read_text()
        for line in readme.splitlines():
            stripped = line.strip()
            # A bare activation command in a code block is the trap; prose
            # explaining not to use it is fine.
            if stripped.startswith((".venv\\Scripts\\Activate", ".\\.venv\\Scripts\\Activate")):
                raise AssertionError(f"README instructs activation: {line!r}")

    def test_readme_gives_the_full_path_pip_invocation(self):
        readme = (ROOT / "README.md").read_text()
        assert ".venv\\Scripts\\python.exe -m pip install -e ." in readme


class TestPathHandling:
    """The transcript failure: the launcher was installed somewhere not on
    PATH, and the user was told to go edit Settings. `wynxo` then did not
    exist, and a successful install looked broken."""

    def test_posix_path_is_added_to_the_shell_rc(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install.sys, "platform", "linux")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setattr(install.Path, "home", staticmethod(lambda: tmp_path))
        assert install._add_to_path_posix(tmp_path / "bin")
        assert str(tmp_path / "bin") in (tmp_path / ".bashrc").read_text()

    def test_posix_path_is_not_added_twice(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setattr(install.Path, "home", staticmethod(lambda: tmp_path))
        install._add_to_path_posix(tmp_path / "bin")
        install._add_to_path_posix(tmp_path / "bin")
        assert (tmp_path / ".bashrc").read_text().count("added by wynxo") == 1

    def test_fish_uses_its_own_syntax(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        monkeypatch.setattr(install.Path, "home", staticmethod(lambda: tmp_path))
        install._add_to_path_posix(tmp_path / "bin")
        config = tmp_path / ".config" / "fish" / "config.fish"
        assert "fish_add_path" in config.read_text()
        assert "export PATH" not in config.read_text()

    def test_windows_prefers_a_directory_already_on_path(self, monkeypatch, tmp_path):
        """WindowsApps is on PATH for every user out of the box, so a shim
        there works with no PATH edit at all."""
        monkeypatch.setattr(install.sys, "platform", "win32")
        windows_apps = tmp_path / "Microsoft" / "WindowsApps"
        windows_apps.mkdir(parents=True)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("PATH", str(windows_apps))
        assert install.user_bin_dir() == windows_apps

    def test_windows_falls_back_when_windowsapps_is_not_on_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install.sys, "platform", "win32")
        (tmp_path / "Microsoft" / "WindowsApps").mkdir(parents=True)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("PATH", "/somewhere/else")
        assert install.user_bin_dir() == tmp_path / "Programs" / "wynxo"


