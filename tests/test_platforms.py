"""Per-platform behaviour, especially Termux -- which is Linux-flavoured but
differs in every path that matters."""

import os
from pathlib import Path

import pytest

from wynxo import platforms


@pytest.fixture
def termux(tmp_path, monkeypatch):
    """A convincing Termux environment."""
    prefix = tmp_path / "usr"
    home = tmp_path / "home"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "tmp").mkdir(parents=True)
    home.mkdir(parents=True)
    (prefix / "bin" / "bash").write_text("#!/bin/sh\n")
    (prefix / "bin" / "bash").chmod(0o755)

    monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
    monkeypatch.setenv("PREFIX", str(prefix))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(prefix / "tmp"))
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(platforms.sys, "platform", "linux")
    return {"prefix": prefix, "home": home}


class TestTermuxDetection:
    def test_detected_from_termux_version(self, termux):
        assert platforms.is_termux()

    def test_detected_from_prefix_when_version_is_stripped(self, termux, monkeypatch):
        # cron and some supervisors sanitise the environment.
        monkeypatch.delenv("TERMUX_VERSION")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        assert platforms.is_termux()

    def test_not_detected_on_a_normal_linux_box(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", str(tmp_path))
        monkeypatch.setattr(os.path, "isdir",
                            lambda p: False if "com.termux" in str(p) else os.path.exists(p))
        assert not platforms.is_termux()

    def test_name_and_description(self, termux):
        assert platforms.name() == "Termux (Android)"
        assert "Termux" in platforms.describe()
        assert "0.118.0" in platforms.describe()


class TestTermuxPaths:
    def test_config_lives_under_the_app_private_home(self, termux):
        config = platforms.config_dir()
        assert str(config).startswith(str(termux["home"]))
        assert config.name == "wynxo"

    def test_data_lives_under_the_app_private_home(self, termux):
        assert str(platforms.data_dir()).startswith(str(termux["home"]))

    def test_temp_dir_is_the_prefix_not_slash_tmp(self, termux):
        """Termux has no /tmp; assuming otherwise fails at runtime."""
        assert platforms.temp_dir() == termux["prefix"] / "tmp"
        assert str(platforms.temp_dir()) != "/tmp"

    def test_config_dir_is_writable(self, termux):
        target = platforms.config_dir()
        target.mkdir(parents=True, exist_ok=True)
        (target / "probe.json").write_text("{}")
        assert (target / "probe.json").exists()


class TestTermuxShell:
    def test_finds_bash_under_prefix(self, termux):
        """Termux binaries are never in /bin, so a bare lookup would miss."""
        shell, flags = platforms.default_shell()
        assert shell == str(termux["prefix"] / "bin" / "bash")
        assert flags == ["-c"]

    def test_respects_an_explicit_shell(self, termux, monkeypatch):
        monkeypatch.setenv("SHELL", str(termux["prefix"] / "bin" / "bash"))
        shell, _ = platforms.default_shell()
        assert shell == str(termux["prefix"] / "bin" / "bash")


class TestTermuxHelp:
    def test_help_targets_the_phone_situation(self, termux):
        help_text = platforms.ollama_server_help()
        assert "0.0.0.0:11434" in help_text
        assert "Wi-Fi" in help_text          # the usual mistake: mobile data
        assert "systemctl" not in help_text  # meaningless on Android


class TestOtherPlatforms:
    def test_windows_prefers_powershell(self, monkeypatch):
        monkeypatch.setattr(platforms.sys, "platform", "win32")
        monkeypatch.setattr(platforms.shutil, "which",
                            lambda exe: f"C:\\{exe}.exe" if exe == "pwsh" else None)
        shell, flags = platforms.default_shell()
        assert "pwsh" in shell
        assert "-Command" in flags

    def test_windows_falls_back_to_comspec(self, monkeypatch):
        monkeypatch.setattr(platforms.sys, "platform", "win32")
        monkeypatch.setattr(platforms.shutil, "which", lambda exe: None)
        monkeypatch.setenv("COMSPEC", "C:\\Windows\\system32\\cmd.exe")
        shell, flags = platforms.default_shell()
        assert "cmd.exe" in shell
        assert flags == ["/c"]

    def test_windows_config_uses_appdata(self, monkeypatch, tmp_path):
        monkeypatch.setattr(platforms.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert platforms.config_dir() == tmp_path / "wynxo"

    def test_macos_uses_application_support(self, monkeypatch):
        monkeypatch.setattr(platforms.sys, "platform", "darwin")
        assert "Application Support" in str(platforms.config_dir())

    def test_linux_respects_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setattr(platforms.sys, "platform", "linux")
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert platforms.config_dir() == tmp_path / "wynxo"


class TestTerminalWidth:
    def test_narrow_detection(self, monkeypatch):
        monkeypatch.setattr(platforms, "terminal_width", lambda default=80: 45)
        assert platforms.is_narrow()
        monkeypatch.setattr(platforms, "terminal_width", lambda default=80: 100)
        assert not platforms.is_narrow()

    def test_width_survives_a_missing_terminal(self, monkeypatch):
        def boom(fallback):
            raise OSError("no tty")
        monkeypatch.setattr(platforms.shutil, "get_terminal_size", boom)
        assert platforms.terminal_width(default=72) == 72
