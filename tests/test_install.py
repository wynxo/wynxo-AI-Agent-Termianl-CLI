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


class TestModelRecommendation:
    @pytest.mark.parametrize("memory,expected", [
        (64, "qwen3-coder:30b"),
        (32, "qwen3-coder:30b"),
        (24, "qwen3-coder:30b"),
        (16, "qwen3:14b"),
        (12, "qwen3:14b"),
        (8, "qwen3:8b"),
        (4, "qwen3:1.7b"),
        (2, "qwen3:1.7b"),
    ])
    def test_scales_with_memory(self, memory, expected, monkeypatch):
        monkeypatch.setattr(install, "total_memory_gb", lambda: memory)
        assert install.recommend_model()[0] == expected

    def test_unknown_memory_picks_a_safe_default(self, monkeypatch):
        monkeypatch.setattr(install, "total_memory_gb", lambda: 0.0)
        model, why = install.recommend_model()
        assert model == "qwen3:8b"
        assert why

    def test_every_model_has_a_reason(self):
        for tag, needs, why in install.MODELS:
            assert ":" in tag and needs > 0 and why


class TestBestInstalled:
    def test_prefers_the_strongest_present(self):
        assert install.best_installed(["qwen3:8b", "qwen3-coder:30b"]) == "qwen3-coder:30b"

    def test_matches_on_family_not_exact_tag(self):
        assert install.best_installed(["qwen3-coder:7b"]) == "qwen3-coder:7b"

    def test_falls_back_to_whatever_is_there(self):
        assert install.best_installed(["llama3:8b"]) == "llama3:8b"

    def test_nothing_installed(self):
        assert install.best_installed([]) is None


class TestEnsureModel:
    """The bug this guards: recommending a model, not pulling it, then
    checking against it anyway."""

    def test_returns_an_installed_model_not_the_recommendation(self, monkeypatch, capsys):
        monkeypatch.setattr(install, "installed_models", lambda: ["qwen3:8b"])
        monkeypatch.setattr(install, "ask", lambda *a, **k: False)
        chosen = install.ensure_model("qwen3:14b", "why", assume_yes=False)
        assert chosen == "qwen3:8b", "must not point the checks at an absent model"

    def test_uses_the_preferred_model_when_present(self, monkeypatch):
        monkeypatch.setattr(install, "installed_models",
                            lambda: ["qwen3:8b", "qwen3-coder:30b"])
        assert install.ensure_model("qwen3-coder:30b", "why", False) == "qwen3-coder:30b"

    def test_family_match_counts_as_present(self, monkeypatch):
        monkeypatch.setattr(install, "installed_models", lambda: ["qwen3-coder:7b"])
        assert install.ensure_model("qwen3-coder:30b", "why", False) == "qwen3-coder:7b"

    def test_returns_none_when_nothing_installed_and_pull_declined(self, monkeypatch):
        monkeypatch.setattr(install, "installed_models", lambda: [])
        monkeypatch.setattr(install, "ask", lambda *a, **k: False)
        assert install.ensure_model("qwen3:8b", "why", False) is None

    def test_pull_failure_falls_back_to_what_exists(self, monkeypatch):
        monkeypatch.setattr(install, "installed_models", lambda: ["qwen3:8b"])
        monkeypatch.setattr(install, "ask", lambda *a, **k: True)
        monkeypatch.setattr(install.subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 1})())
        assert install.ensure_model("qwen3:14b", "why", False) == "qwen3:8b"


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

    def test_launcher_pins_the_interpreter(self, monkeypatch, tmp_path):
        """A shim rather than a symlink, so it works whatever venv is active."""
        monkeypatch.setattr(install.sys, "platform", "linux")
        monkeypatch.setattr(install, "user_bin_dir", lambda: tmp_path / "bin")
        monkeypatch.setattr(install, "ask", lambda *a, **k: True)
        python = tmp_path / "venv" / "bin" / "python"
        launcher = install.link_command(python, tmp_path / "venv", assume_yes=True)
        assert launcher is not None
        body = launcher.read_text()
        assert str(python) in body and "-m wynxo" in body
        assert launcher.stat().st_mode & 0o111, "must be executable"

    def test_declining_the_link_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install, "user_bin_dir", lambda: tmp_path / "bin")
        monkeypatch.setattr(install, "ask", lambda *a, **k: False)
        assert install.link_command(Path("/x/python"), tmp_path, False) is None

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
