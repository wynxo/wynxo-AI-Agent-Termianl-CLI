"""The installer runs on machines we cannot test, so its decision logic is
tested directly. Getting the model choice wrong turns a successful install
into a confusing failure at the last step."""

import importlib.util
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
