from pathlib import Path


def test_windows_launcher_delegates_to_bootstrap():
    source = Path("wynxo/windows_launcher.py").read_text(encoding="utf-8")
    assert "from .bootstrap import main" in source
    assert "raise SystemExit(main() or 0)" in source


def test_installer_windows_link_invokes_python_module_not_generated_exe():
    source = Path("install.py").read_text(encoding="utf-8")
    assert '"{python}" -m wynxo %*' in source
    assert "entry}" not in source.split("if sys.platform == \"win32\":", 1)[1].split("else:", 1)[0]


def test_windows_docs_explain_policy_safe_fallback():
    source = Path("README.md").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe -m wynxo" in source
    assert "Device Guard" in source
