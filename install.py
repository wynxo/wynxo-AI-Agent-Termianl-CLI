#!/usr/bin/env python3
"""Installer for wynxo.

    python3 install.py

Installs the agent and nothing else: a virtualenv, the package, and a `wynxo`
command on your PATH. It does not touch Ollama, does not download models and
does not decide anything about them -- wynxo asks the server what it has and
you pick, which is the only way that stays right as your models change.

    --yes         accept the recommended answer to every prompt
    --no-link     do not put a `wynxo` command on PATH
    --no-ollama   legacy no-op retained for older wrappers
    --venv DIR    virtualenv location (default: .venv)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)
OLLAMA_PORT = 11434

MODELS = [
    ("qwen3-coder:30b", 24, "30B MoE, tool-tuned. The one to want."),
    ("qwen3:14b", 12, "Solid all-rounder, fits a 12GB card."),
    ("qwen3:8b", 8, "Runs on almost anything, CPU included."),
    ("qwen3:1.7b", 4, "Last resort. Fast, but weak at tool calling."),
]


def _enable_windows_vt() -> bool:
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            if handle in (0, -1):
                continue
            mode = wintypes.DWORD()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:
        try:
            os.system("")
            return True
        except Exception:
            return False


class Style:
    def __init__(self) -> None:
        self.on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        if sys.platform == "win32" and self.on:
            self.on = _enable_windows_vt()

    def _wrap(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text
    def bold(self, t): return self._wrap(t, "1")
    def dim(self, t): return self._wrap(t, "2")
    def cyan(self, t): return self._wrap(t, "36")
    def green(self, t): return self._wrap(t, "32")
    def yellow(self, t): return self._wrap(t, "33")
    def red(self, t): return self._wrap(t, "31")


S = Style()
STEP = [0]


def step(title: str) -> None:
    STEP[0] += 1
    print()
    print(S.cyan(S.bold(f"  {STEP[0]}. {title}")))


def ok(msg: str) -> None:
    print(f"     {S.green('OK')}  {msg}")


def warn(msg: str) -> None:
    print(f"     {S.yellow('!')}   {msg}")


def fail(msg: str) -> None:
    print(f"     {S.red('X')}   {msg}")


def info(msg: str) -> None:
    print(S.dim(f"         {msg}"))


def die(msg: str, fix: str = "") -> None:
    print()
    fail(msg)
    if fix:
        for line in fix.splitlines():
            info(line)
    print()
    sys.exit(1)


def ask(question: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        print(f"     {question} {S.dim('[auto: yes]')}")
        return True
    if not sys.stdin.isatty():
        print(f"     {question} {S.dim('[not a terminal: assuming ' + ('yes' if default else 'no') + ']')}")
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"     {question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not answer:
        return default
    return answer in ("y", "yes")


def is_termux() -> bool:
    return bool(os.environ.get("TERMUX_VERSION")) or os.environ.get("PREFIX", "").startswith("/data/data/com.termux")


def platform_name() -> str:
    if is_termux():
        return "Termux (Android)"
    return {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Linux")


def check_python() -> None:
    step("Checking Python")
    version = sys.version_info
    pretty = f"{version.major}.{version.minor}.{version.micro}"
    if version < MIN_PYTHON:
        die(f"Python {pretty} is too old; wynxo needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+.",
            "Linux:   sudo apt install python3\n"
            "macOS:   brew install python\n"
            "Windows: https://python.org/downloads\n"
            "Termux:  pkg install python")
    ok(f"Python {pretty} on {platform_name()} ({platform.machine()})")


def make_venv(venv_dir: Path, assume_yes: bool) -> Path:
    step("Setting up the environment")
    if os.environ.get("VIRTUAL_ENV"):
        current = Path(os.environ["VIRTUAL_ENV"])
        ok(f"already inside a virtualenv: {current}")
        return Path(sys.executable)
    if venv_dir.exists():
        python = venv_python(venv_dir)
        if python.exists():
            ok(f"reusing {venv_dir}")
            return python
        warn(f"{venv_dir} exists but looks broken; recreating it")
        shutil.rmtree(venv_dir, ignore_errors=True)
    print(f"     Creating a virtualenv at {S.bold(str(venv_dir))}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, capture_output=True, text=True, timeout=180)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()[-1:] or [""]
        warn(f"venv creation failed: {detail[0]}")
        if sys.platform.startswith("linux") and not is_termux():
            info("On Debian/Ubuntu this usually means: sudo apt install python3-venv")
        if not ask("Install into your user site-packages instead?", True, assume_yes):
            die("Cannot continue without somewhere to install.")
        return Path(sys.executable)
    except subprocess.TimeoutExpired:
        die("venv creation timed out.")
    python = venv_python(venv_dir)
    ok(f"virtualenv ready at {venv_dir}")
    return python


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def install_wynxo(python: Path, into_user: bool) -> None:
    step("Installing wynxo")
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if into_user:
        command.append("--user")
    command += ["-e", str(ROOT)]
    print(S.dim(f"         {' '.join(command[-2:])}"))
    process = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if process.returncode != 0:
        tail = (process.stderr or process.stdout).strip().splitlines()[-6:]
        die("pip install failed.", "\n".join(tail))
    ok("wynxo and its dependencies installed")
    info("no compiled extensions -- nothing was built from source")
    verify_install(python)


def verify_install(python: Path) -> None:
    import tempfile
    try:
        check = subprocess.run(
            [str(python), "-c", "import wynxo, sys; print(wynxo.__version__); print(sys.executable)"],
            capture_output=True, text=True, timeout=120, cwd=tempfile.gettempdir())
    except (OSError, subprocess.SubprocessError) as exc:
        die(f"Installed, but could not run it: {exc}")
    if check.returncode != 0:
        detail = (check.stderr or check.stdout).strip().splitlines()[-3:]
        die("pip reported success but wynxo will not import.",
            "\n".join(detail) + "\n\nThis usually means pip installed into a different Python than the\none being used. Try again from a clean checkout, or report this.")
    version = check.stdout.strip().splitlines()[0] if check.stdout.strip() else "?"
    ok(f"verified: wynxo {version} imports and runs")


def user_bin_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        windows_apps = base / "Microsoft" / "WindowsApps"
        if windows_apps.is_dir() and on_path(windows_apps):
            return windows_apps
        return base / "Programs" / "wynxo"
    if is_termux():
        prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
        return Path(prefix) / "bin"
    return Path.home() / ".local" / "bin"


def add_to_user_path(directory: Path) -> bool:
    if sys.platform == "win32":
        return _add_to_path_windows(directory)
    return _add_to_path_posix(directory)


def _add_to_path_windows(directory: Path) -> bool:
    try:
        import winreg
    except ImportError:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ) as key:
            try:
                current, kind = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current, kind = "", winreg.REG_EXPAND_SZ
    except OSError:
        return False
    entries = [e for e in str(current).split(";") if e.strip()]
    if any(e.rstrip("\\").lower() == str(directory).rstrip("\\").lower() for e in entries):
        return True
    updated = ";".join(entries + [str(directory)])
    if len(updated) > 1000:
        warn("Your user PATH is very long; not editing it automatically.")
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "Path", 0, kind or winreg.REG_EXPAND_SZ, updated)
    except OSError:
        return False
    try:
        import ctypes
        HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x1A, 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment", SMTO_ABORTIFHUNG, 5000, None)
    except Exception:
        pass
    return True


def _add_to_path_posix(directory: Path) -> bool:
    shell = os.path.basename(os.environ.get("SHELL", "")) or "bash"
    rc = {"zsh": Path.home() / ".zshrc", "fish": Path.home() / ".config" / "fish" / "config.fish", "bash": Path.home() / ".bashrc"}.get(shell, Path.home() / ".profile")
    line = f"fish_add_path {directory}" if shell == "fish" else f'export PATH="{directory}:$PATH"'
    marker = "# added by wynxo installer"
    try:
        existing = rc.read_text(encoding="utf-8") if rc.exists() else ""
        if str(directory) in existing:
            return True
        rc.parent.mkdir(parents=True, exist_ok=True)
        with rc.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{marker}\n{line}\n")
    except OSError:
        return False
    info(f"added to {rc}")
    return True


def on_path(directory: Path) -> bool:
    entries = os.environ.get("PATH", "").split(os.pathsep)
    try:
        resolved = directory.resolve()
    except OSError:
        return False
    for entry in entries:
        try:
            if entry and Path(entry).resolve() == resolved:
                return True
        except OSError:
            continue
    return False


def shell_rc_hint(directory: Path) -> str:
    line = f'export PATH="{directory}:$PATH"'
    shell = os.path.basename(os.environ.get("SHELL", "")) or "bash"
    rc = {"zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish", "bash": "~/.bashrc"}.get(shell, "~/.profile")
    if shell == "fish":
        line = f"fish_add_path {directory}"
    return f"{line}    # add to {rc}"


def entry_point_script(python: Path) -> Path | None:
    into_user = python == Path(sys.executable) and not os.environ.get("VIRTUAL_ENV")
    if into_user:
        code = "import os, sysconfig; print(sysconfig.get_path('scripts', f'{os.name}_user'))"
    else:
        code = "import sysconfig; print(sysconfig.get_path('scripts'))"
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name = "wynxo.exe" if sys.platform == "win32" else "wynxo"
    script = Path(result.stdout.strip()) / name
    return script if script.exists() else None


def link_command(python: Path, venv_dir: Path, assume_yes: bool) -> tuple[Path | None, bool]:
    step("Making `wynxo` available everywhere")
    target = user_bin_dir()
    launcher = target / ("wynxo.cmd" if sys.platform == "win32" else "wynxo")
    entry = entry_point_script(python)
    if not ask(f"Install a `wynxo` command into {target}?", True, assume_yes):
        info("Skipped.")
        return None, False
    if entry is not None and entry.resolve() == launcher.resolve():
        ok(f"already installed at {launcher}")
    else:
        try:
            target.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                body = (f'@echo off\r\n"{entry}" %*\r\n' if entry is not None else f'@echo off\r\n"{python}" -m wynxo %*\r\n')
                launcher.write_text(body, encoding="utf-8")
            else:
                body = (f'#!/bin/sh\nexec "{entry}" "$@"\n' if entry is not None else f'#!/bin/sh\nexec "{python}" -m wynxo "$@"\n')
                launcher.write_text(body, encoding="utf-8")
                launcher.chmod(0o755)
        except OSError as exc:
            warn(f"Could not write {launcher}: {exc}")
            return None, False
        ok(f"installed {launcher}")
    if on_path(target):
        info("it is already on your PATH -- just run `wynxo`")
        return launcher, False
    warn(f"{target} is not on your PATH, so `wynxo` would not be found.")
    if ask("Add it to your PATH now?", True, assume_yes):
        if add_to_user_path(target):
            ok("added to your PATH")
            info("Open a NEW terminal, then run `wynxo`.")
            return launcher, True
        warn("Could not edit your PATH automatically.")
    if sys.platform == "win32":
        info("Add it by hand: Settings -> Edit environment variables -> Path")
    else:
        info(shell_rc_hint(target))
    return launcher, False


def finish(python: Path, venv_dir: Path, launcher: Path | None, path_was_updated: bool) -> None:
    print()
    print(S.green(S.bold("  wynxo is installed.")))
    print()
    if launcher is not None and on_path(launcher.parent):
        print(f"    Run it:  {S.bold('wynxo')}")
    elif launcher is not None and path_was_updated:
        print("    Open a NEW terminal, then run:")
        print(f"      {S.bold('wynxo')}")
        print()
        print(S.dim("    (PATH changes only reach terminals opened afterwards.)"))
        print(S.dim(f"    In this one:  {launcher}"))
    elif launcher is not None:
        print(f"    Run it:  {S.bold(str(launcher))}")
        print(S.dim(f"    Put {launcher.parent} on your PATH for just `wynxo`."))
    else:
        print(f"    Run it:  {S.bold(str(venv_python(venv_dir)) + ' -m wynxo')}")
    print()
    print(S.dim("    On first run wynxo asks where Ollama is, then lists the models"))
    print(S.dim("    that server actually has so you can pick one. Start it in the"))
    print(S.dim("    project you want to work on."))
    print()
    print(S.dim("    Ollama needs to be running:  ollama serve"))
    print(S.dim(f"    wynxo looks at http://127.0.0.1:{OLLAMA_PORT} by default."))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Install wynxo.")
    parser.add_argument("--yes", "-y", action="store_true", help="accept the recommended answer to every prompt")
    parser.add_argument("--no-link", action="store_true", help="do not put a `wynxo` command on PATH")
    parser.add_argument("--no-ollama", action="store_true", help="legacy compatibility flag; ignored")
    parser.add_argument("--venv", default=".venv", help="virtualenv directory")
    args = parser.parse_args()
    print()
    print(S.cyan(S.bold("  wynxo")))
    print(S.dim("  A local AI coding agent. Nothing leaves your machine."))
    check_python()
    venv_dir = (ROOT / args.venv) if not os.path.isabs(args.venv) else Path(args.venv)
    python = make_venv(venv_dir, args.yes)
    install_wynxo(python, into_user=(python == Path(sys.executable) and not os.environ.get("VIRTUAL_ENV")))
    launcher, path_was_updated = ((None, False) if args.no_link else link_command(python, venv_dir, args.yes))
    finish(python, venv_dir, launcher, path_was_updated)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  cancelled")
        sys.exit(130)
