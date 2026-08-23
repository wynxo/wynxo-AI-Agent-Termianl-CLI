#!/usr/bin/env python3
"""One-command setup for wynxo.

    python3 install.py

Checks Python, creates a virtualenv, installs wynxo and its dependencies,
finds or installs Ollama, pulls a model sized to the machine, and runs the
pre-flight checks. Works on Linux, macOS, Windows and Termux.

Nothing here happens silently. Anything that touches the network or writes
outside this directory is described first and asked about, because "curl a
script and run it as root" is not something to do to someone's machine
without telling them.

    --yes        accept the recommended answer to every prompt
    --no-ollama  skip everything Ollama-related; just install wynxo
    --model X    pull X instead of the recommended model
    --venv DIR   virtualenv location (default: .venv)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)
OLLAMA_PORT = 11434

# (tag, gigabytes of RAM/VRAM it wants, one-line reason)
MODELS = [
    ("qwen3-coder:30b", 24, "30B MoE, tool-tuned. The one to want."),
    ("qwen3:14b", 12, "Solid all-rounder, fits a 12GB card."),
    ("qwen3:8b", 8, "Runs on almost anything, CPU included."),
    ("qwen3:1.7b", 4, "Last resort. Fast, but weak at tool calling."),
]


# -- output ----------------------------------------------------------------

def _enable_windows_vt() -> bool:
    """Ask the console to interpret escape sequences, so colour works rather
    than printing as literal `<-[1;32m` noise."""
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


# -- platform --------------------------------------------------------------

def is_termux() -> bool:
    return bool(os.environ.get("TERMUX_VERSION")) or \
        os.environ.get("PREFIX", "").startswith("/data/data/com.termux")


def platform_name() -> str:
    if is_termux():
        return "Termux (Android)"
    return {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Linux")


def total_memory_gb() -> float:
    """Best-effort physical memory, for choosing a model."""
    try:
        if sys.platform == "linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1024 ** 3
        elif sys.platform == "win32":
            # PowerShell/CIM first: wmic is deprecated and simply absent on
            # Windows 11 24H2 and later, where it would silently report 0 and
            # get a 64GB machine recommended an 8B model.
            for command in (
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
            ):
                try:
                    out = subprocess.run(command, capture_output=True, text=True,
                                         timeout=15)
                except (OSError, subprocess.SubprocessError):
                    continue
                digits = [w for w in out.stdout.split() if w.isdigit()]
                if digits:
                    return int(digits[0]) / 1024 ** 3
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0.0


def recommend_model() -> tuple[str, str]:
    memory = total_memory_gb()
    if not memory:
        # Cannot tell; pick the one that runs almost anywhere.
        tag, _, why = MODELS[2]
        return tag, why
    for tag, needs, why in MODELS:
        if memory >= needs:
            return tag, why
    tag, _, why = MODELS[-1]
    return tag, why


# -- steps -----------------------------------------------------------------

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
    """Create the virtualenv and return its python. Falls back to --user.

    Note that nothing here "activates" the virtualenv. Every later step calls
    its interpreter by full path instead, which sidesteps the most common
    Windows failure entirely: PowerShell refuses to run Activate.ps1 under its
    default Restricted execution policy, so activation fails, pip is then
    either missing or the global one, and the whole install appears to do
    nothing.
    """
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
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                       check=True, capture_output=True, text=True, timeout=180)
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

    # Every dependency is pure Python, so this should never need a compiler.
    ok("wynxo and its dependencies installed")
    info("no compiled extensions -- nothing was built from source")

    # Prove it. pip can report success while the result is unusable -- a stale
    # venv, a shadowed name, the wrong interpreter -- and a silent half-install
    # is the hardest kind of failure to diagnose from the outside.
    verify_install(python)


def verify_install(python: Path) -> None:
    """Confirm wynxo is genuinely importable, from outside the source tree.

    Running this from the repository directory would prove nothing: Python
    puts the working directory on sys.path, so `import wynxo` would find the
    source folder whether or not anything was ever installed. Checking from
    elsewhere is what actually distinguishes an install from a checkout.
    """
    import tempfile

    try:
        check = subprocess.run(
            [str(python), "-c",
             "import wynxo, sys; print(wynxo.__version__); print(sys.executable)"],
            capture_output=True, text=True, timeout=120,
            cwd=tempfile.gettempdir())
    except (OSError, subprocess.SubprocessError) as exc:
        die(f"Installed, but could not run it: {exc}")

    if check.returncode != 0:
        detail = (check.stderr or check.stdout).strip().splitlines()[-3:]
        die("pip reported success but wynxo will not import.",
            "\n".join(detail) + "\n\n"
            "This usually means pip installed into a different Python than the\n"
            "one being used. Try again from a clean checkout, or report this.")

    version = check.stdout.strip().splitlines()[0] if check.stdout.strip() else "?"
    ok(f"verified: wynxo {version} imports and runs")


def user_bin_dir() -> Path:
    """Where a user-level command should go on this platform.

    On Windows, WindowsApps is on PATH out of the box for every user, so a
    shim placed there works immediately. Anywhere else needs PATH edited,
    which is the step people never do -- and then `wynxo` is not found and the
    install looks broken even though it succeeded.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA")
                    or (Path.home() / "AppData" / "Local"))
        windows_apps = base / "Microsoft" / "WindowsApps"
        if windows_apps.is_dir() and on_path(windows_apps):
            return windows_apps
        return base / "Programs" / "wynxo"
    if is_termux():
        prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
        return Path(prefix) / "bin"
    return Path.home() / ".local" / "bin"


def add_to_user_path(directory: Path) -> bool:
    """Add ``directory`` to the user's PATH, permanently.

    Telling someone to open Settings and edit environment variables is where a
    "five minute setup" goes to die, so do it for them.
    """
    if sys.platform == "win32":
        return _add_to_path_windows(directory)
    return _add_to_path_posix(directory)


def _add_to_path_windows(directory: Path) -> bool:
    """setx writes HKCU\Environment and survives reboots.

    It truncates at 1024 characters, so read the existing value from the
    registry (not the expanded process PATH, which includes the machine-wide
    half) and refuse rather than corrupt a long PATH.
    """
    try:
        import winreg
    except ImportError:
        return False

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_READ) as key:
            try:
                current, kind = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current, kind = "", winreg.REG_EXPAND_SZ
    except OSError:
        return False

    entries = [e for e in str(current).split(";") if e.strip()]
    if any(e.rstrip("\\").lower() == str(directory).rstrip("\\").lower()
           for e in entries):
        return True

    updated = ";".join(entries + [str(directory)])
    if len(updated) > 1000:
        warn("Your user PATH is very long; not editing it automatically.")
        return False

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "Path", 0, kind or winreg.REG_EXPAND_SZ, updated)
    except OSError:
        return False

    # Tell running shells so a new terminal picks it up without a logout.
    try:
        import ctypes

        HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x1A, 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(
            HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment",
            SMTO_ABORTIFHUNG, 5000, None)
    except Exception:
        pass
    return True


def _add_to_path_posix(directory: Path) -> bool:
    """Append an export line to the shell's rc file, once."""
    shell = os.path.basename(os.environ.get("SHELL", "")) or "bash"
    rc = {"zsh": Path.home() / ".zshrc",
          "fish": Path.home() / ".config" / "fish" / "config.fish",
          "bash": Path.home() / ".bashrc"}.get(shell, Path.home() / ".profile")
    line = (f"fish_add_path {directory}" if shell == "fish"
            else f'export PATH="{directory}:$PATH"')
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
    rc = {"zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish",
          "bash": "~/.bashrc"}.get(shell, "~/.profile")
    if shell == "fish":
        line = f'fish_add_path {directory}'
    return f"{line}    # add to {rc}"


def link_command(python: Path, venv_dir: Path,
                 assume_yes: bool) -> tuple[Path | None, bool]:
    """Put a `wynxo` command somewhere on PATH, so `wynxo` just works.

    A launcher script rather than a symlink: it pins the interpreter, so the
    command keeps working regardless of which virtualenv is active when it
    is run.
    """
    step("Making `wynxo` available everywhere")

    target = user_bin_dir()
    launcher = target / ("wynxo.cmd" if sys.platform == "win32" else "wynxo")

    if not ask(f"Install a `wynxo` command into {target}?", True, assume_yes):
        info("Skipped.")
        return None, False

    try:
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            launcher.write_text(f'@echo off\r\n"{python}" -m wynxo %*\r\n',
                                encoding="utf-8")
        else:
            launcher.write_text(
                f'#!/bin/sh\nexec "{python}" -m wynxo "$@"\n', encoding="utf-8")
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


def find_ollama() -> str | None:
    return shutil.which("ollama")


def ollama_reachable(url: str = f"http://127.0.0.1:{OLLAMA_PORT}") -> str | None:
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=3) as response:
            return json.loads(response.read()).get("version", "unknown")
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def setup_ollama(assume_yes: bool) -> bool:
    """Returns True when a reachable server exists by the end."""
    step("Setting up Ollama")

    if version := ollama_reachable():
        ok(f"Ollama {version} already serving on 127.0.0.1:{OLLAMA_PORT}")
        return True

    binary = find_ollama()
    if binary:
        ok(f"found {binary}, but nothing is serving yet")
        info(f"Start it in another terminal:  {S.bold('ollama serve')}")
        if ask("Start `ollama serve` in the background now?", True, assume_yes):
            return start_ollama()
        return False

    if is_termux():
        warn("Ollama does not run well on Android, and a 30B will not fit on a phone.")
        info("The normal setup is Ollama on a desktop or homelab box, with wynxo")
        info("in Termux talking to it over Wi-Fi. Once that machine is running:")
        info("")
        info("  On that machine:  OLLAMA_HOST=0.0.0.0:11434 ollama serve")
        info("  Here:             wynxo --endpoint 192.168.1.50")
        return False

    if sys.platform == "win32":
        warn("Ollama is not installed.")
        info("Download the installer from https://ollama.com/download")
        info("Then re-run this script.")
        return False

    warn("Ollama is not installed.")
    print()
    info("The official installer is:")
    info(f"    {S.bold('curl -fsSL https://ollama.com/install.sh | sh')}")
    info("It downloads a script from ollama.com and runs it, using sudo to")
    info("place the binary and (on Linux) register a systemd service.")
    print()
    if not ask("Run that now?", False, assume_yes):
        info("Skipped. Install it yourself, then re-run this script.")
        return False

    try:
        process = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True, timeout=900)
    except subprocess.TimeoutExpired:
        fail("The installer timed out.")
        return False
    if process.returncode != 0:
        fail("The installer failed. Install Ollama manually from https://ollama.com")
        return False

    ok("Ollama installed")
    return start_ollama()


def start_ollama() -> bool:
    """Launch `ollama serve` detached, and wait for it to answer."""
    import time

    if not find_ollama():
        return False
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x00000008     # DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except OSError as exc:
        fail(f"Could not start ollama serve: {exc}")
        return False

    for _ in range(20):
        if version := ollama_reachable():
            ok(f"Ollama {version} is serving")
            return True
        time.sleep(0.5)

    warn("Started it, but it has not answered yet. It may still be warming up.")
    return False


def installed_models() -> list[str]:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5) as response:
            return [m["name"] for m in json.loads(response.read()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


def _rank(tag: str) -> int:
    """Position in MODELS, strongest first. Unknown models rank last."""
    for i, (name, _, _) in enumerate(MODELS):
        if tag == name or tag.split(":")[0] == name.split(":")[0]:
            return i
    return len(MODELS)


def best_installed(installed: list[str]) -> str | None:
    """Pick the strongest model already present, by our own ranking."""
    for tag, _, _ in MODELS:
        for name in installed:
            if name == tag or name.split(":")[0] == tag.split(":")[0]:
                return name
    return installed[0] if installed else None


def ensure_model(preferred: str, why: str, assume_yes: bool) -> str | None:
    """Make sure some usable model exists. Returns the one to actually use.

    This has to return what is really there, not what was recommended -- the
    final check runs against this name, and pointing it at a model nobody
    pulled turns a successful install into a confusing failure.
    """
    step("Getting a model")
    installed = installed_models()

    if installed:
        ok(f"already installed: {', '.join(installed[:5])}")
        already = next(
            (m for m in installed
             if m == preferred or m.split(":")[0] == preferred.split(":")[0]), None)
        if already:
            return already

        current = best_installed(installed)
        # Never talk someone down to a weaker model than one they already run.
        # Recommendations are sized from memory, which under-reads a machine
        # with a big GPU.
        if current and _rank(current) < _rank(preferred):
            info(f"{current} is already installed and stronger than the "
                 f"{preferred} this machine's memory suggests. Keeping it.")
            return current
        print(f"     Recommended for this machine: {S.bold(preferred)} -- {why}")
        if not ask(f"Pull {preferred} as well?", False, assume_yes):
            info(f"Keeping {current}.")
            return current
    else:
        memory = total_memory_gb()
        if memory:
            info(f"this machine has about {memory:.0f}GB of memory")
        print(f"     Recommended: {S.bold(preferred)} -- {why}")
        if not ask(f"Pull {preferred} now? (a few GB, takes a while)", True, assume_yes):
            info(f"Skipped. Later:  ollama pull {preferred}")
            return None

    print()
    try:
        process = subprocess.run(["ollama", "pull", preferred], timeout=7200)
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(f"Pull failed: {exc}")
        return best_installed(installed)
    if process.returncode != 0:
        fail(f"`ollama pull {preferred}` failed.")
        return best_installed(installed)
    ok(f"{preferred} ready")
    return preferred


def run_doctor(python: Path, model: str) -> bool:
    """Returns True when the checks found no blockers."""
    step("Checking everything works")
    process = subprocess.run(
        [str(python), "-m", "wynxo", "--doctor",
         "--endpoint", f"127.0.0.1:{OLLAMA_PORT}", "--model", model],
        timeout=600)
    return process.returncode == 0


def finish(python: Path, venv_dir: Path, served: bool, healthy: bool,
           model: str | None, checked: bool, launcher: Path | None = None,
           path_was_updated: bool = False) -> None:
    print()
    if healthy:
        print(S.green(S.bold("  Done. Everything checks out.")))
    elif not checked:
        # Nothing was verified because nothing was asked to be. Not a problem.
        print(S.green(S.bold("  wynxo is installed.")))
        print(S.dim("  Once Ollama is reachable, check it over with:"))
        print(S.dim(f"    {venv_python(venv_dir)} -m wynxo --doctor"))
    else:
        print(S.yellow(S.bold("  Installed, with something left to sort out.")))
        print(S.dim("  The checks above say what. Re-run them any time:"))
        print(S.dim(f"    {venv_python(venv_dir)} -m wynxo --doctor"))
    print()
    if launcher is not None and on_path(launcher.parent):
        print("    Run it from anywhere:")
        print()
        print(f"      {S.bold('wynxo')}")
    elif launcher is not None and path_was_updated:
        print("    Open a NEW terminal, then:")
        print()
        print(f"      {S.bold('wynxo')}")
        print()
        print(S.dim("    (PATH changes only apply to terminals opened after the change.)"))
        print(S.dim("    In this one:"))
        print(S.dim(f"      {launcher}"))
    elif launcher is not None:
        print("    Once " + str(launcher.parent) + " is on your PATH:")
        print()
        print(f"      {S.bold('wynxo')}")
        print()
        print(S.dim("    Until then:"))
        print(S.dim(f"      {launcher}"))
    else:
        print("    Start it with:")
        print()
        print(f"      {S.bold(str(venv_python(venv_dir)) + ' -m wynxo')}")
    print()
    if not served:
        print(S.yellow("    Ollama is not serving yet."))
        print(S.dim("    Start it, or point wynxo at another machine:"))
        print(S.dim("      wynxo --endpoint 192.168.1.50"))
        print()
    print(S.dim("    First run asks where Ollama is and which model to use."))
    print(S.dim("    Then just talk to it. /help lists everything."))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up wynxo.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="accept the recommended answer to every prompt")
    parser.add_argument("--no-ollama", action="store_true",
                        help="only install wynxo; skip Ollama and the model")
    parser.add_argument("--model", help="pull this model instead of the recommended one")
    parser.add_argument("--venv", default=".venv", help="virtualenv directory")
    parser.add_argument("--no-link", action="store_true",
                        help="do not install a `wynxo` command onto PATH")
    args = parser.parse_args()

    print()
    print(S.cyan(S.bold("  wynxo setup")))
    print(S.dim("  A local AI coding agent. Nothing leaves your machine."))

    check_python()

    venv_dir = (ROOT / args.venv) if not os.path.isabs(args.venv) else Path(args.venv)
    python = make_venv(venv_dir, args.yes)
    install_wynxo(python, into_user=(python == Path(sys.executable) and not os.environ.get("VIRTUAL_ENV")))
    launcher, path_was_updated = (
        (None, False) if args.no_link else link_command(python, venv_dir, args.yes))

    served = False
    healthy = False
    preferred, why = recommend_model()
    if args.model:
        preferred, why = args.model, "you asked for this one"

    model: str | None = None
    checked = False
    if not args.no_ollama:
        served = setup_ollama(args.yes)
        if served:
            model = ensure_model(preferred, why, args.yes)
            if model:
                checked = True
                healthy = run_doctor(python, model)

    finish(python, venv_dir, served, healthy, model, checked, launcher,
           path_was_updated)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  cancelled")
        sys.exit(130)
