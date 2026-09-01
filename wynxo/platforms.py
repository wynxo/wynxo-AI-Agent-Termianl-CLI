"""Per-platform behaviour, in one place.

wynxo runs on Linux, macOS, Windows and Termux. Termux is the awkward one:
it is Linux-flavoured but its filesystem lives under an app-private prefix,
it has no ``/tmp``, its shell is not in ``/bin``, and it renders on a phone
screen forty columns wide. Treating it as "just Linux" breaks all four.
"""

from __future__ import annotations

import os
import platform as _platform
import shutil
import sys
from pathlib import Path

TERMUX_PREFIX = "/data/data/com.termux/files/usr"


def is_termux() -> bool:
    """Detect Termux.

    ``TERMUX_VERSION`` is set by the app itself, but is lost by anything that
    sanitises the environment (cron, some supervisors), so the prefix path is
    checked as well.
    """
    if os.environ.get("TERMUX_VERSION"):
        return True
    prefix = os.environ.get("PREFIX", "")
    return prefix.startswith("/data/data/com.termux") or os.path.isdir(TERMUX_PREFIX)


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def name() -> str:
    """A short human-readable platform name for the system prompt."""
    if is_termux():
        return "Termux (Android)"
    if is_windows():
        return "Windows"
    if is_macos():
        return "macOS"
    return _platform.system() or "Linux"


def describe() -> str:
    if is_termux():
        release = os.environ.get("TERMUX_VERSION", "")
        machine = _platform.machine()
        return f"Termux {release} on Android ({machine})".replace("  ", " ")
    return f"{name()} ({_platform.release()})"


# -- filesystem ------------------------------------------------------------

def home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def config_dir() -> Path:
    """Where config.json lives."""
    if is_windows():
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "wynxo"
    if is_macos():
        return Path.home() / "Library" / "Application Support" / "wynxo"
    # Termux included: XDG under the app-private home is correct and writable.
    base = os.environ.get("XDG_CONFIG_HOME") or (home() / ".config")
    return Path(base) / "wynxo"


def data_dir() -> Path:
    """Where sessions and history live."""
    if is_windows():
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "wynxo"
    if is_macos():
        return Path.home() / "Library" / "Application Support" / "wynxo"
    base = os.environ.get("XDG_DATA_HOME") or (home() / ".local" / "share")
    return Path(base) / "wynxo"


def temp_dir() -> Path:
    """Termux has no ``/tmp``; anything assuming otherwise fails at runtime."""
    if is_termux():
        return Path(os.environ.get("TMPDIR") or f"{TERMUX_PREFIX}/tmp")
    return Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")


# -- shell -----------------------------------------------------------------

def default_shell() -> tuple[str, list[str]]:
    """The shell to run commands with, and the flag that takes a command string."""
    if is_windows():
        # PowerShell where available -- cmd.exe quoting is a source of endless
        # subtle breakage, and pwsh/powershell is on every supported Windows.
        for exe in ("pwsh", "powershell"):
            if shutil.which(exe):
                return exe, ["-NoProfile", "-NonInteractive", "-Command"]
        return os.environ.get("COMSPEC", "cmd.exe"), ["/c"]

    shell = os.environ.get("SHELL")
    if shell and shutil.which(shell):
        return shell, ["-c"]

    if is_termux():
        # Termux's binaries are under $PREFIX, never /bin.
        prefix = os.environ.get("PREFIX", TERMUX_PREFIX)
        for exe in (f"{prefix}/bin/bash", f"{prefix}/bin/sh"):
            if os.path.exists(exe):
                return exe, ["-c"]

    for exe in ("bash", "sh"):
        if path := shutil.which(exe):
            return path, ["-c"]
    return "/bin/sh", ["-c"]


# -- terminal --------------------------------------------------------------

def _tty_size():
    """The terminal's real size, asked of the terminal itself.

    ``shutil.get_terminal_size`` consults ``COLUMNS``/``LINES`` *first* and
    only falls back to the ioctl. Those variables are set once, by whatever
    started the process, and nothing updates them when the window changes --
    so anywhere they are exported (a shell with checkwinsize, tmux, CI, a
    test harness) every width wynxo reasons about froze at launch and no
    resize could move it. Streaming then ran off the edge of a narrowed
    window for the rest of the session.

    Asked of the file descriptor instead, the answer is always current.
    stdout first, because that is what is being drawn on; stdin and stderr
    after it, for the case where output is redirected but a terminal is
    still attached. Returns None when none of them is a terminal, which is
    when the environment variables are the right answer after all.
    """
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            fd = stream.fileno()
        except (AttributeError, ValueError, OSError):
            continue
        try:
            size = os.get_terminal_size(fd)
        except (AttributeError, OSError, ValueError):
            continue
        if size.columns > 0 and size.lines > 0:
            return size
    return None


def terminal_width(default: int = 80) -> int:
    size = _tty_size()
    if size is not None:
        return size.columns
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except (OSError, ValueError):
        return default


def terminal_height(default: int = 24) -> int:
    size = _tty_size()
    if size is not None:
        return size.lines
    try:
        return shutil.get_terminal_size((80, default)).lines
    except (OSError, ValueError):
        return default


def is_dumb_terminal() -> bool:
    """TERM=dumb, or no terminal at all.

    prompt_toolkit drops to a plain readline here: no bottom toolbar, no
    redraw, no colours. Chrome that assumes a full-screen renderer has to be
    skipped rather than half-drawn.
    """
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", "unknown"):
        return True
    try:
        return not sys.stdout.isatty()
    except (AttributeError, ValueError):
        return True


def is_narrow() -> bool:
    """A phone in portrait is roughly 40-56 columns. Wide-terminal layout
    (side-by-side tables, boxed banners) becomes unreadable below that."""
    return terminal_width() < 60


def looks_like_a_project(path: Path) -> bool:
    """Whether this directory plausibly contains work to do.

    Someone who installs a `wynxo` command and then runs it from wherever they
    happened to be gets an agent pointed at an install folder or their home
    directory. It works, which is the problem -- it quietly operates on the
    wrong files. Cheap markers, no guessing.
    """
    markers = (
        ".git", "package.json", "pyproject.toml", "setup.py", "Cargo.toml",
        "go.mod", "pom.xml", "build.gradle", "Makefile", "CMakeLists.txt",
        "composer.json", "Gemfile", "requirements.txt", "WYNXO.md",
        "AGENTS.md", "CLAUDE.md", ".wynxo.json",
    )
    try:
        for marker in markers:
            if (path / marker).exists():
                return True
    except OSError:
        return False
    return False


def _system_locations() -> list[tuple[Path, str]]:
    """Directories nobody means to start a project in.

    Taken from the environment rather than by name, because matching a path
    *component* called "windows" or "appdata" flags any project with a
    src/windows/ directory in it -- which is most cross-platform projects --
    and on Windows the temp directory lives under AppData, so it flagged
    practically everything.
    """
    found: list[tuple[Path, str]] = []
    if os.name == "nt":
        for variable, label in (("APPDATA", "your roaming profile"),
                                ("LOCALAPPDATA", "your local profile"),
                                ("SystemRoot", "the Windows directory"),
                                ("ProgramFiles", "Program Files"),
                                ("ProgramFiles(x86)", "Program Files")):
            if raw := os.environ.get(variable):
                found.append((Path(raw), label))
    else:
        for raw, label in (("/usr", "a system directory"),
                           ("/etc", "a system directory"),
                           ("/bin", "a system directory"),
                           ("/sbin", "a system directory"),
                           ("/boot", "a system directory")):
            found.append((Path(raw), label))
    return found


def _inside_a_system_location(resolved: Path) -> str | None:
    """Whether this is a system directory itself -- not merely under one.

    Under is too broad: the Windows temp directory sits inside AppData, and
    people do keep checkouts there. Being *exactly* one of these is the case
    worth refusing.
    """
    if str(resolved) in ("/", "\\"):
        return "this is the filesystem root"
    for location, label in _system_locations():
        try:
            if resolved == location.resolve():
                return f"this is {label}"
        except OSError:
            continue
    return None


def suspicious_workspace(path: Path) -> str | None:
    """A reason this directory is probably not where you meant to work."""
    try:
        resolved = path.resolve()
    except OSError:
        return None

    if resolved == home().resolve():
        return "this is your home directory"

    # The directory wynxo's own launcher was installed into.
    for name in ("wynxo", "wynxo.cmd", "wynxo.exe"):
        if (resolved / name).is_file() and not (resolved / "pyproject.toml").exists():
            return "this is where wynxo itself is installed"

    if system := _inside_a_system_location(resolved):
        return system

    if not looks_like_a_project(resolved):
        return "no project files here (no .git, no package manifest)"
    return None


# -- clipboard ------------------------------------------------------------

def copy_to_clipboard(text: str) -> bool:
    """Put text on the system clipboard, best effort per platform.

    Drag-to-select works normally -- wynxo never captures the mouse -- but
    a whole conversation is easier to take in one piece than to drag over.
    Uses only tools that are already on the machine -- nothing is installed
    -- and falls back along the list until one of them accepts the text.
    """
    import subprocess

    if not text:
        return False
    commands: list[list[str]] = []
    if is_windows():
        # Set-Clipboard handles full Unicode; `clip` is the fallback when
        # PowerShell is not available. The text travels in an environment
        # variable so no codepage can mangle it.
        commands = [
            ["powershell", "-NoProfile", "-NonInteractive",
             "-Command", "$env:WYNXO_COPY | Set-Clipboard"],
            ["clip"],
        ]
    elif is_termux():
        commands = [["termux-clipboard-set"]]
    elif is_macos():
        commands = [["pbcopy"]]
    else:
        for tool in ("wl-copy", "xclip", "xsel"):
            if shutil.which(tool):
                commands = [
                    [tool, "-selection", "clipboard"] if tool == "xclip"
                    else [tool, "--clipboard", "--input"] if tool == "xsel"
                    else [tool],
                ]
                break
    for command in commands:
        env = dict(os.environ)
        data: bytes | None = b""
        if command[0] == "powershell":
            env["WYNXO_COPY"] = text
        else:
            data = text.encode("utf-8", "replace")
        try:
            proc = subprocess.run(command, input=data, env=env,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=10)
        except Exception:
            continue
        if proc.returncode == 0:
            return True
    return False


# -- setup hints -----------------------------------------------------------

def ollama_server_help() -> str:
    """Printed when a connection fails. The remote case trips everyone up."""
    if is_termux():
        return (
            "Ollama only listens on loopback by default, so a server on another\n"
            "machine is unreachable until you tell it otherwise.\n\n"
            "  On the machine running Ollama (not the phone):\n"
            "    OLLAMA_HOST=0.0.0.0:11434 ollama serve\n\n"
            "  Then point wynxo at that machine's LAN address:\n"
            "    wynxo --endpoint 192.168.1.50\n\n"
            "  Find it by running `ip addr` or `ipconfig` on that machine.\n"
            "  The phone must be on the same Wi-Fi, not mobile data."
        )
    if is_windows():
        return (
            "Ollama only listens on loopback by default, so a server on another\n"
            "machine is unreachable until you tell it otherwise.\n\n"
            "  On the remote machine (PowerShell, then restart Ollama):\n"
            "    [Environment]::SetEnvironmentVariable("
            "'OLLAMA_HOST','0.0.0.0:11434','User')"
        )
    return (
        "Ollama only listens on loopback by default, so a server on another\n"
        "machine is unreachable until you tell it otherwise.\n\n"
        "  On the remote machine:\n"
        "    OLLAMA_HOST=0.0.0.0:11434 ollama serve\n"
        "  Or persist it (systemd):\n"
        "    sudo systemctl edit ollama\n"
        "    [Service]\n"
        '    Environment="OLLAMA_HOST=0.0.0.0:11434"'
    )
