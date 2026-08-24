#!/usr/bin/env python3
"""Uninstaller for wynxo.

    python3 uninstall.py

Removes everything the installer put on the machine and nothing else: the
launcher, the source checkout and its virtualenv, the config and data
directories, and the PATH line that was added to your shell profile.

Deliberately self-contained -- it never imports wynxo. An uninstaller that
needs the thing it is uninstalling is useless in exactly the case you most
want one: a half-finished or broken install.

    --yes       accept every prompt
    --dry-run   list what would be removed, change nothing
    --keep-data leave config and sessions in place (for a reinstall)
    --force     remove cloned repositories even with unsaved work in them
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)


# -- output ----------------------------------------------------------------

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


def ask(question: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        print(f"     {question} {S.dim('[auto: yes]')}")
        return True
    if not sys.stdin.isatty():
        shown = "yes" if default else "no"
        print(f"     {question} {S.dim('[not a terminal: assuming ' + shown + ']')}")
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


# -- where things are ------------------------------------------------------
#
# Duplicated from wynxo/platforms.py rather than imported, on purpose: see
# the module docstring. These must stay in step with it.

def is_termux() -> bool:
    return bool(os.environ.get("TERMUX_VERSION")) or \
        os.environ.get("PREFIX", "").startswith("/data/data/com.termux")


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def config_dir() -> Path:
    if is_windows():
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "wynxo"
    if is_macos():
        return Path.home() / "Library" / "Application Support" / "wynxo"
    base = os.environ.get("XDG_CONFIG_HOME") or (home() / ".config")
    return Path(base) / "wynxo"


def data_dir() -> Path:
    if is_windows():
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "wynxo"
    if is_macos():
        return Path.home() / "Library" / "Application Support" / "wynxo"
    base = os.environ.get("XDG_DATA_HOME") or (home() / ".local" / "share")
    return Path(base) / "wynxo"


def source_dir() -> Path:
    """Where get.sh / get.ps1 clone the source to."""
    return Path(os.environ.get("WYNXO_SRC") or (Path.home() / ".wynxo-src"))


def launcher_candidates() -> list[Path]:
    """Every place link_command() might have put a `wynxo` command."""
    out: list[Path] = []
    if is_windows():
        local = Path(os.environ.get("LOCALAPPDATA")
                     or (Path.home() / "AppData" / "Local"))
        out.append(local / "Microsoft" / "WindowsApps" / "wynxo.cmd")
        out.append(local / "Programs" / "wynxo" / "wynxo.cmd")
    else:
        if is_termux():
            prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
            out.append(Path(prefix) / "bin" / "wynxo")
        out.append(Path.home() / ".local" / "bin" / "wynxo")
    return out


def rc_candidates() -> list[Path]:
    """Shell profiles _add_to_path_posix() might have written to.

    All of them, not just the current shell's: someone can install under
    bash and uninstall under zsh, and a stale PATH line in the profile they
    are not using at that moment is exactly the kind of leftover this is
    supposed to prevent.
    """
    return [
        Path.home() / ".bashrc",
        Path.home() / ".zshrc",
        Path.home() / ".profile",
        Path.home() / ".config" / "fish" / "config.fish",
    ]


MARKER = "# added by wynxo installer"


# -- discovery -------------------------------------------------------------

def directory_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def human_size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def git_output(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def unsaved_work(repo: Path) -> str:
    """Why this checkout should not be deleted without asking, if so.

    Two separate hazards, and the second is the one people forget: a clean
    tree can still hold commits that exist nowhere else.
    """
    if not (repo / ".git").exists():
        return ""
    dirty = git_output(repo, "status", "--porcelain")
    if dirty:
        count = len([line for line in dirty.splitlines() if line.strip()])
        return f"{count} uncommitted change(s)"
    # No upstream at all also counts: nothing to have been pushed to.
    unpushed = git_output(repo, "log", "--branches", "--not", "--remotes", "--oneline")
    if unpushed and unpushed.strip():
        count = len(unpushed.strip().splitlines())
        return f"{count} unpushed commit(s)"
    return ""


def cloned_repos() -> list[Path]:
    """The GitHub checkouts /repo made, which are the user's, not ours."""
    root = data_dir() / "repos"
    if not root.is_dir():
        return []
    found: list[Path] = []
    try:
        for owner in sorted(root.iterdir()):
            if not owner.is_dir():
                continue
            for name in sorted(owner.iterdir()):
                if name.is_dir():
                    found.append(name)
    except OSError:
        pass
    return found


# -- removal ---------------------------------------------------------------

def remove_tree(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        info(f"would remove {path}")
        return True
    try:
        shutil.rmtree(path)
        return True
    except OSError as exc:
        fail(f"could not remove {path}: {exc}")
        return False


def remove_file(path: Path, dry_run: bool) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if dry_run:
        info(f"would remove {path}")
        return True
    try:
        path.unlink()
        return True
    except OSError as exc:
        fail(f"could not remove {path}: {exc}")
        return False


def strip_path_line(rc: Path, dry_run: bool) -> bool:
    """Take the installer's PATH line back out of a shell profile.

    Only the marker comment and the single line after it, matched as a pair.
    Anything else in the file is someone's own configuration and is not ours
    to touch -- a careless uninstaller that rewrites a profile is worse than
    one that leaves a line behind.
    """
    try:
        original = rc.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if MARKER not in original:
        return False

    lines = original.splitlines(keepends=True)
    kept: list[str] = []
    index = 0
    removed = False
    while index < len(lines):
        if lines[index].strip() == MARKER:
            index += 1                       # the marker itself
            if index < len(lines):
                index += 1                   # the export/fish_add_path line
            # The installer writes a blank line before the marker; drop it
            # so repeated install/uninstall cycles cannot stack up blanks.
            while kept and not kept[-1].strip():
                kept.pop()
            removed = True
            continue
        kept.append(lines[index])
        index += 1

    if not removed:
        return False
    if dry_run:
        info(f"would remove the PATH line from {rc}")
        return True
    try:
        rc.write_text("".join(kept), encoding="utf-8")
        return True
    except OSError as exc:
        fail(f"could not edit {rc}: {exc}")
        return False


def strip_path_windows(directory: Path, dry_run: bool) -> bool:
    r"""Take a directory back out of the user's PATH in HKCU\Environment.

    Read and write the registry value rather than the expanded process PATH:
    the latter includes the machine-wide half, and writing it back would
    silently copy every system entry into the user's own PATH.
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
                return False
    except OSError:
        return False

    entries = [e for e in str(current).split(";") if e.strip()]
    target = str(directory).rstrip("\\").lower()
    remaining = [e for e in entries if e.rstrip("\\").lower() != target]
    if len(remaining) == len(entries):
        return False
    if dry_run:
        info(f"would remove {directory} from your PATH")
        return True

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "Path", 0, kind or winreg.REG_EXPAND_SZ,
                              ";".join(remaining))
    except OSError:
        return False

    try:
        import ctypes

        HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x1A, 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(
            HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment",
            SMTO_ABORTIFHUNG, 5000, None)
    except Exception:
        pass
    return True


def running_from(path: Path) -> bool:
    """Whether this interpreter lives inside ``path``.

    Matters on Windows only, where a running executable cannot be deleted.
    POSIX unlinks by name and the open inode survives, so removing the tree
    out from under a running script is safe there.
    """
    try:
        Path(sys.executable).resolve().relative_to(path.resolve())
        return True
    except (ValueError, OSError):
        return False


# -- the script ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Uninstall wynxo.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="accept every prompt")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be removed, change nothing")
    parser.add_argument("--keep-data", action="store_true",
                        help="leave config and sessions in place")
    parser.add_argument("--force", action="store_true",
                        help="remove cloned repositories even with unsaved work")
    args = parser.parse_args()

    print()
    print(S.cyan(S.bold("  wynxo uninstaller")))
    if args.dry_run:
        print(S.dim("  Dry run: nothing will be changed."))

    # -- find ---------------------------------------------------------------
    step("Looking for what is installed")

    launchers = [p for p in launcher_candidates() if p.exists() or p.is_symlink()]
    source = source_dir() if source_dir().is_dir() else None
    # macOS puts config and data in the same directory; dict.fromkeys keeps
    # order while collapsing the duplicate, so it is not reported twice.
    data_paths = [p for p in dict.fromkeys([config_dir(), data_dir()]) if p.is_dir()]
    rc_files = [rc for rc in rc_candidates()
                if rc.is_file() and MARKER in _safe_read(rc)]

    found = False
    for launcher in launchers:
        ok(f"launcher      {launcher}")
        found = True
    if source:
        ok(f"source        {source}  {S.dim(human_size(directory_size(source)))}")
        found = True
    for path in data_paths:
        label = "config" if path == config_dir() else "data"
        ok(f"{label:<13} {path}  {S.dim(human_size(directory_size(path)))}")
        found = True
    for rc in rc_files:
        ok(f"PATH line     {rc}")
        found = True

    windows_path_dirs: list[Path] = []
    if is_windows():
        for launcher in launcher_candidates():
            if launcher.parent.name.lower() == "wynxo":
                windows_path_dirs.append(launcher.parent)

    if not found and not windows_path_dirs:
        print()
        ok("Nothing to remove -- wynxo is not installed for this user.")
        print()
        return 0

    # -- protect the user's own work ---------------------------------------
    repos = cloned_repos()
    at_risk: list[tuple[Path, str]] = []
    if repos and not args.keep_data:
        step("Checking cloned repositories")
        for repo in repos:
            reason = unsaved_work(repo)
            if reason:
                at_risk.append((repo, reason))
                warn(f"{repo.name}  {S.yellow(reason)}")
            else:
                info(f"{repo.name}  clean")
        if at_risk and not args.force:
            print()
            fail("Some cloned repositories have work that exists nowhere else.")
            info("These were cloned by /repo and live in wynxo's data directory,")
            info("so removing that directory would take them with it.")
            print()
            for repo, reason in at_risk:
                info(f"{repo}  ({reason})")
            print()
            info("Push or copy them out, then run this again. To delete them")
            info("anyway, re-run with --force.")
            print()
            return 1

    # -- confirm ------------------------------------------------------------
    if not args.dry_run:
        print()
        if not ask("Remove all of the above?", True, args.yes):
            print()
            info("Nothing was changed.")
            print()
            return 1

    # -- remove -------------------------------------------------------------
    step("Removing")

    for launcher in launchers:
        if remove_file(launcher, args.dry_run):
            ok(f"removed {launcher}")

    for rc in rc_files:
        if strip_path_line(rc, args.dry_run):
            ok(f"cleaned the PATH line out of {rc}")

    for directory in windows_path_dirs:
        if strip_path_windows(directory, args.dry_run):
            ok(f"removed {directory} from your PATH")

    if not args.keep_data:
        for path in data_paths:
            if remove_tree(path, args.dry_run):
                ok(f"removed {path}")
    elif data_paths:
        info("kept your config and sessions (--keep-data)")

    deferred = None
    if source:
        if is_windows() and running_from(source):
            # Python holds its own executable open on Windows, so the tree
            # cannot delete itself. Everything else is gone by now; hand
            # back the one command that finishes the job.
            deferred = source
        elif remove_tree(source, args.dry_run):
            ok(f"removed {source}")

    # -- done ---------------------------------------------------------------
    print()
    if args.dry_run:
        print(S.dim("  Dry run finished. Nothing was changed."))
        print()
        return 0

    print(S.green(S.bold("  wynxo has been removed.")))
    print()
    if deferred:
        warn("One directory is left: this uninstaller is running from inside it.")
        info("Windows will not delete a folder holding a running program.")
        print()
        info("Finish with:")
        print(f'      {S.bold(f"Remove-Item -Recurse -Force {deferred}")}')
        print()
    if rc_files:
        info("The PATH change applies to terminals opened from now on.")
        print()
    if args.keep_data:
        info("Your config and sessions were kept. Reinstalling picks them up.")
        print()
    return 0


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        info("Cancelled. Nothing was changed.")
        sys.exit(1)
