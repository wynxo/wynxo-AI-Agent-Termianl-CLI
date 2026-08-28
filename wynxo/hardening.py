"""Runtime hardening shims for cross-platform agent safety.

This module centralizes fixes that affect several tools: safe file replacement,
secret handling, cross-platform shell safety, and sensible project discovery.
"""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import tempfile
from pathlib import Path


def _atomic_write_back(path: Path, text: str, source=None) -> str:
    """Atomically replace a text file while preserving encoding/BOM and mode."""
    encoding = getattr(source, "encoding", "utf-8") if source else "utf-8"
    bom = getattr(source, "bom", b"") if source else b""
    try:
        payload = bom + text.encode(encoding)
        note = ""
    except UnicodeEncodeError:
        payload = text.encode("utf-8")
        note = (
            f" (saved as UTF-8: the new text needs characters "
            f"{encoding} cannot store)"
        ) if encoding != "utf-8" else ""

    path.parent.mkdir(parents=True, exist_ok=True)
    old_mode = None
    try:
        old_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        pass

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if old_mode is not None:
            try:
                os.chmod(temp_name, old_mode)
            except OSError:
                pass
        os.replace(temp_name, path)
        return note
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def windows_hard_refusal(command: str) -> str:
    """Return a reason only for irreversible/high-blast-radius Windows commands."""
    low = command.strip().lower()

    # Ordinary Remove-Item/Set-Content/etc. are *not* hard-refused: they must
    # flow into the normal permission system so legitimate project edits still
    # work. Only whole-drive/system destructive forms are blocked outright.
    whole_drive_remove = re.search(
        r"\bremove-item\b.*(?:-recurse\b.*)?\b(?:[a-z]:\\?$|[a-z]:/??$|\$home(?:\\|/)?$)",
        low,
        re.IGNORECASE,
    )
    if whole_drive_remove or re.search(
        r"\b(?:format-volume|format-disk|format-partition|clear-disk|initialize-disk|remove-partition|set-disk)\b",
        low,
        re.IGNORECASE,
    ):
        return "destructive Windows drive operation"
    if re.search(r"\bdiskpart(?:\.exe)?\b", low, re.IGNORECASE):
        return "disk partition management"
    if re.search(
        r"\b(?:stop-computer|restart-computer|shutdown(?:\.exe)?|restart(?:\.exe)?|halt(?:\.exe)?)\b",
        low,
        re.IGNORECASE,
    ):
        return "shutting down or restarting the machine"
    return ""


_POWERSHELL_SAFE = {
    "get-childitem", "gci", "dir", "ls", "get-content", "gc", "get-location",
    "pwd", "get-item", "gi", "get-command", "gcm", "where.exe", "where",
    "test-path", "select-string", "sls", "get-date", "get-process", "gps",
    "get-service", "get-variable", "get-alias", "get-history", "write-output",
    "echo", "type",
}

_SAFE_SUBCOMMANDS = {
    "git": {"status", "log", "diff", "show", "branch", "remote", "config", "rev-parse", "ls-files", "blame", "describe", "stash"},
    "npm": {"ls", "list", "view", "outdated"},
    "pip": {"list", "show", "freeze"},
    "cargo": {"tree", "metadata"},
}


def windows_is_read_only_command(command: str) -> bool:
    """Conservative Windows read-only classifier for permission decisions."""
    text = command.strip()
    if not text:
        return False
    if re.search(r"&&|\|\||[;|>&<`\n\r]", text):
        return False
    if windows_hard_refusal(text):
        return False

    parts = re.split(r"\s+", text)
    head = parts[0].lower()
    args = parts[1:]

    if head in _POWERSHELL_SAFE:
        return True
    if head in _SAFE_SUBCOMMANDS and args:
        return args[0].lower() in _SAFE_SUBCOMMANDS[head]
    if head in {"python", "py", "node"}:
        return len(args) == 1 and args[0].lower() in {"--version", "-v", "-version"}
    return False


def _visible_path_parts(path: Path) -> list[str]:
    return [part for part in path.parts if part not in {".", ""}]


def install() -> None:
    """Install hardening patches once, idempotently."""
    from .tools import files as files_mod
    from .tools import permissions as permissions_mod
    from .tools import search as search_mod
    from .tools import shell as shell_mod

    if getattr(files_mod, "_WYNXO_HARDENED", False):
        return

    # 1) All text mutations use atomic replacement.
    files_mod._write_back = _atomic_write_back

    # 2) Secret shielding applies to direct file mutations too.
    for cls_name in ("WriteFile", "EditFile", "MultiEdit"):
        cls = getattr(files_mod, cls_name, None)
        if cls is None:
            continue
        original = cls.run
        if getattr(original, "_wynxo_hardened", False):
            continue

        async def guarded_run(self, args, _original=original):
            path_value = getattr(args, "path", None)
            if path_value:
                path = self.resolve_path(path_value)
                if refused := self.shield.blocks(path):
                    from .tools.base import ToolResult
                    return ToolResult.failure(refused)
            return await _original(self, args)

        guarded_run._wynxo_hardened = True
        cls.run = guarded_run

    # 3) Hidden does not mean ignorable. Preserve real project config while
    # skipping repositories, environments and generated dependency trees.
    ignored = set(getattr(files_mod, "IGNORED", set()))
    ignored.difference_update({".github", ".vscode", ".devcontainer", ".husky"})
    ignored.update({
        ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox",
    })

    original_collect = search_mod.Glob._collect
    if not getattr(original_collect, "_wynxo_hardened", False):
        def collect(self, root, pattern):
            out = []
            for path in root.rglob("*"):
                if len(out) > search_mod.MAX_MATCHES * 4:
                    break
                if not path.is_file():
                    continue
                if any(part in ignored for part in _visible_path_parts(path)):
                    continue
                rel = self.relative(path)
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
                    out.append(rel)
            out.sort(
                key=lambda r: -(
                    root.joinpath(r).stat().st_mtime if (root / r).exists() else 0
                )
            )
            return out
        collect._wynxo_hardened = True
        search_mod.Glob._collect = collect

    original_candidates = search_mod.Grep._candidates
    if not getattr(original_candidates, "_wynxo_hardened", False):
        def candidates(self, root, glob):
            out = []
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if any(part in ignored for part in _visible_path_parts(path)):
                    continue
                if glob and not (
                    fnmatch.fnmatch(path.name, glob)
                    or fnmatch.fnmatch(self.relative(path), glob)
                ):
                    continue
                out.append(path)
            return out
        candidates._wynxo_hardened = True
        search_mod.Grep._candidates = candidates

    original_walk = files_mod.ListDir._walk
    if not getattr(original_walk, "_wynxo_hardened", False):
        def walk(self, directory, depth, prefix, out):
            if depth <= 0 or len(out) >= files_mod.MAX_ENTRIES:
                return len(out) >= files_mod.MAX_ENTRIES
            try:
                entries = sorted(
                    (e for e in directory.iterdir() if e.name not in ignored),
                    key=lambda e: (e.is_file(), e.name.lower()),
                )
            except PermissionError:
                out.append(f"{prefix}[permission denied]")
                return False
            for i, entry in enumerate(entries):
                if len(out) >= files_mod.MAX_ENTRIES:
                    return True
                last = i == len(entries) - 1
                branch = "`-- " if last else "|-- "
                if entry.is_symlink():
                    out.append(
                        f"{prefix}{branch}{entry.name}"
                        f"{'/' if entry.is_dir() else ''} -> {self._target(entry)}"
                    )
                    continue
                out.append(f"{prefix}{branch}{entry.name}{'/' if entry.is_dir() else ''}")
                if entry.is_dir():
                    if walk(self, entry, depth - 1, prefix + ("    " if last else "|   "), out):
                        return True
            return False
        walk._wynxo_hardened = True
        files_mod.ListDir._walk = walk

    # 4) Secret-looking shell output is masked before it reaches UI/model.
    original_stream = shell_mod.Shell._stream
    if not getattr(original_stream, "_wynxo_hardened", False):
        async def secure_stream(self, process, timeout):
            callback = self.on_output

            async def safe_callback(line):
                if callback is None:
                    return
                try:
                    safe, _ = self.shield.clean(line)
                    await callback(safe)
                except Exception:
                    return

            self.on_output = safe_callback
            try:
                body, timed_out = await original_stream(self, process, timeout)
            finally:
                self.on_output = callback
            safe_body, _ = self.shield.clean(body)
            return safe_body, timed_out

        secure_stream._wynxo_hardened = True
        shell_mod.Shell._stream = secure_stream

    # 5) Extend the shell classifier for Windows/PowerShell.
    original_hard_refusal = shell_mod.hard_refusal
    def hard_refusal(command):
        base = original_hard_refusal(command)
        if base:
            return base
        if os.name == "nt":
            return windows_hard_refusal(command)
        return (
            windows_hard_refusal(command)
            if re.search(r"\b(?:pwsh|powershell)\b", command, re.I)
            else ""
        )
    shell_mod.hard_refusal = hard_refusal

    original_read_only = permissions_mod.is_read_only_command
    def is_read_only(command):
        if os.name == "nt":
            return windows_is_read_only_command(command)
        return original_read_only(command)
    permissions_mod.is_read_only_command = is_read_only

    files_mod._WYNXO_HARDENED = True
