"""Runtime hardening shims for cross-platform agent safety.

This module deliberately lives at the package boundary so fixes that apply to
multiple tools are centralized while the older tool implementations remain
backwards compatible.  The shims are conservative: uncertain commands prompt,
secret-looking output is masked, and file replacement is atomic.
"""

from __future__ import annotations

import asyncio
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

    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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
    """Return a reason for destructive PowerShell/CMD commands, if any."""
    text = command.strip()
    low = text.lower()

    # Command chaining / redirection is not itself destructive, but the shell
    # permission classifier must not call a compound command read-only.
    destructive = (
        (r"\bremove-item\b", "a PowerShell file/directory removal"),
        (r"\bdel\b(?:\s+/s|\s+/f|\s+/q)?", "a Windows file deletion"),
        (r"\brmdir\b|\brd\b", "a Windows directory deletion"),
        (r"\bclear-content\b", "clearing file contents"),
        (r"\bset-content\b|\badd-content\b|\bout-file\b", "writing file contents"),
        (r"\bmove-item\b|\brename-item\b|\bcopy-item\b", "changing filesystem entries"),
        (r"\bformat-(?:volume|disk|partition)\b|\bformat\b", "formatting a drive/filesystem"),
        (r"\bclear-disk\b|\bremove-partition\b|\binitialize-disk\b|\bset-disk\b", "rewriting disk state"),
        (r"\bstop-computer\b|\brestart-computer\b|\bshutdown(?:\.exe)?\b|\brestart(?:\.exe)?\b", "shutting down or restarting the machine"),
        (r"\bstop-process\b|\bstop-service\b|\brestart-service\b", "stopping or restarting a process/service"),
    )
    for pattern, reason in destructive:
        if re.search(pattern, low, re.IGNORECASE):
            # Make whole-drive deletes/formatting especially explicit.
            if re.search(r"(?:remove-item\s+.*(?:-recurse|/s).*\b(?:[a-z]:[\\/]|\$home)\b|(?:format|format-volume|format-disk).*\b[a-z]:?\b)", low, re.IGNORECASE):
                return "destructive Windows drive operation"
            return reason
    return ""


_POWERSHELL_SAFE = {
    "get-childitem", "gci", "dir", "ls", "get-content", "gc", "get-location",
    "pwd", "get-item", "gi", "get-command", "gcm", "where.exe", "where",
    "test-path", "select-string", "sls", "get-date", "get-process", "gps",
    "get-service", "get-variable", "get-alias", "get-history", "write-output",
    "echo", "type", "python", "py", "node", "git", "npm", "pip", "cargo",
}


def windows_is_read_only_command(command: str) -> bool:
    """Conservative PowerShell read-only classifier for permission prompts."""
    text = command.strip()
    if not text:
        return False
    if re.search(r"&&|\|\||[;|>&<`\n\r]", text):
        return False
    if windows_hard_refusal(text):
        return False
    parts = re.split(r"\s+", text, maxsplit=1)
    head = parts[0].lower()
    if head.startswith("&"):
        return False
    return head in _POWERSHELL_SAFE


def install() -> None:
    """Install hardening patches once, idempotently."""
    from .tools import files as files_mod
    from .tools import permissions as permissions_mod
    from .tools import search as search_mod
    from .tools import shell as shell_mod

    if getattr(files_mod, "_WYNXO_HARDENED", False):
        return

    # 1) All normal text edits go through atomic replacement.
    files_mod._write_back = _atomic_write_back

    # 2) Secret shielding applies to every direct file mutation, not only reads.
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

    # 3) Hidden directories are not synonymous with junk.  Ignore only known
    # generated/VCS directories so .github/.vscode/.devcontainer remain visible.
    ignored = set(getattr(files_mod, "IGNORED", set()))
    ignored.update({".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox"})

    original_collect = search_mod.Glob._collect
    if not getattr(original_collect, "_wynxo_hardened", False):
        def collect(self, root, pattern):
            out = []
            for path in root.rglob("*"):
                if len(out) > search_mod.MAX_MATCHES * 4:
                    break
                if not path.is_file():
                    continue
                if any(part in ignored for part in path.parts):
                    continue
                rel = self.relative(path)
                if __import__("fnmatch").fnmatch.fnmatch(rel, pattern) or __import__("fnmatch").fnmatch.fnmatch(path.name, pattern):
                    out.append(rel)
            out.sort(key=lambda r: -(root.joinpath(r).stat().st_mtime if (root / r).exists() else 0))
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
                if any(part in ignored for part in path.parts):
                    continue
                if glob and not (__import__("fnmatch").fnmatch.fnmatch(path.name, glob) or __import__("fnmatch").fnmatch.fnmatch(self.relative(path), glob)):
                    continue
                out.append(path)
            return out
        candidates._wynxo_hardened = True
        search_mod.Grep._candidates = candidates

    # 4) Mask secret-looking shell output before it reaches the TUI or model.
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

    # 5) Extend the existing Unix command classifier with PowerShell-aware rules.
    original_hard_refusal = shell_mod.hard_refusal
    def hard_refusal(command):
        base = original_hard_refusal(command)
        if base:
            return base
        if os.name == "nt":
            return windows_hard_refusal(command)
        # Also recognize explicit PowerShell syntax when invoked from WSL-like
        # environments, because the user may still have chosen pwsh manually.
        return windows_hard_refusal(command) if re.search(r"\b(?:pwsh|powershell)\b", command, re.I) else ""
    shell_mod.hard_refusal = hard_refusal

    original_read_only = permissions_mod.is_read_only_command
    def is_read_only(command):
        if os.name == "nt":
            return windows_is_read_only_command(command)
        return original_read_only(command)
    permissions_mod.is_read_only_command = is_read_only

    files_mod._WYNXO_HARDENED = True


install()
