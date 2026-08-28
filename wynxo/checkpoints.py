"""Undo for file changes, preserving the exact bytes on disk."""

from __future__ import annotations

import os
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_SNAPSHOTS = 100
MAX_FILE_BYTES = 2_000_000


@dataclass
class Snapshot:
    path: Path
    content: str | None
    """Best-effort text view kept for diffs and backwards compatibility."""
    tool: str
    when: float = field(default_factory=time.time)
    label: str = ""
    raw_bytes: bytes | None = None
    mode: int | None = None

    @property
    def existed(self) -> bool:
        return self.raw_bytes is not None or self.content is not None


class Checkpoints:
    def __init__(self) -> None:
        self._stack: list[Snapshot] = []

    def __len__(self) -> int:
        return len(self._stack)

    def capture(self, path: Path, tool: str, label: str = "") -> None:
        """Record the exact pre-change bytes and mode, without re-encoding."""
        try:
            if path.exists():
                if path.is_dir() or path.stat().st_size > MAX_FILE_BYTES:
                    return
                raw = path.read_bytes()
                mode = stat.S_IMODE(path.stat().st_mode)
                # Keep a decoded representation for existing callers and UI
                # diffs. UTF-8 with surrogateescape is lossless for bytes that
                # are not valid UTF-8, while BOM-aware decodes make common
                # UTF-16/UTF-8-sig project files readable in review output.
                if raw.startswith((b"\\xff\\xfe", b"\\xfe\\xff")):
                    encoding = "utf-16"
                elif raw.startswith(b"\\xef\\xbb\\xbf"):
                    encoding = "utf-8-sig"
                else:
                    encoding = "utf-8"
                content = raw.decode(encoding, errors="surrogateescape")
            else:
                raw = None
                mode = None
                content = None
        except (OSError, UnicodeError):
            return

        self._stack.append(
            Snapshot(path=path, content=content, tool=tool, label=label,
                     raw_bytes=raw, mode=mode)
        )
        if len(self._stack) > MAX_SNAPSHOTS:
            self._stack.pop(0)

    def undo(self) -> tuple[bool, str]:
        """Revert the most recent change, preserving the original bytes/mode."""
        if not self._stack:
            return False, "Nothing to undo."

        snapshot = self._stack.pop()
        name = snapshot.label or snapshot.path.name

        try:
            if snapshot.existed:
                snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                data = snapshot.raw_bytes
                if data is None:
                    data = (snapshot.content or "").encode("utf-8", "surrogateescape")
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{snapshot.path.name}.",
                    suffix=".undo.tmp",
                    dir=str(snapshot.path.parent),
                )
                try:
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                        fh.flush()
                        os.fsync(fh.fileno())
                    if snapshot.mode is not None:
                        try:
                            os.chmod(temp_name, snapshot.mode)
                        except OSError:
                            pass
                    os.replace(temp_name, snapshot.path)
                except Exception:
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass
                    raise
                return True, f"Reverted {name} to its state before {snapshot.tool}."
            if snapshot.path.exists():
                snapshot.path.unlink()
                return True, f"Deleted {name}, which {snapshot.tool} had created."
            return True, f"{name} was already gone."
        except OSError as exc:
            return False, f"Could not undo {name}: {exc}"

    def mark(self) -> int:
        return len(self._stack)

    def changes_since(self, mark: int) -> list[Snapshot]:
        seen: dict[str, Snapshot] = {}
        for snapshot in self._stack[mark:]:
            key = str(snapshot.path)
            if key not in seen:
                seen[key] = snapshot
        return list(seen.values())

    def revert_since(self, mark: int) -> tuple[int, list[str]]:
        problems: list[str] = []
        reverted = 0
        while len(self._stack) > mark:
            ok, message = self.undo()
            if ok:
                reverted += 1
            else:
                problems.append(message)
        return reverted, problems

    def peek(self) -> Snapshot | None:
        return self._stack[-1] if self._stack else None

    def history(self, limit: int = 10) -> list[Snapshot]:
        return list(reversed(self._stack[-limit:]))

    def clear(self) -> None:
        self._stack.clear()
