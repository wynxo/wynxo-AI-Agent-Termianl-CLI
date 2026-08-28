"""Undo for file changes.

Snapshots are deliberately raw bytes, not decoded text. Undo must restore the
file that existed before the agent touched it, including encoding, BOM,
line endings, binary data and executable permissions.
"""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_SNAPSHOTS = 100
MAX_FILE_BYTES = 2_000_000


@dataclass
class Snapshot:
    path: Path
    content: bytes | None
    mode: int | None
    tool: str
    when: float = field(default_factory=time.time)
    label: str = ""

    @property
    def existed(self) -> bool:
        return self.content is not None


class Checkpoints:
    def __init__(self) -> None:
        self._stack: list[Snapshot] = []

    def __len__(self) -> int:
        return len(self._stack)

    def capture(self, path: Path, tool: str, label: str = "") -> None:
        """Record the exact bytes and POSIX permission bits before a change."""
        try:
            if path.exists():
                if path.is_dir() or path.stat().st_size > MAX_FILE_BYTES:
                    return
                content = path.read_bytes()
                mode = stat.S_IMODE(path.stat().st_mode)
            else:
                content = None
                mode = None
        except OSError:
            return

        self._stack.append(
            Snapshot(path=path, content=content, mode=mode, tool=tool, label=label)
        )
        if len(self._stack) > MAX_SNAPSHOTS:
            self._stack.pop(0)

    def undo(self) -> tuple[bool, str]:
        """Revert the most recent change exactly, or report why it failed."""
        if not self._stack:
            return False, "Nothing to undo."

        snapshot = self._stack.pop()
        name = snapshot.label or snapshot.path.name

        try:
            if snapshot.existed:
                snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = snapshot.path.with_name(
                    f".{snapshot.path.name}.wynxo-undo-{os.getpid()}.tmp"
                )
                try:
                    temporary.write_bytes(snapshot.content or b"")
                    if snapshot.mode is not None:
                        try:
                            temporary.chmod(snapshot.mode)
                        except OSError:
                            pass
                    os.replace(temporary, snapshot.path)
                finally:
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
                return True, f"Reverted {name} to its state before {snapshot.tool}."
            if snapshot.path.exists():
                if snapshot.path.is_dir():
                    return False, f"Could not undo {name}: it became a directory."
                snapshot.path.unlink()
                return True, f"Deleted {name}, which {snapshot.tool} had created."
            return True, f"{name} was already gone."
        except OSError as exc:
            return False, f"Could not undo {name}: {exc}"

    def mark(self) -> int:
        return len(self._stack)

    def changes_since(self, mark: int) -> list[Snapshot]:
        seen: dict[str, Snapshot] = {}
        for snapshot in self._stack[max(0, mark):]:
            key = str(snapshot.path)
            if key not in seen:
                seen[key] = snapshot
        return list(seen.values())

    def revert_since(self, mark: int) -> tuple[int, list[str]]:
        problems: list[str] = []
        reverted = 0
        mark = max(0, min(mark, len(self._stack)))
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
        if limit <= 0:
            return []
        return list(reversed(self._stack[-limit:]))

    def clear(self) -> None:
        self._stack.clear()
