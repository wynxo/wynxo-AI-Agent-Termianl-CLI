"""Undo for file changes.

Every write is snapshotted before it happens, so a bad edit is one ``/undo``
away rather than a question of what the file used to look like. Snapshots are
in memory and per-session: this is an undo button, not a backup system, and
pretending otherwise would invite people to rely on it for the wrong thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_SNAPSHOTS = 100
MAX_FILE_BYTES = 2_000_000


@dataclass
class Snapshot:
    path: Path
    content: str | None
    """None means the file did not exist, so undoing means deleting it."""
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
        """Record what ``path`` looks like right now, before it is changed."""
        try:
            if path.exists():
                if path.stat().st_size > MAX_FILE_BYTES:
                    return   # too big to hold; better no undo than an OOM
                content = path.read_text(encoding="utf-8", errors="surrogateescape")
            else:
                content = None
        except OSError:
            return

        self._stack.append(Snapshot(path=path, content=content, tool=tool, label=label))
        if len(self._stack) > MAX_SNAPSHOTS:
            self._stack.pop(0)

    def undo(self) -> tuple[bool, str]:
        """Revert the most recent change. Returns (did_something, message)."""
        if not self._stack:
            return False, "Nothing to undo."

        snapshot = self._stack.pop()
        name = snapshot.label or snapshot.path.name

        try:
            if snapshot.existed:
                snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                with snapshot.path.open("w", encoding="utf-8", newline="",
                                        errors="surrogateescape") as fh:
                    fh.write(snapshot.content or "")
                return True, f"Reverted {name} to its state before {snapshot.tool}."
            if snapshot.path.exists():
                snapshot.path.unlink()
                return True, f"Deleted {name}, which {snapshot.tool} had created."
            return True, f"{name} was already gone."
        except OSError as exc:
            return False, f"Could not undo {name}: {exc}"

    def peek(self) -> Snapshot | None:
        return self._stack[-1] if self._stack else None

    def history(self, limit: int = 10) -> list[Snapshot]:
        return list(reversed(self._stack[-limit:]))

    def clear(self) -> None:
        self._stack.clear()
