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


def _read(path: Path) -> str:
    """The file exactly as capture and undo see it: no newline translation,
    and bytes that are not UTF-8 kept as surrogates so round-trips are
    exact."""
    with path.open("r", encoding="utf-8", errors="surrogateescape",
                   newline="") as handle:
        return handle.read()


@dataclass
class Snapshot:
    path: Path
    content: str | None
    """None means the file did not exist, so undoing means deleting it."""
    tool: str
    when: float = field(default_factory=time.time)
    label: str = ""
    expected: str | None = None
    """What the file held when the agent finished its edit, filled in after
    the write succeeds. Undo refuses when the current content differs from
    this: the file drifted since the agent touched it, which means someone
    else -- almost always the user -- changed it, and undoing would destroy
    that work."""

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
                # newline="" so nothing is translated on the way in. The
                # default translates CRLF to LF, and undo writes back
                # untranslated -- so undoing an edit to a CRLF file
                # converted the whole file to LF, which inside a git repo
                # is every line of it showing as changed.
                #
                # surrogateescape is what makes the rest of it exact: a byte
                # that is not valid UTF-8 becomes a lone surrogate and comes
                # back as the same byte, so a cp1252 or UTF-16 file is
                # restored to what it was rather than to a re-encoding of it.
                with path.open("r", encoding="utf-8",
                               errors="surrogateescape", newline="") as handle:
                    content = handle.read()
            else:
                content = None
        except OSError:
            return

        self._stack.append(Snapshot(path=path, content=content, tool=tool, label=label))
        if len(self._stack) > MAX_SNAPSHOTS:
            self._stack.pop(0)

    def mark_expected(self, path: Path) -> None:
        """Record what ``path`` holds right now as the state the agent left.

        Called after a successful edit. Undo compares the file against this
        before reverting: if it differs, the file changed after the agent
        was done with it, and undoing would overwrite someone else's work.
        """
        for snapshot in reversed(self._stack):
            if snapshot.path == path:
                try:
                    snapshot.expected = _read(snapshot.path)
                except OSError:
                    snapshot.expected = None
                return

    def undo(self) -> tuple[bool, str]:
        """Revert the most recent change, refusing if the file drifted since it."""
        if not self._stack:
            return False, "Nothing to undo."

        snapshot = self._stack.pop()
        name = snapshot.label or snapshot.path.name

        try:
            # The snapshot knows what the agent left behind. Anything else
            # now on disk means someone changed the file after the agent
            # finished -- almost always the user -- and undoing over it
            # would destroy that work silently. Refuse.
            if snapshot.expected is not None and snapshot.path.exists():
                if _read(snapshot.path) != snapshot.expected:
                    return False, (
                        f"{name} changed after it was edited -- not undoing "
                        "over the newer change."
                    )
            # A checkpoint is safe only if the current file still represents
            # the agent's last edit. If the user changed it afterwards, never
            # overwrite that work silently.
            if snapshot.existed and snapshot.path.exists():
                current = _read(snapshot.path)
                if current == snapshot.content:
                    return False, f"{name} is already at its checkpoint state."
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

    def mark(self) -> int:
        """A position to measure a turn's changes from."""
        return len(self._stack)

    def changes_since(self, mark: int) -> list[Snapshot]:
        """The earliest snapshot per file taken after ``mark``.

        Earliest, not latest: three edits to one file during a turn should
        read as one change from how it started, not three overlapping ones.
        """
        seen: dict[str, Snapshot] = {}
        for snapshot in self._stack[mark:]:
            key = str(snapshot.path)
            if key not in seen:
                seen[key] = snapshot
        return list(seen.values())

    def revert_since(self, mark: int) -> tuple[int, list[str]]:
        """Undo everything after ``mark``. Returns (reverted, problems).

        Newest first, so a file written and then edited again lands back on
        what it held before the turn rather than halfway through it.
        """
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
