"""Undo for file changes.

Every write is snapshotted before it happens, so a bad edit is one ``/undo``
away rather than a question of what the file used to look like. Snapshots are
bounded and per-session, and persist to disk with the session so the undo
button still works after a restart -- this is an undo button, not a backup
system, and the disk copy is deliberately capped well below what a real
backup would need.
"""

from __future__ import annotations

import base64
import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import data_dir

MAX_SNAPSHOTS = 100
MAX_FILE_BYTES = 2_000_000
MAX_DISK_BYTES = 20_000_000
"""The persisted undo stack is trimmed to stay under this. A real backup
system would need far more; an undo button needs the last few edits."""


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
    def __init__(self, session_id: str | None = None) -> None:
        self._stack: list[Snapshot] = []
        self._session_id = session_id
        if session_id:
            self._load()

    # -- persistence --------------------------------------------------------

    def _store(self) -> Path | None:
        """Where this session's undo stack lives, if one is configured."""
        if not self._session_id:
            return None
        directory = data_dir() / "checkpoints"
        return directory / f"{self._session_id}.json"

    @staticmethod
    def _encode(content: str | None) -> str | None:
        """Surrogates (from surrogateescape) are not valid JSON, so the
        content is carried as base64 of its UTF-8+surrogateescape bytes."""
        if content is None:
            return None
        raw = content.encode("utf-8", "surrogateescape")
        return base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _decode(encoded: str | None) -> str | None:
        if encoded is None:
            return None
        raw = base64.b64decode(encoded.encode("ascii"))
        return raw.decode("utf-8", "surrogateescape")

    def _save(self) -> None:
        """Write the stack to disk, trimming old snapshots under the budget."""
        path = self._store()
        if path is None:
            return
        # Trim from the oldest until the serialized whole fits -- a giant
        # binary file snapped mid-edit should not wedge the store forever.
        stack = list(self._stack)

        def serialise(snapshots: list[Snapshot]) -> str:
            return json.dumps([
                {"path": str(s.path),
                 "content": self._encode(s.content),
                 "tool": s.tool,
                 "when": s.when,
                 "label": s.label,
                 "expected": self._encode(s.expected)}
                for s in snapshots
            ])

        payload = "[]"
        while stack:
            try:
                payload = serialise(stack)
            except (TypeError, ValueError):
                stack.pop(0)
                continue
            if len(payload) <= MAX_DISK_BYTES:
                break
            stack.pop(0)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        except OSError:
            pass

    def _load(self) -> None:
        """Restore the stack from disk, if a previous session left one."""
        path = self._store()
        if path is None or not path.exists():
            return
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return
        if not isinstance(rows, list):
            return
        self._stack = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            content = self._decode(row.get("content"))
            expected = self._decode(row.get("expected"))
            self._stack.append(Snapshot(
                path=Path(str(row.get("path", ""))),
                content=content,
                tool=str(row.get("tool", "?")),
                when=float(row.get("when", 0.0)),
                label=str(row.get("label", "")),
                expected=expected,
            ))
        self._stack = self._stack[-MAX_SNAPSHOTS:]

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
        self._save()

    def mark_expected(self, path: Path) -> None:
        """Record what ``path`` holds right now as the state the agent left.

        Called after a successful edit. Undo compares the file against this
        before reverting: if it differs, the file changed after the agent
        was done with it, and undoing would overwrite someone else's work.

        Persisted: the whole point of ``expected`` is to survive a restart --
        without it, an undo after resuming the session would silently
        overwrite whatever the user changed in between.
        """
        for snapshot in reversed(self._stack):
            if snapshot.path == path:
                try:
                    snapshot.expected = _read(snapshot.path)
                except OSError:
                    snapshot.expected = None
                self._save()
                return

    def restore(self, snapshot: "Snapshot") -> tuple[bool, str]:
        """Put one file back, refusing if it drifted since the agent left it.

        The check and the write live together here because they were apart
        once and drifted apart with them: undo() had the check, and the
        review flow's step-through wrote the snapshot straight back with no
        check at all, so answering "revert" to a file the user had saved in
        their editor since destroyed that save and reported success.

        Callers own the stack; this owns the file.
        """
        name = snapshot.label or snapshot.path.name
        try:
            # The snapshot knows what the agent left behind. Anything else
            # now on disk means someone changed the file afterwards --
            # almost always the user -- and putting the old one back over
            # it would destroy that work silently.
            if snapshot.expected is not None and snapshot.path.exists():
                if _read(snapshot.path) != snapshot.expected:
                    return False, (
                        f"{name} changed after it was edited -- not undoing "
                        "over the newer change."
                    )
        except OSError as exc:
            return False, f"Could not undo {name}: {exc}"

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

    def undo(self) -> tuple[bool, str]:
        """Revert the most recent change, refusing if the file drifted since it.

        A refusal must not consume the snapshot: if the file drifted (someone
        changed it after the agent was done), undoing over it is refused, but
        the record has to survive so the undo is still possible once the
        conflict is resolved. Popping first meant the first refusal silently
        destroyed the undo history.
        """
        if not self._stack:
            return False, "Nothing to undo."

        snapshot = self._stack[-1]
        name = snapshot.label or snapshot.path.name

        try:
            # A checkpoint is safe only if the current file still represents
            # the agent's last edit. Nothing to do if it is already there.
            if snapshot.existed and snapshot.path.exists():
                if _read(snapshot.path) == snapshot.content:
                    return False, f"{name} is already at its checkpoint state."
        except OSError as exc:
            return False, f"Could not undo {name}: {exc}"

        ok, message = self.restore(snapshot)
        if not ok:
            # A refusal must not consume the snapshot: the undo has to
            # still be possible once the conflict is resolved.
            return ok, message

        # Only now does the record leave the stack.
        self._stack.pop()
        self._save()
        return ok, message

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
                continue
            # undo() refuses without consuming the snapshot, so a drifted
            # file would keep the loop spinning forever. A bulk revert is
            # not a single /undo: skip the blocked change, record why, and
            # keep going with the older ones.
            problems.append(message)
            if len(self._stack) > mark:
                self._stack.pop()
                self._save()
        return reverted, problems

    def peek(self) -> Snapshot | None:
        return self._stack[-1] if self._stack else None

    def history(self, limit: int = 10) -> list[Snapshot]:
        return list(reversed(self._stack[-limit:]))

    def clear(self) -> None:
        self._stack.clear()
        with contextlib.suppress(OSError):
            if path := self._store():
                path.unlink(missing_ok=True)
