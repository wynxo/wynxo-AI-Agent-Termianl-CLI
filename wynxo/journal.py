"""An append-only record of a session, for reading after something goes wrong.

Every prompt, reply, tool call, tool result and error is written as one JSON
object per line, with a timestamp. That format matters: a crash mid-write
costs you the last line and nothing else, and the file stays greppable and
tailable while the session is still running.

It is a debugging aid, not telemetry. It never leaves the machine, and
``/log off`` stops it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import data_dir

MAX_FIELD = 4_000
"""Per-field cap. A 200KB file dumped into a tool result would otherwise make
the log useless to read and slow to write."""

KEEP_SESSIONS = 20


@dataclass
class Journal:
    session_id: str
    path: Path | None = None
    enabled: bool = True
    _failed: bool = field(default=False, repr=False)

    @classmethod
    def open(cls, session_id: str, enabled: bool = True) -> "Journal":
        journal = cls(session_id=session_id, enabled=enabled)
        if not enabled:
            return journal
        try:
            directory = data_dir() / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            journal.path = directory / f"{stamp}-{session_id}.jsonl"
            journal.write("session", cwd=str(Path.cwd()), pid=os.getpid())
            prune(directory)
        except OSError:
            # Never let logging stop the agent working.
            journal.enabled = False
            journal._failed = True
        return journal

    def write(self, kind: str, **fields: Any) -> None:
        if not self.enabled or self.path is None:
            return
        record = {"t": round(time.time(), 3), "kind": kind}
        for key, value in fields.items():
            record[key] = _trim(value)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            self.enabled = False
            self._failed = True

    # -- the events worth recording ---------------------------------------

    def user(self, text: str) -> None:
        self.write("user", text=text)

    def assistant(self, text: str, tokens: int = 0, seconds: float = 0.0) -> None:
        self.write("assistant", text=text, tokens=tokens, seconds=round(seconds, 2))

    def thinking(self, text: str) -> None:
        self.write("thinking", text=text)

    def tool(self, name: str, arguments: dict) -> None:
        self.write("tool", name=name, args=arguments)

    def tool_result(self, name: str, ok: bool, output: str) -> None:
        self.write("tool_result", name=name, ok=ok, output=output)

    def stage(self, name: str, detail: str = "") -> None:
        self.write("stage", name=name, detail=detail)

    def error(self, message: str) -> None:
        self.write("error", message=message)

    def note(self, message: str, **fields: Any) -> None:
        self.write("note", message=message, **fields)

    # -- reading back ------------------------------------------------------

    def tail(self, count: int = 40) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        out = []
        for line in lines[-count:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def size(self) -> int:
        try:
            return self.path.stat().st_size if self.path else 0
        except OSError:
            return 0


def _trim(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_FIELD:
        return value[:MAX_FIELD] + f"… [{len(value) - MAX_FIELD} more characters]"
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_trim(v) for v in value[:50]]
    return value


def prune(directory: Path, keep: int = KEEP_SESSIONS) -> None:
    """Keep the most recent sessions; a log directory that grows without
    bound is a bug report waiting to happen."""
    try:
        logs = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for old in logs[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def recent(limit: int = 10) -> list[Path]:
    directory = data_dir() / "logs"
    if not directory.is_dir():
        return []
    try:
        return sorted(directory.glob("*.jsonl"),
                      key=lambda p: -p.stat().st_mtime)[:limit]
    except OSError:
        return []
