"""Long-term memory, as two markdown files the agent maintains itself.

    <project>/.wynxo/memory.md   what it learned about this codebase
    <config>/user.md             what it learned about you, everywhere

The design constraint that matters is speed. An agent that does a retrieval
pass before every turn feels slow on local hardware, and embeddings would
mean a vector store, a model to build them, and a background index -- all of
which cost more than they return at this size.

So memory is boring on purpose: two capped markdown files, read once at
startup, inlined into the system prompt. Reading them costs a couple of
milliseconds and a few hundred tokens, and the model can edit them with the
same file tools it uses for everything else. No index, no staleness, no
lag.

The cap is what keeps that true. Past the limit the oldest entries are
dropped rather than allowed to grow into the context budget.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import config_dir

PROJECT_DIR = ".wynxo"
PROJECT_FILE = "memory.md"
USER_FILE = "user.md"

MAX_PROJECT_CHARS = 8_000
MAX_USER_CHARS = 4_000
MAX_ENTRY_CHARS = 500

PROJECT_HEADER = """# Project memory

Facts about this codebase worth carrying between sessions: conventions,
gotchas, where things live, decisions and why. Keep entries short and true.
Delete anything that stops being accurate.
"""

USER_HEADER = """# About the user

Preferences and working habits that hold across every project: tools they
use, how they like things explained, what to avoid. Keep it short.
"""


@dataclass
class MemoryFile:
    path: Path
    header: str
    limit: int

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def body(self) -> str:
        """The content without the boilerplate header."""
        text = self.read()
        if not text.strip():
            return ""
        lines = [
            line for line in text.splitlines()
            # Drop the header prose so it is not re-fed to the model as fact.
            if not line.startswith("#") or line.startswith("##")
        ]
        stripped = "\n".join(lines).strip()
        for phrase in ("Facts about this codebase", "Preferences and working habits",
                       "Keep entries short", "Keep it short", "Delete anything"):
            stripped = "\n".join(
                l for l in stripped.splitlines() if phrase not in l)
        return stripped.strip()

    def entries(self) -> list[str]:
        return [line.strip() for line in self.body().splitlines()
                if line.strip().startswith(("-", "*"))]

    def append(self, note: str) -> tuple[bool, str]:
        """Add one entry. Returns (added, message)."""
        note = " ".join(note.split())[:MAX_ENTRY_CHARS].rstrip(".")
        if not note:
            return False, "Empty note."

        existing = self.entries()
        if _already_known(note, existing):
            return False, "Already remembered something equivalent."

        text = self.read() or self.header
        if not text.endswith("\n"):
            text += "\n"
        text += f"- {note}\n"
        text = self._trim(text)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(text, encoding="utf-8")
        except OSError as exc:
            return False, f"Could not write {self.path}: {exc}"
        return True, note

    def forget(self, pattern: str) -> tuple[int, str]:
        """Drop entries matching ``pattern`` (case-insensitive substring)."""
        text = self.read()
        if not text:
            return 0, "Nothing remembered yet."
        needle = pattern.strip().lower()
        kept, dropped = [], 0
        for line in text.splitlines():
            if line.strip().startswith(("-", "*")) and needle in line.lower():
                dropped += 1
                continue
            kept.append(line)
        if not dropped:
            return 0, f"Nothing matching {pattern!r}."
        try:
            self.path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
        except OSError as exc:
            return 0, f"Could not write {self.path}: {exc}"
        return dropped, f"Forgot {dropped} entr{'y' if dropped == 1 else 'ies'}."

    def _trim(self, text: str) -> str:
        """Keep the file under its cap by dropping the oldest entries.

        This is what stops memory quietly becoming a context tax.
        """
        if len(text) <= self.limit:
            return text
        lines = text.splitlines()
        head = [l for l in lines if not l.strip().startswith(("-", "*"))]
        items = [l for l in lines if l.strip().startswith(("-", "*"))]
        while items and len("\n".join(head + items)) > self.limit:
            items.pop(0)
        return "\n".join(head + items).rstrip() + "\n"


# Function words carry no meaning here and only dilute the overlap: without
# stripping them, "this project uses pytest" and "the project uses pytest and
# is run with pytest" score as different notes.
_STOPWORDS = frozenset("""
a an the this that these those it its is are was were be been being am
and or but if then so as of in on at to for from with by about into over
you your we our i my me they them their he she his her
do does did doing done have has had having can could should would will
not no nor only just also very really quite than there here when while
""".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _already_known(note: str, existing: list[str]) -> bool:
    """Cheap near-duplicate check: no embeddings, just content-word overlap.

    Numbers are treated as distinguishing rather than as ordinary words.
    "port 8080" and "port 9090" share every other token, and collapsing them
    would silently lose a fact -- which is worse than keeping a near-duplicate.
    """
    words = _content_words(note)
    if not words:
        return False
    note_numbers = _numbers(note)
    for line in existing:
        other = _content_words(line)
        if not other:
            continue
        if _numbers(line) != note_numbers:
            continue
        overlap = len(words & other) / max(len(words), len(other))
        if overlap >= 0.8:
            return True
    return False


class Memory:
    """Both memory files, and how they reach the model."""

    def __init__(self, workspace: Path, user_dir: Path | None = None):
        self.workspace = workspace
        self.project = MemoryFile(
            path=workspace / PROJECT_DIR / PROJECT_FILE,
            header=PROJECT_HEADER, limit=MAX_PROJECT_CHARS)
        self.user = MemoryFile(
            path=(user_dir or config_dir()) / USER_FILE,
            header=USER_HEADER, limit=MAX_USER_CHARS)

    def file_for(self, scope: str) -> MemoryFile:
        return self.user if scope.strip().lower() in ("user", "me", "global") \
            else self.project

    def remember(self, note: str, scope: str = "project") -> tuple[bool, str]:
        return self.file_for(scope).append(note)

    def forget(self, pattern: str, scope: str = "project") -> tuple[int, str]:
        return self.file_for(scope).forget(pattern)

    def counts(self) -> tuple[int, int]:
        return len(self.project.entries()), len(self.user.entries())

    def prompt_section(self) -> str:
        """What gets inlined into the system prompt. Empty when there is
        nothing worth saying, so a fresh project pays nothing."""
        parts = []
        if user := self.user.body():
            parts.append(f"### About the user\n\n{user}")
        if project := self.project.body():
            parts.append(f"### About this project\n\n{project}")
        if not parts:
            return ""
        return (
            "\n## Memory\n\n"
            "Things you learned earlier and wrote down. Treat them as true "
            "unless what you see now contradicts them -- in which case fix the "
            "memory with the `remember` tool.\n\n"
            + "\n\n".join(parts) + "\n"
        )

    def touched_recently(self, seconds: float = 5.0) -> bool:
        """Whether either file changed just now, so the caller can reload."""
        now = time.time()
        for memory_file in (self.project, self.user):
            try:
                if memory_file.exists() and now - memory_file.path.stat().st_mtime < seconds:
                    return True
            except OSError:
                continue
        return False
