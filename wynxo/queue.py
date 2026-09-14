"""Type-ahead: what you write while the agent is still working.

A local 30B can take a minute to answer, and having nowhere to put the next
thought during that minute is the difference between a tool you converse with
and one you wait on. Keystrokes arriving mid-turn are collected here and run
in order when the turn ends.

The collector shares the terminal with the key watcher, so it lives in the
same raw-mode reader rather than opening a second one -- two things reading
stdin is how keystrokes go missing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rich.cells import cell_len


def _tail_to_width(text: str, width: int, ellipsis: str) -> str:
    """Keep the newest end of ``text`` inside a terminal-cell budget.

    ``len`` is the wrong unit for terminal UI: CJK characters and many emoji
    occupy two cells, while combining marks occupy none. The queue preview is
    painted into the live status row, so exceeding the real cell width makes
    the terminal wrap and leaves visual debris behind.
    """
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text

    marker = ellipsis if cell_len(ellipsis) <= width else ""
    suffix = ""
    for char in reversed(text):
        candidate = char + suffix
        if cell_len(marker + candidate) > width:
            break
        suffix = candidate
    return marker + suffix


@dataclass
class Pending:
    """Messages typed while a turn was running."""

    items: deque[str] = field(default_factory=deque)
    draft: str = ""
    """The line being typed right now, not yet submitted."""

    # -- editing -----------------------------------------------------------

    def key(self, char: str) -> str | None:
        """Feed one keystroke. Returns a completed line, or None.

        Only printable characters, backspace and enter are handled: anything
        else belongs to the key watcher's bindings and is left alone.
        """
        if char in ("\r", "\n"):
            # Whitespace-only input is still empty, but do not silently alter
            # a real line. A queued code fragment or shell command can
            # legitimately care about its surrounding whitespace.
            line, self.draft = self.draft, ""
            if line.strip():
                self.items.append(line)
                return line
            return None
        if char in ("\x7f", "\b"):
            self.draft = self.draft[:-1]
            return None
        if char == "\x15":            # ctrl-u, clear the line
            self.draft = ""
            return None
        if char.isprintable():
            self.draft += char
        return None

    # -- state -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items) or bool(self.draft)

    def take(self) -> str | None:
        """The next queued message, oldest first."""
        return self.items.popleft() if self.items else None

    def clear(self) -> str:
        """Drop everything. Returns an exact human-readable description."""
        count = len(self.items)
        had_draft = bool(self.draft)
        self.items.clear()
        self.draft = ""

        if count and had_draft:
            noun = "message" if count == 1 else "messages"
            return f"{count} queued {noun} and draft dropped"
        if count:
            noun = "message" if count == 1 else "messages"
            return f"{count} queued {noun} dropped"
        if had_draft:
            return "draft dropped"
        return ""

    def preview(self, width: int = 40, ellipsis: str = "\u2026") -> str:
        """What to show in the status bar while a turn runs."""
        if self.draft:
            return _tail_to_width(self.draft, width, ellipsis)
        if self.items:
            return f"{len(self.items)} queued"
        return ""

    def summary(self) -> list[str]:
        return list(self.items)
