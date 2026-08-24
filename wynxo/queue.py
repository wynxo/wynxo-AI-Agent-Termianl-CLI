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
            line, self.draft = self.draft.strip(), ""
            if line:
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
        """Drop everything. Returns what was dropped, for reporting."""
        count = len(self.items)
        self.items.clear()
        self.draft = ""
        return f"{count} queued message(s) dropped" if count else ""

    def preview(self, width: int = 40, ellipsis: str = "\u2026") -> str:
        """What to show in the status bar while a turn runs."""
        if self.draft:
            text = self.draft
            if len(text) > width:
                text = ellipsis + text[-(width - len(ellipsis)):]
            return text
        if self.items:
            return f"{len(self.items)} queued"
        return ""

    def summary(self) -> list[str]:
        return list(self.items)
