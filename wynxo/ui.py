"""Terminal rendering.

Windows note: rich handles the VT sequences, but the box-drawing and arrow
glyphs used elsewhere in TUIs do not survive the default Windows console
font. Everything here sticks to ASCII plus a small set of glyphs that are
checked against the active encoding at startup.
"""

from __future__ import annotations

import sys
from typing import Iterable

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

ACCENT = "bright_cyan"
MUTED = "grey58"
GOOD = "green"
BAD = "red"
WARN = "yellow"


def _supports_unicode() -> bool:
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" in encoding:
        return True
    try:
        "•─".encode(encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class Glyphs:
    def __init__(self, unicode_ok: bool):
        if unicode_ok:
            self.bullet, self.arrow, self.tick = "•", "→", "✓"
            self.cross, self.gear, self.dot = "✗", "●", "·"
        else:
            self.bullet, self.arrow, self.tick = "*", "->", "+"
            self.cross, self.gear, self.dot = "x", "o", "."


class UI:
    def __init__(self, theme: str = "dark", show_thinking: bool = True):
        self.console = Console(
            highlight=False,
            soft_wrap=False,
            # legacy_windows=False forces modern VT output on Windows Terminal.
            legacy_windows=False if sys.platform == "win32" else None,
        )
        self.g = Glyphs(_supports_unicode())
        self.show_thinking = show_thinking
        self.code_theme = "monokai" if theme == "dark" else "friendly"

    # -- chrome ------------------------------------------------------------

    def banner(self, model: str, endpoint: str, effort: str, workspace: str) -> None:
        title = Text()
        title.append("wynxo", style=f"bold {ACCENT}")
        title.append("  a local coding agent", style=MUTED)

        table = Table.grid(padding=(0, 2))
        table.add_column(style=MUTED)
        table.add_column()
        table.add_row("model", Text(model, style="bold"))
        table.add_row("server", Text(endpoint))
        table.add_row("effort", Text(effort, style=f"bold {ACCENT}"))
        table.add_row("project", Text(workspace))

        self.console.print()
        self.console.print(Panel(Group(title, "", table), border_style=ACCENT, padding=(1, 2)))
        self.console.print(
            Text("  /help for commands, /effort to change gear, Ctrl-C to interrupt", style=MUTED)
        )
        self.console.print()

    def rule(self, label: str = "") -> None:
        self.console.print(Rule(label, style=MUTED))

    # -- messages ----------------------------------------------------------

    def info(self, message: str) -> None:
        self.console.print(Text(f"  {message}", style=MUTED))

    def warn(self, message: str) -> None:
        self.console.print(Text(f"  ! {message}", style=WARN))

    def error(self, message: str) -> None:
        self.console.print()
        self.console.print(Panel(Text(message), title="error", border_style=BAD, padding=(0, 1)))

    def success(self, message: str) -> None:
        self.console.print(Text(f"  {self.g.tick} {message}", style=GOOD))

    def assistant_markdown(self, text: str) -> None:
        if not text.strip():
            return
        self.console.print()
        self.console.print(Markdown(text, code_theme=self.code_theme))
        self.console.print()

    def thinking(self, text: str) -> None:
        if not (self.show_thinking and text.strip()):
            return
        preview = text.strip()
        if len(preview) > 1600:
            preview = preview[:1600] + "\n[...]"
        self.console.print(
            Panel(
                Text(preview, style="italic " + MUTED),
                title="thinking",
                title_align="left",
                border_style=MUTED,
                padding=(0, 1),
            )
        )

    # -- tools -------------------------------------------------------------

    def tool_start(self, name: str, summary: str) -> None:
        line = Text()
        line.append(f"  {self.g.gear} ", style=ACCENT)
        line.append(name, style=f"bold {ACCENT}")
        if summary:
            line.append(f"  {summary[:100]}", style=MUTED)
        self.console.print(line)

    def tool_result(self, name: str, ok: bool, display: str, output: str) -> None:
        if display.startswith(("--- ", "diff --git")) or "\n+++ " in display[:200]:
            self.diff(display)
            return
        text = (display or output).strip()
        if not text:
            return
        first = text.splitlines()[0]
        extra = len(text.splitlines()) - 1
        line = Text("    ")
        line.append(self.g.tick if ok else self.g.cross, style=GOOD if ok else BAD)
        line.append(" ")
        line.append(first[:150], style="" if ok else BAD)
        if extra > 0:
            line.append(f"  (+{extra} lines)", style=MUTED)
        self.console.print(line)

    def diff(self, text: str) -> None:
        if not text.strip():
            return
        body = Text()
        for line in text.splitlines()[:120]:
            if line.startswith("+++") or line.startswith("---"):
                body.append(line + "\n", style=MUTED)
            elif line.startswith("+"):
                body.append(line + "\n", style=GOOD)
            elif line.startswith("-"):
                body.append(line + "\n", style=BAD)
            elif line.startswith("@@"):
                body.append(line + "\n", style=ACCENT)
            else:
                body.append(line + "\n", style=MUTED)
        self.console.print(Panel(body, border_style=MUTED, padding=(0, 1)))

    def todos(self, rendered: str) -> None:
        if not rendered.strip():
            return
        body = Text()
        for line in rendered.splitlines():
            if line.startswith("[x]"):
                body.append(line + "\n", style=f"{MUTED} strike")
            elif line.startswith("[>]"):
                body.append(line + "\n", style=f"bold {ACCENT}")
            else:
                body.append(line + "\n")
        self.console.print(Panel(body, title="plan", title_align="left", border_style=MUTED, padding=(0, 1)))

    def code(self, text: str, language: str = "text") -> None:
        self.console.print(Syntax(text, language, theme=self.code_theme, word_wrap=True))

    # -- transient ---------------------------------------------------------

    def status(self, message: str) -> Status:
        return self.console.status(Text(message, style=MUTED), spinner="dots")

    def stream_chunk(self, text: str) -> None:
        self.console.print(text, end="", markup=False, highlight=False)

    def table(self, columns: Iterable[str], rows: Iterable[Iterable[str]], title: str = "") -> None:
        table = Table(title=title or None, border_style=MUTED, title_style=f"bold {ACCENT}")
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        self.console.print(table)

    def stats(self, usage, elapsed: float, effort: str, context_pct: float) -> None:
        bits = [
            f"{effort}",
            f"{usage.completion_tokens} tok",
            f"{usage.tokens_per_second():.0f} tok/s" if usage.tokens_per_second() else "",
            f"{elapsed:.1f}s",
            f"ctx {context_pct:.0f}%",
        ]
        self.console.print(
            Text("  " + f" {self.g.dot} ".join(b for b in bits if b), style=MUTED)
        )
