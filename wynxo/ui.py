"""Terminal rendering.

Windows note: rich handles the VT sequences, but the box-drawing and arrow
glyphs used elsewhere in TUIs do not survive the default Windows console
font. Everything here sticks to ASCII plus a small set of glyphs that are
checked against the active encoding at startup.
"""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Iterable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .platforms import is_narrow, terminal_width

ACCENT = "bright_cyan"
BAR_STYLE = "on grey23"
"""The pinned bar's background. A filled strip is what separates a status bar
from just another line that scrolled past."""
BAR_ACCENT = "bright_cyan"
BAR_DIM = "grey62"
MIN_ACTIVITY_WIDTH = 16
"""Cells kept for the activity text before the stats start claiming space."""
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
        self.unicode = unicode_ok
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
        self.narrow = is_narrow()
        """Phone-width terminals get a stacked layout instead of tables."""
        self.width = terminal_width()

    # -- chrome ------------------------------------------------------------

    def banner(self, model: str, endpoint: str, effort: str, workspace: str) -> None:
        """A single line of identity, then a rule.

        Assembled by priority rather than truncated: on a narrow terminal the
        server address goes before the project path does, because the path is
        the one you actually need to see.
        """
        server = endpoint.split(" (")[0].replace("http://", "").replace("https://", "")
        parts = [model, effort, self.shorten_path(workspace), server]

        separator = f"  {self.g.dot}  "
        budget = self.width - 10
        shown: list[str] = []
        for part in parts:
            candidate = shown + [part]
            if len(separator.join(candidate)) > budget:
                continue
            shown = candidate

        head = Text()
        head.append("  wynxo", style=f"bold {ACCENT}")
        for i, part in enumerate(shown):
            head.append(separator, style=MUTED)
            head.append(part, style="bold" if i == 0 else "")

        self.console.print()
        self.console.print(head, overflow="ellipsis", no_wrap=True)
        self.console.print(Rule(style=MUTED, characters="\u2500"))

    def wake(self, pet, name: str) -> None:
        """A short wake-up before the header.

        Two thirds of a second, once per session, and skipped entirely when
        animation is off or nothing is watching. Anything longer is a thing
        you wait through rather than enjoy.
        """
        if not (pet and pet.enabled and pet.animate and self.console.is_terminal):
            return
        from .pet import Mood

        sequence = [(Mood.SAD, 0.09), (Mood.IDLE, 0.09), (Mood.THINKING, 0.09),
                    (Mood.READING, 0.09), (Mood.HAPPY, 0.16), (Mood.IDLE, 0.0)]
        self.console.print()
        with Live("", console=self.console, refresh_per_second=20,
                  transient=True) as live:
            for mood, pause in sequence:
                pet.react(mood)
                line = Text("  ")
                line.append(pet.face(advance=False), style=f"bold {pet.style()}")
                line.append(f"  {name}", style=MUTED)
                live.update(line)
                if pause:
                    time.sleep(pause)
        pet.rest()

    def shorten_path(self, path: str) -> str:
        """~/code/proj rather than /home/you/code/proj, and never the full
        thing when it would push everything else off the line."""
        import os

        home = os.path.expanduser("~")
        if path.startswith(home):
            path = "~" + path[len(home):]
        budget = max(18, self.width // 3)
        if len(path) <= budget:
            return path
        parts = path.replace("\\", "/").split("/")
        return ".../" + "/".join(parts[-2:]) if len(parts) > 2 else path[-budget:]

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
        limit = max(24, self.width - 6) if self.narrow else 10_000
        body = Text()
        for line in (l[:limit] for l in text.splitlines()[:120]):
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

    def code_line(self, line: str, language: str = "text") -> None:
        """One highlighted line, indented, with no block chrome.

        Syntax() would draw its own background band per line and stack into a
        ragged column, so the lexer is used directly instead.
        """
        try:
            rendered = Text.from_ansi(line) if "\x1b" in line else Text(line)
            if language not in ("text", ""):
                from rich.syntax import Syntax as _S

                lexer = _S.get_lexer(_S("", language), line)
                rendered = Text()
                for token, value in lexer.get_tokens(line):
                    if value.endswith("\n"):
                        value = value[:-1]
                    if value:
                        rendered.append(value, style=self._token_style(token))
        except Exception:
            rendered = Text(line)
        self.console.print(Text("  ") + rendered, highlight=False)

    def _token_style(self, token) -> str:
        from pygments.token import (Comment, Error, Keyword, Name, Number,
                                    Operator, String)

        for kind, style in ((Comment, MUTED), (String, "green"), (Number, "cyan"),
                            (Keyword, "magenta"), (Name.Function, "bright_blue"),
                            (Name.Class, "bright_blue"), (Operator, "bright_white"),
                            (Error, BAD)):
            if token in kind:
                return style
        return ""

    # -- transient ---------------------------------------------------------

    def status(self, message: str) -> Status:
        return self.console.status(Text(message, style=MUTED), spinner="dots")

    def stream_chunk(self, text: str) -> None:
        self.console.print(text, end="", markup=False, highlight=False)

    def table(self, columns: Iterable[str], rows: Iterable[Iterable[str]], title: str = "") -> None:
        columns = list(columns)
        rows = [[str(c) for c in row] for row in rows]

        if self.narrow:
            # Stack each row as a labelled block; a grid this wide would wrap
            # into unreadable confetti on a phone.
            if title:
                self.console.print(Text(title, style=f"bold {ACCENT}"))
            for row in rows:
                head, *rest = row
                self.console.print(Text(head, style="bold"))
                for label, value in zip(columns[1:], rest):
                    if value:
                        self.console.print(
                            Text(f"  {label}: ", style=MUTED) + Text(value))
            self.console.print()
            return

        table = Table(title=title or None, border_style=MUTED, title_style=f"bold {ACCENT}")
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
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


class CodeStreamer:
    """Renders streamed assistant text, highlighting code as it arrives.

    Each code line is syntax-highlighted and printed once, as soon as it is
    complete. An earlier version printed a dim preview and then rewound the
    cursor to replace it with a highlighted block, which crashed outright on
    rich 15 (no ``Control.clear_lines``) and was fragile regardless: cursor
    arithmetic goes wrong the moment a line wraps, the block scrolls, or the
    pinned status bar redraws underneath it.

    Printing each line once, already highlighted, has none of those failure
    modes and looks the same -- code appearing a line at a time, in colour.
    """

    def __init__(self, ui: "UI"):
        self.ui = ui
        self.buffer = ""
        self.in_code = False
        self.language = "text"
        self.started = False

    def feed(self, text: str) -> None:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line)

    def _line(self, line: str) -> None:
        fence = line.lstrip().startswith("```")

        if fence:
            if self.in_code:
                self.in_code = False
            else:
                self.in_code = True
                self.language = _language(line.lstrip()[3:].strip())
                self._ensure_started()
            return

        self._ensure_started()
        if self.in_code:
            self.ui.code_line(line, self.language)
        else:
            self.ui.console.print(line, markup=False, highlight=False)

    def _ensure_started(self) -> None:
        if not self.started:
            self.ui.console.print()
            self.started = True

    def finish(self) -> str:
        """Flush a trailing partial line. Output has already gone out."""
        if self.buffer:
            self._line(self.buffer)
            self.buffer = ""
        self.in_code = False
        if self.started:
            self.ui.console.print()
        return ""


def _language(tag: str) -> str:
    """Normalise a fence tag to something pygments knows."""
    tag = (tag or "text").split()[0].lower() if tag else "text"
    return {"sh": "bash", "shell": "bash", "console": "bash", "py": "python",
            "js": "javascript", "ts": "typescript", "yml": "yaml",
            "": "text"}.get(tag, tag)


class ThoughtStreamer:
    """Streams the model's reasoning as dim, indented, wrapped prose.

    Reasoning arrives as a flood of tiny fragments with no line structure at
    all, so it cannot be printed per-chunk like content: it has to be
    accumulated and broken at word boundaries, or it turns into one
    unreadable line the width of the transcript.
    """

    def __init__(self, ui: "UI", indent: str = "    "):
        self.ui = ui
        self.indent = indent
        self.line = ""
        self.pending = ""
        """A trailing partial word. Fragments split mid-word constantly --
        "what auth" then ".py does" -- and treating each fragment's pieces as
        whole words inserts a space into the middle of every one."""
        self.width = max(28, ui.width - len(indent) - 2)

    def feed(self, text: str) -> None:
        self.pending += text.replace("\r", "")
        while True:
            newline = self.pending.find("\n")
            space = self.pending.rfind(" ")
            if newline == -1 and space == -1:
                return
            if newline != -1 and (space == -1 or newline < space):
                self._words(self.pending[:newline])
                self._flush()
                self.pending = self.pending[newline + 1:]
                continue
            self._words(self.pending[: space + 1])
            self.pending = self.pending[space + 1:]
            return

    def _words(self, text: str) -> None:
        for word in text.split():
            candidate = f"{self.line} {word}".strip()
            if len(candidate) > self.width:
                self._flush()
                candidate = word
            self.line = candidate

    def _flush(self) -> None:
        if self.line:
            self.ui.console.print(Text(self.indent + self.line, style=MUTED),
                                  highlight=False)
            self.line = ""

    def finish(self) -> None:
        if self.pending:
            self._words(self.pending)
            self.pending = ""
        self._flush()
        self.ui.console.print()


class ActivityBar:
    """The pinned bar: what is happening, and the tokens as they arrive.

    It holds the bottom line while output scrolls above it, and is styled to
    match the prompt's toolbar so the two read as one continuous strip -- the
    bar is there while you type, and stays there while the answer streams.
    """

    SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
    SPINNER_ASCII = "|/-\\"

    def __init__(self, ui: "UI", effort: str, hint: str = "", model: str = "",
                 pet=None):
        self.ui = ui
        self.effort = effort
        self.hint = hint
        self.model = model
        self.pet = pet
        """When present it replaces the spinner: the face carries the same
        information -- something is happening, and roughly what."""
        self.activity = "thinking"
        self.detail = ""
        self.tokens = 0
        self.context_pct = 0.0
        self.started = time.monotonic()
        self._live: Live | None = None
        self._frame = 0

    # -- content -----------------------------------------------------------

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def rate(self) -> float:
        seconds = self.elapsed()
        return self.tokens / seconds if seconds > 0.4 and self.tokens else 0.0

    def _segments(self) -> list[tuple[str, str]]:
        """(text, style) pairs, most important first, for a fit-aware build."""
        out: list[tuple[str, str]] = []
        if self.tokens:
            out.append((f"{self.tokens} tok", "bold"))
        if rate := self.rate():
            out.append((f"{rate:.0f} tok/s", ""))
        out.append((f"{self.elapsed():.0f}s", ""))
        out.append((self.effort, ""))
        if self.context_pct:
            out.append((f"ctx {self.context_pct:.0f}%", ""))
        return out

    def _render(self) -> Text:
        self._frame += 1
        width = max(20, self.ui.width)

        left = Text(style=BAR_STYLE)
        if self.pet is not None and self.pet.enabled:
            left.append(" ")
            left.append(self.pet.padded(), style=f"bold {self.pet.style()}")
            left.append(" ")
        else:
            frames = self.SPINNER if self.ui.g.unicode else self.SPINNER_ASCII
            left.append(f" {frames[self._frame % len(frames)]} ",
                        style=f"bold {BAR_ACCENT}")
        left.append(self.activity, style="bold")
        if self.detail:
            left.append("  ")
            left.append(self.detail, style=BAR_DIM)

        # The stats claim their space before the activity text does. The token
        # counter is the point of this bar, so a long file path must lose its
        # tail rather than push the numbers off the end.
        candidates = self._segments()
        if self.hint:
            candidates.append((self.hint, BAR_DIM))
        stats_budget = max(0, width - MIN_ACTIVITY_WIDTH)

        stats = Text(style=BAR_STYLE)
        for text, style in candidates:
            piece = Text(style=BAR_STYLE)
            if stats.cell_len:
                piece.append(f"  {self.ui.g.dot}  ", style=BAR_DIM)
            piece.append(text, style=style)
            if stats.cell_len + piece.cell_len + 1 > stats_budget:
                break
            stats.append_text(piece)
        stats.append(" ")

        room = width - stats.cell_len
        if left.cell_len > room:
            left.truncate(max(1, room - 1), overflow="ellipsis")
        left.append(" " * max(1, room - left.cell_len))
        left.append_text(stats)
        return left

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self.ui.console.is_terminal:
            return
        self._live = Live(self._render(), console=self.ui.console,
                          refresh_per_second=12, transient=True)
        self._live.start()

    def update(self, activity: str | None = None, detail: str | None = None,
               tokens: int | None = None, context_pct: float | None = None) -> None:
        if activity is not None:
            self.activity = activity
            if self.pet is not None:
                self.pet.set_activity(activity)
        if detail is not None:
            self.detail = detail
        if tokens is not None:
            self.tokens = tokens
        if context_pct is not None:
            self.context_pct = context_pct
        self.refresh()

    def add_token(self, count: int = 1) -> None:
        """One streamed chunk is one token, near enough, for a live counter.
        The exact figure arrives with the final chunk and replaces it."""
        self.tokens += count
        self.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def stop(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            with contextlib.suppress(Exception):
                live.stop()

    def __enter__(self) -> "ActivityBar":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
