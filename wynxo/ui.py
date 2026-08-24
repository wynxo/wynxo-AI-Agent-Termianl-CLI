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

from rich.box import ASCII as ASCII_BOX, ROUNDED
from rich.cells import cell_len
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

from .theme import Palette, resolve as resolve_theme

# Module-level names kept for the many call sites that reference them. They are
# rebound when a UI is constructed, so a palette change reaches everything
# without threading a theme object through every render function.
PALETTE: Palette = resolve_theme("purple")
ACCENT = PALETTE.accent
MUTED = PALETTE.muted
GOOD = PALETTE.good
WARN = PALETTE.warn
BAD = PALETTE.bad
BAR_STYLE = f"on {PALETTE.bar_bg}"
BAR_ACCENT = PALETTE.bar_accent
BAR_DIM = PALETTE.bar_dim
MIN_ACTIVITY_WIDTH = 16
"""Cells kept for the activity text before the stats start claiming space."""


def apply_palette(palette: Palette) -> None:
    """Rebind the module colours. Called once when the UI is built."""
    global PALETTE, ACCENT, MUTED, GOOD, WARN, BAD, BAR_STYLE, BAR_ACCENT, BAR_DIM
    PALETTE = palette
    ACCENT = palette.accent
    MUTED = palette.muted
    GOOD = palette.good
    WARN = palette.warn
    BAD = palette.bad
    BAR_STYLE = f"on {palette.bar_bg}"
    BAR_ACCENT = palette.bar_accent
    BAR_DIM = palette.bar_dim


def _supports_unicode() -> bool:
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" in encoding:
        return True
    try:
        "•─".encode(encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


ASCII_FALLBACK = {
    "\u2022": "*", "\u2192": "->", "\u2190": "<-", "\u2191": "up",
    "\u2193": "down", "\u2713": "+", "\u2717": "x", "\u25cf": "o",
    "\u00b7": ".", "\u2026": "...", "\u2014": "-", "\u2013": "-",
    "\u276f": ">", "\u2502": "|", "\u2500": "-", "\u256d": "+",
    "\u256e": "+", "\u2570": "+", "\u256f": "+", "\u201c": '"',
    "\u201d": '"', "\u2018": "'", "\u2019": "'",
}
"""Replacements for the glyphs used in labels the caller cannot re-word."""


def to_ascii(text: str) -> str:
    """Down-convert a display string for a terminal that cannot draw it.

    Anything outside the table is dropped rather than shown as a question
    mark, which is what an ASCII locale would otherwise print.
    """
    out = []
    for char in text:
        if char in ASCII_FALLBACK:
            out.append(ASCII_FALLBACK[char])
        elif ord(char) < 128:
            out.append(char)
    return "".join(out)


class Glyphs:
    def __init__(self, unicode_ok: bool):
        self.unicode = unicode_ok
        if unicode_ok:
            self.bullet, self.arrow, self.tick = "•", "→", "✓"
            self.cross, self.gear, self.dot = "✗", "●", "·"
            # Rounded box corners, for the input field the prompt sits in.
            self.tl, self.tr, self.bl, self.br = "╭", "╮", "╰", "╯"
            self.hbar, self.vbar, self.ellipsis = "─", "│", "…"
        else:
            self.bullet, self.arrow, self.tick = "*", "->", "+"
            self.cross, self.gear, self.dot = "x", "o", "."
            self.tl, self.tr, self.bl, self.br = "+", "+", "+", "+"
            self.hbar, self.vbar, self.ellipsis = "-", "|", "..."


class UI:
    def __init__(self, theme: str = "purple", show_thinking: bool = True):
        self.console = Console(
            highlight=False,
            soft_wrap=False,
            # legacy_windows=False forces modern VT output on Windows Terminal.
            legacy_windows=False if sys.platform == "win32" else None,
        )
        self.g = Glyphs(_supports_unicode())
        # rich's default panel box is Unicode; ASCII terminals need the
        # plain one or every border renders as question marks.
        self.box = ROUNDED if self.g.unicode else ASCII_BOX
        self.palette = resolve_theme(theme)
        apply_palette(self.palette)
        self.show_thinking = show_thinking
        self.code_theme = self.palette.code_theme
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
        prefix = "  wynxo"
        budget = self.width - 1
        shown: list[str] = []
        for part in parts:
            candidate = shown + [part]
            room = cell_len(prefix) + sum(
                cell_len(separator) + cell_len(p) for p in candidate)
            if room > budget:
                continue
            shown = candidate

        head = Text()
        head.append(prefix, style=f"bold {ACCENT}")
        for i, part in enumerate(shown):
            head.append(separator, style=MUTED)
            head.append(part, style="bold" if i == 0 else "")

        self.console.print()
        self.console.print(head, overflow="ellipsis", no_wrap=True)
        self.console.print(Rule(style=MUTED, characters=self.g.hbar))

    def clear(self) -> None:
        """Clear the screen and scrollback, so a session starts on a clean page.

        Scrollback too: clearing only the visible rows leaves the previous
        session one scroll away, which is worse than not clearing at all.
        """
        if not self.console.is_terminal:
            return
        self.console.clear()
        self.console.file.write("\x1b[3J")   # erase saved lines
        self.console.file.flush()

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
        self.console.print(Rule(label, style=MUTED, characters=self.g.hbar))

    # -- messages ----------------------------------------------------------

    def info(self, message: str) -> None:
        self.console.print(Text(f"  {message}", style=MUTED))

    def warn(self, message: str) -> None:
        self.console.print(Text(f"  ! {message}", style=WARN))

    def error(self, message: str) -> None:
        self.console.print()
        self.console.print(Panel(Text(message), title="error", border_style=BAD, box=self.box,
                  padding=(0, 1)))

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
                box=self.box,
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
        self.console.print(Panel(body, border_style=MUTED, box=self.box, padding=(0, 1)))

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
        self.console.print(Panel(body, title="plan", title_align="left", border_style=MUTED,
                  box=self.box, padding=(0, 1)))

    def code(self, text: str, language: str = "text") -> None:
        self.console.print(Syntax(text, language, theme=self.code_theme, word_wrap=True))

    def code_line(self, line: str, language: str = "text",
                  indent: str = "  ") -> None:
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
        self.console.print(Text(indent) + rendered, highlight=False)

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

        table = Table(title=title or None, border_style=MUTED, box=self.box,
                      title_style=f"bold {ACCENT}")
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def stats(self, usage, elapsed: float, effort: str, context_pct: float) -> None:
        speed = usage.tokens_per_second()
        bits = [
            f"{effort}",
            f"{usage.completion_tokens} tok",
            f"{speed:.0f} tok/s" if speed else "",
            f"{elapsed:.1f}s",
            f"ctx {context_pct:.0f}%",
        ]
        bits = [b for b in bits if b]

        # Narrow terminals get the short version rather than a wrapped line.
        # Effort and context also sit in the pinned bar, so they go first.
        sep = f" {self.g.dot} "

        def line(parts):
            return "  " + sep.join(parts)

        for drop in (f"{effort}", f"ctx {context_pct:.0f}%", f"{speed:.0f} tok/s"):
            if len(line(bits)) <= self.width:
                break
            if drop in bits:
                bits.remove(drop)
        self.console.print(Text(line(bits), style=MUTED))


class CodeStreamer:
    """Renders streamed assistant text as it arrives.

    Prose is written out word by word as soon as each word is complete, with
    wrapping done here rather than by the terminal -- so text flows the way it
    is generated instead of appearing a whole line at a time when a newline
    finally shows up. A model writing one long paragraph used to produce
    nothing at all until it finished.

    Fenced code is different: it is highlighted per line, once the line is
    whole, because a half-written line cannot be lexed.
    """

    def __init__(self, ui: "UI", indent: str = "", style: str = "",
                 code: bool = True):
        self.ui = ui
        self.indent = indent
        self.style = style
        self.code = code
        """False for reasoning: a model's scratchpad is full of stray
        backticks and half-fences that are not code blocks."""
        self.buffer = ""
        self.column = 0
        self.in_code = False
        self.language = "text"
        self.started = False
        self.width = max(30, ui.width - len(indent) - 1)

    # -- entry point -------------------------------------------------------

    def feed(self, text: str) -> None:
        self.buffer += text.replace("\r", "")
        while self.buffer:
            newline = self.buffer.find("\n")
            if newline != -1:
                self._segment(self.buffer[:newline], end_of_line=True)
                self.buffer = self.buffer[newline + 1:]
                continue
            # No newline yet. Emit whole words and hold the partial one, so a
            # word never appears split across a flush.
            space = self.buffer.rfind(" ")
            if space == -1:
                return
            self._segment(self.buffer[: space + 1], end_of_line=False)
            self.buffer = self.buffer[space + 1:]
            return

    # -- the two modes -----------------------------------------------------

    def _segment(self, text: str, end_of_line: bool) -> None:
        if self.code and (self.in_code or text.lstrip().startswith("```")):
            self._code_segment(text, end_of_line)
            return
        self._prose(text)
        if end_of_line:
            self._newline()

    def _code_segment(self, text: str, end_of_line: bool) -> None:
        # Fences only ever matter on a complete line.
        if not end_of_line:
            self.buffer = text + self.buffer
            return
        if text.lstrip().startswith("```"):
            if self.in_code:
                self.in_code = False
            else:
                self.in_code = True
                self.language = _language(text.lstrip()[3:].strip())
                self._ensure_started()
            return
        self._ensure_started()
        self.ui.code_line(text, self.language, indent=self.indent + "  ")

    def _prose(self, text: str) -> None:
        for word in _words(text):
            if word.isspace():
                if self.column:
                    self._write(word)
                continue
            if self.column and self.column + len(word) > self.width:
                self._newline()
            if not self.column:
                self._write(self.indent)
            self._write(word)

    # -- output ------------------------------------------------------------

    def _write(self, text: str) -> None:
        self._ensure_started()
        if self.style:
            self.ui.console.print(text, style=self.style, end="",
                                  markup=False, highlight=False)
        else:
            self.ui.console.file.write(text)
            self.ui.console.file.flush()
        self.column += len(text)

    def _newline(self) -> None:
        if self.started:
            self.ui.console.file.write("\n")
            self.ui.console.file.flush()
        self.column = 0

    def _ensure_started(self) -> None:
        if not self.started:
            self.ui.console.print()
            self.started = True

    def finish(self) -> str:
        if self.buffer:
            self._segment(self.buffer, end_of_line=True)
            self.buffer = ""
        if self.column:
            self._newline()
        self.in_code = False
        if self.started:
            self.ui.console.print()
        return ""


def _words(text: str):
    """Split into words and the whitespace between them, keeping both."""
    current = ""
    for char in text:
        if char == " ":
            if current:
                yield current
                current = ""
            yield " "
        else:
            current += char
    if current:
        yield current


def _language(tag: str) -> str:
    """Normalise a fence tag to something pygments knows."""
    tag = (tag or "text").split()[0].lower() if tag else "text"
    return {"sh": "bash", "shell": "bash", "console": "bash", "py": "python",
            "js": "javascript", "ts": "typescript", "yml": "yaml",
            "": "text"}.get(tag, tag)


class ThoughtStreamer(CodeStreamer):
    """The model's reasoning: same flow, indented and dimmed, no code blocks.

    Reasoning is not code even when it contains backticks, so fence handling
    is off -- a stray ``` in a scratchpad would otherwise swallow the rest of
    the thought into a syntax highlighter.
    """

    def __init__(self, ui: "UI", indent: str = "    "):
        super().__init__(ui, indent=indent, style=MUTED, code=False)


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
        self.queued = ""
        """What the user is typing, or how many messages are waiting."""
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
        if self.queued:
            # What you are typing beats what the agent is doing: you need to
            # see your own keystrokes, and the detail is still one line up.
            left.append("  ")
            left.append(f"\u203a {self.queued}", style=f"bold {BAR_ACCENT}")
        elif self.detail:
            left.append("  ")
            left.append(self.detail, style=BAR_DIM)

        # The stats claim their space before the activity text does. The token
        # counter is the point of this bar, so a long file path must lose its
        # tail rather than push the numbers off the end.
        candidates = self._segments()
        if self.hint:
            candidates.append((self.hint, BAR_DIM))
        # While the user is typing, their own keystrokes are the most
        # important thing on the line, so the left side claims more of it and
        # the stats give way rather than the other way round.
        floor = min(width - 20, MIN_ACTIVITY_WIDTH + len(self.queued) + 6) \
            if self.queued else MIN_ACTIVITY_WIDTH
        stats_budget = max(0, width - max(MIN_ACTIVITY_WIDTH, floor))

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
