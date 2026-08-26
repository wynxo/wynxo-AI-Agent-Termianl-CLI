"""Terminal rendering.

Windows note: rich handles the VT sequences, but the box-drawing and arrow
glyphs used elsewhere in TUIs do not survive the default Windows console
font. Everything here sticks to ASCII plus a small set of glyphs that are
checked against the active encoding at startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import sys
import time
from typing import Iterable

from rich.box import ASCII as ASCII_BOX, ROUNDED
from rich.cells import cell_len
from rich.console import Console, Group
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
def _ansi_of(style: str) -> str:
    """One rich style as the escape that turns it on."""
    from rich.style import Style

    try:
        return Style.parse(style).render("\x00").split("\x00")[0]
    except Exception:
        return ""


CODE_SPAN = "#e6c07b"
"""`inline code` in the model's prose. Warm against the purple, so it reads
as a name rather than as emphasis."""

MIN_ACTIVITY_WIDTH = 16
"""Cells kept for the activity text before the stats start claiming space."""


# Modules that do `from .ui import ACCENT, MUTED, ...`. That kind of import
# binds a *copy* of the name in the importing module, so rebinding ours here
# never reached them -- which is why changing the theme used to require a
# restart. Pushing the new values into them is blunt, but it is one place and
# the alternative is threading a palette object through seventy call sites.
# Per module, exactly the names it does `from .ui import ...` for. Listed
# rather than discovered, because a name can mean something else elsewhere:
# cli.py also imports WARN from .status, where it is a status tag and not a
# colour, and blindly overwriting it printed a raw "[#f0c674]" on screen
# instead of "[ WARN ]". Keep these in step with the imports.
_COLOUR_CONSUMERS = {
    "wynxo.cli": ("ACCENT", "MUTED"),
    "wynxo.wizard": ("ACCENT", "MUTED"),
    "wynxo.doctor": ("ACCENT", "BAD", "GOOD", "MUTED", "WARN"),
}


def apply_palette(palette: Palette) -> None:
    """Rebind the module colours, everywhere they were imported to."""
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

    import sys as _sys

    here = globals()
    for module_name, names in _COLOUR_CONSUMERS.items():
        module = _sys.modules.get(module_name)
        if module is None:
            continue          # not imported in this run; nothing to update
        for name in names:
            if hasattr(module, name):
                setattr(module, name, here[name])


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitise(text: str) -> str:
    """Strip control characters out of text wynxo did not write itself.

    Model output, file contents and command output all end up on screen, and
    a terminal *acts* on escape sequences in them: ESC[2J clears the screen
    and takes the scrollback with it, ESC]0; renames the window. It does not
    take a hostile model -- a log with colour codes in it, a terminal
    recording, a test fixture, and the agent echoing any of them back is
    enough.

    rich neutralises this when it is handed a plain string, and does not
    when it is handed a Text, a Syntax or a Markdown, which is most of what
    this module builds. So it is done here, once, at the point where
    somebody else's text becomes something to draw.

    Newlines and tabs stay. Nothing else in the C0 range is meant literally,
    carriage returns included: a line rewriting itself makes no sense in a
    transcript that is a list of finished lines.
    """
    return _CONTROL.sub("", text)


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


class _SaidOnce:
    """What a spinner becomes where nothing can repaint.

    Carries the same shape as rich's Status -- a context manager with an
    update() -- so no caller has to know which one it got.
    """

    def __init__(self, ui: "UI", message: str):
        self.ui = ui
        self._say(message)

    def _say(self, message: str) -> None:
        if message:
            self.ui.console.print(Text(f"  {message}", style=MUTED))

    def update(self, status=None, **_kwargs) -> None:
        if status is not None:
            self._say(status if isinstance(status, str) else str(status))

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def __enter__(self) -> "_SaidOnce":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


class UI:
    def __init__(self, theme: str = "purple", show_thinking: bool = True):
        self.console = Console(
            highlight=False,
            soft_wrap=False,
            # legacy_windows=False forces modern VT output on Windows Terminal.
            legacy_windows=False if sys.platform == "win32" else None,
        )
        self.g = Glyphs(_supports_unicode())
        self.bar: "ActivityBar | None" = None
        """The pinned bar, while a turn is running. Streamed text has
        to be handed to it rather than written straight out, or its
        repaint erases whatever landed on its row."""
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
        self.live_ok = True
        """Whether a rich Live may drive the screen. False under the chat
        layout, where the activity bar is drawn into the pinned status row
        instead -- a Live there would emit cursor moves into the transcript
        buffer and shred it."""
        self.on_refresh: "Callable[[], None] | None" = None
        """Set by the chat layout so an in-place update repaints the pane."""

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
        if not self.live_ok:
            # The chat layout's transcript is a buffer of finished lines, and
            # a Live writes cursor moves and carriage returns to redraw in
            # place -- which land in that buffer as literal "?25l" and "^M"
            # rather than as an animation. One still frame instead.
            self.console.print()
            self.console.print(f"  {pet.padded()}  {pet.name} is awake",
                               style=pet.style())
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
        text = sanitise(text)
        self.console.print()
        self.console.print(Markdown(text, code_theme=self.code_theme))
        self.console.print()

    def thinking(self, text: str) -> None:
        if not (self.show_thinking and text.strip()):
            return
        preview = sanitise(text).strip()
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
        text = sanitise(display or output).strip()
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

    def tool_output(self, line: str) -> None:
        """One line from a command that is still running.

        Dimmed and indented past the tool line so a long build reads as
        something happening underneath the step, rather than as the agent's
        own words. Truncated per line: a stray 5000-column line from a
        minifier would otherwise wrap into a screenful.
        """
        text = sanitise(line).rstrip()
        if not text.strip():
            return
        limit = max(24, self.width - 8)
        self.console.print(Text("      " + text[:limit], style=MUTED))

    def diff(self, text: str) -> None:
        if not text.strip():
            return
        text = sanitise(text)
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
        self.console.print(Syntax(sanitise(text), language,
                                  theme=self.code_theme, word_wrap=True))

    def highlight(self, line: str, language: str = "text") -> Text:
        """One line, syntax-highlighted, with no block chrome.

        Syntax() would draw its own background band per line and stack into a
        ragged column, so the lexer is used directly instead.

        Safe on a half-written line: pygments will mis-lex an unterminated
        string or a keyword that is still being typed, and both correct
        themselves on the next character. That is the price of showing code
        as it arrives rather than a line at a time.
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
        return rendered

    def code_line(self, line: str, language: str = "text",
                  indent: str = "  ") -> None:
        self.console.print(Text(indent) + self.highlight(line, language),
                           highlight=False)

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

    def status(self, message: str):
        """A spinner while something slow happens.

        rich's Status is a Live: it hides the cursor, redraws in place and
        carriage-returns over itself. Sent to a buffer of finished lines --
        which is what the console is under the chat layout -- those arrive as
        literal "?25l", "?25h" and "^M" in the middle of the conversation.
        That is what /model looked like: two spinners' worth of escape codes
        printed into the transcript before the picker even opened.

        So where a Live cannot go, the message is simply said once. It is the
        same information, minus the animation.
        """
        if not self.live_ok:
            return _SaidOnce(self, message)
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
                 code: bool = True, literal: bool = False):
        self.ui = ui
        self.indent = indent
        self.style = style
        self.code = code
        """False for reasoning: a model's scratchpad is full of stray
        backticks and half-fences that are not code blocks."""
        self.literal = literal
        """True when the text is known to be a file's contents. Prose drops
        whitespace at the start of a line, which is right for a sentence and
        destroys the indentation of every line of Python."""
        self.buffer = ""
        self.column = 0
        self.in_code = False
        self.language = "text"
        self.started = False
        self.width = max(30, ui.width - len(indent) - 1)
        self.partial = ""
        """The half-written code line, while inside a fence."""
        self.word = ""
        """The word being typed, so a wrap can carry it down whole."""
        self.in_span = False
        """Inside a `code span`. Reset at every newline, so an unpaired
        backtick colours one line rather than the rest of the answer."""
        self.in_bold = False
        self.in_heading = False
        self.pending_star = ""
        """A single asterisk, held for one character to see whether a second
        follows. `2 * 3` has to survive; `**bold**` has to work."""
        self.pending_hashes = ""
        self._pen_shown = ""
        """The style the terminal is currently set to, when writing to it
        directly. Only used where there is no bar to redraw through."""
        self.marks_code = not literal
        """Whether backticks mean anything here. Inside a file being written
        they are just characters."""
        self.line = self._blank()
        """The line being written. While the activity bar is up this is the
        bar's lead line rather than terminal output, so a partial line and a
        repainting bar can share the screen."""

    # -- entry point -------------------------------------------------------

    def feed(self, text: str) -> None:
        # Everything streamed goes through here -- the answer, the reasoning,
        # a file being written -- so it is the one place worth cleaning.
        self.buffer += sanitise(text)
        while self.buffer:
            newline = self.buffer.find("\n")
            if newline != -1:
                self._segment(self.buffer[:newline], end_of_line=True)
                self.buffer = self.buffer[newline + 1:]
                continue

            # Everything else goes out the moment it arrives -- code and
            # prose alike, one character at a time. Holding back the partial
            # word is what made replies land in jumps, and holding back the
            # partial line is what made a function appear all at once when
            # the model finally pressed enter.
            held = self._held_length()
            if held >= len(self.buffer):
                return
            self._segment(self.buffer[:len(self.buffer) - held],
                          end_of_line=False)
            self.buffer = self.buffer[len(self.buffer) - held:]
            return

    def _held_length(self) -> int:
        """How much of the tail cannot be shown yet.

        Exactly one thing can't: text that might turn out to be a fence.
        Three backticks at the start of a line stop being characters and
        become a marker, and once one has been printed as prose there is no
        taking it back -- which is how "``" and then "`" leaked onto the
        screen either side of a code block.

        So a line that has begun with a backtick waits, and nothing else
        does. The wait is at most a couple of characters: the moment the
        line turns out to be `inline code` rather than a fence it is
        released, and a fence's own line is never printed anyway.
        """
        if not self.code:
            return 0
        # A fence only means anything at the start of a line. Mid-expression
        # -- inside a string, in a comment -- a backtick is just a character,
        # and waiting on it would stall the stream.
        if self.in_code:
            if self.partial.strip():
                return 0
        elif self.line.plain:
            return 0
        stripped = self.buffer.lstrip(" \t")
        if not stripped:
            return 0
        if stripped.startswith("```") or "```".startswith(stripped):
            # Either a fence, or still too short to tell. Its whole line
            # goes: the rest of an opening fence is the language name, which
            # is a label rather than text to show.
            return len(self.buffer)
        return 0

    # -- the two modes -----------------------------------------------------

    def _segment(self, text: str, end_of_line: bool) -> None:
        if self.code and (self.in_code or self._opens_a_fence(text)):
            self._code_segment(text, end_of_line)
            return
        if self.literal:
            self._literal(text)
            if end_of_line:
                self._newline()
            return
        self._prose(text)
        if end_of_line:
            self._flush_marks()
            self._newline()

    def _opens_a_fence(self, text: str) -> bool:
        """Whether this segment is a fence opening, rather than prose.

        The segment has to be the start of its line. A chunk boundary can
        fall anywhere, so "see ```` ``` ```` in the docs" arrives as a
        segment beginning with three backticks without being a fence at all.
        """
        return not self.line.plain and text.lstrip().startswith("```")

    def _code_segment(self, text: str, end_of_line: bool) -> None:
        if not end_of_line:
            # A partial fence never reaches here: _held_length keeps the
            # whole of a line that begins with a backtick until its newline
            # arrives, so anything unterminated is code being written now.
            self.partial += text
            self._show_partial()
            return

        whole = self.partial + text
        self.partial = ""
        if whole.lstrip().startswith("```"):
            self._clear_partial()
            if self.in_code:
                self.in_code = False
            else:
                self.in_code = True
                self.language = _language(whole.lstrip()[3:].strip())
                self._ensure_started()
            return
        self._ensure_started()
        self._clear_partial()
        self.ui.code_line(whole, self.language, indent=self.indent + "  ")

    def _show_partial(self) -> None:
        """Put the half-written line in the live region, highlighted.

        The bar redraws in place, so the line can grow a character at a time
        without each version being left behind in the scrollback. Without a
        bar there is nowhere to redraw, so the line waits for its newline --
        the old behaviour, which is correct when nothing is pinned.
        """
        if self.ui.bar is None:
            return
        line = Text(self.indent + "  ")
        line.append_text(self.ui.highlight(self.partial, self.language))
        self.ui.bar.set_lead(line)

    def _clear_partial(self) -> None:
        if self.ui.bar is not None:
            self.ui.bar.set_lead(None)

    def _literal(self, text: str) -> None:
        """Every character exactly as written, indentation included.

        No reflowing: a line of code means what its leading whitespace says,
        and a line too long for the terminal is broken at the edge rather
        than rearranged into something that no longer parses.
        """
        for char in text:
            if self.column >= self.width:
                self._newline()
            if not self.column and self.indent:
                self._write(self.indent)
            self._write(char)

    def _prose(self, text: str) -> None:
        """One character at a time, wrapping without splitting words.

        Emitting whole words is easier and reads worse: the answer arrives in
        little jumps, and a model that pauses mid-word appears to have
        stopped. Writing every character as it lands is what makes the reply
        look like it is being typed.

        Wrapping is the reason this is not simply "print the character". The
        line width is only exceeded part-way through a word, and by then the
        word is already on screen -- so it is lifted off the end of the line
        and carried down to the next one, which is what a word processor does
        and what the eye expects.
        """
        for char in text:
            if self.marks_code:
                consumed, char = self._mark(char)
                if consumed:
                    continue

            if char.isspace():
                self.word = ""
                if self.column:
                    self._write(char)
                continue

            if self.column + 1 > self.width:
                if (self.word and self._rewritable
                        and len(self.word) + len(self.indent) < self.width):
                    self._carry_word_down()
                else:
                    # Either the word is longer than the line, or the line has
                    # already gone to the terminal and cannot be taken back.
                    self._newline()
            if not self.column:
                self._write(self.indent)
            self._write(char)
            self.word += char

    def _mark(self, char: str) -> tuple[bool, str]:
        """Handle the markdown a model actually writes, one character at a time.

        `code`, **bold**, and a ## heading. Per character rather than per
        finished line, because a line is not finished when it is drawn --
        by the time it is, its characters have gone to the terminal and
        cannot be restyled.

        Returns (consumed, char): the mark itself is swallowed, everything
        else comes back to be written.
        """
        # A heading is decided by what starts the line. The hashes are held
        # rather than written, because "#" is also just a character and only
        # "## " makes it a heading.
        if self.pending_hashes:
            if char == "#" and len(self.pending_hashes) < 6:
                self.pending_hashes += "#"
                return True, char
            if char == " ":
                self.in_heading = True
                self.pending_hashes = ""
                return True, char
            held, self.pending_hashes = self.pending_hashes, ""
            self._prose_out(held)
        elif char == "#" and not self.column and not self.line.plain:
            self.pending_hashes = "#"
            return True, char

        if self.pending_star:
            self.pending_star = False
            if char == "*":
                self.in_bold = not self.in_bold
                return True, char
            self._prose_out("*")          # a lone asterisk, meant literally
        elif char == "*":
            self.pending_star = True
            return True, char

        if char == "`":
            self.in_span = not self.in_span
            return True, char
        return False, char

    def _flush_marks(self) -> None:
        """Write out anything held back that turned out to be literal.

        A line ending on a single "*" was waiting to see whether a second
        followed. None ever does, and without this the asterisk was simply
        dropped.
        """
        held, self.pending_hashes = self.pending_hashes, ""
        if self.pending_star:
            self.pending_star = False
            held += "*"
        if held:
            self._prose_out(held)

    def _prose_out(self, text: str) -> None:
        """Write held-back characters through the ordinary prose path."""
        for char in text:
            if self.column + 1 > self.width:
                self._newline()
            if not self.column:
                self._write(self.indent)
            self._write(char)
            self.word += char

    @property
    def _rewritable(self) -> bool:
        """Whether the line in progress can still be changed.

        While the activity bar is up the line lives inside it and is redrawn
        on every repaint, so a word can be lifted off the end. Written
        straight to the terminal it is already gone, and the only honest
        thing left is to break at the edge.
        """
        return self.ui.bar is not None

    def _carry_word_down(self) -> None:
        """Move the half-written word to the next line, taking it with us."""
        keep = self.line.plain[: len(self.line.plain) - len(self.word)]
        carried = self.word
        self.line = self._blank(keep)
        self.column = cell_len(keep)
        self._newline()
        self.word = ""
        self._write(self.indent)
        self._write(carried)
        self.word = carried

    # -- output ------------------------------------------------------------

    def _blank(self, text: str = "") -> Text:
        """A fresh line, styled once for all of it."""
        return Text(text, style=self.style or "")

    @property
    def _pen(self) -> str:
        """The style for what is being written right now."""
        if self.in_span:
            return CODE_SPAN
        if self.in_heading:
            return f"bold {ACCENT}"
        if self.in_bold:
            return "bold"
        return ""

    def _stylize(self, style: str, start: int, end: int) -> None:
        """Style a slice, growing the previous span where it can.

        A code span arrives one character at a time like everything else,
        and a span per character is what made a coloured line cost an escape
        pair per letter. Adjacent characters in the same style are one span.
        """
        spans = self.line.spans
        if spans:
            last = spans[-1]
            if last.end == start and last.style == style:
                try:
                    spans[-1] = last._replace(end=end)
                    return
                except AttributeError:
                    pass          # a rich that does not use a NamedTuple
        self.line.stylize(style, start, end)

    def _pen_change(self) -> str:
        """The escape needed to bring the terminal to the current pen."""
        wanted = self._pen
        if wanted == self._pen_shown:
            return ""
        self._pen_shown = wanted
        if not wanted:
            return "\x1b[0m"
        return _ansi_of(wanted)

    def _write(self, text: str) -> None:
        """Add to the line in progress.

        The activity bar is a rich Live: it repaints by erasing the rows it
        owns, so anything written straight to the terminal on its row is gone
        at the next repaint. That is how "Done, wrote out.txt." came out as
        "out.txt." after a tool call, and why the two-space indent kept
        vanishing from the first line of an answer.

        So while the bar is up, a half-finished line lives *inside* it, as a
        lead line drawn above the status strip. It still grows a word at a
        time; it just grows somewhere the repaint can see. The moment the
        line is complete it is printed normally and scrolls up out of the
        live region like any other output.
        """
        self._ensure_started()
        # Plain unless something is emphasised: the line carries the style,
        # not each character. Appending with a style creates a span per
        # call, and calls are per character now -- which turned every
        # streamed line into one escape pair per letter, ten bytes of colour
        # for each byte of text, all of it kept in the transcript and
        # re-rendered on every repaint.
        start = len(self.line.plain)
        self.line.append(text)
        if pen := self._pen:
            self._stylize(pen, start, start + len(text))
        self.column += cell_len(text)
        if self.ui.bar is not None:
            self.ui.bar.set_lead(self.line)
        else:
            # Straight to the terminal, so the colour has to be written as
            # well as chosen -- and only when it changes, or every character
            # would carry its own escape pair.
            self.ui.console.file.write(self._pen_change() + text)
            self.ui.console.file.flush()

    def _newline(self) -> None:
        if self.started:
            if self.ui.bar is not None:
                self.ui.bar.set_lead(None)
                self.ui.console.print(self.line, markup=False, highlight=False,
                                      soft_wrap=True)
            else:
                self.ui.console.file.write(
                    ("\x1b[0m" if self._pen_shown else "") + "\n")
                self.ui.console.file.flush()
        # Emphasis is reset: a line is where it ends, so one stray backtick
        # or asterisk cannot colour the rest of the answer. What is *held*
        # is not reset here -- a wrap is not the end of a line, and a
        # half-seen "*" may still pair with the next character.
        self.line = self._blank()
        self.column = 0
        self.in_span = self.in_bold = self.in_heading = False
        self._pen_shown = ""

    def _ensure_started(self) -> None:
        if not self.started:
            self.ui.console.print()
            self.started = True

    def finish(self) -> str:
        if self.buffer or self.partial:
            self._segment(self.buffer, end_of_line=True)
            self.buffer = ""
        self._flush_marks()
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


METER_BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
"""Lower-eighth block through full block."""

METER_WIDTH = 3


def effort_meter(effort: str, unicode_ok: bool = True) -> str:
    """A fixed-width gauge for an effort level.

    Fixed width on purpose: a meter that grew with the level would shift
    everything after it in the bar every time you pressed Ctrl-E.
    """
    from .effort import ORDER

    try:
        rank = ORDER.index(effort)
    except ValueError:
        return " " * METER_WIDTH
    # rank+1 of len(ORDER), so the lowest level still shows something: a
    # meter that is blank at `low` reads as broken rather than as low.
    fraction = (rank + 1) / len(ORDER)

    if not unicode_ok:
        # Three characters that read as increasing intensity, all one cell.
        step = min(METER_WIDTH, max(1, round(fraction * METER_WIDTH)))
        return (".:!"[step - 1] * step).ljust(METER_WIDTH)

    filled = fraction * METER_WIDTH
    out = []
    for slot in range(METER_WIDTH):
        share = min(1.0, max(0.0, filled - slot))
        if share <= 0:
            out.append(" ")
        else:
            index = min(len(METER_BLOCKS) - 1,
                        max(0, round(share * (len(METER_BLOCKS) - 1))))
            out.append(METER_BLOCKS[index])
    return "".join(out)


SURGE_FRAMES = (
    "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588",
    "\u2588\u2587\u2586\u2585\u2584\u2583\u2582\u2581",
)


def celebrate(ui: "UI", label: str, level: int, steps: int) -> None:
    """A band of colour for stepping up, drawn once.

    The animated surge needs a repainting widget, which the chat layout does
    not have -- its transcript is a list of finished lines. This is the same
    idea in one line: the band is longer and brighter the further up you
    went, so the top of the scale still reads as an event.
    """
    from rich.text import Text as _T

    sweep = [(255, 120, 200), (255, 96, 190), (246, 74, 186), (228, 64, 190),
             (204, 62, 200), (176, 70, 214), (150, 84, 226), (132, 104, 236)]
    span = max(8, min(46, 8 + level * 7))
    block = "█" if ui.g.unicode else "#"
    row = _T("  ")
    for i in range(span):
        r, g, b = sweep[(i + level * 2) % len(sweep)]
        row.append(block, style=f"#{r:02x}{g:02x}{b:02x}")
    row.append(f"  {label}", style="bold #ff78c8")
    ui.console.print(row)


async def surge(ui: "UI", label: str, style: str, width: int = 34) -> None:
    """A short wave across the line, for stepping up to max or ultra.

    Drawn on one line and then rewritten, so it costs a line of scrollback
    rather than a screenful. Skipped when there is no terminal to animate --
    a pipe would otherwise collect every frame as separate output.
    """
    if not ui.console.is_terminal:
        return
    if not ui.live_ok:
        # Same reason: in-place redrawing has nowhere to happen here.
        ui.console.print(f"  {label}", style=f"bold {style}")
        return
    blocks = SURGE_FRAMES[0] if ui.g.unicode else "-=#"
    span = min(width, max(10, ui.width - 20))
    with Live("", console=ui.console, refresh_per_second=30,
              transient=True) as live:
        for step in range(span + 6):
            bar = Text("  ")
            for cell in range(span):
                distance = abs(cell - step)
                if distance < len(blocks):
                    bar.append(blocks[len(blocks) - 1 - distance], style=style)
                else:
                    bar.append(" ")
            bar.append(f"  {label}", style=f"bold {style}")
            live.update(bar)
            await asyncio.sleep(0.012)


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
        self.animate = True
        """Off means a still bar: no sweep, no dots. Follows the same
        setting the companion's animation does."""
        self.plan: str = ""
        """The current todo list, rendered. Held in the live region and
        redrawn in place, rather than printed again on every update -- the
        plan is one thing that changes, not a stream of panels."""
        self.plan_done_frame = 0
        """Non-zero while the completion animation is playing."""
        self.lead: Text | None = None
        """A half-written line of the answer, drawn just above the strip.

        Streamed prose cannot be written to the terminal while the bar owns
        that row -- the next repaint erases it. Carrying the partial line
        inside the live region instead lets the answer arrive a word at a
        time without fighting the bar for the same cells."""
        self.started = time.monotonic()
        self._live: Live | None = None
        self._frame = 0

    # -- content -----------------------------------------------------------

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def rate(self) -> float:
        seconds = self.elapsed()
        return self.tokens / seconds if seconds > 0.4 and self.tokens else 0.0

    def effort_meter(self) -> str:
        """A little gauge that fills up as the effort level rises.

        The level already has a name in the bar; this is there so the change
        is visible at a glance without reading a word -- pick ultra and the
        strip visibly leans on it.
        """
        return effort_meter(self.effort, self.ui.g.unicode)

    THINKING_DOTS = 4
    """Cycle length for the trailing dots. Four reads as a rhythm; more
    looks like the line is loading rather than the model is working."""

    def _activity_text(self) -> Text:
        """The activity word, with a highlight travelling through it.

        A static word next to a spinner still reads as stalled -- the spinner
        turns whether or not anything is happening. Moving the emphasis
        through the word itself is a second, independent sign of life, and it
        costs one Text per frame.
        """
        word = self.activity
        out = Text(style=BAR_STYLE)
        if not word:
            return out

        if not self.ui.g.unicode or not self.animate:
            out.append(word, style="bold")
            return out

        # One bright cell sweeping left to right, with a lit tail behind it.
        span = len(word) + 6
        head = self._frame % span
        for index, char in enumerate(word):
            distance = head - index
            if distance == 0:
                out.append(char, style=f"bold {BAR_ACCENT}")
            elif 0 < distance <= 2:
                out.append(char, style="bold")
            else:
                out.append(char, style=BAR_DIM if distance < 0 else "bold")

        if word == "thinking":
            dots = (self._frame // 3) % (self.THINKING_DOTS + 1)
            out.append(self.ui.g.dot * dots, style=BAR_ACCENT)
            out.append(" " * (self.THINKING_DOTS - dots))
        return out

    def _segments(self) -> list[tuple[str, str]]:
        """(text, style) pairs, most important first, for a fit-aware build."""
        out: list[tuple[str, str]] = []
        if self.tokens:
            out.append((f"{self.tokens} tok", "bold"))
        if rate := self.rate():
            out.append((f"{rate:.0f} tok/s", ""))
        out.append((f"{self.elapsed():.0f}s", ""))
        out.append((f"{self.effort_meter()} {self.effort}", ""))
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
        left.append_text(self._activity_text())
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
        if not self.ui.live_ok:
            return       # the chat layout paints this row itself
        if not self.ui.console.is_terminal:
            return
        self._live = Live(self, console=self.ui.console,
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

    PLAN_DONE_FRAMES = 8
    """Long enough to register at 12fps, short enough not to be in the way."""

    def set_plan(self, rendered: str) -> None:
        self.plan = rendered or ""
        self.plan_done_frame = 0
        self.refresh()

    def plan_is_complete(self) -> bool:
        """Every line ticked, and there was at least one."""
        lines = [ln for ln in self.plan.splitlines() if ln.strip()]
        steps = [ln for ln in lines if ln.lstrip().startswith(("[ ]", "[>]", "[x]"))]
        return bool(steps) and all(ln.lstrip().startswith("[x]") for ln in steps)

    async def finish_plan(self) -> None:
        """Tick the whole plan, hold it for a beat, then take it away.

        The point is to show the thing completing rather than having it
        vanish between frames -- a plan that simply disappears reads as
        having been abandoned.
        """
        if not self.plan:
            return
        for frame in range(1, self.PLAN_DONE_FRAMES + 1):
            self.plan_done_frame = frame
            self.refresh()
            await asyncio.sleep(0.06)
        self.plan = ""
        self.plan_done_frame = 0
        self.refresh()

    def _plan_panel(self):
        """The pinned plan, or None when there is nothing to show."""
        if not self.plan:
            return None
        g = self.ui.g
        body = Text()
        lines = [ln for ln in self.plan.splitlines() if ln.strip()]
        for line in lines:
            stripped = line.lstrip()
            if self.plan_done_frame or stripped.startswith("[x]"):
                body.append(f" {g.tick} ", style=GOOD)
                body.append(stripped[3:].strip() + "\n", style=f"{MUTED} strike")
            elif stripped.startswith("[>]"):
                body.append(f" {g.gear} ", style=f"bold {ACCENT}")
                body.append(stripped[3:].strip() + "\n", style=f"bold {ACCENT}")
            else:
                body.append("   ")
                body.append(stripped[3:].strip() + "\n" if stripped.startswith("[ ]")
                            else stripped + "\n", style=MUTED)

        body.rstrip()
        done = sum(1 for ln in lines if ln.lstrip().startswith("[x]"))
        total = len(lines)
        if self.plan_done_frame:
            done = total
        title = f"plan  {done}/{total}"
        # The completion frames pulse the border so the tick registers.
        border = GOOD if self.plan_done_frame else ACCENT
        if self.plan_done_frame and self.plan_done_frame % 2 == 0:
            border = ACCENT
        return Panel(body, title=title, title_align="left", border_style=border,
                     box=self.ui.box, padding=(0, 1))

    def set_lead(self, line: Text | None) -> None:
        """Show (or clear) the line of the answer currently being written."""
        self.lead = line
        self.refresh()

    def refresh(self) -> None:
        if self.ui.on_refresh is not None:
            # Chat layout: the bar lives in the pinned status row, so a
            # repaint is the pane's job rather than Live's.
            self.ui.on_refresh()
        if self._live is not None:
            # Nudge Live to repaint now. The renderable is self, so the
            # content is recomputed either way -- this only skips the wait
            # for the next scheduled refresh.
            self._live.refresh()

    def _renderable(self):
        """The pinned block: plan on top, then the line being written, then
        the status strip. Everything here is redrawn in place."""
        parts = []
        if (panel := self._plan_panel()) is not None:
            parts.append(panel)
        if self.lead is not None and self.lead.plain:
            parts.append(self.lead)
        parts.append(self._render())
        return parts[0] if len(parts) == 1 else Group(*parts)

    def __rich_console__(self, console, options):
        """Re-render on every refresh, not just when something calls update().

        Live was being handed the *result* of _renderable() -- a finished
        Text object. Auto-refresh then redrew that same frozen object twelve
        times a second, so the elapsed clock only moved when a token happened
        to arrive and call update(). A model that spends thirty seconds in
        prompt evaluation before its first token showed a stopped clock for
        all thirty of them. Handing Live the bar itself makes each refresh
        recompute.
        """
        yield self._renderable()

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
