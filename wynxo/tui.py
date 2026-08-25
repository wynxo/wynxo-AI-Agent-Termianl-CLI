"""A chat-shaped terminal UI: transcript above, composer pinned at the bottom.

The classic REPL prints a prompt, you type, the answer scrolls out beneath
it, and a fresh prompt is drawn below that. It works, but it reads nothing
like the tools this is meant to feel like. Every chat application, and every
terminal agent worth the comparison, puts the composer at the bottom of the
screen and lets the conversation flow above it -- the input stays exactly
where your hands already are, and it stays there while the model is working
rather than vanishing for the duration of the turn.

The design that makes this tractable is that nothing about wynxo's rendering
changes. rich still draws every panel, diff, spinner and code block exactly
as before; it just draws into a buffer instead of onto the terminal, and the
resulting ANSI is what the transcript pane shows. So this file is a
container, not a second UI, and the two modes cannot drift apart.

Scrolling is deliberately simple: rich is given the terminal's width and
wraps as it always has, so a transcript line is already at most one screen
row and showing the last N of them is exactly right. No wrapping arithmetic
of our own, and nothing to get wrong when the window is resized.
"""

from __future__ import annotations

import asyncio
import io
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import ColorDepth
from rich.console import Console

MIN_WIDTH = 20
MAX_SCROLLBACK = 4_000
"""Transcript lines kept. Past this the oldest go, because the whole point
of holding them is to scroll back through a session, not to grow without
limit on a phone."""


class Transcript:
    """The conversation, as lines of ANSI, with a rich Console writing in.

    The Console is the same one the rest of wynxo already draws with, so
    panels, diffs, syntax highlighting and the pet all arrive here without
    any of that code knowing the difference.
    """

    def __init__(self, width: int = 80):
        self._buffer = io.StringIO()
        self.lines: list[str] = []
        self.width = max(MIN_WIDTH, width)
        self.console = self._make_console()
        self.on_change: Callable[[], None] | None = None

    def _make_console(self) -> Console:
        return Console(
            file=self._buffer,
            force_terminal=True,          # keep colour: it is going to a screen
            color_system="truecolor",
            highlight=False,
            soft_wrap=False,
            width=self.width,
            # Otherwise rich re-reads the real terminal and picks its height,
            # which has nothing to do with the pane it is drawing into.
            height=10_000,
        )

    def resize(self, width: int) -> None:
        width = max(MIN_WIDTH, width)
        if width == self.width:
            return
        self.width = width
        # Lines already written keep the width they were wrapped at. Rewrapping
        # them would mean re-rendering panels and diffs from source, which is
        # not something rich keeps around -- and a terminal does no better.
        self.console.width = width

    def drain(self) -> None:
        """Move whatever rich has written into the visible transcript."""
        text = self._buffer.getvalue()
        if not text:
            return
        self._buffer.seek(0)
        self._buffer.truncate(0)
        # A trailing newline means "the line ended", not "there is a blank
        # line after it" -- splitting naively would double every gap.
        pieces = text.split("\n")
        if pieces and pieces[-1] == "":
            pieces.pop()
        self.lines.extend(pieces)
        if len(self.lines) > MAX_SCROLLBACK:
            del self.lines[: len(self.lines) - MAX_SCROLLBACK]
        if self.on_change is not None:
            self.on_change()

    def visible(self, height: int, offset: int = 0) -> list[str]:
        """The slice of the transcript to show, newest at the bottom."""
        if height <= 0:
            return []
        end = len(self.lines) - max(0, offset)
        end = max(0, min(end, len(self.lines)))
        return self.lines[max(0, end - height):end]

    def max_offset(self, height: int) -> int:
        return max(0, len(self.lines) - height)

    def clear(self) -> None:
        self.lines.clear()
        if self.on_change is not None:
            self.on_change()


class ChatUI:
    """The full-screen layout: transcript, status strip, composer.

    The application runs continuously rather than once per prompt. That is
    what keeps the composer on screen while a turn is running -- the classic
    prompt has to be released before the agent can print anything, which is
    precisely why it disappears for the length of every answer.

    Submitted lines go onto a queue for the caller's worker to pick up, so
    typing the next message while the current one is still being answered
    does the obvious thing instead of being swallowed.
    """

    COMPOSER_ROWS = 3      # top border, the line you type on, bottom border
    STATUS_ROWS = 1

    def __init__(self, status: Callable[[], str] | None = None,
                 completer=None, on_interrupt: Callable[[], None] | None = None,
                 unicode: bool = True, accent: str = "ansimagenta",
                 width: int | None = None):
        # Started at the real width rather than a default: the banner is
        # drawn before the application has rendered once, and a rule wrapped
        # at 80 in a 120-column terminal stays that way for the session --
        # lines keep the width they were written at.
        self.transcript = Transcript(width or _terminal_width())
        self.transcript.on_change = self._changed
        self.submissions: asyncio.Queue[str] = asyncio.Queue()
        self.scroll = 0
        """Rows scrolled back from the bottom. Zero follows the newest."""
        self._status = status or (lambda: "")
        self._on_interrupt = on_interrupt
        self._unicode = unicode
        self._accent = accent
        self._closed = False

        self.buffer = Buffer(multiline=False, completer=completer,
                             complete_while_typing=True,
                             accept_handler=self._accept)
        self.app = self._build()

    # -- geometry ----------------------------------------------------------

    def size(self) -> tuple[int, int]:
        try:
            size = self.app.output.get_size()
            return max(MIN_WIDTH, size.columns), max(4, size.rows)
        except Exception:
            return 80, 24

    def transcript_rows(self) -> int:
        _, rows = self.size()
        return max(1, rows - self.COMPOSER_ROWS - self.STATUS_ROWS)

    # -- rendering ---------------------------------------------------------

    def _transcript_fragments(self):
        # Drained here rather than only when a caller remembers to flush:
        # every repaint goes through this, so anything rich has written is
        # on screen by definition and no write can be left stranded in the
        # buffer waiting for the next call that happens to flush.
        self.transcript.drain()
        width, _ = self.size()
        self.transcript.resize(width)
        rows = self.transcript_rows()
        # Clamped on every render rather than only when scrolling: the
        # transcript grows underneath, and a stale offset would drift the
        # view off the end of it.
        self.scroll = min(self.scroll, self.transcript.max_offset(rows))
        lines = self.transcript.visible(rows, self.scroll)
        # Padded at the top so a short conversation sits just above the
        # composer rather than stranded at the top of an empty screen. That
        # is where a chat window puts it, and it keeps the newest line in the
        # same place on screen whether there are three lines or three hundred.
        if len(lines) < rows:
            lines = [""] * (rows - len(lines)) + lines
        return ANSI("\n".join(lines))

    def _status_fragments(self):
        text = self._status()
        if self.scroll > 0:
            marker = "  ^ scrolled back -- End to follow again"
            text = f"{text}{marker}" if text else marker.strip()
        return ANSI(text)

    def _edge(self, top: bool):
        def render():
            width, _ = self.size()
            if self._unicode:
                left, right, bar = ("╭", "╮", "─") if top else ("╰", "╯", "─")
            else:
                left, right, bar = ("+", "+", "-")
            return [(f"class:edge", left + bar * max(0, width - 2) + right)]
        return render

    def _build(self) -> Application:
        transcript = Window(
            content=FormattedTextControl(self._transcript_fragments,
                                         focusable=False),
            wrap_lines=False,      # rich already wrapped to the exact width
            dont_extend_height=False,
        )
        status = Window(
            content=FormattedTextControl(self._status_fragments,
                                         focusable=False),
            height=self.STATUS_ROWS,
        )
        composer = Window(
            content=BufferControl(buffer=self.buffer,
                                  input_processors=[]),
            height=1,
            get_line_prefix=lambda *_: [("class:prompt", "│ > ")],
        )
        layout = Layout(HSplit([
            transcript,
            status,
            Window(content=FormattedTextControl(self._edge(True)), height=1),
            composer,
            Window(content=FormattedTextControl(self._edge(False)), height=1),
        ]), focused_element=composer)

        return Application(
            layout=layout,
            key_bindings=self._keys(),
            full_screen=True,
            mouse_support=False,
            color_depth=ColorDepth.TRUE_COLOR,
            erase_when_done=True,
        )

    # -- input -------------------------------------------------------------

    def _accept(self, buff: Buffer) -> bool:
        text = buff.text
        self.submissions.put_nowait(text)
        # False: do not keep the text in the buffer. The transcript is where
        # what you said belongs, and the composer should be empty and ready.
        return False

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()
        scrolling = Condition(lambda: True)

        @keys.add("c-c")
        def _(event):
            # Interrupts the turn rather than killing the app: the whole
            # reason the composer stays on screen is that the session
            # survives the answer being cut short.
            if self._on_interrupt is not None:
                self._on_interrupt()

        @keys.add("c-d")
        def _(event):
            if not self.buffer.text:
                self.submissions.put_nowait("/quit")

        @keys.add("pageup", filter=scrolling)
        def _(event):
            self.scroll = min(
                self.transcript.max_offset(self.transcript_rows()),
                self.scroll + max(1, self.transcript_rows() - 1))

        @keys.add("pagedown", filter=scrolling)
        def _(event):
            self.scroll = max(0, self.scroll - max(1, self.transcript_rows() - 1))

        @keys.add("end")
        def _(event):
            self.scroll = 0

        return keys

    # -- the api the repl uses ---------------------------------------------

    def _changed(self) -> None:
        if self.scroll == 0:
            return            # already following the bottom
        # Something new arrived while scrolled back. Hold position rather
        # than yanking the view down mid-read.

    def flush(self) -> None:
        """Publish anything rich has drawn, and repaint."""
        self.transcript.drain()
        self.invalidate()

    def invalidate(self) -> None:
        if self._closed:
            return
        try:
            self.app.invalidate()
        except Exception:
            pass

    async def next_message(self) -> str:
        return await self.submissions.get()

    async def repaint_loop(self, interval: float = 0.1) -> None:
        """Keep the screen live while a turn streams.

        Streaming writes land in the buffer between keystrokes, and without
        this the pane would only repaint when the user happened to press
        something -- so an answer would arrive in silent jumps.
        """
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                self.invalidate()
        except asyncio.CancelledError:
            pass

    def exit(self) -> None:
        self._closed = True
        try:
            self.app.exit()
        except Exception:
            pass


def render_to_ansi(renderable, width: int) -> str:
    """One rich renderable as a single line of ANSI, for the status row.

    Its own Console because the transcript's is mid-stream: writing the bar
    through that would interleave a repainting widget with the conversation
    it is supposed to sit beneath.
    """
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=True, color_system="truecolor",
                      highlight=False, soft_wrap=False,
                      width=max(MIN_WIDTH, width), height=4)
    try:
        console.print(renderable, end="")
    except Exception:
        return ""
    return sink.getvalue().split("\n")[0]


def _terminal_width(default: int = 80) -> int:
    import shutil

    try:
        return shutil.get_terminal_size((default, 24)).columns
    except (OSError, ValueError):
        return default


def usable() -> bool:
    """Whether this terminal can host the chat layout.

    It needs a real terminal on both ends -- something to draw on and
    something to read keys from -- and a TERM that admits to handling escape
    sequences. Anywhere else falls back to the scrolling prompt, which works
    on anything.
    """
    import os
    import sys

    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError, OSError):
        return False
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", "unknown"):
        return False
    if not term:
        return sys.platform == "win32"
    return True
