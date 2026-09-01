"""The application layout: header, transcript, composer, footer.

Until this existed the REPL was a ``PromptSession.prompt_async()`` -- a
readline prompt rendered inline at the cursor. Nothing owned the composer's
geometry, so nothing could keep it anywhere: prompt_toolkit drew it wherever
the cursor happened to be and then emitted newlines until its toolbar
reached the last row. The blank rows in between were the "floating,
half-open, oversized" box. It was not a styling problem; there was no
layout to style.

The contract, in the units the allocator actually uses:

    HEADER      fixed       exactly one row
    TRANSCRIPT  flexible    every spare row on the screen
    COMPOSER    natural     its own content, capped, then scrolls inside
    FOOTER      fixed       exactly one row

That contract is enforceable because of how ``HSplit._divide_heights``
works. It starts every child at its ``min``, grows each toward ``preferred``,
and then -- this is the part that matters -- keeps growing children toward
``max``, cycling by weight, until the screen is full. So *any* child whose
``max`` exceeds its ``preferred`` will absorb spare rows.

``Dimension(min=1, max=8)`` looks like "between one and eight rows". It is
really ``preferred=1, max=8``, which invites the allocator to inflate it to
eight the moment there is room -- a tall empty box under a short
conversation. Every fixed row here therefore reports ``Dimension.exact``
(min == preferred == max), and the composer reports ``exact`` of its *own
measured content height*. That leaves exactly one child in the whole tree
whose max exceeds its preferred -- the transcript -- so it is the only place
spare rows can land.

Overlays (the plan, the pet, toasts, the completion menu) are Floats. A
Float is drawn over its parent and reports no height at all, which is what
keeps them from moving the composer no matter how tall they get.
"""

from __future__ import annotations

import asyncio
import io
import math
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.layout import (ConditionalContainer, Float,
                                   FloatContainer, HSplit, Layout, Window)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output.defaults import create_output


class _Wheel(FormattedTextControl):
    """A read-only control that answers the scroll wheel.

    The transcript is not focusable -- focus belongs to the composer and
    stays there -- and prompt_toolkit routes mouse events to the control
    under the pointer rather than to the focused one, so the wheel has to be
    handled here or it does nothing at all.
    """

    def __init__(self, *args, on_scroll=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_scroll = on_scroll

    def mouse_handler(self, mouse_event):
        if self._on_scroll is None:
            return NotImplemented
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._on_scroll(3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._on_scroll(-3)
            return None
        # Everything else (clicks, drags) is left alone so it can fall
        # through to the terminal rather than being swallowed here.
        return NotImplemented

def _visible_width(text: str) -> int:
    """Cells a styled string occupies, ignoring its escape sequences."""
    import re

    from rich.cells import cell_len

    return cell_len(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text))


MIN_WIDTH = 20
MAX_SCROLLBACK = 20_000
"""Lines kept. Past this the oldest go; a terminal does the same."""


class _Sink(io.StringIO):
    """Where Rich writes, and the only door into the transcript.

    Rich has two output paths, and wrapping only one of them was a bug you
    could watch happen: ``Console.print`` went through the wrapper and
    appeared immediately, while the streamers -- which write to
    ``console.file`` directly, a character at a time, because that is how you
    colour a partial line -- accumulated here silently and did not become
    rows until some unrelated print flushed them. Streaming arrived in
    lumps.

    Draining on write instead of on print means there is one door rather
    than two, and no Rich code path can be added later that quietly misses
    it.
    """

    def __init__(self, on_write):
        super().__init__()
        self._on_write = on_write

    def write(self, text: str) -> int:
        written = super().write(text)
        self._on_write()
        return written


class Transcript:
    """The conversation, as lines of ANSI, with a rich Console writing in.

    The Console is the same kind the rest of wynxo already draws with, so
    panels, diffs, syntax highlighting and the pet all arrive here without
    any of that code knowing it is no longer writing to a terminal.
    """

    def __init__(self, width: int = 80):
        self._buffer = _Sink(self._sink_wrote)
        self._draining = False
        self.lines: list[str] = []
        self.width = max(MIN_WIDTH, width)
        self.console = self._make_console()
        self.on_change: Callable[[int], None] | None = None
        self.on_resize: Callable[[int], None] | None = None
        """Told the new width when the pane changes size. The UI drawing in
        here follows it: prompt_toolkit's application takes SIGWINCH for as
        long as it runs and never gives it back, so nothing else can hear a
        resize while the layout is up."""

    def _make_console(self):
        from .ui import SafeConsole

        return SafeConsole(
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
        # Lines already written keep the width they were wrapped at.
        # Rewrapping them would mean re-rendering panels and diffs from
        # source, which rich does not keep around -- and a terminal does no
        # better with its own scrollback.
        self.console.width = width
        if self.on_resize is not None:
            self.on_resize(width)

    def _sink_wrote(self) -> None:
        """Rich wrote something. Turn whole lines into rows straight away.

        A partial last line is left in the sink: a streamer writes a word at
        a time, and promoting a half-written line to a row would make every
        word its own row.
        """
        if self._draining:
            return          # a drain that prints would re-enter
        if "\n" in self._buffer.getvalue():
            self.drain()

    def drain(self) -> None:
        """Move whatever rich has written into the visible transcript."""
        text = self._buffer.getvalue()
        if not text:
            return
        # Only complete lines become rows; whatever follows the last newline
        # is a partial line still being written, and goes back in the sink.
        head, sep, tail = text.rpartition("\n")
        if not sep:
            return
        self._buffer.seek(0)
        self._buffer.truncate(0)
        self._draining = True
        try:
            if tail:
                self._buffer.write(tail)
        finally:
            self._draining = False
        # A trailing CR is a line ending, not content. Windows subprocess
        # output, a CRLF file being written and a model echoing one all
        # arrive as \r\n; splitting on \n alone left the \r on the end of
        # every row, and it survives into the rendered fragments as a literal
        # character -- where a terminal reads it as "return to column 0" and
        # the row is overdrawn by whatever comes next.
        pieces = [line[:-1] if line.endswith("\r") else line
                  for line in head.split("\n")]
        self.lines.extend(pieces)
        if len(self.lines) > MAX_SCROLLBACK:
            del self.lines[: len(self.lines) - MAX_SCROLLBACK]
        if self.on_change is not None:
            self.on_change(len(pieces))

    def visible(self, height: int, offset: int = 0) -> list[str]:
        """The slice to show, newest at the bottom."""
        if height <= 0:
            return []
        end = max(0, min(len(self.lines) - max(0, offset), len(self.lines)))
        return self.lines[max(0, end - height):end]

    def max_offset(self, height: int) -> int:
        return max(0, len(self.lines) - height)

    def clear(self) -> None:
        self.lines.clear()
        if self.on_change is not None:
            self.on_change(0)


class ChatLayout:
    """The full-screen application, and the geometry it guarantees.

    Exposes ``prompt_async()`` with the same shape ``PromptSession`` has, so
    the REPL loop reads one line at a time exactly as it always did. The
    difference is that the application keeps running between those calls --
    which is what lets the composer stay on screen while a turn is being
    answered, instead of being torn down and redrawn for every prompt.
    """

    HEADER_ROWS = 1
    FOOTER_ROWS = 1
    RULE_ROWS = 1
    """The thin separator between the transcript and the composer."""
    COMPOSER_MAX_ROWS = 8
    """Past this the composer scrolls internally rather than growing. Without
    a ceiling a pasted essay would push the footer off the bottom."""
    TODO_WIDTH = 36
    """Wide enough for the plan panel corner.py renders: its 34-cell inner
    width plus the two border columns. Narrower and the panel's right border
    is clipped off, which reads as a broken box rather than a narrow one.
    tests/test_layout_geometry.py pins the two together."""
    TODO_MAX_ROWS = 12
    OVERLAY_MAX_ROWS = 26
    """The overlay carries the live edit as well as the plan now, so it needs
    the rows a diff needs. Still a Float: however tall it gets it reports no
    height into the vertical split and cannot move the composer."""

    def __init__(self, *, completer=None, unicode: bool = True,
                 accent: str = "ansimagenta",
                 header: Callable[[], str] | None = None,
                 footer: Callable[[], str] | None = None,
                 overlay: Callable[[], list[str]] | None = None,
                 key_bindings: KeyBindings | None = None,
                 on_interrupt: Callable[[], None] | None = None,
                 width: int | None = None, height: int | None = None):
        self.unicode = unicode
        self.accent = accent
        self._header = header or (lambda: "")
        self._footer = footer or (lambda: "")
        self._overlay = overlay or (lambda: [])
        self._extra_keys = key_bindings
        self.on_interrupt = on_interrupt
        """What Ctrl-C does when no modal owns it. See the binding below --
        without one, Ctrl-C is inert for the whole session."""

        self._forced_size = (width, height)
        self.transcript = Transcript(width or 80)
        self.transcript.on_change = self._content_arrived

        self.scroll = 0
        """Rows scrolled back from the bottom. Zero means pinned to the
        newest output, which is when auto-follow is on."""
        self.unread = 0
        """Lines that arrived while scrolled back, for the jump-to-bottom
        hint. Reset the moment the view returns to the bottom."""

        self.buffer = Buffer(completer=completer, complete_while_typing=True,
                             multiline=True, accept_handler=self._accept)
        self._submitted: asyncio.Queue[str] = asyncio.Queue()

        # -- modals ---------------------------------------------------------
        # Everything that used to open a *second* prompt_toolkit Application
        # runs in this one instead. Two applications cannot share a terminal:
        # the second takes the output, and on exit leaves the first with a
        # screen state its renderer believes is already correct -- which is
        # why /model shredded the header, overwrote the composer and left the
        # footer behind.
        self._picker: dict | None = None
        """Title, choices, cursor and the future waiting on the answer."""
        self._ask: dict | None = None
        """A one-line question borrowing the composer. Its future takes the
        next submission, so the main loop cannot swallow the answer."""

        self.opening = True
        """Whether the transcript still holds only the welcome.

        It decides where short content sits. A conversation belongs at the
        bottom, against the composer, growing upward -- but the opening
        screen is not a conversation, and the same rule left the logo and
        the greeting pinned to the bottom under ten rows of nothing. Cleared
        by the first thing the user sends."""

        self.mouse_on = True
        """Mouse reporting. On, the wheel scrolls the transcript; off, the
        terminal does its own drag-select and copy. Toggled with F2."""

        self.app = self._build()

    # -- the size the layout is reasoning about ---------------------------

    def regions(self) -> dict[str, "Window"]:
        """Each structural region by name, for geometry assertions."""
        return {
            "header": self._header_window,
            "transcript": self._transcript_window,
            "rule": self._rule_window,
            "composer": self._composer_window,
            "footer": self._footer_window,
        }

    def size(self) -> tuple[int, int]:
        """(columns, rows). Forced values win, for tests at a fixed size."""
        width, height = self._forced_size
        if width and height:
            return width, height
        try:
            got = self.app.output.get_size()
            return (width or got.columns, height or got.rows)
        except Exception:
            return (width or 80, height or 24)

    # -- geometry ----------------------------------------------------------

    def header_rows(self) -> int:
        """One row, until the screen cannot spare it.

        Below a certain height the fixed furniture alone costs more rows than
        the terminal has, and a split whose minimums exceed its height is not
        a squeezed layout -- ``HSplit._divide_heights`` answers None and
        prompt_toolkit draws an entirely blank screen. So the furniture is
        shed, in the order it can be done without: the rule first (it only
        separates two things that are still adjacent), then the header, then
        the footer. The composer and one row of conversation are what is
        actually being used and are the last things to go.
        """
        _, height = self.size()
        return self.HEADER_ROWS if height >= 4 else 0

    def rule_rows(self) -> int:
        _, height = self.size()
        return self.RULE_ROWS if height >= 5 else 0

    def footer_rows(self) -> int:
        _, height = self.size()
        return self.FOOTER_ROWS if height >= 3 else 0

    def composer_rows(self) -> int:
        """Rows the input needs right now: its own content, capped.

        Reported as an *exact* dimension by the window below, so the
        allocator has no slack to hand it. This is the number that used to
        be a range -- and a range is what let it inflate into a tall empty
        box whenever the conversation was short.
        """
        width, height = self.size()
        room = max(1, width - 5)          # border, gutter, caret
        rows = 0
        for line in (self.buffer.text or "").split("\n"):
            rows += max(1, math.ceil(len(line) / room))
        # Never more than the screen can seat alongside the fixed rows. On a
        # short terminal a five-line paste asked for more rows than existed,
        # the sum of the split's minimums exceeded the height, and
        # _divide_heights answered None -- which prompt_toolkit renders as an
        # entirely blank screen. The composer gives way there; the
        # alternative is showing nothing at all.
        seatable = max(1, height - self.header_rows() - self.rule_rows()
                       - self.footer_rows())
        return max(1, min(rows, self.COMPOSER_MAX_ROWS, seatable))

    def transcript_rows(self) -> int:
        """What the transcript is entitled to: everything left over."""
        _, height = self.size()
        fixed = (self.header_rows() + self.rule_rows()
                 + self.composer_rows() + self.footer_rows())
        return max(0, height - fixed)

    # -- scrolling ---------------------------------------------------------

    def _content_arrived(self, added: int) -> None:
        """New output. Follow it, or hold the reader's place."""
        if self.scroll > 0:
            # Scrolled back: keep the same lines under the eye rather than
            # yanking to the newest. The slice is taken from the end, so the
            # offset has to grow by exactly what arrived.
            self.scroll += added
            self.unread += added
            self.scroll = min(self.scroll,
                              self.transcript.max_offset(self.transcript_rows()))
        self.invalidate()

    def scroll_by(self, rows: int) -> None:
        """Negative scrolls toward the newest, positive back through history."""
        ceiling = self.transcript.max_offset(self.transcript_rows())
        self.scroll = max(0, min(self.scroll + rows, ceiling))
        if self.scroll == 0:
            self.unread = 0
        self.invalidate()

    def to_bottom(self) -> None:
        self.scroll = 0
        self.unread = 0
        self.invalidate()

    def following(self) -> bool:
        return self.scroll == 0

    # -- content -----------------------------------------------------------

    def _transcript_fragments(self):
        # A resize changes how wide new output should be wrapped. The check
        # rides on the render because prompt_toolkit re-renders on resize
        # anyway, and a width that lags by a frame wraps a whole panel wrong.
        width, _ = self.size()
        self.transcript.resize(width)
        rows = self.transcript_rows()
        lines = self.transcript.visible(rows, self.scroll)
        # Top-padded, so a short conversation sits at the bottom of its
        # region against the composer rather than floating under the header.
        # The welcome is the exception: it is a title card, not the tail of
        # a conversation, and bottom-aligning it put the logo in the last
        # few rows under a screen of empty space.
        spare = max(0, rows - len(lines))
        above = spare // 2 if self.opening else spare
        return ANSI("\n".join([""] * above + lines + [""] * (spare - above)))

    def _header_fragments(self):
        return ANSI(self._header())

    def _rule_fragments(self):
        width, _ = self.size()
        bar = "─" if self.unicode else "-"
        if self.scroll > 0:
            tag = f" ↓ {self.unread} new " if self.unread else " ↓ more below "
            if not self.unicode:
                tag = tag.replace("↓", "v")
            keep = max(0, width - len(tag) - 2)
            return ANSI(f"\x1b[2m{bar * 2}\x1b[0m\x1b[1m{tag}\x1b[0m"
                        f"\x1b[2m{bar * keep}\x1b[0m")
        return ANSI(f"\x1b[2m{bar * width}\x1b[0m")

    def _footer_fragments(self):
        text = self._footer()
        # Shift-drag first, because it is the answer that needs no mode and
        # no wynxo-specific knowledge: every mainstream terminal -- xterm,
        # Windows Terminal, iTerm, GNOME Terminal, kitty -- bypasses mouse
        # reporting while shift is held and does its own selection. The hint
        # used to name only F2, so it taught a toggle nobody would guess at
        # for something the terminal already does.
        note = (" shift+drag to select  ·  F2 mouse off" if self.mouse_on
                else " drag to select  ·  F2 mouse on")
        width, _ = self.size()

        plain = _visible_width(text)
        if plain + len(note) + 1 <= width:
            text = text + " " * max(1, width - plain - len(note)) + note
        return ANSI(text)

    def _overlay_fragments(self):
        return ANSI("\n".join(self._overlay()))

    def _overlay_width(self) -> int:
        """Wide enough for whatever the overlay is currently holding, capped
        so it never crowds the conversation on a narrow terminal."""
        width, _ = self.size()
        widest = max((len(row) for row in self._overlay()), default=0)
        # The screen is the hard limit. TODO_WIDTH is a preference, and as a
        # floor it won a 20-column terminal outright -- a 36-column float on
        # a screen that has 20.
        room = max(8, width - 4)
        return min(room, max(min(self.TODO_WIDTH, room), min(widest + 2, room)))

    def _overlay_height(self) -> int:
        """The float's own height. It is *not* part of the vertical split, so
        whatever this returns cannot move the composer or the footer."""
        _, height = self.size()
        # Never more than half the screen: an overlay that covers the
        # conversation it is annotating has stopped being an overlay.
        room = max(1, min(height - 2, max(3, height // 2)))
        return max(0, min(len(self._overlay()), self.OVERLAY_MAX_ROWS, room))

    def _composer_prefix(self, line_number: int, wrap_count: int):
        """The caret, on the first row only; continuation rows get a gutter
        of the same width so multi-line input stays aligned under it."""
        if line_number == 0 and wrap_count == 0:
            if self._ask is not None:
                # The question replaces the caret, so it is obvious what the
                # composer is asking for rather than what it usually accepts.
                return [("class:caret", f"{self._ask['question']} ")]
            return [("class:caret", " ❯ " if self.unicode else " > ")]
        return [("class:caret", "   ")]

    # -- input -------------------------------------------------------------

    def _accept(self, buffer: Buffer) -> bool:
        # Whatever else this is, the welcome is over.
        self.opening = False
        if self._ask is not None and not self._ask["future"].done():
            # A question is borrowing the composer. Its answer belongs to
            # whoever asked, not to the main loop -- both await the same
            # composer, and a shared queue would hand it to whichever
            # happened to be waiting first.
            self._ask["future"].set_result(buffer.text)
        else:
            self._submitted.put_nowait(buffer.text)
        return False        # False: clear the buffer after accepting

    async def ask(self, question: str, default: str = "") -> str:
        """Ask for a line of text in the composer, inside this application."""
        loop = asyncio.get_running_loop()
        self._ask = {"question": question, "future": loop.create_future()}
        if default:
            self.buffer.text = default
            self.buffer.cursor_position = len(default)
        self.focus_composer()
        self.invalidate()
        try:
            return await self._ask["future"]
        finally:
            self._ask = None
            self.buffer.reset()
            self.invalidate()

    def cancel_ask(self) -> None:
        """Abandon the open question, as Ctrl-C.

        Raised into whoever is awaiting rather than returned as a value:
        every caller of ask() already has an ``except (EOFError,
        KeyboardInterrupt)`` written for exactly this, and the permission
        prompt turns it into Decision.ABORT. Those handlers were unreachable
        -- nothing under the layout ever raised -- so Ctrl-C at a blocking
        permission prompt did nothing at all, which is the moment somebody
        most wants a way out.

        EOFError rather than KeyboardInterrupt, though both are caught:
        KeyboardInterrupt is a BaseException, and asyncio does not contain
        those the way it contains ordinary exceptions -- setting one on a
        future propagates it out through the event loop itself, which tore
        down the whole application instead of the question. EOFError is also
        the more honest description: the answer is never coming.
        """
        ask = self._ask
        if ask is not None and not ask["future"].done():
            ask["future"].set_exception(EOFError("the question was abandoned"))

    async def pick(self, title: str, choices: list, default: int = 0):
        """An arrow-key chooser, drawn as an overlay in this application.

        Returns the chosen ``Choice.value``, or None when escaped.
        """
        loop = asyncio.get_running_loop()
        self._picker = {"title": title, "choices": choices,
                        "index": max(0, min(default, len(choices) - 1)),
                        "future": loop.create_future()}
        self.invalidate()
        try:
            return await self._picker["future"]
        finally:
            self._picker = None
            # Focus goes straight back to the composer: the point of closing
            # a picker is to carry on typing, not to hunt for the caret.
            self.focus_composer()
            self.invalidate()

    def picking(self) -> bool:
        return self._picker is not None

    def asking(self) -> bool:
        """Whether a question is currently borrowing the composer."""
        return self._ask is not None

    def _picker_fragments(self):
        picker = self._picker
        if picker is None:
            return []
        cursor = "\u276f" if self.unicode else ">"
        out: list[tuple[str, str]] = []
        if picker["title"]:
            out.append(("class:picker.title", f"  {picker['title']}\n"))
        width, height = self.size()
        room = max(3, min(len(picker["choices"]), height - 8))
        first = max(0, min(picker["index"] - room // 2,
                           len(picker["choices"]) - room))
        for offset, choice in enumerate(picker["choices"][first:first + room]):
            i = first + offset
            selected = i == picker["index"]
            out.append(("class:picker.cursor", f"  {cursor} " if selected else "    "))
            label = getattr(choice, "label", str(choice))
            style = "class:picker.selected" if selected else "class:picker.row"
            out.append((style, label))
            if getattr(choice, "badge", ""):
                out.append(("class:picker.badge", f"  {choice.badge}"))
            if getattr(choice, "hint", ""):
                out.append(("class:picker.hint", f"  {choice.hint}"))
            out.append(("", "\n"))
        out.append(("class:picker.hint",
                    "  up/down to move  enter to choose  esc to cancel"))
        return to_formatted_text(out)

    def _picker_height(self) -> int:
        if self._picker is None:
            return 0
        _, height = self.size()
        return min(len(self._picker["choices"]) + 2, max(5, height - 6))

    async def prompt_async(self, *_args, default: str = "", **_kwargs) -> str:
        """The next submitted line. Same shape as PromptSession.prompt_async."""
        if default:
            self.buffer.text = default
            self.buffer.cursor_position = len(default)
        self.focus_composer()
        return await self._submitted.get()

    def submit(self, text: str) -> None:
        """Hand the main loop a line nobody typed.

        Ctrl-C twice on an empty composer leaves through here as "/quit",
        so quitting runs the same shutdown an explicit /quit does rather
        than a second teardown path that has to be kept in step with it.
        """
        self._submitted.put_nowait(text)

    def focus_composer(self) -> None:
        """Put focus (and therefore the cursor) back in the input.

        Called after every turn, tool, error and resize. The composer is the
        only focusable window in the tree, so there is nowhere else for the
        cursor to end up -- but focus can be lost outright when a float
        closes, and typing into nothing is the worst failure mode there is.
        """
        try:
            self.app.layout.focus(self._composer_window)
        except Exception:
            pass

    def invalidate(self) -> None:
        try:
            self.app.invalidate()
        except Exception:
            pass

    # -- the tree ----------------------------------------------------------

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()
        picking = Condition(self.picking)
        typing = Condition(lambda: self._picker is None)

        @keys.add("up", filter=picking)
        def _(event):
            self._picker["index"] = max(0, self._picker["index"] - 1)

        @keys.add("down", filter=picking)
        def _(event):
            self._picker["index"] = min(len(self._picker["choices"]) - 1,
                                        self._picker["index"] + 1)

        @keys.add("enter", filter=picking)
        def _(event):
            picker = self._picker
            if not picker["future"].done():
                picker["future"].set_result(
                    picker["choices"][picker["index"]].value)

        @keys.add("escape", filter=picking)
        @keys.add("c-c", filter=picking)
        def _(event):
            if not self._picker["future"].done():
                self._picker["future"].set_result(None)

        asking = Condition(self.asking)

        @keys.add("c-c", filter=asking)
        @keys.add("escape", filter=asking)
        def _(event):
            self.cancel_ask()

        # Ctrl-C everywhere else. This application holds the terminal in
        # prompt_toolkit's raw mode, which clears ISIG -- so the driver never
        # raises SIGINT here and the handler the session installs before
        # every turn could not fire. With the two modal bindings above
        # filtered off, nothing claimed the key at all: Ctrl-C did nothing
        # while the agent worked, and nothing at the composer either, in the
        # layout the app starts by default and under a status bar that says
        # "^C stop". The owner decides what it means; the layout only has to
        # deliver it.
        #
        # Its own condition rather than `typing`: prompt_toolkit runs the
        # last binding whose filter passes, so a filter that is merely "no
        # picker" would win over the `asking` one above and steal Ctrl-C
        # from an open question.
        unclaimed = Condition(lambda: self._picker is None and self._ask is None)

        @keys.add("c-c", filter=unclaimed)
        def _(event):
            if self.on_interrupt is not None:
                self.on_interrupt()

        @keys.add("f2")
        def _(event):
            """Hand the mouse back to the terminal, or take it again.

            With reporting on, the wheel scrolls the transcript but the
            terminal never sees a drag, so click-and-drag selection does
            nothing. Off, selection and copy work exactly as they do in any
            other program. Neither is right all the time, so it is a toggle
            rather than a decision made once for everyone.
            """
            self.mouse_on = not self.mouse_on
            self.invalidate()

        @keys.add("pageup")
        def _(event):
            self.scroll_by(max(1, self.transcript_rows() - 1))

        @keys.add("pagedown")
        def _(event):
            self.scroll_by(-max(1, self.transcript_rows() - 1))

        @keys.add("escape", "end")
        def _(event):
            self.to_bottom()

        @keys.add("enter", filter=typing)
        def _(event):
            # Enter submits; Alt-Enter (below) inserts a newline. A composer
            # that needed a modifier to send would be wrong for a chat.
            event.current_buffer.validate_and_handle()

        @keys.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        if self._extra_keys is not None:
            from prompt_toolkit.key_binding import merge_key_bindings

            return merge_key_bindings([keys, self._extra_keys])
        return keys

    def _build(self) -> Application:
        # HEADER -- exact. Cannot grow, cannot absorb.
        header = Window(
            content=FormattedTextControl(self._header_fragments,
                                         focusable=False),
            height=lambda: Dimension.exact(self.header_rows()),
            style="class:header",
        )

        # TRANSCRIPT -- the one flexible child in the entire tree. Its max is
        # unbounded and its preferred is zero, so _divide_heights hands it
        # every row the fixed children did not claim.
        transcript = Window(
            content=_Wheel(self._transcript_fragments, focusable=False,
                           on_scroll=self.scroll_by),
            wrap_lines=False,          # rich already wrapped to the width
            height=Dimension(min=0, preferred=0, weight=1),
        )

        # RULE -- exact.
        rule = Window(
            content=FormattedTextControl(self._rule_fragments,
                                         focusable=False),
            height=lambda: Dimension.exact(self.rule_rows()),
        )

        # COMPOSER -- exact, of its own measured content. This is the whole
        # fix: a *range* here (min=1, max=8) reports preferred=1 and max=8,
        # and the allocator's third pass grows any such child toward its max
        # until the screen is full. That is the tall empty box. An exact
        # dimension has no slack to grow into.
        self._composer_control = BufferControl(buffer=self.buffer,
                                               focusable=True)
        composer = Window(
            content=self._composer_control,
            height=lambda: Dimension.exact(self.composer_rows()),
            wrap_lines=True,
            get_line_prefix=self._composer_prefix,
            style="class:composer",
        )
        self._composer_window = composer

        # FOOTER -- exact, and last, so it is on the bottom row by
        # construction rather than by being pushed there.
        footer = Window(
            content=FormattedTextControl(self._footer_fragments,
                                         focusable=False),
            height=lambda: Dimension.exact(self.footer_rows()),
            style="class:footer",
        )

        # Held by name so callers -- the geometry tests above all -- can
        # identify each region by identity rather than by guessing from a
        # style string. The transcript and the rule both carry no style,
        # so style is not an identifier.
        self._header_window = header
        self._transcript_window = transcript
        self._rule_window = rule
        self._footer_window = footer

        body = HSplit([header, transcript, rule, composer, footer])

        # Overlays are Floats: drawn over the transcript, reporting no height
        # into the split above. However tall the plan gets, the composer does
        # not move.
        root = FloatContainer(
            content=body,
            floats=[
                Float(top=1, right=0,
                      width=self._overlay_width, height=self._overlay_height,
                      content=Window(
                          content=FormattedTextControl(self._overlay_fragments,
                                                       focusable=False),
                          wrap_lines=False)),
                # The completion menu is a float too, so suggestions draw
                # over the transcript and reserve nothing. Reserving rows for
                # a menu is what put six blank rows under the old prompt.
                Float(xcursor=True, ycursor=True,
                      content=CompletionsMenu(max_height=8, scroll_offset=1)),
                # bottom=3 clears the composer (1), the footer (1) and the
                # rule (1). At 2 the picker's last row landed on the rule and
                # the two drew over each other.
                Float(bottom=3, left=2, right=2,
                      height=self._picker_height,
                      content=ConditionalContainer(
                          Window(content=FormattedTextControl(
                              self._picker_fragments, focusable=False),
                              style="class:picker"),
                          filter=Condition(self.picking))),
            ],
        )

        return Application(
            layout=Layout(root, focused_element=composer),
            key_bindings=self._keys(),
            full_screen=True,
            # A filter, not a flag. Mouse reporting is what lets the wheel
            # scroll the transcript, and also what stops the terminal from
            # ever seeing a drag -- which is why selection appeared broken.
            # F2 hands it back.
            mouse_support=Condition(lambda: self.mouse_on),
            erase_when_done=True,
            style=self._style(),
            output=create_output(stdout=None),
        )

    def _style(self):
        from prompt_toolkit.styles import Style

        return Style.from_dict({
            "header": "bold",
            "footer": f"bg:#1c1c1c {self.accent}",
            "caret": f"bold {self.accent}",
            "composer": "",
            "picker": "bg:#161616",
            "picker.title": f"bold {self.accent}",
            "picker.cursor": f"bold {self.accent}",
            "picker.selected": f"bold {self.accent}",
            "picker.row": "",
            "picker.badge": "#7f7f7f",
            "picker.hint": "#7f7f7f",
        })

    # -- running -----------------------------------------------------------

    def flush_to_terminal(self) -> None:
        """Write the transcript out to the real terminal and stop capturing.

        For the paths that never reach ``run()``: a failed connection, a
        wizard that bails, a crash during start-up. The banner and the error
        explaining what went wrong are already in the transcript by then, and
        without this they would be dropped on the floor -- the application
        that would have drawn them never started.
        """
        import sys

        self.transcript.drain()
        if self.transcript.lines:
            sys.stdout.write("\n".join(self.transcript.lines) + "\n")
            sys.stdout.flush()
        self.transcript.lines.clear()

    async def run(self) -> None:
        await self.app.run_async()

    def stop(self) -> None:
        try:
            if self.app.is_running:
                self.app.exit()
        except Exception:
            pass
