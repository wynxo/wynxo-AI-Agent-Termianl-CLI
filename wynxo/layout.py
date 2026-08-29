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
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (Float, FloatContainer, HSplit, Layout,
                                   Window)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output.defaults import create_output

MIN_WIDTH = 20
MAX_SCROLLBACK = 20_000
"""Lines kept. Past this the oldest go; a terminal does the same."""


class Transcript:
    """The conversation, as lines of ANSI, with a rich Console writing in.

    The Console is the same kind the rest of wynxo already draws with, so
    panels, diffs, syntax highlighting and the pet all arrive here without
    any of that code knowing it is no longer writing to a terminal.
    """

    def __init__(self, width: int = 80):
        self._buffer = io.StringIO()
        self.lines: list[str] = []
        self.width = max(MIN_WIDTH, width)
        self.console = self._make_console()
        self.on_change: Callable[[int], None] | None = None

    def _make_console(self):
        from rich.console import Console

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
        # Lines already written keep the width they were wrapped at.
        # Rewrapping them would mean re-rendering panels and diffs from
        # source, which rich does not keep around -- and a terminal does no
        # better with its own scrollback.
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

    def __init__(self, *, completer=None, unicode: bool = True,
                 accent: str = "ansimagenta",
                 header: Callable[[], str] | None = None,
                 footer: Callable[[], str] | None = None,
                 overlay: Callable[[], list[str]] | None = None,
                 key_bindings: KeyBindings | None = None,
                 width: int | None = None, height: int | None = None):
        self.unicode = unicode
        self.accent = accent
        self._header = header or (lambda: "")
        self._footer = footer or (lambda: "")
        self._overlay = overlay or (lambda: [])
        self._extra_keys = key_bindings

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

    def composer_rows(self) -> int:
        """Rows the input needs right now: its own content, capped.

        Reported as an *exact* dimension by the window below, so the
        allocator has no slack to hand it. This is the number that used to
        be a range -- and a range is what let it inflate into a tall empty
        box whenever the conversation was short.
        """
        width, _ = self.size()
        room = max(1, width - 4)          # border, gutter, caret
        rows = 0
        for line in (self.buffer.text or "").split("\n"):
            rows += max(1, math.ceil(len(line) / room))
        return max(1, min(rows, self.COMPOSER_MAX_ROWS))

    def transcript_rows(self) -> int:
        """What the transcript is entitled to: everything left over."""
        _, height = self.size()
        fixed = (self.HEADER_ROWS + self.RULE_ROWS
                 + self.composer_rows() + self.FOOTER_ROWS)
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
        pad = ["" for _ in range(max(0, rows - len(lines)))]
        return ANSI("\n".join(pad + lines))

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
        return ANSI(self._footer())

    def _overlay_fragments(self):
        return ANSI("\n".join(self._overlay()))

    def _overlay_height(self) -> int:
        """The float's own height. It is *not* part of the vertical split, so
        whatever this returns cannot move the composer or the footer."""
        return max(0, min(len(self._overlay()), self.TODO_MAX_ROWS))

    def _composer_prefix(self, line_number: int, wrap_count: int):
        """The caret, on the first row only; continuation rows get a gutter
        of the same width so multi-line input stays aligned under it."""
        if line_number == 0 and wrap_count == 0:
            return [("class:caret", "❯ " if self.unicode else "> ")]
        return [("class:caret", "  ")]

    # -- input -------------------------------------------------------------

    def _accept(self, buffer: Buffer) -> bool:
        self._submitted.put_nowait(buffer.text)
        return False        # False: clear the buffer after accepting

    async def prompt_async(self, *_args, default: str = "", **_kwargs) -> str:
        """The next submitted line. Same shape as PromptSession.prompt_async."""
        if default:
            self.buffer.text = default
            self.buffer.cursor_position = len(default)
        self.focus_composer()
        return await self._submitted.get()

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

        @keys.add("pageup")
        def _(event):
            self.scroll_by(max(1, self.transcript_rows() - 1))

        @keys.add("pagedown")
        def _(event):
            self.scroll_by(-max(1, self.transcript_rows() - 1))

        @keys.add("escape", "end")
        def _(event):
            self.to_bottom()

        @keys.add("enter")
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
            height=Dimension.exact(self.HEADER_ROWS),
            style="class:header",
        )

        # TRANSCRIPT -- the one flexible child in the entire tree. Its max is
        # unbounded and its preferred is zero, so _divide_heights hands it
        # every row the fixed children did not claim.
        transcript = Window(
            content=FormattedTextControl(self._transcript_fragments,
                                         focusable=False),
            wrap_lines=False,          # rich already wrapped to the width
            height=Dimension(min=0, preferred=0, weight=1),
        )

        # RULE -- exact.
        rule = Window(
            content=FormattedTextControl(self._rule_fragments,
                                         focusable=False),
            height=Dimension.exact(self.RULE_ROWS),
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
            height=Dimension.exact(self.FOOTER_ROWS),
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
                      width=self.TODO_WIDTH, height=self._overlay_height,
                      content=Window(
                          content=FormattedTextControl(self._overlay_fragments,
                                                       focusable=False),
                          wrap_lines=False)),
                # The completion menu is a float too, so suggestions draw
                # over the transcript and reserve nothing. Reserving rows for
                # a menu is what put six blank rows under the old prompt.
                Float(xcursor=True, ycursor=True,
                      content=CompletionsMenu(max_height=8, scroll_offset=1)),
            ],
        )

        return Application(
            layout=Layout(root, focused_element=composer),
            key_bindings=self._keys(),
            full_screen=True,
            mouse_support=True,
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
