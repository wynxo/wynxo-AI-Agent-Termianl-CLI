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
import string
import time
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (Float, FloatContainer, HSplit,
                                   Layout, Window)
from prompt_toolkit.layout.menus import CompletionsMenu
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


def _output():
    """The screen to draw on, or a stand-in when there is no screen.

    Constructing an Application builds the platform's output object there
    and then, and on Windows that means opening a console handle. Without
    one -- a CI runner, a service, anything started by pythonw -- it raises
    NoConsoleScreenBufferError from the constructor, so merely *building*
    the layout was fatal.

    A stand-in keeps the object constructible everywhere. Nothing is drawn
    through it, which is correct: with no console there is nothing to draw
    on, and usable() has already sent a real session down the scrolling
    path.
    """
    try:
        from prompt_toolkit.output.defaults import create_output

        return create_output()
    except Exception:
        from prompt_toolkit.output import DummyOutput

        return DummyOutput()


_ANSWER_KEYS = string.ascii_letters + string.digits
"""The keys a one-press answer can be. Everything else a question might
receive -- backspace, the arrows, Ctrl-C -- belongs to the composer."""


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

    HEADER_ROWS = 2        # the identity line, and a rule under it
    COMPOSER_ROWS = 3      # top border, the line you type on, bottom border
    STATUS_ROWS = 1        # the floor: the activity bar on its own
    MAX_STATUS_ROWS = 14
    """The ceiling. The pinned block grows to fit a plan and the line being
    written, but never so far that there is no conversation left to read."""

    def __init__(self, status: Callable[[], str] | None = None,
                 completer=None, on_interrupt: Callable[[], None] | None = None,
                 on_thinking: Callable[[], None] | None = None,
                 on_tools: Callable[[], None] | None = None,
                 unicode: bool = True, accent: str = "ansimagenta",
                 width: int | None = None,
                 header: Callable[[], str] | None = None):
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
        self._header = header or (lambda: "")
        self._on_interrupt = on_interrupt
        # Ctrl-O and Ctrl-T used to reach the session only through a
        # KeyWatcher thread reading the tty behind this application's back.
        # Two readers of one terminal means each byte goes to whichever wins
        # the race, so the keys worked intermittently and stole characters
        # out of the composer while they were at it. Bound here instead.
        self._on_thinking = on_thinking
        self._on_tools = on_tools
        self._unicode = unicode
        self._accent = accent
        self._closed = False
        self.question = ""
        self.answers: dict[str, str] = {}
        self.answer: "asyncio.Future[str] | None" = None
        self.picker: dict | None = None
        self.picked: "asyncio.Future[str | None] | None" = None
        self.on_resize: "Callable[[int], None] | None" = None
        """Told the new width when the window changes, so whatever wraps
        text for this pane can be told too."""
        self._last_width = 0
        self._status_lines = 1
        """Rows the pinned block took last time it was drawn."""
        self.typed: "asyncio.Future[str] | None" = None
        """Set while a line of free text is being read, by prompt()."""
        self.default = ""
        """The answer a bare enter takes, where the question names one."""

        # Keep the composer single-line semantically, but render it as a
        # wrapped viewport. prompt_toolkit then scrolls horizontally/vertically
        # to the cursor instead of letting long input vanish behind the edge.
        self.buffer = Buffer(multiline=False, completer=completer,
                             complete_while_typing=True,
                             accept_handler=self._accept)
        self.app = self._build()

    # -- geometry ----------------------------------------------------------

    def size(self) -> tuple[int, int]:
        """The screen, in columns and rows.

        Asked of the application only while it is actually running. Reading
        `app.output` before that builds the platform's output object, and on
        Windows that means opening a console handle -- which a test runner
        or a CI job does not have. Off the terminal size otherwise, which
        every platform answers without side effects.
        """
        if self.app.is_running:
            try:
                size = self.app.output.get_size()
                return self._measured(max(MIN_WIDTH, size.columns),
                                      max(4, size.rows))
            except Exception:
                pass
        return self._measured(max(MIN_WIDTH, _terminal_width()),
                              max(4, _terminal_height()))

    def _measured(self, width: int, rows: int) -> tuple[int, int]:
        """Announce a width that has changed since the last measurement.

        Everything drawn into the transcript is wrapped by rich before it
        gets here, at whatever width the UI was told about -- and the pane
        does not wrap, it truncates. So a window made narrower mid-session
        cut the right-hand end off every line written after it until the
        session was restarted.
        """
        if width != self._last_width:
            self._last_width = width
            if self.on_resize is not None:
                self.on_resize(width)
        return width, rows

    def status_rows(self) -> int:
        """How many rows the pinned block needs, as of the last repaint.

        The previous frame's height rather than this one's: prompt_toolkit
        settles the layout before it asks for content, and re-rendering the
        bar here to measure it would advance its spinner twice per frame.
        One frame of lag at ten frames a second is not visible.
        """
        _, rows = self.size()
        room = max(1, rows - self.HEADER_ROWS - self.COMPOSER_ROWS - 3)
        return max(self.STATUS_ROWS,
                   min(self._status_lines, self.MAX_STATUS_ROWS, room))

    def transcript_rows(self) -> int:
        _, rows = self.size()
        return max(1, rows - self.HEADER_ROWS - self.COMPOSER_ROWS
                   - self.status_rows())

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
        if picker := self._picker_lines(width):
            # Picker rows are rendered in the transcript pane, but must never
            # push the newest conversation rows out of the pane. Reserve their
            # rows from the visible slice instead of appending and truncating
            # the combined list (which used to hide the newest messages).
            available = max(0, rows - len(picker))
            lines = self.transcript.visible(available, self.scroll)
            lines = [""] * max(0, available - len(lines)) + lines + picker
        if len(lines) < rows:
            lines = [""] * (rows - len(lines)) + lines
        return ANSI("\n".join(lines[-rows:]))

    def _header_fragments(self):
        """The identity line, kept on screen.

        It used to be the first thing printed into the conversation, which
        meant it scrolled away after a page and the one line saying which
        model and which project you are talking to was gone for the rest of
        the session.
        """
        return ANSI(self._header())

    def _rule_fragments(self):
        width, _ = self.size()
        bar = "─" if self._unicode else "-"
        return [("class:edge", bar * max(0, width))]

    def _status_fragments(self):
        text = self._status()
        if self.scroll > 0:
            marker = "  ^ scrolled back -- End to follow again"
            text = f"{text}\n{marker}" if text else marker.strip()
        # Keep the status content bounded before prompt_toolkit lays out the
        # screen. Otherwise a verbose status/plan can consume the entire
        # viewport and leave the composer no room to render.
        _, rows = self.size()
        room = max(1, rows - self.HEADER_ROWS - self.COMPOSER_ROWS - 1)
        status_lines = text.splitlines() if text else [""]
        status_lines = status_lines[-min(self.MAX_STATUS_ROWS, room):]
        text = "\n".join(status_lines)
        self._status_lines = len(status_lines)
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
            height=lambda: self.status_rows(),
        )
        composer_control = BufferControl(buffer=self.buffer,
                                          input_processors=[])
        composer = Window(
            content=composer_control,
            height=1,
            wrap_lines=True,
            dont_extend_height=False,
            get_line_prefix=lambda *_: [("class:prompt", self._composer_prefix())],
        )
        # The composer is a fixed bottom block. A one-line Window lets long
        # input disappear past the right edge; a three-line block with a
        # scrolling BufferControl keeps the caret and the newest text visible
        # while leaving the bottom border immovable.
        composer_frame = HSplit([
            Window(content=FormattedTextControl(self._edge(True)), height=1),
            composer,
            Window(content=FormattedTextControl(self._edge(False)), height=1),
        ])
        body = HSplit([
            Window(content=FormattedTextControl(self._header_fragments),
                   height=1),
            Window(content=FormattedTextControl(self._rule_fragments),
                   height=1),
            transcript,
            status,
            composer_frame,
        ])

        # The completer had nowhere to draw. A Buffer with a completer set
        # will happily compute suggestions and show none of them unless the
        # layout contains a menu to float over it -- which is why /mo… stopped
        # offering /model the moment the composer moved into this layout.
        layout = Layout(
            FloatContainer(
                content=body,
                floats=[Float(xcursor=True, ycursor=True,
                              content=CompletionsMenu(max_height=8,
                                                      scroll_offset=1))],
            ),
            focused_element=composer,
        )

        return Application(
            layout=layout,
            key_bindings=self._keys(),
            full_screen=True,
            mouse_support=False,
            color_depth=ColorDepth.TRUE_COLOR,
            erase_when_done=True,
            output=_output(),
        )

    # -- input -------------------------------------------------------------

    def _composer_prefix(self) -> str:
        """What sits to the left of the cursor: the usual caret, or the
        question currently waiting for an answer."""
        if self.asking or self.typing:
            return f"│ {self.question} "
        return "│ > "

    def _accept(self, buff: Buffer) -> bool:
        text = buff.text
        if self.typing:
            self._resolve_typed(text.strip())
            return False
        if self.asking:
            # Typed out in full and entered, rather than answered with one
            # key. Matched on the first letter so "yes" and "y" agree.
            chosen = text.strip().lower()
            for key in self.answers:
                if chosen == key or (chosen and chosen[0] == key):
                    self._resolve(key)
                    return False
            # Nothing typed, and the question named a safe answer for that:
            # enter takes it. Questions that must not be answered by a
            # reflex -- granting a permission above all -- name no default,
            # and enter does nothing.
            if not chosen and self.default:
                self._resolve(self.default)
            return False
        self.submissions.put_nowait(text)
        # False: do not keep the text in the buffer. The transcript is where
        # what you said belongs, and the composer should be empty and ready.
        return False

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()
        scrolling = Condition(lambda: True)

        asking = Condition(lambda: self.asking)

        def answer_or_type(event) -> None:
            # A single key answers only from an empty composer. Once there
            # is text in it you are writing a sentence, not answering: a
            # question offering [a]lways would otherwise be granted by the
            # "a" in "hello again", and silently granting a permission is
            # the worst thing this could get wrong.
            if self.buffer.text:
                self.buffer.insert_text(event.data)
                return
            key = str(event.data).lower()
            if key in self.answers:
                self._resolve(key)
            else:
                # Not an answer, so it is the beginning of a typed one.
                self.buffer.insert_text(event.data)

        # Bound one letter at a time rather than as <any>. <any> matches
        # every key press there is, and marked eager it won hands down over
        # every other binding -- so with a question up, backspace inserted a
        # literal "^?", Ctrl-C inserted "^C", the arrows did nothing, and a
        # typo could not be corrected or the question escaped. The prompt
        # read "[y] yes  [a] always  [n] no  [q] stop: hello^?^?^C" and the
        # only way out was to kill the process.
        #
        # Naming the keys that can actually be answers leaves everything
        # else to the composer's ordinary editing bindings, which is where
        # it belonged.
        for character in _ANSWER_KEYS:
            keys.add(character, filter=asking, eager=True)(answer_or_type)

        picking = Condition(lambda: self.picking)

        @keys.add("up", filter=picking, eager=True)
        def _(event):
            count = len(self.picker["options"])
            self.picker["index"] = (self.picker["index"] - 1) % count

        @keys.add("down", filter=picking, eager=True)
        def _(event):
            count = len(self.picker["options"])
            self.picker["index"] = (self.picker["index"] + 1) % count

        @keys.add("enter", filter=picking, eager=True)
        def _(event):
            if self.picked is not None and not self.picked.done():
                index = self.picker["index"]
                option = self.picker["options"][index]
                # A row may carry a third element: what the caller gets back,
                # when that is not the text on screen. /resume shows "2h ago"
                # and needs a session id.
                self.picked.set_result(option[2] if len(option) > 2
                                       else option[0])

        @keys.add("escape", filter=picking, eager=True)
        def _(event):
            if self.picked is not None and not self.picked.done():
                self.picked.set_result(None)

        @keys.add("c-c")
        def _(event):
            if self.picking:
                if not self.picked.done():
                    self.picked.set_result(None)
                return
            if self.typing:
                self._resolve_typed("")     # cancelled, so nothing typed
                return
            if self.asking:
                # "stop" where the question offers it, and otherwise an
                # answer no branch matches, which every caller reads as
                # abort. Either way Ctrl-C gets you out.
                self._resolve("q" if "q" in self.answers else "")
                return
            # Interrupts the turn rather than killing the app: the whole
            # reason the composer stays on screen is that the session
            # survives the answer being cut short.
            if self._on_interrupt is not None:
                self._on_interrupt()

        @keys.add("c-o")
        def _(event):
            """Show or hide the model's thinking, at the prompt or mid-turn."""
            if self._on_thinking is not None:
                self._on_thinking()

        @keys.add("c-t")
        def _(event):
            """Full tool output, or the summary."""
            if self._on_tools is not None:
                self._on_tools()

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

    # -- choosing, with arrows, without a second application ---------------

    async def choose(self, title: str, options: list[tuple],
                     current: str = "") -> str | None:
        """An arrow-key picker drawn at the foot of the conversation.

        The standalone picker is its own prompt_toolkit application, and one
        cannot run inside another -- so in this layout every settings command
        would have degraded to printing a table, which is the behaviour this
        project has twice been asked to stop doing. Drawn here instead, with
        the same keys.

        Returns the chosen value, or None if the user pressed escape.
        """
        if not options:
            return None
        self.picker = {
            "title": title,
            "options": options,
            "index": max(0, next((i for i, option in enumerate(options)
                                  if option[0] == current), 0)),
        }
        self.picked = asyncio.get_event_loop().create_future()
        self.invalidate()
        try:
            return await self.picked
        finally:
            self.picker = None
            self.picked = None
            self.invalidate()

    @property
    def picking(self) -> bool:
        return self.picked is not None and not self.picked.done()

    def _picker_lines(self, width: int) -> list[str]:
        """The open picker, with the highlighted row alive.

        The selected row cycles through the sweep while it sits there, so
        moving down the list is something you watch rather than something
        you infer from a moved caret. Everything else stays dim, which is
        what makes the moving one read as selected.
        """
        picker = self.picker
        if not picker:
            return []
        dim, reset = "\x1b[38;5;247m", "\x1b[0m"
        mark = "❯" if self._unicode else ">"
        phase = int(time.monotonic() * 12)

        title = _rgb(_SWEEP[phase % len(_SWEEP)])
        lines = [f"{title}  {picker['title']}{reset}"]
        for i, option in enumerate(picker["options"]):
            name, hint = option[0], option[1]
            if i == picker["index"]:
                # Offset per character so the colour runs along the word.
                lit = "".join(
                    f"{_rgb(_SWEEP[(phase + n) % len(_SWEEP)])}{ch}"
                    for n, ch in enumerate(f"{mark} {name}")
                )
                body = f"{lit}{reset}"
                if hint:
                    body += f"  {dim}{hint}{reset}"
            else:
                body = f"  {dim}{name}{reset}"
                if hint:
                    body += f"  {dim}{hint}{reset}"
            lines.append("  " + body)
        lines.append(f"{dim}  arrows move  ·  enter chooses  ·  esc cancels{reset}")
        return lines

    # -- asking a question without a second application --------------------

    async def ask(self, question: str, answers: dict[str, str],
                  default: str = "") -> str:
        """Put a question in the composer row and wait for the answer.

        A second prompt_toolkit application cannot run inside this one --
        attempting it leaves the layout half-drawn, the border gone and the
        question typed over the composer, which is exactly what the first
        version did. So the question is asked through the composer that is
        already here: single keys answer immediately, and anything typed and
        entered is matched too.
        """
        self.question = question
        self.answers = answers
        self.default = default
        self.answer = asyncio.get_event_loop().create_future()
        self.invalidate()
        try:
            return await self.answer
        except asyncio.CancelledError:
            # The session is going away, not the user declining. Answered as
            # "stop" so the turn unwinds, then re-raised so shutdown carries
            # on unwinding too.
            if self.answer is not None and not self.answer.done():
                self.answer.cancel()
            raise
        finally:
            self.question = ""
            self.answers = {}
            self.default = ""
            self.answer = None
            self.invalidate()

    async def prompt(self, question: str, default: str = "") -> str:
        """Read a line of free text, in the composer that is already here.

        Same reason as ask(): editing a commit message through a second
        prompt_toolkit application tore the layout apart. The default is put
        in the composer to be edited rather than described, which is what
        makes it a starting point instead of a thing to retype.
        """
        self.question = question
        self.typed = asyncio.get_event_loop().create_future()
        self.buffer.text = default
        self.buffer.cursor_position = len(default)
        self.invalidate()
        try:
            return await self.typed
        except asyncio.CancelledError:
            if self.typed is not None and not self.typed.done():
                self.typed.cancel()
            raise
        finally:
            self.question = ""
            self.typed = None
            self.buffer.text = ""
            self.invalidate()

    def _resolve(self, key: str) -> None:
        if self.answer is None or self.answer.done():
            return
        self.answer.set_result(key)

    def _resolve_typed(self, text: str) -> None:
        if self.typed is None or self.typed.done():
            return
        self.typed.set_result(text)

    @property
    def asking(self) -> bool:
        return self.answer is not None and not self.answer.done()

    @property
    def typing(self) -> bool:
        """Waiting for a line of free text rather than for an answer."""
        return self.typed is not None and not self.typed.done()

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


def render_to_ansi(renderable, width: int, max_rows: int = 1) -> str:
    """A rich renderable as ANSI, for the pinned rows.

    Its own Console because the transcript's is mid-stream: writing the bar
    through that would interleave a repainting widget with the conversation
    it is supposed to sit beneath.

    max_rows matters. The header is one line and always will be, but the
    pinned block below the conversation is the activity bar *and* whatever
    sits above it -- the plan, the line currently being written. Keeping
    only the first line meant that with a plan up, the pinned row showed the
    panel's top border and nothing else: no items, and no activity bar
    either, for as long as the plan lived.
    """
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=True, color_system="truecolor",
                      highlight=False, soft_wrap=False,
                      width=max(MIN_WIDTH, width), height=max(4, max_rows + 2))
    try:
        console.print(renderable, end="")
    except Exception:
        return ""
    lines = sink.getvalue().split("\n")
    if len(lines) <= max_rows:
        return "\n".join(lines)
    # Keep the end: the activity bar is the last line of the block, and it
    # is the part that must never be pushed off.
    return "\n".join(lines[-max_rows:])


# The same pink-through-violet sweep the logo uses. Imported lazily to keep
# tui.py independent of the logo module, which imports asciiart in turn.
_SWEEP = [
    (255, 120, 200), (255,  96, 190), (246,  74, 186), (228,  64, 190),
    (204,  62, 200), (176,  70, 214), (150,  84, 226), (132, 104, 236),
    (150,  84, 226), (176,  70, 214), (204,  62, 200), (228,  64, 190),
    (246,  74, 186), (255,  96, 190), (255, 120, 200), (255, 150, 205),
]


def _rgb(colour: tuple[int, int, int]) -> str:
    r, g, b = colour
    return f"\x1b[38;2;{r};{g};{b}m"


def _terminal_width(default: int = 80) -> int:
    import shutil

    try:
        return shutil.get_terminal_size((default, 24)).columns
    except (OSError, ValueError):
        return default


def _terminal_height(default: int = 24) -> int:
    import shutil

    try:
        return shutil.get_terminal_size((80, default)).lines
    except (OSError, ValueError):
        return default


MIN_ROWS = ChatUI.HEADER_ROWS + ChatUI.COMPOSER_ROWS + ChatUI.STATUS_ROWS + 2
"""The furniture, plus two rows of conversation worth reading. Below this
prompt_toolkit gives up and draws "Window too small..." instead of the
layout, which is not something to leave a user staring at."""


def usable() -> bool:
    """Whether this terminal can host the chat layout.

    It needs a real terminal on both ends -- something to draw on and
    something to read keys from -- and a TERM that admits to handling escape
    sequences. Anywhere else falls back to the scrolling prompt, which works
    on anything.

    Height counts too. Header, status row and composer are pinned, so a very
    short window (a split pane, a phone in landscape) has nothing left for
    the conversation: at five rows the whole screen was replaced by
    prompt_toolkit's "Window too small...". The scrolling prompt has no
    such floor, so that is where a small window goes.
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
    if _terminal_height() < MIN_ROWS:
        return False
    if not term:
        return sys.platform == "win32"
    return True
