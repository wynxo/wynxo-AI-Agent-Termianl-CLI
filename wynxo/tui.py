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
import re
import string
import time
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (Float, FloatContainer, HSplit,
                                   Layout, Window)
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style, default_ui_style, merge_styles
from rich.console import Console

MIN_WIDTH = 20
MAX_SCROLLBACK = 4_000

DEFAULT_CHROME = {
    "edge": "ansibrightblack",
    "toast-ok": "ansibrightcyan",
    "toast-fail": "ansired",
    "pet": "",
    "todo": "",
    "todo-title": "bold ansibrightcyan",
    "ok": "ansigreen",
    "fail": "ansired",
    "active": "ansibrightcyan",
}
"""The prompt_toolkit style classes the live chrome draws with.

Swapped wholesale by apply_theme() so /theme repaints the borders, toasts,
plan panel and activity markers live instead of only the rich transcript.
"""

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
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
            # Notify after the cap so the callback sees the actual current
            # content length, including any discarded oldest rows.
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


class _HistoryAlreadyLoaded:
    """The stand-in for a history-loading task that will never run.

    Only what load_history_if_not_yet_loaded checks is implemented, and it
    only ever reports 'done'."""

    def done(self) -> bool:
        return True

    def result(self) -> None:
        return None

    def add_done_callback(self, _cb) -> None:
        pass

    def cancel(self) -> bool:
        return False


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
    ACTIVITY_ROWS = 1      # activity is rendered inside the fixed footer row
    TODO_WIDTH = 38        # fixed top-right todo panel width on wide screens
    TODO_MAX_ROWS = 10
    COMPOSER_ROWS = 3      # floor of the composer: top border, one input row, bottom border
    COMPOSER_MAX_ROWS = 6  # the composer's ceiling; beyond this it scrolls inside
    FOOTER_ROWS = 1        # the status strip is exactly one row, always
    ACTIVITY_MAX = 96      # prevent provider/tool text from flooding the chrome
    STATUS_ROWS = FOOTER_ROWS  # compatibility name; status is no longer variable-height

    # The layout contract, in the units the allocator actually uses:
    #
    #   OUTPUT   flexible  -- the transcript is the only child that may grow
    #   COMPOSER natural   -- its height is its content, capped at MAX
    #   FOOTER   fixed     -- exactly one row, whatever is happening
    #
    # prompt_toolkit's HSplit hands spare rows to any child whose reported
    # max exceeds its preferred size (containers.py _divide_heights cycles
    # children by weight toward max after preferred is met). So every fixed
    # row here reports an exact dimension -- min == preferred == max -- and
    # the composer's max is capped at its content height by
    # dont_extend_height. There is then no child left that can absorb spare
    # rows except the transcript, which is where they belong.

    def __init__(self, status: Callable[[], str] | None = None,
                 completer=None, on_interrupt: Callable[[], None] | None = None,
                 on_thinking: Callable[[], None] | None = None,
                 on_tools: Callable[[], None] | None = None,
                 on_dictate: Callable[[], None] | None = None,
                 unicode: bool = True, accent: str = "ansimagenta",
                 width: int | None = None,
                 chrome: dict[str, str] | None = None,
                 header: Callable[[], str] | None = None,
                 pet_state: Callable[[], str] | None = None,
                 pet_enabled: Callable[[], bool] | None = None,
                 pet_animate: Callable[[], bool] | None = None):
        # Started at the real width rather than a default: the banner is
        # drawn before the application has rendered once, and a rule wrapped
        # at 80 in a 120-column terminal stays that way for the session --
        # lines keep the width they were written at.
        self.transcript = Transcript(width or _terminal_width())
        self.transcript.on_change = self._changed
        self._scroll_content_length = 0
        self.submissions: asyncio.Queue[str] = asyncio.Queue()
        self.scroll = 0
        """Rows scrolled back from the bottom. Zero follows the newest."""
        self._status = status or (lambda: "")
        self._header = header or (lambda: "")
        self.activity = ""
        self.activity_ok = True
        self.activity_started = 0.0
        self.activity_pulse = 0
        self._on_interrupt = on_interrupt
        # Ctrl-O and Ctrl-T used to reach the session only through a
        # KeyWatcher thread reading the tty behind this application's back.
        # Two readers of one terminal means each byte goes to whichever wins
        # the race, so the keys worked intermittently and stole characters
        # out of the composer while they were at it. Bound here instead.
        self._on_thinking = on_thinking
        self._on_tools = on_tools
        # Ctrl-R: one speech-to-text round. The session itself lives in the
        # CLI; the layout only needs to know whom to poke and to keep the
        # composer focused while it runs.
        self._on_dictate = on_dictate
        self._unicode = unicode
        self._accent = accent
        self._chrome = {**DEFAULT_CHROME, **(chrome or {})}
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
        self.todo_rendered = ""
        self.todo_title = ""
        self.todo_frame = 0
        self.todo_mode = "expanded"
        """expanded | compact | hidden -- the plan panel's collapse state."""
        self._todo_last_tick = 0.0
        """Rows the pinned block took last time it was drawn."""
        self._pet_state = pet_state
        self._pet_enabled = pet_enabled
        self._pet_animate = pet_animate
        self._pet_frame = 0
        self._last_pet_tick = time.monotonic()
        self._last_activity_tick = self._last_pet_tick
        self._toast: tuple[str, float] | None = None
        self._toast_life = 3.5
        self._toast_queue: list[tuple[str, bool]] = []
        self._float_budget: int | None = None
        """Rows the top-right block may use, set by the float's height
        measurement each render. None until a layout has measured it, so
        direct calls (tests, previews) render everything."""
        """Seconds a notification stays before it clears itself."""
        self.typed: "asyncio.Future[str] | None" = None
        """Set while a line of free text is being read, by prompt()."""
        self.default = ""
        """The answer a bare enter takes, where the question names one."""

        # Keep the composer single-line semantically (Enter submits; a
        # newline needs Alt-Enter), but render it as a wrapped viewport.
        # prompt_toolkit then scrolls horizontally/vertically to the cursor
        # instead of letting long input vanish behind the edge. History is
        # kept per session so Up/Down walk what you already sent.
        self.buffer = Buffer(multiline=False, completer=completer,
                             complete_while_typing=False,
                             history=InMemoryHistory(),
                             accept_handler=self._accept)
        # prompt_toolkit loads history through a background task that needs
        # the running application's event loop. InMemoryHistory has nothing
        # to load, and this layout is regularly built before any loop exists
        # (tests, previews, measuring the layout) -- so mark it loaded and
        # keep create_content loop-free. Accepted lines still append to the
        # history through the buffer's own accept path.
        self.buffer._load_history_task = _HistoryAlreadyLoaded()
        self.app = self._build()

    # -- geometry ----------------------------------------------------------

    def _raw_size(self) -> tuple[int, int]:
        """Read terminal dimensions without announcing a resize callback."""
        if self.app.is_running:
            try:
                size = self.app.output.get_size()
                return max(MIN_WIDTH, size.columns), max(4, size.rows)
            except Exception:
                pass
        return max(MIN_WIDTH, _terminal_width()), max(4, _terminal_height())

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

    def composer_content_rows(self, width: int | None = None) -> int:
        """How many rows the input actually needs, right now.

        Measured from the BufferControl itself, with the same width,
        wrapping and line-prefix the real render will use -- so what the
        composer is allocated is exactly what its content needs, never a
        constant guessed at and never a row of blank space inside the box.
        Capped at COMPOSER_MAX_ROWS; past that the BufferControl scrolls
        internally and keeps the caret in view.
        """
        if width is None:
            width, _ = self.size()
        try:
            rows = self._composer_control.preferred_height(
                max(MIN_WIDTH, width), self.COMPOSER_MAX_ROWS,
                True, self._composer_line_prefix)
        except Exception:
            rows = 1
        return max(1, min(self.COMPOSER_MAX_ROWS, rows or 1))

    def composer_frame_rows(self) -> int:
        """The composer block's natural height: content plus its borders.

        In a very short terminal the furniture must still fit. The input
        control is capped to the rows left after the header, footer and one
        readable output row; prompt_toolkit's small-window fallback is worse
        than showing fewer composer lines.
        """
        _, rows = self.size()
        available = max(1, rows - self.HEADER_ROWS - self.FOOTER_ROWS - 1 - 2)
        return min(self.composer_content_rows(), available) + 2

    def _composer_line_prefix(self, *_):
        return [("class:prompt", self._composer_prefix())]

    def status_rows(self) -> int:
        """Compatibility accessor for the former variable-height status.

        The footer is intentionally fixed now, so this always returns one.
        """
        return self.FOOTER_ROWS

    def transcript_rows(self) -> int:
        """Rows left for the conversation: everything the fixed furniture
        and the composer's current natural height do not take."""
        _, rows = self.size()
        return max(1, rows - self.HEADER_ROWS - self.FOOTER_ROWS
                   - self.composer_frame_rows())

    # -- rendering ---------------------------------------------------------

    def _transcript_fragments(self):
        # Drained here rather than only when a caller remembers to flush:
        # every repaint goes through this, so anything rich has written is
        # on screen by definition and no write can be left stranded in the
        # buffer waiting for the next call that happens to flush.
        self._drain_transcript()

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

    def _pet_lines(self) -> list[str]:
        """The living pet scene, or [] when it is off.

        Rendered from the mood the CLI reports and the motion scene that
        mood maps to, advanced one frame per repaint -- the same render
        that already animates the pet's face, so no extra timer exists.
        """
        if self._pet_state is None or self._pet_enabled is None \
                or self._pet_animate is None:
            return []
        try:
            if not self._pet_enabled():
                return []
            mood = self._pet_state()
        except Exception:
            return []
        if not mood:
            return []
        from .motion import scene_for, select
        scene = scene_for(mood)
        frames = select(scene, unicode=self._unicode, width=30,
                        reduced=not self._pet_animate())
        now = time.monotonic()
        # Repaint cadence is not guaranteed (a quiet terminal may repaint
        # irregularly), so advance on elapsed time rather than on render count.
        # This keeps animation speed stable without creating another scheduler.
        if self._pet_animate():
            elapsed_steps = int((now - self._last_pet_tick) / 0.12)
            self._pet_frame += max(1, elapsed_steps)
            if elapsed_steps > 0:
                self._last_pet_tick += elapsed_steps * 0.12
        index = (self._pet_frame // 2) % len(frames) if len(frames) > 1 else 0
        return [line.rstrip() for line in frames[index].split("\n")
                if line.strip()]

    def _toast_line(self) -> str:
        """The current notification, or "" once it has lived its life."""
        if not self._toast:
            return ""
        text, at = self._toast
        if time.monotonic() - at > self._toast_life:
            self._toast = None
            if self._toast_queue:
                next_text, next_ok = self._toast_queue.pop(0)
                self._toast_ok = next_ok
                self._toast = (next_text, time.monotonic())
                return next_text
            return ""
        return text

    def notify(self, text: str, ok: bool = True) -> None:
        """Queue a transient notification in the floating chrome.

        Notifications are deliberately not written to the transcript: they
        are useful now, but must not consume history or alter composer geometry.
        """
        if not text:
            return
        if self._toast is not None:
            self._toast_queue.append((text, ok))
            return
        self._toast = (text, time.monotonic())
        self._toast_ok = ok
        self.invalidate()

    def set_todo_mode(self, mode: str) -> None:
        """expanded | compact | hidden. Hidden removes the panel but keeps
        the pet; the composer and transcript are untouched either way."""
        if mode in ("expanded", "compact", "hidden"):
            self.todo_mode = mode
            self.invalidate()

    def _todo_fragments(self):
        """The top-right block: toast, pet, then the plan panel.

        It is a float, so it reserves nothing: no matter how much is drawn
        here, the composer keeps its row, the transcript keeps its rows,
        and a busy panel cannot push anything around.

        Every entry is a (style, text) fragment. prompt_toolkit treats a
        bare list as StyleAndTextTuples and unpacks each element into
        (style, text, *rest), so a plain string would have its first
        character read as a color -- the ``✓`` in a toast crashed the whole
        renderer with "Wrong color format '✓'".

        When the float's measured budget is smaller than the block, the
        panel falls back to its one-line form rather than letting the
        window clip a box with its bottom edge missing.
        """
        budget = self._float_budget
        out: list[tuple[str, str]] = []
        if toast := self._toast_line():
            out.append(("class:toast-ok" if getattr(self, "_toast_ok", True)
                        else "class:toast-fail", toast))
        if self.todo_mode != "hidden":
            room = None if budget is None else max(0, budget - len(out))
            for line in self._pet_lines()[:room]:
                out.append(("class:pet", line))
        if self.todo_mode in ("expanded", "compact") and (
                budget is None or len(out) < budget):
            panel = self._todo_panel()
            room = None if budget is None else max(0, budget - len(out))
            if budget is not None and self.todo_mode == "expanded" \
                    and len(panel) > room:
                panel = self._todo_compact_line()
            out.extend(panel[:room])
        return out

    def _todo_panel(self) -> list[tuple[str, str]]:
        """The plan panel: a bordered box, or the one-line compact form.

        Each row comes back as (style, text) fragments -- markers get their
        own color (green done, red failed, accent active), everything else
        stays in the default body style.
        """
        summary = self._todo_summary()
        if summary is None:
            return []
        title, summary_text = summary

        if self.todo_mode == "compact":
            return self._todo_compact_line()

        inner = self.TODO_WIDTH - 2
        edge = "─" if self._unicode else "-"
        if self._unicode:
            bar, tl, tr, bl, br = "│", "╭", "╮", "╰", "╯"
        else:
            bar, tl, tr, bl, br = "|", "+", "+", "+", "+"
        head = f" ✦ {title} · {summary_text} "
        top = tl + edge + head + edge * max(0, inner - len(head) - 2) + tr
        out: list[tuple[str, str]] = [("class:edge", top)]
        capped = [line for line in self.todo_rendered.splitlines()
                  if line.strip()][: self.TODO_MAX_ROWS - 2]
        for line in capped:
            stripped = line.strip()
            if stripped.startswith("[x]"):
                marker, label, mclass = "✓", stripped[3:].strip(), "class:ok"
            elif stripped.startswith("[!]"):
                marker, label, mclass = "✕", stripped[3:].strip(), "class:fail"
            elif stripped.startswith("[>]"):
                marker = ("✧", "⋆", "✦", "♡")[self.todo_frame % 4]
                label, mclass = stripped[3:].strip(), "class:active"
            else:
                marker, label, mclass = "·", stripped, "class:todo"
            # One fragment per row: the flat fragment list must map 1:1 to
            # visual lines so the float's wrapped window and any test can
            # trust the shape. The whole row takes the marker's class -- a
            # done row reads green, a failed one red, the active one accent.
            row = f" {marker} {label}"
            pad = max(0, inner - len(row) - 2)
            out.append((mclass, bar + row + " " * pad + bar))
        out.append(("class:edge", bl + edge * max(0, inner - 2) + br))
        return out

    def _float_height(self) -> int:
        """How tall the top-right block may be, bounded by the terminal.

        On a short screen the block shrinks instead of overflowing: floats
        overlay, so a block taller than the window would draw over the
        composer, which is the one thing this panel must never do. The
        budget is remembered so _todo_fragments can trim its content to it
        -- the window clips whatever overflows, and a panel whose bottom
        border was cut off reads as a rendering bug.
        """
        _, rows = self.size()
        height = 1
        if self._toast:
            height += 1
        if self.todo_mode != "hidden":
            height += 4          # the pet scene block
        if self.todo_rendered:
            height += 1 if self.todo_mode == "compact" else self.TODO_MAX_ROWS
        budget = max(3, min(height, max(3, rows // 2)))
        self._float_budget = budget
        return budget

    def _todo_summary(self) -> tuple[str, str] | None:
        """(title, "done/total · N failed") for the rendered plan, or None."""
        lines = [line for line in self.todo_rendered.splitlines() if line.strip()]
        if not lines:
            return None
        capped = lines[: self.TODO_MAX_ROWS - 2]
        done = sum(line.lstrip().startswith("[x]") for line in capped)
        failed = sum(line.lstrip().startswith("[!]") for line in capped)
        title = self.todo_title or "plan"
        summary = f"{done}/{len(capped)}" + (f" · {failed} failed" if failed else "")
        return title, summary

    def _todo_compact_line(self) -> list[tuple[str, str]]:
        """The one-line panel form, used directly when the full box cannot
        fit the measured float budget."""
        summary = self._todo_summary()
        if summary is None:
            return []
        title, text = summary
        return [("class:todo-title", f"✦ {title} · {text}")]

    def set_todos(self, rendered: str, title: str = "") -> None:
        self.todo_rendered = rendered or ""
        self.todo_title = title or ""
        self.todo_frame += 1
        self.invalidate()

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
        """Compatibility alias for integrations that rendered the old status.

        It intentionally returns the new fixed-height footer content; the old
        multi-row status contract is gone so status changes cannot reflow the
        composer.
        """
        return self._footer_fragments()

    def apply_theme(self, chrome: dict[str, str] | None) -> None:
        """Repaint the live chrome (borders, toasts, plan, activity) with a
        new palette, without rebuilding the application."""
        if chrome:
            self._chrome = {**DEFAULT_CHROME, **chrome}
        self.app.style = merge_styles([
            Style.from_dict(self._chrome),
            default_ui_style(),
        ])
        self.invalidate()

    def set_activity(self, text: str, ok: bool = True) -> None:
        """Update the single bounded activity strip without changing layout.

        The activity row is intentionally stateful: while a command is active
        it gets a tiny pulse, while completed work gets a stable check/cross.
        It is still one row, so activity cannot reflow the transcript or input.
        """
        self.activity = " ".join(str(text).splitlines()).strip()[:self.ACTIVITY_MAX]
        self.activity_ok = ok
        self.activity_started = time.monotonic()
        self.activity_pulse = 0
        self.invalidate()

    def _activity_fragments(self):
        width, _ = self.size()
        text = self.activity
        if len(text) > width - 4:
            text = text[: max(0, width - 7)] + "..."
        return ANSI(text)

    def _footer_fragments(self):
        """The one status row, under the composer.

        Fixed height by construction: whatever arrives here is flattened to
        a single line, so no state change -- a stage, a tool result, thinking
        toggling, the plan growing -- can ever move the composer or resize
        the conversation above it. A longer history of what happened lives
        in the transcript and in /log, not in this row.
        """
        text = " ".join(self._status().splitlines()).strip()
        if self.activity:
            # The live bar already shows the current tool during a turn; only
            # prepend the strip when the status does not already carry it, or
            # the footer would say the same thing twice.
            plain = _ANSI_RE.sub("", text)
            if not plain.startswith(self.activity[:24]):
                text = self.activity + (f"   {text}" if text else "")
        if self.scroll > 0:
            marker = "^ scrolled back -- End to follow again"
            text = f"{text}   {marker}" if text else marker
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
        footer = Window(
            content=FormattedTextControl(self._footer_fragments,
                                         focusable=False),
            height=1,               # exact: min == preferred == max == 1
            dont_extend_height=True,
        )
        composer_control =        self._composer_control = BufferControl(
            buffer=self.buffer, input_processors=[])

        composer = Window(
            content=self._composer_control,
            # No explicit preferred: with preferred left unset the Window
            # derives it from the BufferControl's real wrapped content, and
            # dont_extend_height caps max at that same figure. The composer
            # therefore grows exactly as the input needs -- one row empty,
            # more for wrapped or multi-line input -- and the allocator has
            # no slack row it could ever hand this window. Past
            # COMPOSER_MAX_ROWS the control scrolls internally and keeps the
            # caret visible.
            height=Dimension(min=1, max=self.COMPOSER_MAX_ROWS),
            wrap_lines=True,
            dont_extend_height=True,
            get_line_prefix=self._composer_line_prefix,
        )
        # Natural height, not a fixed one: the frame is exactly its content
        # (borders plus whatever the input needs this frame), so it can
        # neither collapse the input into one row nor swell into the blank
        # box that a max > preferred dimension invites the allocator to fill.
        composer_frame = HSplit([
            Window(content=FormattedTextControl(self._edge(True)), height=1,
                   dont_extend_height=True),
            composer,
            Window(content=FormattedTextControl(self._edge(False)), height=1,
                   dont_extend_height=True),
        ], height=lambda: Dimension.exact(self.composer_frame_rows()))
        header = FloatContainer(
            content=Window(content=FormattedTextControl(self._header_fragments),
                           height=1),
            floats=[Float(
                right=0,
                top=0,
                width=self.TODO_WIDTH,
                height=self._float_height,
                content=Window(content=FormattedTextControl(self._todo_fragments),
                               wrap_lines=True),
            )],
        )
        body = HSplit([
            header,
            Window(content=FormattedTextControl(self._rule_fragments),
                   height=1, dont_extend_height=True),
            # Activity is rendered inside the footer's fixed row; keeping it
            # out of the HSplit preserves the established five-child layout
            # and guarantees it can never consume transcript/composer space.
            # The only flexible child: every spare row the screen has ends
            # up here, because no other child reports a max above its
            # preferred size. Streaming, thinking, tool activity and status
            # changes all render inside fixed or content-driven rows and
            # cannot move anything.
            transcript,
            composer_frame,
            footer,
        ], height=Dimension(min=0, preferred=0, weight=1))

        # The completer had nowhere to draw. A Buffer with a completer set
        # will happily compute suggestions and show none of them unless the
        # layout contains a menu to float over it -- which is why /mo… stopped
        # offering /model the moment the composer moved into this layout.
        # CompletionsMenu is a float: it draws over the transcript and
        # reserves nothing.
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
            # The top-right block's classes. Merged with the default UI
            # style so the composer, cursor and menus keep their normal
            # appearance. Accent-colored by default; /theme minimal maps
            # these to plain styles instead.
            style=merge_styles([
                Style.from_dict(self._chrome),
                default_ui_style(),
            ]),
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

        # Alt-Enter puts a newline in the composer instead of submitting.
        # prompt_toolkit represents Alt-Enter as escape followed by Enter in
        # the terminal stream; binding the sequence directly keeps ordinary
        # Escape available for cancelling completion/pickers.
        @keys.add("escape", "enter")
        def _(event):
            self.buffer.insert_text("\n")

        # Ctrl-R: one speech-to-text round. The transcript lands in the
        # composer to be reviewed and sent by hand.
        @keys.add("c-r")
        def _(event):
            if self._on_dictate is not None:
                self._on_dictate()

        return keys

    # -- focus -------------------------------------------------------------

    def refocus(self) -> None:
        """Put the caret back in the composer.

        Nothing in this layout is meant to hold focus for long: a turn
        ending, a tool finishing, a question being answered or a speech
        transcription landing should all leave the user typing, not hunting
        for the input with a mouse. Idempotent when already focused.
        """
        if self._closed:
            return
        try:
            for window in self.app.layout.find_all_windows():
                if window.content.__class__ is BufferControl:
                    self.app.layout.focus(window)
                    return
        except Exception:
            return

    # -- layout diagnostics --------------------------------------------------

    def layout_report(self) -> str:
        """Where the vertical space is going, one row per component.

        Developer-only, but cheap enough to keep: the composer bugs of the
        past all came from guessing which widget held which rows, and this
        is the answer measured from the real layout objects at the real
        size, not from the constants.
        """
        width, rows = self.size()
        frame = self.composer_frame_rows()
        content = self.composer_content_rows(width)
        output = self.transcript_rows()
        return "\n".join([
            f"root      {rows}x{width}",
            f"header    {self.HEADER_ROWS} (identity + rule)",
            f"output    {output}",
            f"composer  {frame} (content {content} + 2 borders, cap {self.COMPOSER_MAX_ROWS})",
            f"footer    {self.FOOTER_ROWS}",
            f"check     {self.HEADER_ROWS + output + frame + self.FOOTER_ROWS} <= {rows}",
        ])

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

    def _drain_transcript(self) -> None:
        """Drain rich output while preserving the user's scroll anchor."""
        previous = len(self.transcript.lines)
        self.transcript.drain()
        current = len(self.transcript.lines)
        if self.scroll > 0 and current > previous:
            self.scroll += current - previous
        self._scroll_content_length = current

    def _changed(self) -> None:
        # Transcript.drain invokes this after appending. The actual anchor
        # adjustment is performed by _drain_transcript, where the old length
        # is still available; this callback remains for external clear/change
        # notifications and repaint hooks.
        self._scroll_content_length = len(self.transcript.lines)

    def flush(self) -> None:
        """Publish anything rich has drawn, and repaint."""
        self._drain_transcript()
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
        self.question = ""
        self.answers = {}
        self.default = ""
        for future in (self.answer, self.picked, self.typed):
            if future is not None and not future.done():
                future.cancel()
        self.answer = self.picked = self.typed = None
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


MIN_ROWS = (ChatUI.HEADER_ROWS + ChatUI.COMPOSER_ROWS
            + ChatUI.FOOTER_ROWS + 2)
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
