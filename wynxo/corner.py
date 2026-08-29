"""The plan, pinned to the top-right corner while the transcript scrolls.

The plan used to ride along inside the bottom live region, which meant it
sat between the answer and the composer and moved every time either of them
changed size. Pinning it to a corner needs somewhere on the screen that
scrolling cannot reach, and a terminal has exactly one such mechanism that
does not involve taking the whole screen over: DECSTBM, the scrolling
region.

``\\x1b[{top};{bottom}r`` tells the terminal to confine scrolling to those
rows. Everything above ``top`` then stays where it is no matter how much
output goes by, so the panel can be drawn there once and repainted only when
it changes. The terminal's own scrollback, mouse selection and copy all keep
working inside the region, which is the whole reason for doing it this way
rather than with a full-screen application.

Three things make this safe to leave switched on:

* It is only armed on a real terminal that is tall enough to spare the rows.
  A short window keeps every row for the conversation.
* The region is released in a ``finally``, and again from an ``atexit`` hook.
  A terminal left with a region set is a terminal where the top rows never
  scroll again, which outlives the process and is not something to risk on a
  crash path.
* Every write is one ``\\x1b7 ... \\x1b8`` pair -- save cursor, paint, put it
  back -- so it can happen between any two lines of output without the
  writer noticing.
"""

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass

from .platforms import is_dumb_terminal, terminal_height, terminal_width

MIN_HEIGHT = 16
"""Below this the panel is not worth the rows it costs."""

MIN_WIDTH = 60
"""Narrower than this and the panel would crowd the transcript it sits
beside rather than sitting out of its way."""

MAX_ITEMS = 8
"""Longer plans are summarised rather than drawn in full: the corner is for
knowing where you are, not for reading the whole list."""


@dataclass
class Item:
    text: str
    state: str          # "done", "active" or "todo"


def parse(rendered: str) -> list[Item]:
    """The todo lines, as items. Anything unrecognised is a plain step."""
    items: list[Item] = []
    for line in rendered.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[x]"):
            items.append(Item(stripped[3:].strip(), "done"))
        elif stripped.startswith("[>]"):
            items.append(Item(stripped[3:].strip(), "active"))
        elif stripped.startswith("[ ]"):
            items.append(Item(stripped[3:].strip(), "todo"))
        else:
            items.append(Item(stripped, "todo"))
    return items


class CornerPlan:
    """Owns the reserved rows and what is painted into them."""

    def __init__(self, ui, width: int = 34):
        self.ui = ui
        self.width = width
        self.items: list[Item] = []
        self.armed = False
        self._rows = 0
        self._pulse = 0
        """Counts up while the plan is completing, so the border can flash."""
        self._released = True

    # -- capability --------------------------------------------------------

    def usable(self) -> bool:
        """Whether this terminal can spare the rows and honour the region."""
        if is_dumb_terminal() or os.environ.get("WYNXO_NO_CORNER"):
            return False
        return (terminal_height() >= MIN_HEIGHT
                and terminal_width() >= MIN_WIDTH)

    # -- the reserved region ----------------------------------------------

    def _write(self, text: str) -> None:
        try:
            self.ui.console.file.write(text)
            self.ui.console.file.flush()
        except (OSError, ValueError):
            # The stream went away (a closed pipe, a torn-down test). Losing
            # the panel is not worth taking the turn down with it.
            self.armed = False

    def arm(self) -> None:
        """Reserve the top rows and start painting there."""
        if self.armed or not self.usable():
            return
        rows = self.panel_height()
        if rows <= 0:
            return
        self._rows = rows
        height = terminal_height()
        # Scrolling is confined to everything below the panel. The cursor is
        # then parked inside that region, because a cursor left above it
        # would write into the reserved rows on the very next line.
        self._write(f"\x1b[{rows + 1};{height}r\x1b[{height};1H")
        self.armed = True
        self._released = False
        atexit.register(self.release)

    def release(self) -> None:
        """Give the whole screen back to scrolling.

        Idempotent, and safe to call from atexit after the console has gone:
        a terminal left with a scrolling region set keeps that region after
        wynxo exits, which would break the user's shell.
        """
        if self._released:
            return
        self._released = True
        self.armed = False
        try:
            self.ui.console.file.write("\x1b[r")
            self.ui.console.file.flush()
        except (OSError, ValueError):
            pass

    def panel_height(self) -> int:
        """Rows the panel needs, given what is in it."""
        if not self.items:
            return 0
        shown = min(len(self.items), MAX_ITEMS)
        extra = 1 if len(self.items) > MAX_ITEMS else 0
        # + the titled top rule, the progress bar, and the bottom rule. This
        # must match len(lines()) exactly: it decides how many rows are held
        # out of the scroll, and a panel taller than its region would have
        # its last row scroll away with the transcript.
        return shown + extra + 3

    # -- painting ----------------------------------------------------------

    def set(self, rendered: str) -> None:
        """Replace the plan. Re-arms when the number of rows changed."""
        items = parse(rendered)
        if items == self.items:
            return
        was = self._rows
        self.items = items
        if not items:
            self.clear()
            return
        if not self.armed or self.panel_height() != was:
            # A different number of rows means a different region.
            self.release()
            self.arm()
        self.paint()

    def clear(self) -> None:
        """Take the panel down and hand the rows back."""
        if not self.armed:
            self.items = []
            return
        rows = self._rows
        self.release()
        # Wipe what was drawn, or it stays on screen as a frozen ghost with
        # nothing left to repaint it.
        blank = " " * (self.width + 2)
        out = ["\x1b7"]
        left = max(1, terminal_width() - self.width - 1)
        for row in range(1, rows + 1):
            out.append(f"\x1b[{row};{left}H{blank}")
        out.append("\x1b8")
        self._write("".join(out))
        self.items = []
        self._rows = 0

    def pulse(self, frame: int) -> None:
        """Border emphasis while the plan finishes."""
        self._pulse = frame
        self.paint()

    SPARKLES = "·✧✦✧"
    """Frames for the mark beside a finished plan. Cycled by the pulse, so
    the panel visibly celebrates for a beat instead of just going quiet."""

    def progress(self) -> str:
        """A filled bar for the fraction of the plan that is done.

        The count in the title says the same thing, but a bar says it without
        being read -- which is the point of a panel you glance at rather than
        study.
        """
        if not self.items:
            return ""
        done = sum(1 for i in self.items if i.state == "done")
        span = self.width - 2
        filled = round(span * done / len(self.items))
        if self.ui.g.unicode:
            return "━" * filled + "─" * (span - filled)
        return "=" * filled + "-" * (span - filled)

    def lines(self) -> list[str]:
        """The panel's text, already fitted to its width and styled."""
        from .ui import ACCENT, FAINT, GOOD, MUTED

        g = self.ui.g
        inner = self.width
        done = sum(1 for i in self.items if i.state == "done")
        total = len(self.items)
        finishing = bool(self._pulse)
        border = GOOD if finishing and self._pulse % 2 else ACCENT

        def rule(left: str, right: str, label: str = "") -> str:
            if label:
                bar = g.hbar * max(0, inner - len(label) - 3)
                return (f"[{border}]{left}{g.hbar}[/] [bold {ACCENT}]{label}[/] "
                        f"[{border}]{bar}{right}[/]")
            return f"[{border}]{left}{g.hbar * inner}{right}[/]"

        mark = ""
        if finishing:
            mark = " " + self.SPARKLES[self._pulse % len(self.SPARKLES)] \
                if g.unicode else " *"
        out = [rule(g.tl, g.tr, f"PLAN {done}/{total}{mark}")]

        # The bar sits directly under the title, where the eye lands first.
        filled_style = GOOD if done == total and total else ACCENT
        out.append(f"[{border}]{g.vbar}[/] [{filled_style}]{self.progress()}[/] "
                   f"[{border}]{g.vbar}[/]")

        shown = self.items[:MAX_ITEMS]
        for item in shown:
            if item.state == "done" or finishing:
                glyph, style = g.tick, f"{MUTED} strike"
            elif item.state == "active":
                glyph, style = g.gear, f"bold {ACCENT}"
            else:
                glyph, style = g.dot, MUTED
            room = inner - 4
            text = item.text if len(item.text) <= room else item.text[:room - 1] + g.ellipsis
            pad = " " * (room - len(text))
            out.append(f"[{border}]{g.vbar}[/] [{style}]{glyph} {text}[/]{pad} "
                       f"[{border}]{g.vbar}[/]")
        if len(self.items) > MAX_ITEMS:
            rest = f"+{len(self.items) - MAX_ITEMS} more"
            pad = " " * (inner - len(rest) - 3)
            out.append(f"[{border}]{g.vbar}[/] [{FAINT}]{rest}[/]{pad} "
                       f"[{border}]{g.vbar}[/]")
        out.append(rule(g.bl, g.br))
        return out

    def paint(self) -> None:
        """Draw the panel into the reserved rows, leaving the cursor put."""
        if not self.armed or not self.items:
            return
        from rich.text import Text

        left = max(1, terminal_width() - self.width - 1)
        chunks = ["\x1b7"]
        for row, markup in enumerate(self.lines(), start=1):
            segments = self.ui.console.render(Text.from_markup(markup))
            painted = "".join(
                segment.text if segment.style is None
                else segment.style.render(segment.text)
                for segment in segments if segment.text != "\n")
            chunks.append(f"\x1b[{row};{left}H{painted}\x1b[0m")
        chunks.append("\x1b8")
        self._write("".join(chunks))

    def repaint_after_resize(self) -> None:
        """The region is measured in absolute rows, so a resize invalidates
        it. Re-arm against the new size rather than leaving a region that
        points at rows the window no longer has."""
        if not self.armed:
            return
        self.release()
        self.arm()
        self.paint()


def panel_lines(rendered: str, ui) -> list[str]:
    """The plan as ANSI lines, for the chat layout's overlay Float.

    The corner panel paints itself with absolute cursor moves, which is the
    right mechanism inside a scrolling terminal and the wrong one inside a
    full-screen application -- there the plan is a Float, and a Float wants
    text, not positioning. Same rendering either way, so the two views of the
    plan cannot drift apart.
    """
    from rich.text import Text

    panel = CornerPlan(ui)
    panel.items = parse(rendered)
    if not panel.items:
        return []
    out = []
    for markup in panel.lines():
        segments = ui.console.render(Text.from_markup(markup))
        out.append("".join(
            segment.text if segment.style is None
            else segment.style.render(segment.text)
            for segment in segments if segment.text != "\n"))
    return out
