"""The application's visual shell: header, rail, cards, status bar.

One place for the chrome, so the interface is a single application rather
than a pile of widgets that happen to share a terminal. Everything here is
a renderable -- nothing prints, nothing reaches for the console -- which is
what lets the home screen compose the same pieces the live transcript uses
one at a time.

Three rules the whole shell keeps:

* One line of border, or none. A panel earns its outline by holding
  something you would otherwise mistake for the transcript; the navigation
  rail, the suggestions and the illustration hold nothing of the sort and
  are drawn without one.
* The accent is the only colour with an opinion. Borders are ``faint``,
  labels are ``muted``, and the violet is spent on the four or five things
  that are actually the subject: the wordmark, the active rail item, the
  caret, the state marks.
* Nothing is filled. Backgrounds belong to the terminal, so the shell sits
  on whatever the user's own theme paints behind it instead of stamping a
  rectangle over it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.box import Box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import portrait
from .platforms import terminal_height

__all__ = ["header", "rail", "chip", "user_message", "assistant_card",
           "suggestions", "input_box", "status_bar", "home", "Metrics"]


# A hairline box. rich's ROUNDED draws a heavier top and bottom than this
# design wants beside a dark illustration, and ASCII_BOX is the fallback the
# UI already picks for terminals that cannot draw the corners.
THIN = Box(
    "╭─┬╮\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "╰─┴╯\n"
)


# -- the wordmark ------------------------------------------------------------
#
# Five letters, drawn rather than typed. Six pixel rows pack into three
# terminal rows with the half-block glyphs the illustration uses, which is
# the point: the name is made of the same material as the picture under it,
# so the header belongs to the screen rather than sitting on top of it.

_LETTERS = {
    "w": ("#...#",
          "#...#",
          "#.#.#",
          ".#.#.",
          ".....",
          "....."),
    "y": ("#..#",
          "#..#",
          "#..#",
          ".###",
          "...#",
          "###."),
    "n": ("###.",
          "#..#",
          "#..#",
          "#..#",
          "....",
          "...."),
    "x": ("#..#",
          ".##.",
          ".##.",
          "#..#",
          "....",
          "...."),
    "o": (".##.",
          "#..#",
          "#..#",
          ".##.",
          "....",
          "...."),
}

WORDMARK_ROWS = 3
WORDMARK_CELLS = sum(len(g[0]) for g in _LETTERS.values()) + len(_LETTERS) - 1


def wordmark(palette) -> Text:
    """"wynxo" as pixels: three rows, one cell per pixel."""
    grid = [""] * 6
    for index, letter in enumerate("wynxo"):
        glyph = _LETTERS[letter]
        for row in range(6):
            grid[row] += ("." if index else "") + glyph[row]

    out = Text(no_wrap=True)
    for top, bottom in zip(grid[0::2], grid[1::2], strict=True):
        if out.plain:
            out.append("\n")
        for a, b in zip(top, bottom, strict=True):
            lit = (a == "#", b == "#")
            out.append({(True, True): "█", (True, False): "▀",
                        (False, True): "▄", (False, False): " "}[lit],
                       style=palette.accent if any(lit) else "")
    return out


def plain_wordmark(palette) -> Text:
    """The name as letters, for a terminal that cannot draw the pixels."""
    return Text("wynxo", style=f"bold {palette.accent}")


# -- header ------------------------------------------------------------------

TAGLINE = ("your local ai companion", "think · build · explore · together")


def header(ui, version: str) -> Panel:
    """The wordmark, a hairline separator, what this is, and the version.

    Compact on purpose: three rows of content and a rule top and bottom.
    A header is read once, so anything in it that changes belongs on the
    status bar instead -- which is where the model and the mode live.
    """
    palette = ui.palette
    mark = wordmark(palette) if ui.g.unicode else plain_wordmark(palette)

    lines = Text(no_wrap=True)
    for index, line in enumerate(TAGLINE):
        if index:
            lines.append("\n")
        lines.append(_glyphs(ui, line),
                     style=palette.text if not index else palette.muted)

    grid = Table.grid(expand=True)
    grid.add_column(width=WORDMARK_CELLS if ui.g.unicode else 5)
    grid.add_column(width=3)
    grid.add_column(ratio=1)
    grid.add_column(justify="right")
    grid.add_row(mark,
                 Text(f" {ui.g.vbar} ", style=palette.faint),
                 lines,
                 Text(version, style=palette.faint))
    return Panel(grid, box=_box(ui), border_style=palette.faint,
                 padding=(0, 1))


# -- navigation rail ---------------------------------------------------------

NAV = (("chat", "◈", "*"), ("tools", "✦", "+"),
       ("files", "▤", "#"), ("system", "◎", "o"),
       ("settings", "✧", "~"), ("help", "?", "?"))
"""Section, icon, and what the icon is on a terminal without the glyph.

Not a menu. wynxo is one conversation and every one of these is reached by
typing, so the rail says what the application is made of rather than
offering somewhere to click -- which is why it has no border, no highlight
box, and nothing that looks pressable."""

RAIL_CELLS = 13
RAIL_FROM = 90
"""How wide a terminal has to be before the rail is worth its columns.

It costs fifteen, and between about seventy-five and ninety those fifteen
are the difference between the illustration being drawn and being dropped.
The character outranks the rail: the rail says what the application is made
of, and the character is what it is."""


def rail(ui, active: str = "chat") -> Table:
    """The sections, with the current one marked on its left edge.

    Terminal-native rather than web-native: an indicator bar and a change
    of weight, not a rounded pill. The active row is the only one carrying
    the accent, so the eye finds it without anything having to glow.
    """
    palette = ui.palette
    grid = Table.grid(padding=(0, 0))
    grid.add_column(width=1)      # the indicator
    grid.add_column(width=1)      # a cell of air
    grid.add_column(width=2)      # the icon, padded so a wide glyph fits
    grid.add_column(width=RAIL_CELLS - 4)
    for name, icon, fallback in NAV:
        here = name == active
        grid.add_row(
            Text(ui.g.vbar if here and ui.g.unicode else
                 ("|" if here else " "), style=palette.accent),
            Text(" "),
            Text(icon if ui.g.unicode else fallback,
                 style=palette.accent if here else palette.faint),
            Text(name, style=f"bold {palette.accent}" if here
                 else palette.muted),
        )
    return grid


# -- conversation pieces -----------------------------------------------------

RESTING = ("ready", "idle", "waiting")
"""States that are not an activity, so they take no ellipsis. "thinking..."
says something is happening; "ready..." says nothing is, at length."""


def state_label(companion: str) -> str:
    return companion if companion in RESTING else f"{companion}..."


def chip(ui, label: str) -> Text:
    """A small outlined tag: what the companion is doing, in a word.

    Drawn by hand rather than as a Panel because it has to hug its label --
    a Panel expands to the column it is in, and a "thinking…" bubble as
    wide as the conversation is not a bubble, it is a banner.
    """
    palette = ui.palette
    g = ui.g
    body = _glyphs(ui, label)
    line = Text(no_wrap=True)
    line.append(f"{g.tl}{g.hbar}", style=palette.faint)
    line.append(f" {body} ", style=palette.muted)
    line.append(f"{g.hbar * 2}{g.tr}", style=palette.faint)
    edge = Text(no_wrap=True)
    edge.append(g.bl + g.hbar * (line.cell_len - 2) + g.br,
                style=palette.faint)
    return Text("\n", no_wrap=True).join([line, edge])


def user_message(ui, text: str, note: str = "") -> Panel:
    """What was asked, in its own outline.

    The caret is the same one the composer draws, so the line you typed and
    the line it becomes are the same shape in the same column. ``note`` is
    the aside the queue drain adds -- kept as its own argument rather than
    glued onto the text, so it stays muted and does not turn into part of
    what you said.
    """
    palette = ui.palette
    body = Text()
    body.append(f"{ui.g.caret} ", style=f"bold {palette.accent}")
    first, *rest = text.splitlines() or [""]
    body.append(first, style=f"bold {palette.text}")
    for line in rest:
        body.append("\n  " + line, style=palette.text)
    if note:
        body.append(f"   {note}", style=palette.faint)
    # Sized to what was said, not to the terminal. A full-width outline
    # around the word "hi" is a banner, and the transcript would be a
    # column of them.
    return Panel(body, box=_box(ui), border_style=palette.faint,
                 padding=(0, 1), expand=False)


def assistant_card(ui, text: str) -> Panel:
    """An answer inside a hairline outline."""
    palette = ui.palette
    body = Text(no_wrap=False)
    for index, line in enumerate(text.splitlines()):
        if index:
            body.append("\n")
        body.append(_glyphs(ui, line), style=palette.text)
    return Panel(body, box=_box(ui), border_style=palette.faint,
                 padding=(0, 1))


def suggestions(ui, items) -> Group:
    """Commands worth knowing, as a list rather than as a panel.

    No outline: it sits between two outlined things and a third border
    between them turns the column into a stack of boxes, which is the one
    thing this layout is trying not to be.
    """
    palette = ui.palette
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=11)
    grid.add_column(ratio=1)
    for command, what in items:
        grid.add_row(Text(command, style=palette.accent_dim),
                     Text(what, style=palette.muted))
    return Group(Text("suggestions:", style=palette.faint), grid)


DEFAULT_SUGGESTIONS = (
    ("/help", "show all commands"),
    ("/tools", "list available tools"),
    ("/theme", "change appearance"),
    ("/clear", "clear the screen"),
)


def input_box(ui, placeholder: str = "Type a message or command...") -> Panel:
    """The composer, drawn where the real one will open.

    On the home screen this is the shape of the thing you are about to type
    into; in the running application prompt_toolkit draws the same box
    around a live buffer. Same box, same column, same caret.
    """
    palette = ui.palette
    body = Text(no_wrap=True, overflow="ellipsis")
    body.append(f"{ui.g.caret}  ", style=f"bold {palette.accent}")
    body.append(placeholder, style=palette.faint)
    return Panel(body, box=_box(ui), border_style=palette.accent_dim,
                 padding=(0, 1))


# -- status bar --------------------------------------------------------------

@dataclass(frozen=True)
class Metrics:
    """What the status bar says. A value object so the bar can be rendered
    from a live REPL, from the home screen, and from a test without any of
    the three having to know where the others get their facts."""

    model: str
    mode: str = "agent"
    companion: str = "idle"
    marks: tuple[str, ...] = ()
    """Roles for the small state indicators on the right, in order."""


def status_bar(ui, metrics: Metrics) -> Panel:
    """A thin outlined strip across the application.

    Deliberately the same three facts as the reference and in the same
    order: which model is answering, what mode it is in, and what the
    companion is doing. They are the three things that are true of the
    whole session rather than of one message.
    """
    palette = ui.palette
    line = Table.grid(expand=True)
    line.add_column(ratio=1)
    line.add_column(justify="right")
    line.add_row(status_text(ui, metrics), marks(ui, metrics.marks))
    return Panel(line, box=_box(ui), border_style=palette.faint,
                 padding=(0, 1))


def status_text(ui, metrics: Metrics) -> Text:
    """model / mode / companion, labelled, as one line."""
    palette = ui.palette
    out = Text(no_wrap=True, overflow="ellipsis")
    facts = (("model", metrics.model), ("mode", metrics.mode),
             ("companion", metrics.companion))
    for index, (label, value) in enumerate(facts):
        if index:
            out.append(f"   {ui.g.dot}   ", style=palette.faint)
        out.append(f"{label}: ", style=palette.faint)
        out.append(value, style=palette.muted)
    return out


def marks(ui, roles) -> Text:
    """The small state indicators. Three at most, and never a sentence."""
    palette = ui.palette
    out = Text(no_wrap=True)
    for index, role in enumerate(roles or ("accent", "accent_dim", "faint")):
        if index:
            out.append(" ")
        out.append(ui.g.gear, style=palette.role(role))
    return out


# -- the home screen ---------------------------------------------------------

GREETING = ("Hey! 👋",
            "What do you want to do? I can help with coding,",
            "tools, files, or just chat. Let me know!")


def home(ui, *, model: str, version: str, mode: str = "agent",
         companion: str = "ready", greeting=GREETING,
         placeholder: str = "Type a message or command...",
         items=DEFAULT_SUGGESTIONS) -> Group:
    """The whole screen, composed once.

    The illustration is the subject and everything else is measured around
    it, in that order: the picture takes what the terminal's *height* can
    afford, the rail takes a fixed thirteen columns when there is room for
    one, and the conversation takes the rest. Sizing the picture from the
    width instead is what makes a hero image push the content it is meant
    to introduce off the bottom of a short terminal.

    The two columns end on the same row. The picture is the taller of the
    two by design, so the conversation is spaced to meet it -- the input
    box sits on the illustration's baseline rather than floating halfway up
    a column with a dozen dead rows under it.
    """
    show_rail = ui.width >= RAIL_FROM
    column_rows = _column_rows(greeting, items)
    art = _illustration(ui, show_rail, column_rows)

    body = Table.grid(expand=True, padding=(0, 0))
    if show_rail:
        body.add_column(width=RAIL_CELLS)
        body.add_column(width=2)
    if art is not None:
        body.add_column(width=art[1])
        body.add_column(width=3)
    body.add_column(ratio=1)

    slack = max(0, (art[2] if art else 0) - column_rows)
    column = Group(
        chip(ui, state_label(companion)),
        Text(""),
        assistant_card(ui, "\n".join(greeting)),
        Text(""),
        suggestions(ui, items),
        *[Text("") for _ in range(slack + 1)],
        input_box(ui, placeholder),
    )

    row: list = []
    if show_rail:
        row += [rail(ui), Text("")]
    if art is not None:
        row += [art[0], Text("")]
    row.append(column)
    body.add_row(*row)

    return Group(header(ui, version), Text(""), body, Text(""),
                 status_bar(ui, Metrics(model=model, mode=mode,
                                        companion=companion)))


CHAT_CELLS = 40
"""The least the conversation column will accept: enough for a suggestion
and its description to share a row, and for the greeting to wrap sensibly.

Everything above it goes to the illustration until that reaches its native
width, and only then does the conversation get wider. That order is the
design: the character is the subject, and a reserve generous enough to
protect the greeting's original line breaks cost the picture ten columns
and, below about a hundred, dropped it entirely."""

BALANCE = 12
"""How many rows taller than the conversation the picture may be.

There has to be a limit and it has to be a small number. Without one the
illustration takes the whole terminal on a tall screen and the greeting
floats in the middle of a column of nothing; with too small a one the
character shrinks on exactly the wide terminals that could show him
properly. Twelve keeps him the tallest thing on the screen and keeps the
input box on his baseline."""

HEIGHT_OVERHEAD = 13
"""Rows the screen spends on everything that is not the two columns: the
header, the status bar, the blank rows between them, and one row of air
under the whole thing."""


def _column_rows(greeting, items) -> int:
    """How tall the conversation column is, before any spacing.

    Counted rather than measured. Every piece in it is a known number of
    rows -- an outline costs two, a blank one -- and knowing the total
    before anything is rendered is what lets the illustration be sized to
    match it in the same pass.
    """
    return (2                       # the state chip
            + 1
            + 2 + len(greeting)     # the greeting, in its outline
            + 1
            + 1 + len(items)        # the suggestions and their heading
            + 1
            + 3)                    # the input box


def _illustration(ui, show_rail: bool, column_rows: int):
    """The picture, how many columns it took, and how many rows -- or nothing.

    Dropped rather than shrunk past the point where it stops being a
    person: a smudge where a character should be says less than giving the
    conversation the whole width.
    """
    if not portrait.fits(ui.width, ui.g.unicode):
        return None
    spare = ui.width - CHAT_CELLS - 3 - ((RAIL_CELLS + 2) if show_rail else 0)
    # Height, from two directions. The terminal is the hard limit; the
    # conversation column is the soft one, because a picture much taller
    # than the words beside it stops being a composition and becomes a
    # poster with a note stuck to it.
    rows_available = min(terminal_height() - HEIGHT_OVERHEAD,
                         column_rows + BALANCE)
    cells = min(portrait.NATIVE_CELLS, spare,
                portrait.cells_for(max(0, rows_available)))
    if cells < portrait.MIN_CELLS:
        return None
    lines = portrait.rows(cells, ui.palette)
    return Group(*lines), cells, len(lines)


# -- shared helpers ----------------------------------------------------------

def _box(ui) -> Box:
    return THIN if ui.g.unicode else ui.box


def _glyphs(ui, text: str) -> str:
    """Copy written with Unicode in it, on a terminal that may not have it."""
    from .ui import to_ascii

    return text if ui.g.unicode else to_ascii(text)
