"""The application's visual shell: header, cards, and session chrome.

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

from rich.box import ASCII, Box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import portrait, sprite
from .companion import State
from .platforms import terminal_height

__all__ = ["header", "rail", "chip", "user_message", "assistant_card",
           "suggestions", "capability_panel", "hero_card", "boot_frame",
           "input_box", "status_bar", "home", "Metrics"]


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
    """A compact brand bar that reads cleanly in a real terminal.

    The old header spent three rows on a pixel wordmark that looked like
    damaged text in many fonts. The brand is more legible as type; the
    personality can live in the optional companion and the colour palette.
    """
    palette = ui.palette
    mark = Text("WYNXO" if ui.g.unicode else "wynxo",
                style=f"bold {palette.accent}")

    lines = Text(no_wrap=True)
    for index, line in enumerate(TAGLINE):
        if index:
            lines.append("\n")
        lines.append(_glyphs(ui, line),
                     style=palette.text if not index else palette.muted)

    grid = Table.grid(expand=True)
    grid.add_column(width=5)
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
        body.append(_glyphs(ui, line), style=f"bold {palette.text}" if index == 0 else palette.muted)
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
        grid.add_row(Text(command, style=f"bold {palette.accent}"),
                     Text(what, style=palette.muted))
    return Group(Text("suggestions:", style=palette.faint), grid)


def welcome_card(ui, greeting=()) -> Panel:
    """The first thing the user reads: purpose, then the useful promise.

    This is deliberately content-first. It gives the landing screen one
    strong card instead of several decorative boxes competing with the
    composer that prompt_toolkit owns below it.
    """
    palette = ui.palette
    lines = list(greeting) or ["Welcome to wynxo."]
    body = Text(no_wrap=False)
    for index, line in enumerate(lines):
        if index:
            body.append("\n")
        body.append(_glyphs(ui, line),
                    style=f"bold {palette.text}" if index == 0
                    else palette.muted)
    body.append("\n\n")
    body.append(f"local-first  {ui.g.dot}  scoped  {ui.g.dot}  "
                "ask before writes",
                style=palette.faint)
    return Panel(body, title="new session", box=_box(ui),
                 border_style=palette.accent_dim, padding=(1, 2))


def quick_panel(ui, items) -> Panel:
    """A small command palette preview, without pretending to be clickable."""
    palette = ui.palette
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=11)
    grid.add_column(ratio=1)
    for command, what in items[:6]:
        grid.add_row(Text(command, style=f"bold {palette.accent}"),
                     Text(what, style=palette.muted))
    return Panel(Group(Text("start with a slash command", style=palette.faint),
                       Text(""), grid),
                 title="quick start", box=_box(ui),
                 border_style=palette.faint, padding=(1, 1))


CAPABILITIES = (
    ("inspect", "files + search"),
    ("build", "edit + verify"),
    ("operate", "apps + GitHub"),
    ("remember", "long-term context"),
    ("guard", "approval first"),
)


def capability_panel(ui, items=CAPABILITIES) -> Panel:
    """Show the useful surface area without pretending it is clickable.

    The landing screen should explain why this is more than a chat prompt.
    These are deliberately short verbs and honest boundaries: the agent can
    work across the local project and installed apps, while writes and other
    consequential actions still pass through permission checks.
    """
    palette = ui.palette
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=10)
    grid.add_column(ratio=1)
    for name, description in items:
        grid.add_row(Text(name, style=f"bold {palette.accent}"),
                     Text(description, style=palette.muted))
    return Panel(Group(Text("one agent, many workflows", style=palette.faint),
                       Text(""), grid),
                 title="capabilities", box=_box(ui),
                 border_style=palette.faint, padding=(1, 1))


HERO_MIN_WIDTH = 110
HERO_MIN_HEIGHT = 46
HERO_PANEL_WIDTH = 44
HERO_MIN_CELLS = 32
HERO_MAX_CELLS = 38


def _hero_cells(ui) -> int:
    """Choose a painted scene size that leaves the launchpad breathing room."""
    # The painting is deliberately kept below its native maximum here. A
    # larger terminal gains whitespace and readable copy, not a mascot whose
    # pixels are simply bigger. This is the same scene as the live companion,
    # just given the room a landing screen can afford.
    return min(HERO_MAX_CELLS, max(HERO_MIN_CELLS, ui.width // 4))


def _hero_status(ui, companion: str = "ready") -> Text:
    """The small status ribbon under the painted scene."""
    palette = ui.palette
    out = Text()
    out.append(f"{ui.g.busy}  ", style=f"bold {palette.bar_accent}")
    out.append(state_label(companion).upper(), style=f"bold {palette.text}")
    out.append("   ", style=palette.faint)
    out.append("local copilot", style=palette.muted)
    return out


def hero_card(ui, model: str = "", companion: str = "ready") -> Panel:
    """The full painted identity of wynxo.

    This is the visual anchor the landing screen was missing. It uses the
    hand-painted truecolour scene rather than approximating a face with ASCII,
    then keeps the copy underneath deliberately quiet so the art remains the
    first thing the eye sees. The live task scene uses the smaller animated
    sprite; this one is a still, high-resolution introduction to the same
    character.
    """
    palette = ui.palette
    art = portrait.rows(_hero_cells(ui), palette)
    caption = Text("your local copilot, ready to work", style=palette.faint)
    if model:
        caption.append(f"  {ui.g.dot}  {ui.shorten_model(model, 22)}",
                       style=palette.faint)
    body = Group(*art, Text(""), _hero_status(ui, companion), caption)
    return Panel(body, title="WYNXO // COPILOT", box=_box(ui),
                 border_style=palette.accent, padding=(0, 1))


BOOT_PHASES = (
    ("calibrating the workspace", "scope locked"),
    ("arming the local tools", "approval gates online"),
    ("warming the copilot", "memory ready"),
    ("wynxo is ready", "your move"),
)


def _progress_line(ui, frame: int, total: int) -> Text:
    """A restrained progress glow for the short boot reveal."""
    palette = ui.palette
    total = max(1, total)
    progress = min(1.0, max(0.0, (frame + 1) / total))
    width = 28
    filled = int(round(width * progress))
    line = Text()
    line.append("  ", style=palette.faint)
    line.append(ui.g.hbar * filled, style=palette.bar_accent)
    line.append(ui.g.dot * (width - filled), style=palette.faint)
    line.append(f"  {int(progress * 100):3d}%", style=palette.muted)
    return line


def boot_frame(ui, model: str, workspace: str, frame: int = 0,
               total: int = 8) -> Panel:
    """One frame of the fast, transient launch reveal.

    The image itself stays stable so it reads as art, while the progress
    ribbon and phase copy supply motion. A moving scanline over every pixel
    would look like a broken terminal capture; one controlled accent is
    enough to make the hand-off feel alive.
    """
    palette = ui.palette
    phase, detail = BOOT_PHASES[min(len(BOOT_PHASES) - 1,
                                    (frame * len(BOOT_PHASES)) // max(1, total))]
    title = Text()
    title.append("WYNXO", style=f"bold {palette.accent}")
    title.append("  //  LOCAL COPILOT", style=f"bold {palette.text}")
    title.append(f"  {ui.g.spark}", style=palette.bar_accent)

    if (ui.width >= HERO_MIN_WIDTH and terminal_height() >= HERO_MIN_HEIGHT
            and ui.g.unicode
            and portrait.fits(ui.width, ui.g.unicode)):
        art = portrait.rows(_hero_cells(ui), palette)
    elif sprite.fits(ui.width, ui.g.unicode):
        art = sprite.rows(State.THINKING, frame // 2, palette)
    else:
        art = []

    meta = Text(f"  {phase}", style=f"bold {palette.text}")
    meta.append(f"  {ui.g.dot}  {detail}", style=palette.muted)
    location = Text("  ", style=palette.faint)
    if workspace:
        location.append(ui.shorten_path(workspace), style=palette.faint)
    body = Group(title, location, Text(""), *art, Text(""), meta,
                 _progress_line(ui, frame, total))
    return Panel(body, title="connecting", box=_box(ui),
                 border_style=palette.accent_dim if frame % 2 else palette.accent,
                 padding=(1, 2))


def companion_card(ui, companion: str) -> Panel:
    """A restrained companion preview for users who opt into the mascot.

    The full raster portrait is beautiful in source but too visually dense
    for a terminal landing screen. The same half-block sprite used by the
    live task view is clearer, cheaper to render, and keeps the companion
    visually connected to actual work.
    """
    palette = ui.palette
    try:
        state = companion if isinstance(companion, State) else State(
            str(companion).strip().lower())
    except ValueError:
        state = State.IDLE
    art = sprite.rows(state, 0, palette)
    body = Group(*art, Text(""),
                 Text(state_label(companion), style=f"bold {palette.text}"),
                 Text("appears while wynxo works", style=palette.faint))
    return Panel(body, title="companion", box=_box(ui),
                 border_style=palette.accent_dim, padding=(0, 1))


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
    room = max(1, ui.width - 11)  # frame, padding, gap, and state marks
    if out.cell_len > room:
        suffix = f" {ui.g.dot} {metrics.mode} {ui.g.dot} {metrics.companion}"
        model_room = max(1, room - Text(suffix).cell_len - 7)
        out = Text("model: ", style=palette.faint, no_wrap=True, overflow="crop")
        out.append(ui.shorten_model(metrics.model, model_room), style=palette.text)
        out.append(suffix, style=palette.muted)
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

GREETING = ("Let's build something.",
            "Bring an idea, a bug, or a question.",
            "We'll work through it together.")


def home(ui, *, model: str, version: str, mode: str = "agent",
         companion: str = "ready", greeting=GREETING,
         placeholder: str = "Type a message or command...",
         items=DEFAULT_SUGGESTIONS, workspace: str = "",
         show_companion: bool = True, show_art: bool = False,
         show_static_controls: bool = True) -> Group:
    """The whole screen, composed once.

    The home screen is a launchpad, not a fake screenshot of the prompt.
    The real composer and toolbar belong to prompt_toolkit, so callers can
    hide the static controls when this is followed by an interactive prompt.
    ``show_static_controls=True`` remains available for render-only callers
    and backwards compatibility.
    """
    # A normal 24-row terminal needs space for the real composer too. Use the
    # same content hierarchy, just with one column and no decorative panels.
    # Interactive callers still have to place a live three-row composer
    # underneath this scrollback. The extra headroom keeps a 34-row terminal
    # on the compact composition instead of making the prompt arrive after a
    # one-line scroll.
    compact_height = 38 if not show_static_controls else 34
    if terminal_height() < compact_height:
        return compact_home(ui, model=model, version=version, mode=mode,
                            companion=companion, workspace=workspace,
                            greeting=greeting, items=items,
                            show_static_controls=show_static_controls)

    # On a genuinely wide, tall terminal the painted scene is the identity
    # anchor. It is deliberately a different composition from the compact
    # launchpad: the left side explains the product, the right side gives it
    # a face. The CLI opts into this; render-only callers keep the lighter
    # historical composition unless they ask for the art explicitly.
    if (show_art and ui.width >= HERO_MIN_WIDTH
            and terminal_height() >= HERO_MIN_HEIGHT
            and portrait.fits(ui.width, ui.g.unicode)):
        body = Table.grid(expand=True, padding=(0, 2))
        body.add_column(ratio=1)
        body.add_column(width=HERO_PANEL_WIDTH)
        main = Group(welcome_card(ui, greeting), Text(""),
                     capability_panel(ui), Text(""), quick_panel(ui, items))
        body.add_row(main, hero_card(ui, model, companion))
    elif ui.width >= 82:
        # Two purposeful columns on wide screens: the welcome message and the
        # command palette. No fake rail, no floating status chip, and no
        # giant illustration pushing the thing the user came to type off-screen.
        body = Table.grid(expand=True, padding=(0, 2))
        body.add_column(ratio=1)
        body.add_column(width=36)
        main = Group(welcome_card(ui, greeting), Text(""),
                     capability_panel(ui))
        side: list = []
        # The compact sprite is shown only when explicitly enabled and when
        # the terminal can afford its full card without wrapping.
        if ((show_companion or show_art) and ui.width >= 82
                and terminal_height() >= 40
                and sprite.fits(ui.width, ui.g.unicode)):
            side.append(companion_card(ui, companion))
            side.append(Text(""))
        side.append(quick_panel(ui, items))
        body.add_row(main, Group(*side))
    else:
        body = Group(welcome_card(ui, greeting), Text(""),
                     suggestions(ui, items))

    parts: list = [header(ui, version), workspace_line(ui, workspace),
                   Text(""), body, Text(""),
                   Text(f"  ready  {ui.g.dot}  type a task below, or start with /help",
                        style=ui.palette.muted)]
    if show_static_controls:
        # Render-only callers can still request the complete preview. The
        # interactive CLI passes False so prompt_toolkit owns these rows and
        # they are never printed twice.
        parts.extend([Text(""), input_box(ui, placeholder), Text(""),
                      status_bar(ui, Metrics(model=model, mode=mode,
                                             companion=companion))])
    return Group(*parts)


def workspace_line(ui, workspace: str) -> Text:
    line = Text(no_wrap=True, overflow="ellipsis")
    if workspace:
        line.append("  workspace  ", style=ui.palette.faint)
        line.append(ui.shorten_path(workspace), style=ui.palette.muted)
    return line


def compact_home(ui, *, model, version, mode, companion, workspace,
                 greeting, items, show_static_controls: bool = True) -> Group:
    """An uncluttered welcome that leaves room to type on short screens."""
    palette = ui.palette
    title = Text("  WYNXO", style=f"bold {palette.accent}")
    title.append(f"  {version}", style=palette.faint)
    title.append(f"   {ui.g.dot}   local-first coding agent", style=palette.muted)
    state = Text("  READY", style=f"bold {palette.good}")
    state.append(f"  {ui.g.dot}  describe what you want to build",
                 style=palette.muted)
    welcome = Text("  " + _glyphs(ui, greeting[0] if greeting else "Welcome."),
                   style=f"bold {palette.text}")
    shortcuts = Text("  ", no_wrap=True, overflow="ellipsis")
    for index, (command, _) in enumerate(items):
        if index:
            shortcuts.append("   ", style=palette.faint)
        shortcuts.append(command, style=palette.accent)
    visual: list = []
    # A short terminal cannot carry the full painted scene, but it can still
    # carry a real terminal illustration. Keeping the sprite in the compact
    # composition means the art does not appear for half a second during boot
    # and then disappear the moment the prompt opens.
    if terminal_height() >= 30 and sprite.fits(ui.width, ui.g.unicode):
        visual = [Text(""), Group(*sprite.rows(State.IDLE, 0, palette)),
                  Text("")]
    parts: list = [title, workspace_line(ui, workspace), Text(""), state,
                 welcome,
                 Text("  Describe a task to get started.", style=palette.muted),
                 *visual, shortcuts,
                 Text(f"  ready  {ui.g.dot}  Tab completes slash commands",
                      style=palette.faint)]
    if show_static_controls:
        parts.extend([Text(""), status_bar(ui, Metrics(
            model=model, mode=mode, companion=companion))])
    return Group(*parts)


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
    cells = min(portrait.MAX_CELLS, spare,
                portrait.cells_for(max(0, rows_available)))
    if cells < portrait.MIN_CELLS:
        return None
    lines = portrait.rows(cells, ui.palette)
    return Group(*lines), cells, len(lines)


# -- shared helpers ----------------------------------------------------------

def _box(ui) -> Box:
    return THIN if ui.g.unicode else ASCII


def _glyphs(ui, text: str) -> str:
    """Copy written with Unicode in it, on a terminal that may not have it."""
    from .ui import to_ascii

    return text if ui.g.unicode else to_ascii(text)
