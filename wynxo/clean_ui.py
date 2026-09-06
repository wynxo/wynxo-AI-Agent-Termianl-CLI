"""Typography and alignment cleanup for the Wynxo product shell.

The product shell intentionally aims for a polished terminal-native look, but
terminal fonts disagree about several decorative Unicode glyphs.  This layer
keeps the same information architecture while using width-safe primitives and
simpler spacing so the UI stays crisp in real monospace terminals.
"""

from __future__ import annotations

from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.formatted_text.html import html_escape
from rich.cells import cell_len
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from . import product_ui
from . import shell
from . import ui as ui_mod
from .platforms import is_dumb_terminal, terminal_height

_INSTALLED = False
_PREV: dict[str, object] = {}


def _box(ui):
    return shell.THIN if ui.g.unicode else shell.ASCII


def _feature(ui, title: str, description: str) -> Group:
    """Plain text feature block with no font-dependent decorative icon."""
    palette = ui.palette
    heading = Text(title.upper(), style=f"bold {palette.accent}")
    detail = Text(description, style=palette.muted)
    return Group(heading, detail)


def _home(self, model: str, workspace: str, *, mode: str = "agent",
          companion: str = "ready", version: str = "",
          show_companion: bool = False, show_art: bool = False,
          show_static_controls: bool = False) -> None:
    """Render the same home screen with cleaner monospace typography."""
    self.refresh_size()
    if is_dumb_terminal() or self.narrow or not self.console.is_terminal:
        return _PREV["home"](
            self, model, workspace, mode=mode, companion=companion,
            version=version, show_companion=show_companion, show_art=show_art,
            show_static_controls=show_static_controls,
        )

    palette = self.palette
    width = self.width
    height = terminal_height()

    self.console.print()

    brand = Text()
    brand.append("WYNXO", style=f"bold {palette.accent}")
    brand.append(f"  {version or __version__}", style=palette.faint)
    brand.append("  -  local-first coding agent", style=palette.muted)
    self.console.print(brand)

    location = Text()
    location.append("workspace  ", style=palette.faint)
    location.append(self.shorten_path(workspace), style=palette.muted)
    self.console.print(location)
    self.console.print()

    self.console.print(Text("Welcome to Wynxo", style=f"bold {palette.text}"))
    self.console.print(Text(
        "Your local-first coding agent. Describe what you want to build.",
        style=palette.muted,
    ))

    if width >= 96 and height >= 28:
        self.console.print()
        features = Table.grid(expand=True, padding=(0, 3))
        for _ in range(4):
            features.add_column(ratio=1)
        features.add_row(
            _feature(self, "Build", "Create, edit, and\nrefactor code."),
            _feature(self, "Explore", "Search and understand\nyour codebase."),
            _feature(self, "Plan", "Break down complex\ntasks."),
            _feature(self, "Automate", "Run tools and\naccomplish more."),
        )
        self.console.print(features)

    self.console.print()
    self.console.print(Text("Quick start", style=f"bold {palette.accent}"))

    quick = Table.grid(padding=(0, 2))
    quick.add_column(width=11)
    quick.add_column()
    for command, description in (
        ("/help", "Show commands"),
        ("/tools", "List available tools"),
        ("/theme", "Change the theme"),
        ("/status", "Show session status"),
        ("/clear", "Clear the terminal"),
    ):
        quick.add_row(
            Text(command, style=f"bold {palette.accent}"),
            Text(description, style=palette.muted),
        )
    self.console.print(quick)
    self.console.print()


def _message_grid(ui, who: str, text: str, *, note: str = ""):
    """Simple two-column message block that cannot drift by glyph width."""
    palette = ui.palette
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1)

    rail = Text(ui.g.vbar, style=palette.accent_dim)
    head = Text()
    head.append(who.upper(), style=f"bold {palette.accent}")
    head.append(f"  {product_ui._clock()}", style=palette.faint)

    body = Text(ui_mod.sanitise(text), style=palette.text)
    if note:
        body.append(f"  {ui_mod.sanitise(note)}", style=palette.faint)

    grid.add_row(rail, Group(head, body))
    return grid


def _user_line(self, text: str, note: str = "") -> None:
    if self.narrow or not self.g.unicode:
        return _PREV["user_line"](self, text, note)
    self.console.boundary()
    self.console.print(_message_grid(self, "You", text, note=note))


def _assistant_heading(ui) -> None:
    palette = ui.palette
    ui.console.boundary()
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1)
    rail = Text(ui.g.vbar, style=palette.accent)
    head = Text()
    head.append("WYNXO", style=f"bold {palette.accent}")
    head.append(f"  {product_ui._clock()}", style=palette.faint)
    grid.add_row(rail, head)
    ui.console.print(grid)


def _tool_call(self, name: str, target: str, detail: str = "",
               ok: bool = True) -> None:
    if self.narrow or not self.g.unicode:
        return _PREV["tool_call"](self, name, target, detail, ok)

    from . import cli as cli_mod

    palette = self.palette
    line = Text()
    line.append(self.g.tick if ok else self.g.cross,
                style=palette.good if ok else palette.bad)
    line.append("  ")
    line.append(cli_mod.verb(name), style=f"bold {palette.accent}")
    if target:
        line.append(f"  {ui_mod.sanitise(target)[:120]}", style=palette.text)
    if detail:
        line.append(f"  {ui_mod.sanitise(detail)[:120]}",
                    style=palette.muted if ok else palette.bad)

    self.console.boundary()
    self.console.print(Panel(
        line,
        title=" Tool ",
        title_align="left",
        box=_box(self),
        border_style=palette.faint,
        padding=(0, 2),
    ))


def _todos(self, rendered: str) -> None:
    if not rendered.strip():
        return
    if self.narrow or not self.g.unicode:
        return _PREV["todos"](self, rendered)

    from . import cli as cli_mod

    steps = cli_mod.plan_steps(ui_mod.sanitise(rendered))
    if not steps:
        return

    palette = self.palette
    body = Text()
    for state, text in steps:
        if body.plain:
            body.append("\n")
        if state == "done":
            marker, tone = self.g.step_done, palette.good
        elif state == "now":
            marker, tone = self.g.step_now, palette.accent
        else:
            marker, tone = self.g.step_todo, palette.muted
        body.append(f"{marker} ", style=tone)
        body.append(text, style=palette.text if state == "now" else palette.muted)

    self.console.boundary()
    self.console.print(Panel(
        body,
        title=f" Plan  {len(steps)} step{'s' if len(steps) != 1 else ''} ",
        title_align="left",
        box=_box(self),
        border_style=palette.faint,
        padding=(0, 2),
    ))


def _prompt_message(self) -> HTML:
    if is_dumb_terminal() or not self.ui.g.unicode or self.ui.width < 60:
        return _PREV["prompt_message"](self)

    palette = self.ui.palette
    g = self.ui.g
    width = max(30, self.ui.width)

    left = product_ui._prompt_hint(self)
    right = "Alt+Enter newline  |  Ctrl+C stop"
    room = width - cell_len(left) - cell_len(right)
    if room >= 2:
        hint = left + (" " * room) + right
    else:
        left_room = max(8, width - cell_len(right) - 2)
        hint = product_ui._trim_cells(left, left_room)
        if cell_len(hint) + cell_len(right) + 2 <= width:
            hint += "  " + right

    top = g.tl + g.hbar * max(2, width - 2) + g.tr
    return HTML(
        '<style fg="%s">%s</style>\n'
        '<style fg="%s">%s</style>\n'
        '<style fg="%s">%s</style> '
        '<b><style fg="%s">%s</style></b> '
        % (
            palette.faint, html_escape(hint),
            palette.accent, html_escape(top),
            palette.accent, html_escape(g.vbar),
            palette.accent, html_escape(g.caret),
        )
    )


def _status_parts(self, width: int) -> tuple[str, str]:
    """Fit status text deterministically and preserve the right-hand state."""
    g = self.ui.g
    left = self._status_line().replace(" · ", " | ")
    workspace = self.ui.shorten_path(str(self.workspace))
    right = f"READY  {g.vbar}  {workspace}"

    inner = max(1, width - 4)
    gap = 2
    if cell_len(left) + cell_len(right) + gap > inner:
        left_room = max(8, inner - cell_len(right) - gap)
        left = product_ui._trim_cells(left, left_room)
    if cell_len(left) + cell_len(right) + gap > inner:
        right_room = max(8, inner - cell_len(left) - gap)
        right = product_ui._trim_cells(right, right_room)
    return left, right


def _bottom_toolbar(self):
    if is_dumb_terminal() or not self.ui.g.unicode or self.ui.width < 60:
        return _PREV["bottom_toolbar"](self)

    palette = self.ui.palette
    g = self.ui.g
    width = max(30, self.ui.width)
    left, right = _status_parts(self, width)

    inner = max(1, width - 2)
    content_room = max(0, inner - 2)
    gap = max(1, content_room - cell_len(left) - cell_len(right))
    body = " " + left + (" " * gap) + right
    if cell_len(body) < inner:
        body += " " * (inner - cell_len(body))
    elif cell_len(body) > inner:
        body = product_ui._trim_cells(body, inner)

    frame = ui_mod._ansi_of(palette.accent)
    muted = ui_mod._ansi_of(palette.muted)
    good = ui_mod._ansi_of(palette.good)
    reset = "\x1b[0m"

    divider = g.vbar + g.hbar * max(2, width - 2) + g.vbar
    bottom = g.bl + g.hbar * max(2, width - 2) + g.br

    status = f"{muted}{body}{reset}"
    if "READY" in body:
        status = status.replace("READY", f"{good}READY{muted}", 1)

    value = (
        f"{frame}{divider}{reset}\n"
        f"{frame}{g.vbar}{reset}{status}{frame}{g.vbar}{reset}\n"
        f"{frame}{bottom}{reset}"
    )
    return ANSI(value)


def install() -> None:
    """Install typography cleanup after :mod:`wynxo.product_ui`."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import cli as cli_mod

    _PREV.update({
        "home": ui_mod.UI.home,
        "user_line": ui_mod.UI.user_line,
        "tool_call": ui_mod.UI.tool_call,
        "todos": ui_mod.UI.todos,
        "prompt_message": cli_mod.Repl._prompt_message,
        "bottom_toolbar": cli_mod.Repl._bottom_toolbar,
    })

    ui_mod.UI.home = _home
    ui_mod.UI.user_line = _user_line
    ui_mod.UI.tool_call = _tool_call
    ui_mod.UI.todos = _todos

    cli_mod.Repl._prompt_message = _prompt_message
    cli_mod.Repl._bottom_toolbar = _bottom_toolbar

    # product_ui's streaming callbacks resolve this helper through their module
    # globals, so replacing it here also cleans the live-response heading.
    product_ui._assistant_heading = _assistant_heading

    _INSTALLED = True
