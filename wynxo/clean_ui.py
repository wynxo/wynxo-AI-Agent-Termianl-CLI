"""Calm, width-safe terminal presentation for Wynxo.

This layer deliberately keeps the interaction surface flatter than the older
product shell. A terminal agent should feel like a conversation with useful
status, not a dashboard made from nested boxes:

* messages use a small prefix instead of timestamped rails,
* tool and plan events are compact transcript lines instead of panels,
* the composer has one quiet hint line and one prompt line,
* session status stays on one dim toolbar line.

The agent loop, permissions, provider behavior, completion, and routing are not
changed here. This module only owns presentation.
"""

from __future__ import annotations

from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.formatted_text.html import html_escape
from rich.cells import cell_len
from rich.console import Group
from rich.table import Table
from rich.text import Text

from . import __version__
from . import product_ui
from . import ui as ui_mod
from .platforms import is_dumb_terminal, terminal_height

_INSTALLED = False
_PREV: dict[str, object] = {}


def _rule(ui, width: int | None = None) -> Text:
    """A quiet separator that stays inside the active terminal width."""
    width = max(8, (width or ui.width) - 1)
    return Text(ui.g.hbar * width, style=ui.palette.faint, no_wrap=True)


def _feature(ui, title: str, description: str) -> Group:
    palette = ui.palette
    return Group(
        Text(title, style=palette.accent),
        Text(description, style=palette.muted),
    )


def _context_row(ui, model: str, workspace: str, mode: str) -> Table:
    """Compact session metadata for the home screen."""
    palette = ui.palette
    row = Table.grid(padding=(0, 2), expand=False)
    row.add_column(no_wrap=True)
    row.add_column(no_wrap=True)
    row.add_column(no_wrap=False)

    model_name = ui_mod.sanitise(model or "local model")
    mode_name = ui_mod.sanitise(mode or "agent")
    path = ui.shorten_path(workspace)

    row.add_row(
        Text.assemble(("model  ", palette.faint), (model_name, palette.muted)),
        Text.assemble(("mode  ", palette.faint), (mode_name, palette.muted)),
        Text.assemble(("workspace  ", palette.faint), (path, palette.muted)),
    )
    return row


def _home(self, model: str, workspace: str, *, mode: str = "agent",
          companion: str = "ready", version: str = "",
          show_companion: bool = False, show_art: bool = False,
          show_static_controls: bool = False) -> None:
    """Render a flat, productivity-first launch screen."""
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
    brand.append("  local-first", style=palette.muted)
    self.console.print(brand)
    self.console.print(_context_row(self, model, workspace, mode))
    self.console.print(_rule(self, width))
    self.console.print()

    self.console.print(Text("Ask normally. Work directly.", style=palette.text))
    self.console.print(Text(
        "Chat, inspect a project, edit files, run tools, or manage a larger task "
        "without switching to a dashboard-style interface.",
        style=palette.muted,
    ))

    if width >= 96 and height >= 27:
        self.console.print()
        features = Table.grid(expand=True, padding=(0, 3))
        for _ in range(4):
            features.add_column(ratio=1)
        features.add_row(
            _feature(self, "build", "create, edit, and\nrefactor code"),
            _feature(self, "explore", "search and understand\nyour codebase"),
            _feature(self, "plan", "break down larger\ntasks clearly"),
            _feature(self, "operate", "run local tools and\ninstalled apps"),
        )
        self.console.print(features)

    self.console.print()
    quick = Table.grid(padding=(0, 2))
    quick.add_column(width=10, no_wrap=True)
    quick.add_column()
    for command, description in (
        ("/chat", "tool-free conversation"),
        ("/code", "local project work"),
        ("/github", "GitHub repository workflow"),
        ("/help", "commands and shortcuts"),
    ):
        quick.add_row(
            Text(command, style=palette.accent),
            Text(description, style=palette.muted),
        )
    self.console.print(quick)
    self.console.print()


def _user_line(self, text: str, note: str = "") -> None:
    """Render user input as a transcript line, not a card."""
    if self.narrow or not self.g.unicode:
        return _PREV["user_line"](self, text, note)

    palette = self.palette
    self.console.boundary()

    body = Text(no_wrap=False)
    body.append(f"{self.g.caret} ", style=f"bold {palette.accent}")
    lines = ui_mod.sanitise(text).splitlines() or [""]
    body.append(lines[0], style=palette.text)
    for line in lines[1:]:
        body.append("\n  " + line, style=palette.text)
    if note:
        body.append(f"  {ui_mod.sanitise(note)}", style=palette.faint)
    self.console.print(body)


def _assistant_heading(ui) -> None:
    """Small response marker with no timestamp or vertical rail."""
    palette = ui.palette
    ui.console.boundary()
    head = Text(no_wrap=True)
    head.append("◆ " if ui.g.unicode else "* ", style=palette.accent_dim)
    head.append("wynxo", style=palette.muted)
    ui.console.print(head)


def _tool_call(self, name: str, target: str, detail: str = "",
               ok: bool = True) -> None:
    """Keep frequent tool events compact enough to scan as a transcript."""
    palette = self.palette
    operation = ui_mod.verb(name)
    target = ui_mod.sanitise(target)[:160]
    detail = ui_mod.sanitise(detail)[:200]

    line = Text(no_wrap=False)
    line.append(self.g.tick if ok else self.g.cross,
                style=palette.good if ok else palette.bad)
    line.append("  ")
    line.append(operation, style=palette.accent_dim if ok else palette.bad)
    if target:
        line.append("  ")
        line.append(target, style=palette.text)
    if detail:
        line.append("\n   ")
        line.append(detail, style=palette.faint if ok else palette.bad)

    self.console.boundary()
    self.console.print(line)


def _todos(self, rendered: str) -> None:
    """Render the plan as a light transcript block instead of another panel."""
    if not rendered.strip():
        return
    if self.narrow or not self.g.unicode:
        return _PREV["todos"](self, rendered)

    from . import cli as cli_mod

    steps = cli_mod.plan_steps(ui_mod.sanitise(rendered))
    if not steps:
        return

    palette = self.palette
    done = sum(1 for state, _ in steps if state == "done")

    header = Text()
    header.append("plan", style=palette.muted)
    header.append(f"  {done}/{len(steps)}", style=palette.faint)

    body = Text()
    for index, (state, text) in enumerate(steps):
        if index:
            body.append("\n")
        if state == "done":
            marker, tone = self.g.step_done, palette.good
        elif state == "now":
            marker, tone = self.g.step_now, palette.accent
        else:
            marker, tone = self.g.step_todo, palette.faint
        body.append(f"{marker} ", style=tone)
        body.append(text, style=palette.text if state == "now" else palette.muted)

    self.console.boundary()
    self.console.print(Group(header, body))


def _prompt_message(self) -> HTML:
    """One hint line and one prompt line: no composer box."""
    if is_dumb_terminal() or not self.ui.g.unicode or self.ui.width < 60:
        return _PREV["prompt_message"](self)

    palette = self.ui.palette
    hint = product_ui._prompt_hint(self)
    if hint:
        return HTML(
            '<style fg="%s">%s</style>\n'
            '<b><style fg="%s">%s</style></b> '
            % (
                palette.faint,
                html_escape(hint),
                palette.accent,
                html_escape(self.ui.g.caret),
            )
        )
    return HTML(
        '<b><style fg="%s">%s</style></b> '
        % (palette.accent, html_escape(self.ui.g.caret))
    )


def _status_parts(self, width: int) -> tuple[str, str]:
    """Fit useful session state onto one quiet toolbar line."""
    g = self.ui.g
    left = self._status_line().replace(" · ", f"  {g.dot}  ")
    workspace = self.ui.shorten_path(str(self.workspace))
    right = f"ready  {g.dot}  {workspace}"

    room = max(1, width)
    gap = 2
    if cell_len(left) + cell_len(right) + gap > room:
        left_room = max(8, room - cell_len(right) - gap)
        left = product_ui._trim_cells(left, left_room)
    if cell_len(left) + cell_len(right) + gap > room:
        right_room = max(8, room - cell_len(left) - gap)
        right = product_ui._trim_cells(right, right_room)
    return left, right


def _bottom_toolbar(self):
    """Single-line status instead of a three-line framed dock."""
    if is_dumb_terminal() or not self.ui.g.unicode or self.ui.width < 60:
        return _PREV["bottom_toolbar"](self)

    palette = self.ui.palette
    width = max(30, self.ui.width)
    left, right = _status_parts(self, width)

    gap = max(2, width - cell_len(left) - cell_len(right))
    body = left + (" " * gap) + right
    if cell_len(body) > width:
        body = product_ui._trim_cells(body, width)

    faint = ui_mod._ansi_of(palette.faint)
    muted = ui_mod._ansi_of(palette.muted)
    good = ui_mod._ansi_of(palette.good)
    reset = "\x1b[0m"

    value = f"{faint}{body}{reset}"
    # Keep the model/session side one step stronger than the empty spacing and
    # workspace chrome without introducing another accent colour.
    if left:
        value = value.replace(f"{faint}{left}", f"{muted}{left}{faint}", 1)
    if "ready" in body:
        value = value.replace("ready", f"{good}ready{faint}", 1)
    return ANSI(value)


def install() -> None:
    """Install the calm terminal shell after :mod:`wynxo.product_ui`."""
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

    # product_ui's streaming callbacks resolve this helper through module
    # globals, so replacing it keeps live responses consistent too.
    product_ui._assistant_heading = _assistant_heading

    _INSTALLED = True
