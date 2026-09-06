"""Crisp, width-safe product shell for Wynxo.

This layer is intentionally conservative about terminal rendering. It keeps
Wynxo's product-style hierarchy while avoiding font-fragile iconography and
making the composer, tool cards, plans, messages, and status dock fit
predictably in real monospace terminals.
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

# The clean shell uses a two-cell rail plus one space between columns. Keep the
# reply body on that same baseline. This constant is shared by normal markdown
# rendering and live streaming so the two paths cannot drift apart again.
MESSAGE_INDENT = "   "


def _box(ui):
    return shell.THIN if ui.g.unicode else shell.ASCII


def _rule(ui, width: int | None = None) -> Text:
    """A quiet separator that stays inside the active terminal width."""
    width = max(8, (width or ui.width) - 1)
    return Text(ui.g.hbar * width, style=ui.palette.faint, no_wrap=True)


def _feature(ui, title: str, description: str) -> Group:
    palette = ui.palette
    return Group(
        Text(title.upper(), style=f"bold {palette.accent}"),
        Text(description, style=palette.muted),
    )


def _context_row(ui, model: str, workspace: str, mode: str, width: int) -> Table:
    """Compact session metadata that still fits when names get long."""
    palette = ui.palette
    model_name = product_ui._trim_cells(
        ui_mod.sanitise(model or "local model"), max(12, min(34, width // 3)))
    mode_name = product_ui._trim_cells(ui_mod.sanitise(mode or "agent"), 16)
    path = ui.shorten_path(workspace)

    if width < 88:
        row = Table.grid(padding=(0, 2), expand=False)
        row.add_column(no_wrap=True)
        row.add_column(no_wrap=True)
        row.add_row(
            Text.assemble(("MODEL  ", palette.faint), (model_name, palette.muted)),
            Text.assemble(("MODE  ", palette.faint), (mode_name, palette.muted)),
        )
        path_room = max(18, width - 12)
        row.add_row(
            Text.assemble(
                ("WORKSPACE  ", palette.faint),
                (product_ui._trim_cells(path, path_room), palette.muted),
            ),
            Text(""),
        )
        return row

    row = Table.grid(padding=(0, 2), expand=False)
    row.add_column(no_wrap=True)
    row.add_column(no_wrap=True)
    row.add_column(no_wrap=False)
    path_room = max(20, width - cell_len(model_name) - cell_len(mode_name) - 30)
    row.add_row(
        Text.assemble(("MODEL  ", palette.faint), (model_name, palette.muted)),
        Text.assemble(("MODE  ", palette.faint), (mode_name, palette.muted)),
        Text.assemble(
            ("WORKSPACE  ", palette.faint),
            (product_ui._trim_cells(path, path_room), palette.muted),
        ),
    )
    return row


def _home(self, model: str, workspace: str, *, mode: str = "agent",
          companion: str = "ready", version: str = "",
          show_companion: bool = False, show_art: bool = False,
          show_static_controls: bool = False) -> None:
    """Render a clean, productivity-first launch screen."""
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
    brand.append("  LOCAL-FIRST CODING AGENT", style=palette.muted)
    self.console.print(brand)
    self.console.print(_context_row(self, model, workspace, mode, width))
    self.console.print(_rule(self, width))
    self.console.print()

    self.console.print(Text("What are we building?", style=f"bold {palette.text}"))
    self.console.print(Text(
        "Describe a task in plain language. Wynxo can inspect the project, "
        "edit files, run tools, and keep the work moving.",
        style=palette.muted,
    ))

    if width >= 96 and height >= 27:
        self.console.print()
        features = Table.grid(expand=True, padding=(0, 3))
        for _ in range(4):
            features.add_column(ratio=1)
        features.add_row(
            _feature(self, "Build", "Create, edit, and\nrefactor code."),
            _feature(self, "Explore", "Search and understand\nyour codebase."),
            _feature(self, "Plan", "Break down larger\ntasks clearly."),
            _feature(self, "Automate", "Run local tools and\ninstalled apps."),
        )

        self.console.print(Panel(
            features,
            box=_box(self),
            border_style=palette.faint,
            padding=(0, 2),
        ))

    self.console.print()
    quick = Table.grid(padding=(0, 2))
    quick.add_column(width=11, no_wrap=True)
    quick.add_column()
    for command, description in (
        ("/help", "Commands and keyboard shortcuts"),
        ("/tools", "Available local tools"),
        ("/status", "Model, mode, context, and session state"),
        ("/apps", "Browse applications installed on this machine"),
        ("/theme", "Switch terminal theme"),
    ):
        quick.add_row(
            Text(command, style=f"bold {palette.accent}"),
            Text(description, style=palette.muted),
        )

    self.console.print(Panel(
        quick,
        title=" Quick start ",
        title_align="left",
        box=_box(self),
        border_style=palette.faint,
        padding=(0, 2),
    ))
    self.console.print()


def _message_grid(ui, who: str, text: str, *, note: str = "",
                  assistant: bool = False):
    """Two-column message block with a stable rail and readable hierarchy."""
    palette = ui.palette
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1)

    rail_tone = palette.accent if assistant else palette.accent_dim
    rail = Text(ui.g.vbar, style=rail_tone)

    head = Text()
    head.append(who.upper(), style=f"bold {palette.accent if assistant else palette.text}")
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


def _assistant_markdown(self, text: str) -> None:
    """Render a complete reply on the same baseline as the clean heading."""
    if not text.strip():
        return
    if self.narrow or not self.g.unicode:
        return _PREV["assistant_markdown"](self, text)

    _assistant_heading(self)
    streamer = ui_mod.CodeStreamer(self, indent=MESSAGE_INDENT)
    streamer.feed(text)
    streamer.finish()
    self.console.print()


def _tool_call(self, name: str, target: str, detail: str = "",
               ok: bool = True) -> None:
    """Render a tool event without depending on cli.py presentation helpers."""
    palette = self.palette
    operation = ui_mod.verb(name).replace("_", " ")
    target = ui_mod.sanitise(target)[:160]
    detail = ui_mod.sanitise(detail)[:200]

    body = Text()
    body.append(self.g.tick if ok else self.g.cross,
                style=palette.good if ok else palette.bad)
    body.append("  ")
    body.append(operation, style=f"bold {palette.accent}")
    if target:
        body.append("\n   ")
        body.append(target, style=palette.text)
    if detail:
        body.append("\n   ")
        body.append(detail, style=palette.muted if ok else palette.bad)

    self.console.boundary()
    self.console.print(Panel(
        body,
        title=" Tool  done " if ok else " Tool  failed ",
        title_align="left",
        box=_box(self),
        border_style=palette.faint if ok else palette.bad,
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
    done = 0
    for state, text in steps:
        if body.plain:
            body.append("\n")
        if state == "done":
            marker, tone = self.g.step_done, palette.good
            done += 1
        elif state == "now":
            marker, tone = self.g.step_now, palette.accent
        else:
            marker, tone = self.g.step_todo, palette.muted
        body.append(f"{marker} ", style=tone)
        body.append(text, style=palette.text if state == "now" else palette.muted)

    self.console.boundary()
    self.console.print(Panel(
        body,
        title=f" Plan  {done}/{len(steps)} complete ",
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
    right = "Enter send   Alt+Enter newline"
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
    """Fit session status while preserving the workspace and READY state."""
    g = self.ui.g
    left = self._status_line().replace(" · ", "  |  ")
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


async def _on_content(self, text: str) -> None:
    """Stream replies on exactly the same baseline as non-streamed replies."""
    if not text:
        return
    async with self._status_lock:
        self.tokens += 1
        if not self._streaming:
            self._end_stream()
            _assistant_heading(self.ui)
            self.streamer = ui_mod.CodeStreamer(self.ui, indent=MESSAGE_INDENT)
            self._streaming = True
        if self.bar is not None:
            self.bar.update(activity="writing", detail="",
                            tokens=self.tokens,
                            state=self._writing_state())
        self.typed.feed(self._write_content, text)


def install() -> None:
    """Install the width-safe product shell after :mod:`wynxo.product_ui`."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import cli as cli_mod

    _PREV.update({
        "home": ui_mod.UI.home,
        "user_line": ui_mod.UI.user_line,
        "assistant_markdown": ui_mod.UI.assistant_markdown,
        "tool_call": ui_mod.UI.tool_call,
        "todos": ui_mod.UI.todos,
        "prompt_message": cli_mod.Repl._prompt_message,
        "bottom_toolbar": cli_mod.Repl._bottom_toolbar,
        "on_content": cli_mod.TerminalCallbacks.on_content,
    })

    ui_mod.UI.home = _home
    ui_mod.UI.user_line = _user_line
    ui_mod.UI.assistant_markdown = _assistant_markdown
    ui_mod.UI.tool_call = _tool_call
    ui_mod.UI.todos = _todos

    cli_mod.Repl._prompt_message = _prompt_message
    cli_mod.Repl._bottom_toolbar = _bottom_toolbar
    cli_mod.TerminalCallbacks.on_content = _on_content

    # product_ui's streaming callbacks resolve this helper through module
    # globals, so replacing it also keeps any fallback response path visually
    # consistent with the clean shell.
    product_ui._assistant_heading = _assistant_heading

    _INSTALLED = True
