"""Product-style terminal chrome for Wynxo.

This module layers the polished, chat-first shell over the battle-tested
prompt/streaming engine in :mod:`wynxo.cli`. It deliberately leaves the
agent, provider, permissions, tools, and prompt-session lifecycle alone.

The important split is:
- cli.py owns behaviour.
- this module owns presentation.

Keeping the redesign here makes it possible to iterate on the visual shell
without turning the already-large REPL into a second UI toolkit.
"""

from __future__ import annotations

from datetime import datetime

from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.formatted_text.html import html_escape
from rich.cells import cell_len
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from . import shell
from . import ui as ui_mod
from .platforms import is_dumb_terminal, terminal_height

_INSTALLED = False
_ORIGINALS: dict[str, object] = {}


def _clock() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def _box(ui):
    return shell.THIN if ui.g.unicode else shell.ASCII


def _trim_cells(text: str, room: int, marker: str = "…") -> str:
    if room <= 0:
        return ""
    if cell_len(text) <= room:
        return text
    marker = marker if cell_len(marker) <= room else "." * room
    budget = max(0, room - cell_len(marker))
    out = ""
    for char in reversed(text):
        width = cell_len(char)
        if width > budget:
            break
        out = char + out
        budget -= width
    return marker + out


def _icon(ui, glyph: str, *, tone: str | None = None) -> Text:
    """A tiny terminal-native square like the mockup's message/tool icons."""
    g = ui.g
    tone = tone or ui.palette.accent
    body = glyph[:2].ljust(2)
    out = Text(no_wrap=True)
    if ui.g.unicode:
        out.append(g.tl + g.hbar * 2 + g.tr + "\n", style=tone)
        out.append(g.vbar, style=tone)
        out.append(body, style=f"bold {tone}")
        out.append(g.vbar + "\n", style=tone)
        out.append(g.bl + g.hbar * 2 + g.br, style=tone)
    else:
        out.append("+--+\n", style=tone)
        out.append("|", style=tone)
        out.append(body, style=f"bold {tone}")
        out.append("|\n+--+", style=tone)
    return out


def _message_grid(ui, who: str, glyph: str, text: str, *,
                  note: str = "", assistant: bool = False):
    palette = ui.palette
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=6)
    grid.add_column(ratio=1)

    head = Text()
    head.append(who, style=f"bold {palette.accent}")
    head.append(f"  {_clock()}", style=palette.faint)

    body = Text()
    lines = text.splitlines() or [""]
    for index, line in enumerate(lines):
        if index:
            body.append("\n")
        body.append(line, style=palette.text if assistant else f"bold {palette.text}")
    if note:
        body.append(f"   {note}", style=palette.faint)

    grid.add_row(_icon(ui, glyph), Group(head, body))
    return grid


def _assistant_heading(ui) -> None:
    palette = ui.palette
    ui.console.boundary()
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=6)
    grid.add_column(ratio=1)
    head = Text()
    head.append("Wynxo", style=f"bold {palette.accent}")
    head.append(f"  {_clock()}", style=palette.faint)
    grid.add_row(_icon(ui, ">_"), head)
    ui.console.print(grid)


def _feature(ui, glyph: str, title: str, description: str) -> Group:
    palette = ui.palette
    title_line = Text()
    title_line.append(f"[{glyph}] ", style=palette.accent_dim)
    title_line.append(title, style=f"bold {palette.accent}")
    return Group(title_line, Text(description, style=palette.muted))


def _home(self, model: str, workspace: str, *, mode: str = "agent",
          companion: str = "ready", version: str = "",
          show_companion: bool = False, show_art: bool = False,
          show_static_controls: bool = False) -> None:
    """Draw the product launch screen without faking the live composer."""
    self.refresh_size()
    if is_dumb_terminal() or self.narrow or not self.console.is_terminal:
        return _ORIGINALS["home"](
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
    brand.append(f"  {self.g.dot}  local-first coding agent", style=palette.muted)
    self.console.print(brand)

    place = Text()
    place.append("workspace  ", style=palette.faint)
    place.append(self.shorten_path(workspace), style=palette.muted)
    self.console.print(place)
    self.console.print()

    welcome = Text()
    welcome.append("Welcome to Wynxo!", style=f"bold {palette.text}")
    welcome.append("\nYour local-first coding agent, ready when you are.",
                   style=palette.muted)
    welcome.append("\nStart a new session. Describe what you want to build.",
                   style=palette.faint)
    self.console.print(welcome)

    if width >= 96 and height >= 28:
        self.console.print()
        features = Table.grid(expand=True, padding=(0, 2))
        for _ in range(4):
            features.add_column(ratio=1)
        features.add_row(
            _feature(self, ">_", "Build", "Create, edit, and\nrefactor code."),
            _feature(self, "?", "Explore", "Search and understand\nyour codebase."),
            _feature(self, self.g.task, "Plan", "Break down complex\ntasks."),
            _feature(self, self.g.gear, "Automate", "Run tools and\naccomplish more."),
        )
        self.console.print(features)

    self.console.print()
    self.console.print(Text("Quick start", style=f"bold {palette.accent}"))

    commands = (
        ("/help", "Show help and available commands"),
        ("/tools", "List available tools"),
        ("/theme", "Change the color theme"),
        ("/clear", "Clear the terminal"),
        ("/status", "Show current session status"),
    )
    quick = Table.grid(padding=(0, 2))
    quick.add_column(width=12)
    quick.add_column()
    for command, description in commands:
        quick.add_row(Text(command, style=f"bold {palette.accent}"),
                      Text(description, style=palette.muted))
    self.console.print(quick)

    self.console.print()
    footer = Text()
    footer.append("Ready. Type a message to start building.", style=palette.faint)
    if width >= 82:
        hint = "Enter to send  ·  Alt+Enter for newline"
        gap = max(2, width - cell_len(footer.plain) - cell_len(hint) - 1)
        footer.append(" " * gap)
        footer.append(hint, style=palette.faint)
    self.console.print(footer, overflow="crop", no_wrap=True)
    self.console.print()


def _user_line(self, text: str, note: str = "") -> None:
    if self.narrow or not self.g.unicode:
        return _ORIGINALS["user_line"](self, text, note)
    self.console.boundary()
    self.console.print(_message_grid(self, "You", "○", text, note=note))


def _assistant_markdown(self, text: str) -> None:
    if not text.strip():
        return
    if self.narrow or not self.g.unicode:
        return _ORIGINALS["assistant_markdown"](self, text)

    _assistant_heading(self)
    streamer = ui_mod.CodeStreamer(self, indent="      ")
    streamer.feed(text)
    streamer.finish()
    self.console.print()


def _tool_call(self, name: str, target: str, detail: str = "",
               ok: bool = True) -> None:
    if self.narrow or not self.g.unicode:
        return _ORIGINALS["tool_call"](self, name, target, detail, ok)

    palette = self.palette
    self.console.boundary()

    line = Text()
    line.append(self.g.tick if ok else self.g.cross,
                style=palette.good if ok else palette.bad)
    line.append("  ")
    line.append(ui_mod.verb(name).replace("_", " "),
                style=f"bold {palette.accent}")
    if target:
        line.append(f"  {ui_mod.sanitise(target)[:120]}", style=palette.text)
    if detail:
        line.append(f"  {ui_mod.sanitise(detail)[:120]}",
                    style=palette.muted if ok else palette.bad)

    card = Table.grid(expand=True, padding=(0, 1))
    card.add_column(width=6)
    card.add_column(ratio=1)
    card.add_row(_icon(self, self.g.gear, tone=palette.accent_dim), line)

    self.console.print(Panel(
        card,
        title=" Tools ",
        title_align="left",
        box=_box(self),
        border_style=palette.faint,
        padding=(0, 1),
    ))


def _todos(self, rendered: str) -> None:
    if not rendered.strip():
        return
    if self.narrow or not self.g.unicode:
        return _ORIGINALS["todos"](self, rendered)

    # Imported lazily: cli imports UI, so importing cli at module import would
    # create a cycle.
    from . import cli as cli_mod

    steps = cli_mod.plan_steps(ui_mod.sanitise(rendered))
    if not steps:
        return

    palette = self.palette
    body = Text()
    for index, (state, text) in enumerate(steps, start=1):
        if body.plain:
            body.append("\n")
        if state == "done":
            marker, tone = self.g.tick, palette.good
        elif state == "now":
            marker, tone = "●", palette.accent
        else:
            marker, tone = "○", palette.muted
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


def _prompt_hint(repl) -> str:
    from . import cli as cli_mod

    note, repl._prompt_note = cli_mod.live_note(repl._prompt_note)
    if note:
        return note
    typed = cli_mod.command_hints(cli_mod._composer_text(repl))
    if typed:
        return "  ".join(typed)
    return "Describe a task, or type /help for commands."


def _prompt_message(self) -> HTML:
    if is_dumb_terminal() or not self.ui.g.unicode or self.ui.width < 60:
        return _ORIGINALS["prompt_message"](self)

    palette = self.ui.palette
    g = self.ui.g
    width = max(30, self.ui.width)

    left = _prompt_hint(self)
    right = "Enter send  ·  Alt+Enter newline"
    room = width - cell_len(left) - cell_len(right)
    if room >= 2:
        hint = left + (" " * room) + right
    else:
        hint = _trim_cells(left, max(8, width - cell_len(right) - 2))
        if cell_len(hint) + cell_len(right) + 2 <= width:
            hint += "  " + right

    edge = g.tl + g.hbar * max(2, width - 2) + g.tr
    caret = self.ui.g.caret
    return HTML(
        '<style fg="%s">%s</style>\n'
        '<style fg="%s">%s</style>\n'
        '<style fg="%s">%s</style> '
        '<b><style fg="%s">%s</style></b> '
        % (
            palette.faint, html_escape(hint),
            palette.accent, html_escape(edge),
            palette.accent, html_escape(g.vbar),
            palette.accent, html_escape(caret),
        )
    )


def _prompt_rail(self) -> HTML:
    if is_dumb_terminal() or not self.ui.g.unicode or self.ui.width < 60:
        return _ORIGINALS["prompt_rail"](self)
    return HTML('<style fg="%s">%s</style>'
                % (self.ui.palette.accent, html_escape(self.ui.g.vbar)))


def _status_row(self, width: int) -> tuple[str, str]:
    g = self.ui.g
    left = self._status_line()
    workspace = self.ui.shorten_path(str(self.workspace))
    right = f"● READY  {g.vbar}  {workspace}"

    inner = max(1, width - 4)  # borders and one space on each side
    min_gap = 2
    if cell_len(left) + cell_len(right) + min_gap > inner:
        left_room = max(8, inner - cell_len(right) - min_gap)
        left = _trim_cells(left, left_room)
    if cell_len(left) + cell_len(right) + min_gap > inner:
        right_room = max(8, inner - cell_len(left) - min_gap)
        right = _trim_cells(right, right_room)
    return left, right


def _bottom_toolbar(self):
    if is_dumb_terminal() or not self.ui.g.unicode or self.ui.width < 60:
        return _ORIGINALS["bottom_toolbar"](self)

    palette = self.ui.palette
    g = self.ui.g
    width = max(30, self.ui.width)
    left, right = _status_row(self, width)

    inner = max(1, width - 2)
    body_room = max(0, inner - 2)
    gap = max(1, body_room - cell_len(left) - cell_len(right))
    body = " " + left + (" " * gap) + right
    if cell_len(body) < inner:
        body += " " * (inner - cell_len(body))
    elif cell_len(body) > inner:
        body = _trim_cells(body, inner)

    frame = ui_mod._ansi_of(palette.accent)
    muted = ui_mod._ansi_of(palette.muted)
    good = ui_mod._ansi_of(palette.good)
    reset = "\x1b[0m"

    divider = "├" + g.hbar * max(2, width - 2) + "┤"
    bottom = g.bl + g.hbar * max(2, width - 2) + g.br

    # Paint the READY mark green without making the whole right-hand status
    # strip green. ANSI is used because prompt_toolkit sanitises literal ESC
    # bytes in plain formatted text.
    status = f"{muted}{body}{reset}"
    ready_token = "● READY"
    if ready_token in body:
        status = status.replace(ready_token, f"{good}{ready_token}{muted}", 1)

    value = (
        f"{frame}{divider}{reset}\n"
        f"{frame}{g.vbar}{reset}{status}{frame}{g.vbar}{reset}\n"
        f"{frame}{bottom}{reset}"
    )
    return ANSI(value)


async def _on_content(self, text: str) -> None:
    """Stream the answer under a stable Wynxo message header."""
    if not text:
        return
    async with self._status_lock:
        self.tokens += 1
        if not self._streaming:
            self._end_stream()
            _assistant_heading(self.ui)
            self.streamer = ui_mod.CodeStreamer(self.ui, indent="      ")
            self._streaming = True
        if self.bar is not None:
            self.bar.update(activity="writing", detail="",
                            tokens=self.tokens,
                            state=self._writing_state())
        self.typed.feed(self._write_content, text)


async def _on_todos(self, rendered: str) -> None:
    """Keep the live plan, but commit one clean plan card to the transcript."""
    steps = []
    if rendered.strip():
        from . import cli as cli_mod
        steps = cli_mod.plan_steps(ui_mod.sanitise(rendered))
    if steps and not getattr(self, "_product_plan_printed", False):
        self.ui.todos(rendered)
        self._product_plan_printed = True
    await _ORIGINALS["on_todos"](self, rendered)
    if steps and all(state == "done" for state, _ in steps):
        self._product_plan_printed = False


def install() -> None:
    """Install the product shell once, after :mod:`wynxo.cli` is imported."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import cli as cli_mod

    _ORIGINALS.update({
        "home": ui_mod.UI.home,
        "user_line": ui_mod.UI.user_line,
        "assistant_markdown": ui_mod.UI.assistant_markdown,
        "tool_call": ui_mod.UI.tool_call,
        "todos": ui_mod.UI.todos,
        "prompt_message": cli_mod.Repl._prompt_message,
        "prompt_rail": cli_mod.Repl._prompt_rail,
        "bottom_toolbar": cli_mod.Repl._bottom_toolbar,
        "on_content": cli_mod.TerminalCallbacks.on_content,
        "on_todos": cli_mod.TerminalCallbacks.on_todos,
    })

    ui_mod.UI.home = _home
    ui_mod.UI.user_line = _user_line
    ui_mod.UI.assistant_markdown = _assistant_markdown
    ui_mod.UI.tool_call = _tool_call
    ui_mod.UI.todos = _todos

    cli_mod.Repl._prompt_message = _prompt_message
    cli_mod.Repl._prompt_rail = _prompt_rail
    cli_mod.Repl._bottom_toolbar = _bottom_toolbar
    cli_mod.TerminalCallbacks.on_content = _on_content
    cli_mod.TerminalCallbacks.on_todos = _on_todos

    _INSTALLED = True
