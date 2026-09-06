"""Minimal, terminal-native product shell for Wynxo.

The shell deliberately avoids decorative boxes and font-fragile iconography.
Hierarchy comes from spacing, typography, and a small number of status marks:
closer to a serious coding CLI than a dashboard rendered inside a terminal.
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
from .platforms import is_dumb_terminal

_INSTALLED = False
_PREV: dict[str, object] = {}

# Replies, tools, and plans all begin on the same quiet two-cell content rail.
MESSAGE_INDENT = "  "


def _trim(text: str, room: int) -> str:
    return product_ui._trim_cells(ui_mod.sanitise(text), max(1, room))


def _meta_line(ui, model: str, workspace: str, mode: str, width: int) -> Group:
    palette = ui.palette
    model_name = _trim(model or "local model", max(12, min(36, width // 3)))
    mode_name = _trim(mode or "agent", 14)
    path = _trim(ui.shorten_path(workspace), max(18, width - 4))

    session = Text()
    session.append(model_name, style=palette.muted)
    session.append("  /  ", style=palette.faint)
    session.append(mode_name, style=palette.muted)
    return Group(session, Text(path, style=palette.faint))


def _home(self, model: str, workspace: str, *, mode: str = "agent",
          companion: str = "ready", version: str = "",
          show_companion: bool = False, show_art: bool = False,
          show_static_controls: bool = False) -> None:
    """A calm launch screen: identity, context, useful entry points, nothing else."""
    self.refresh_size()
    if is_dumb_terminal() or self.narrow or not self.console.is_terminal:
        return _PREV["home"](
            self, model, workspace, mode=mode, companion=companion,
            version=version, show_companion=show_companion, show_art=show_art,
            show_static_controls=show_static_controls,
        )

    palette = self.palette
    width = self.width

    self.console.print()

    brand = Text()
    brand.append("WYNXO", style=f"bold {palette.accent}")
    brand.append(f"  {version or __version__}", style=palette.faint)
    brand.append("    local-first coding agent", style=palette.muted)
    self.console.print(brand)
    self.console.print(_meta_line(self, model, workspace, mode, width))

    self.console.print()
    self.console.print(Text("Ready to build.", style=f"bold {palette.text}"))
    self.console.print(Text(
        "Describe the outcome. Wynxo can inspect the project, edit files, run "
        "tests, and use installed applications.",
        style=palette.muted,
    ))

    self.console.print()
    quick = Table.grid(padding=(0, 2), expand=False)
    quick.add_column(width=10, no_wrap=True)
    quick.add_column(no_wrap=False)
    for command, description in (
        ("/help", "commands + keyboard shortcuts"),
        ("/tools", "agent capabilities"),
        ("/apps", "installed applications"),
        ("/status", "model, mode, context + session"),
    ):
        quick.add_row(
            Text(command, style=f"bold {palette.accent}"),
            Text(description, style=palette.faint),
        )
    self.console.print(quick)
    self.console.print()


def _message_block(ui, who: str, text: str, *, note: str = "",
                   assistant: bool = False) -> Group:
    palette = ui.palette
    head = Text()
    head.append(who.lower(), style=(f"bold {palette.accent}" if assistant
                                     else f"bold {palette.muted}"))
    head.append(f"  {product_ui._clock()}", style=palette.faint)

    body = Text(MESSAGE_INDENT + ui_mod.sanitise(text), style=palette.text)
    if note:
        body.append(f"  {ui_mod.sanitise(note)}", style=palette.faint)
    return Group(head, body)


def _user_line(self, text: str, note: str = "") -> None:
    if self.narrow or not self.g.unicode:
        return _PREV["user_line"](self, text, note)
    self.console.boundary()
    self.console.print(_message_block(self, "you", text, note=note))


def _assistant_heading(ui) -> None:
    ui.console.boundary()
    head = Text()
    head.append("wynxo", style=f"bold {ui.palette.accent}")
    head.append(f"  {product_ui._clock()}", style=ui.palette.faint)
    ui.console.print(head)


def _assistant_markdown(self, text: str) -> None:
    if not text.strip():
        return
    if self.narrow or not self.g.unicode:
        return _PREV["assistant_markdown"](self, text)

    _assistant_heading(self)
    streamer = ui_mod.CodeStreamer(self, indent=MESSAGE_INDENT)
    streamer.feed(text)
    streamer.finish()
    self.console.print()


def _tool_label(name: str) -> str:
    return ui_mod.verb(name).replace("_", " ").strip()


def _tool_call(self, name: str, target: str, detail: str = "",
               ok: bool = True) -> None:
    """One compact operation record instead of a card around every tool call."""
    palette = self.palette
    operation = _tool_label(name)
    target = ui_mod.sanitise(target)[:180]
    detail = ui_mod.sanitise(detail)[:220]

    line = Text()
    line.append("[ok]" if ok else "[x] ",
                style=palette.good if ok else palette.bad)
    line.append("  ")
    line.append(operation, style=f"bold {palette.text}")
    if target:
        line.append("  ")
        line.append(target, style=palette.muted)

    self.console.boundary()
    self.console.print(line)
    if detail:
        self.console.print(Text(MESSAGE_INDENT + detail,
                                style=palette.faint if ok else palette.bad))


def _todos(self, rendered: str) -> None:
    if not rendered.strip():
        return
    if self.narrow or not self.g.unicode:
        return _PREV["todos"](self, rendered)

    steps = ui_mod.plan_steps(ui_mod.sanitise(rendered))
    if not steps:
        return

    palette = self.palette
    done = sum(1 for state, _ in steps if state == "done")
    body = Text()
    body.append("plan", style=f"bold {palette.text}")
    body.append(f"  {done}/{len(steps)}\n", style=palette.faint)

    for index, (state, text) in enumerate(steps):
        marker, tone = {
            "done": ("[x]", palette.good),
            "now": ("[>]", palette.accent),
            "todo": ("[ ]", palette.faint),
        }[state]
        body.append(MESSAGE_INDENT)
        body.append(marker, style=tone)
        body.append("  ")
        body.append(text, style=palette.text if state == "now" else palette.muted)
        if index != len(steps) - 1:
            body.append("\n")

    self.console.boundary()
    self.console.print(body)


def _prompt_message(self) -> HTML:
    if is_dumb_terminal() or self.ui.width < 48:
        return _PREV["prompt_message"](self)

    palette = self.ui.palette
    width = max(30, self.ui.width)
    left = product_ui._prompt_hint(self)
    right = "Enter send  /  Alt+Enter newline"

    room = width - cell_len(left) - cell_len(right)
    if room >= 3:
        hint = left + (" " * room) + right
    else:
        left_room = max(10, width - cell_len(right) - 3)
        hint = product_ui._trim_cells(left, left_room)
        if cell_len(hint) + cell_len(right) + 3 <= width:
            hint += "   " + right

    return HTML(
        '<style fg="%s">%s</style>\n'
        '<b><style fg="%s">&gt;</style></b> '
        % (palette.faint, html_escape(hint), palette.accent)
    )


def _status_parts(self, width: int) -> tuple[str, str]:
    left = self._status_line().replace(" · ", " / ")
    workspace = self.ui.shorten_path(str(self.workspace))
    right = f"ready  /  {workspace}"

    inner = max(1, width)
    gap = 3
    if cell_len(left) + cell_len(right) + gap > inner:
        left = product_ui._trim_cells(
            left, max(10, inner - cell_len(right) - gap))
    if cell_len(left) + cell_len(right) + gap > inner:
        right = product_ui._trim_cells(
            right, max(10, inner - cell_len(left) - gap))
    return left, right


def _bottom_toolbar(self):
    if is_dumb_terminal() or self.ui.width < 48:
        return _PREV["bottom_toolbar"](self)

    palette = self.ui.palette
    width = max(30, self.ui.width)
    left, right = _status_parts(self, width)
    gap = max(1, width - cell_len(left) - cell_len(right))
    body = left + (" " * gap) + right
    if cell_len(body) > width:
        body = product_ui._trim_cells(body, width)

    muted = ui_mod._ansi_of(palette.muted)
    good = ui_mod._ansi_of(palette.good)
    reset = "\x1b[0m"
    rendered = f"{muted}{body}{reset}"
    if "ready" in body:
        rendered = rendered.replace("ready", f"{good}ready{muted}", 1)
    return ANSI(rendered)


async def _on_content(self, text: str) -> None:
    """Streaming and completed replies use the same two-cell content baseline."""
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
    """Install the minimal shell after :mod:`wynxo.product_ui`."""
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

    product_ui._assistant_heading = _assistant_heading
    _INSTALLED = True
