"""Optional visual dashboard chrome for the scrolling WYNXO CLI.

The core CLI deliberately remains a normal scrollback-friendly terminal
application. This module only replaces the startup banner with a richer,
reference-inspired dashboard: dark cards, a compact navigation rail, a large
catboy-at-a-laptop illustration, and a clean status footer.

It is installed by bootstrap before cli imports UI, so existing UI methods,
tests, streaming, tools, and prompt behaviour stay untouched.
"""
from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# A deliberately terminal-native illustration. It is not a huge cat face:
# the silhouette has ears, hair/head, neck, shoulders, arms, and a laptop.
# Keeping it as text also means it works over SSH and never needs an image
# renderer or a graphical dependency.
_CATBOY = (
    "            /\\        /\\            \n"
    "           /##\\______/##\\           \n"
    "          /##############\\          \n"
    "         /###  o    o  ###\\         \n"
    "        |###     __     ###|        \n"
    "        |###   \\____/   ###|        \n"
    "         \\###  ____   ###/         \n"
    "          \\##############/          \n"
    "           \\####/\\####/            \n"
    "            |###|  |###|             \n"
    "        ____|###|__|###|____         \n"
    "      /\\    \\  /\\  /    /\\      \n"
    "     /  \\____\\/  \\/____/  \\     \n"
    "    /        /  /\\  \\        \\    \n"
    "   /________/__/  \\__\\________\\   \n"
    "        ___/  \\____/  \\___        \n"
    "      _/____________________\\_      \n"
    "     |   W Y N X O  •  CODE   |     \n"
    "     |_________________________|     "
)


def _nav_item(icon: str, label: str, active: bool, accent: str, muted: str) -> Text:
    line = Text()
    line.append("▌ " if active else "  ", style=f"bold {accent}" if active else muted)
    line.append(f"{icon}  ", style=accent if active else muted)
    line.append(label, style=f"bold {accent}" if active else muted)
    return line


def _banner(self, model: str, endpoint: str, effort: str, workspace: str,
            pet=None, greeting: str = "") -> None:
    """Reference-inspired startup dashboard, while keeping the CLI model intact."""
    del endpoint, pet, greeting

    accent = self.palette.accent
    muted = self.palette.muted
    faint = self.palette.faint
    good = self.palette.good
    bar_bg = self.palette.bar_bg

    # Header: intentionally compact, like an application shell rather than
    # the old multiline terminal title card.
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    left = Text()
    left.append("wynxo", style=f"bold {accent}")
    left.append("  |  your local ai companion", style=muted)
    sub = Text("think  ·  build  ·  explore  ·  together", style=faint)
    header.add_row(left, Text("v0.2.0", style=faint))
    header.add_row(sub, Text("LOCAL", style=good))

    nav = Table.grid(padding=(0, 1))
    nav.add_column(width=3)
    nav.add_column()
    items = [
        ("◌", "chat", True),
        ("⚒", "tools", False),
        ("□", "files", False),
        ("▣", "system", False),
        ("⚙", "settings", False),
        ("?", "help", False),
    ]
    for icon, label, active in items:
        nav.add_row(_nav_item(icon, label, active, accent, muted))

    nav_panel = Panel(
        nav,
        title=Text(" WYNXO ", style=f"bold {accent}"),
        border_style=faint,
        box=self.box,
        padding=(1, 1),
    )

    art = Text()
    for line in _CATBOY.splitlines():
        art.append(line + "\n", style=f"bold {accent}")
    art.append("\n")
    art.append("     ◈ companion", style=f"bold {accent}")
    art.append("   · sitting at the desk", style=faint)

    hero = Panel(
        art,
        title=Text(" companion / idle ", style=faint),
        border_style=accent,
        box=self.box,
        padding=(0, 1),
    )

    model_short = self.shorten_model(model, max(24, self.width // 2))
    path_short = self.shorten_path(workspace)

    chat = Table.grid(padding=(0, 0))
    chat.add_column()
    bubble = Text()
    bubble.append("thinking...", style=muted)
    chat.add_row(Panel(bubble, border_style=faint, box=self.box, padding=(0, 1)))

    prompt_hint = Text()
    prompt_hint.append("❯ ", style=accent)
    prompt_hint.append("hi", style="bold")
    chat.add_row(Panel(prompt_hint, border_style=accent, box=self.box, padding=(0, 1)))

    intro = Text()
    intro.append("Hey!  ", style=f"bold {accent}")
    intro.append("Your local AI companion is ready.\n", style="")
    intro.append("Ask me to code, inspect files, run tools, or just chat.", style=muted)
    chat.add_row(Panel(intro, border_style=faint, box=self.box, padding=(1, 1)))

    suggestions = Text()
    suggestions.append("suggestions:\n", style=f"bold {accent}")
    suggestions.append("❯ /help     ", style=muted)
    suggestions.append("show all commands\n", style=faint)
    suggestions.append("❯ /tools    ", style=muted)
    suggestions.append("list available tools\n", style=faint)
    suggestions.append("❯ /theme    ", style=muted)
    suggestions.append("change appearance\n", style=faint)
    suggestions.append("❯ /clear    ", style=muted)
    suggestions.append("clear the screen", style=faint)
    chat.add_row(Panel(suggestions, border_style=faint, box=self.box, padding=(1, 1)))

    main = Table.grid(expand=True, padding=(0, 1))
    main.add_column(width=20)
    main.add_column(ratio=1)
    main.add_row(nav_panel, Table.grid())
    # Put the hero beside the conversation, matching the reference's
    # left-character / right-chat composition.
    main = Table.grid(expand=True, padding=(0, 1))
    main.add_column(width=20)
    main.add_column(ratio=1)
    right = Table.grid(expand=True)
    right.add_column(ratio=1)
    right.add_row(chat)
    main.add_row(Table.grid(), right)

    # The hero is deliberately rendered as a second left column block after
    # the nav. This keeps the terminal layout readable even when the width is
    # smaller than the reference screenshot.
    left_stack = Group(nav_panel, hero)
    main = Table.grid(expand=True, padding=(0, 1))
    main.add_column(width=38)
    main.add_column(ratio=1)
    main.add_row(left_stack, chat)

    footer = Text()
    footer.append(" model: ", style=faint)
    footer.append(model_short, style=muted)
    footer.append("    |    mode: ", style=faint)
    footer.append("agent", style=accent)
    footer.append("    |    workspace: ", style=faint)
    footer.append(path_short, style=muted)
    footer.append("    |    companion: ", style=faint)
    footer.append("idle  ●○○○", style=accent)

    self.console.print()
    self.console.print(Panel(header, border_style=faint, box=self.box, padding=(0, 1)))
    self.console.print(main)
    self.console.print(Panel(footer, border_style=faint, box=self.box, padding=(0, 1)))
    self.console.print()


def install() -> None:
    """Install the dashboard banner without changing the rest of UI."""
    from .ui import UI
    UI.banner = _banner
