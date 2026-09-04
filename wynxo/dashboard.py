"""Polished terminal dashboard chrome for WYNXO.

The dashboard is deliberately terminal-native: no fake GUI, no giant ASCII
mascot, and no extra runtime dependency. It uses Rich panels and a compact
half-block pixel illustration so the startup screen feels like an application
rather than a collection of debug boxes.
"""
from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# 40x20 pixel-art canvas.  A human silhouette comes first; the ears are small
# accents and the laptop is clearly in front of the character.
# Two source rows become one terminal row via upper/lower half blocks.
_ART = [
    "0000000000000000000011111111000000000000",
    "0000000000000000011111111111110000000000",
    "0000000000000000111111111111111000000000",
    "0000000000000011111111111111111100000000",
    "0000000000000111111111111111111110000000",
    "0000000000001111111111111111111111000000",
    "0000000000011111111111111111111111100000",
    "0000000000111111111111111111111111110000",
    "0000000001111111111111111111111111111000",
    "0000000011111111111111111111111111111100",
    "0000000111111111111111111111111111111110",
    "0000000111111111111111111111111111111110",
    "0000000111111111111111111111111111111110",
    "0000000011111111111111111111111111111100",
    "0000000011111111111111111111111111111100",
    "0000000001111111111111111111111111111000",
    "0000000000111111111111111111111111110000",
    "0000000000111111111111111111111111110000",
    "0000000000111111111111111111111111110000",
    "0000000000011111111111111111111111100000",
    "0000000000011111111111111111111111100000",
    "0000000000111111111111111111111111110000",
    "0000000001111111111111111111111111111000",
    "0000000011111111111111111111111111111100",
    "0000000111111111111111111111111111111110",
    "0000001111111111111111111111111111111111",
    "0000011111111111111111111111111111111111",
    "0000111111111111111111111111111111111111",
    "0001111111111111111111111111111111111111",
    "0011111111111111111111111111111111111111",
    "0011111111111111111111111111111111111111",
    "0111111111111111111111111111111111111111",
    "0111111111111111111111111111111111111111",
    "1111111111111111111111111111111111111111",
    "1111111111111111111111111111111111111111",
    "1111111111111111111111111111111111111111",
    "1111111111111111111111111111111111111111",
    "1111111111111111111111111111111111111111",
    "1111111111111111111111111111111111111111",
    "1111111111111111111111111111111111111111",
]

# Overlay a face, hair/ears, hoodie, arms and laptop onto the silhouette.
# Each overlay is sparse so the silhouette remains recognizable at a glance.
_OVERLAY = {
    (2, 16): "2", (2, 23): "2", (3, 15): "2", (3, 24): "2",
    (4, 14): "2", (4, 25): "2", (5, 15): "2", (5, 24): "2",
    (6, 16): "1", (6, 23): "1",
    (8, 17): "3", (8, 22): "3",
    (9, 16): "3", (9, 18): "4", (9, 21): "4", (9, 23): "3",
    (10, 17): "5", (10, 22): "5",
    (11, 18): "3", (11, 21): "3",
    (12, 19): "6", (12, 20): "6",
    (14, 12): "2", (14, 27): "2",
    (16, 11): "2", (16, 28): "2",
    (18, 10): "2", (18, 29): "2",
    (21, 10): "7", (21, 29): "7",
    (22, 9): "7", (22, 30): "7",
    (23, 8): "7", (23, 31): "7",
    (24, 7): "7", (24, 32): "7",
    (25, 6): "7", (25, 33): "7",
    (26, 5): "7", (26, 34): "7",
    (27, 5): "7", (27, 34): "7",
    (28, 6): "7", (28, 33): "7",
    (29, 7): "7", (29, 32): "7",
    (30, 8): "7", (30, 31): "7",
    (31, 9): "7", (31, 30): "7",
    (32, 10): "7", (32, 29): "7",
    (33, 11): "7", (33, 28): "7",
    (34, 12): "8", (34, 27): "8",
    (35, 13): "8", (35, 26): "8",
    (36, 14): "8", (36, 25): "8",
    (37, 15): "8", (37, 24): "8",
}

_STYLES = {
    "1": "accent_dim",
    "2": "accent",
    "3": "text",
    "4": "bar_accent",
    "5": "muted",
    "6": "text",
    "7": "bar_bg",
    "8": "faint",
}


def _pixel_art(palette) -> Text:
    pixels = [list(row) for row in _ART]
    for (y, x), value in _OVERLAY.items():
        if 0 <= y < len(pixels) and 0 <= x < len(pixels[y]):
            pixels[y][x] = value

    def style(value: str) -> str:
        return palette.role(_STYLES.get(value, "accent_dim"))

    out = Text()
    for y in range(0, len(pixels), 2):
        top = pixels[y]
        bottom = pixels[y + 1]
        for a, b in zip(top, bottom):
            # Treat the base silhouette as transparent and only render the
            # character/laptop overlays. This avoids a solid purple rectangle.
            if a == "0" and b == "0":
                out.append(" ")
            elif a == "0":
                out.append("▄", style=style(b))
            elif b == "0":
                out.append("▀", style=style(a))
            elif a == b:
                out.append("█", style=style(a))
            else:
                out.append("▀", style=f"{style(a)} on {style(b)}")
        out.append("\n")
    return out


def _nav(palette) -> Panel:
    rows = [
        ("◈", "chat", True),
        ("◇", "tools", False),
        ("□", "files", False),
        ("▣", "system", False),
        ("⚙", "settings", False),
        ("?", "help", False),
    ]
    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)
    table.add_column(width=11)
    for icon, label, active in rows:
        table.add_row(
            Text("▌" if active else " ", style=f"bold {palette.accent}"),
            Text(f"{icon}  {label}", style=f"bold {palette.accent}" if active else palette.muted),
        )
    return Panel(
        table,
        title=Text(" WYNXO ", style=f"bold {palette.accent}"),
        border_style=palette.faint,
        box=ROUNDED,
        padding=(1, 0),
    )


def _banner(self, model: str, endpoint: str, effort: str, workspace: str,
            pet=None, greeting: str = "") -> None:
    """Render the startup shell; normal prompt/streaming remains untouched."""
    del endpoint, pet, greeting

    p = self.palette
    width = max(70, self.width)
    model_short = self.shorten_model(model, max(20, width // 3))
    path_short = self.shorten_path(workspace)

    brand = Text()
    brand.append("wynxo", style=f"bold {p.accent}")
    brand.append("  |  your local ai companion", style=p.text)

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(brand, Text("v0.2.0", style=p.muted))
    header.add_row(
        Text("think  ·  build  ·  explore  ·  together", style=p.muted),
        Text("● LOCAL", style=f"bold {p.good}"),
    )

    hero_art = _pixel_art(p)
    caption = Text()
    caption.append("  ◈ ", style=f"bold {p.accent}")
    caption.append("companion", style=f"bold {p.accent}")
    caption.append("  ·  ready", style=p.muted)
    hero = Panel(
        Group(hero_art, caption),
        title=Text(" companion ", style=p.muted),
        border_style=p.accent,
        box=ROUNDED,
        padding=(0, 1),
    )

    welcome = Text()
    welcome.append("welcome back\n", style=f"bold {p.accent}")
    welcome.append("Your local AI workspace is ready.\n\n", style=p.text)
    welcome.append("model    ", style=p.muted)
    welcome.append(model_short + "\n", style=p.text)
    welcome.append("workspace", style=p.muted)
    welcome.append("  " + path_short + "\n", style=p.text)
    welcome.append("effort   ", style=p.muted)
    welcome.append(effort + "\n", style=p.accent)
    welcome_panel = Panel(
        welcome,
        title=Text(" session ", style=p.muted),
        border_style=p.faint,
        box=ROUNDED,
        padding=(1, 1),
    )

    commands = Text()
    commands.append("quick commands\n", style=f"bold {p.accent}")
    for command, description in (
        ("/help", "show every command"),
        ("/tools", "see available tools"),
        ("/theme", "change the appearance"),
        ("/clear", "clear the screen"),
    ):
        commands.append(f"  {command:<10}", style=p.text)
        commands.append(description + "\n", style=p.muted)

    command_panel = Panel(
        commands,
        title=Text(" shortcuts ", style=p.muted),
        border_style=p.faint,
        box=ROUNDED,
        padding=(1, 1),
    )

    right = Group(welcome_panel, command_panel)
    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(width=28)
    body.add_column(ratio=1)
    body.add_row(_nav(p), right)

    footer = Text()
    footer.append(" model ", style=p.muted)
    footer.append(model_short, style=p.text)
    footer.append("   │   ", style=p.faint)
    footer.append("mode ", style=p.muted)
    footer.append("agent", style=p.accent)
    footer.append("   │   ", style=p.faint)
    footer.append("companion ", style=p.muted)
    footer.append("idle", style=p.accent)
    footer.append("   │   ", style=p.faint)
    footer.append("↑↓ move   Enter select   Ctrl+C stop", style=p.muted)

    self.console.print()
    self.console.print(Panel(header, border_style=p.faint, box=ROUNDED, padding=(0, 1)))
    self.console.print(body)
    self.console.print(Panel(footer, border_style=p.faint, box=ROUNDED, padding=(0, 1)))
    self.console.print()


def install() -> None:
    """Install the dashboard banner without replacing the main UI renderer."""
    from .ui import UI
    UI.banner = _banner
