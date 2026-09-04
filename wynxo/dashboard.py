"""Polished terminal dashboard chrome for WYNXO.

The dashboard is terminal-native and intentionally keeps the real REPL intact.
It adds a compact app shell, navigation rail, session cards, and a procedural
half-block pixel illustration of a human catboy sitting behind a laptop.
"""
from __future__ import annotations

import math

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Pixel labels.  The renderer packs two vertical pixels into one terminal cell
# using ▀/▄/█.  This gives much more detail than ASCII line art while staying
# completely dependency-free and working over SSH.
_PIXEL_STYLES = {
    "H": "accent_dim",   # hair / ears
    "h": "accent",       # hair highlights
    "S": "text",         # skin
    "E": "bar_accent",   # eyes
    "M": "muted",        # mouth
    "O": "accent_dim",   # hoodie
    "o": "muted",        # hoodie shadow
    "L": "bar_bg",       # laptop bezel
    "l": "faint",        # laptop body
    "C": "bar_accent",   # screen content
    "D": "faint",        # desk
    "T": "warn",         # mug / small desk accent
}


def _make_catboy(width: int = 40, height: int = 40) -> list[list[str]]:
    """Create a recognizable seated catboy silhouette in a small pixel canvas."""
    px = [["0"] * width for _ in range(height)]

    def put(x: int, y: int, value: str) -> None:
        if 0 <= x < width and 0 <= y < height:
            px[y][x] = value

    def ellipse(cx: float, cy: float, rx: float, ry: float, value: str) -> None:
        for y in range(max(0, int(cy - ry)), min(height, int(cy + ry + 1))):
            for x in range(max(0, int(cx - rx)), min(width, int(cx + rx + 1))):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                    put(x, y, value)

    def rect(x0: int, y0: int, x1: int, y1: int, value: str) -> None:
        for y in range(max(0, y0), min(height, y1 + 1)):
            for x in range(max(0, x0), min(width, x1 + 1)):
                put(x, y, value)

    # Hoodie / seated torso.
    ellipse(19, 29, 14, 10, "O")
    ellipse(10, 30, 6, 7, "o")
    ellipse(29, 30, 6, 7, "o")

    # Small neck behind the face.
    rect(17, 19, 22, 24, "S")

    # Human head and hair mass.
    ellipse(19, 12, 10, 9, "S")
    ellipse(19, 10, 11, 9, "H")
    ellipse(12, 12, 5, 7, "H")
    ellipse(26, 12, 5, 7, "H")

    # Cat ears: small triangles, not a giant cat head.
    for i in range(6):
        for x in range(14 - i, 14 + i + 1):
            put(x, 3 + i, "H")
        for x in range(24 - i, 24 + i + 1):
            put(x, 3 + i, "H")
    for i in range(4):
        for x in range(15 - i // 2, 15 + i // 2 + 1):
            put(x, 5 + i, "h")
        for x in range(23 - i // 2, 23 + i // 2 + 1):
            put(x, 5 + i, "h")

    # Fringe / hair falling over the forehead.
    for x, depth in ((13, 5), (15, 4), (17, 3), (19, 4), (21, 3), (23, 4), (25, 5)):
        for y in range(7, 7 + depth):
            put(x, y, "H")
    for x in (16, 20, 24):
        put(x, 9, "h")

    # Eyes, nose, mouth: restrained so it reads as a face rather than a mask.
    put(16, 13, "E")
    put(17, 13, "E")
    put(22, 13, "E")
    put(23, 13, "E")
    put(19, 15, "M")
    put(20, 15, "M")

    # Hoodie strings.
    for y in range(20, 26):
        put(17, y, "h")
        put(22, y, "h")
    put(17, 26, "h")
    put(22, 26, "h")

    # Forearms reaching around the laptop.
    for y in range(27, 34):
        for x in range(7 + (y - 27) // 2, 16):
            put(x, y, "o")
        for x in range(23, 32 - (y - 27) // 2):
            put(x, y, "o")

    # Hands on the keyboard.
    ellipse(14, 31, 5, 2, "S")
    ellipse(25, 31, 5, 2, "S")

    # Laptop screen in front of the body.
    rect(10, 25, 29, 31, "L")
    rect(12, 26, 27, 29, "C")
    # Screen lines / cursor.
    for x in range(14, 25, 3):
        put(x, 27, "h")
    for x in range(14, 22, 4):
        put(x, 28, "H")
    put(23, 28, "E")

    # Laptop base.
    for y in range(32, 35):
        for x in range(7, 33):
            put(x, y, "l" if y != 34 else "L")
    for x in range(12, 28, 2):
        put(x, 33, "h")

    # Desk edge and tiny mug for environmental context.
    rect(4, 35, 35, 36, "D")
    rect(31, 31, 34, 35, "D")
    rect(32, 29, 34, 31, "T")

    return px


def _pixel_art(palette) -> Text:
    pixels = _make_catboy()

    def style(value: str) -> str:
        role = _PIXEL_STYLES.get(value, "accent_dim")
        return palette.role(role)

    out = Text()
    for y in range(0, len(pixels), 2):
        for a, b in zip(pixels[y], pixels[y + 1]):
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
    items = [
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
    for icon, label, active in items:
        table.add_row(
            Text("▌" if active else " ", style=f"bold {palette.accent}"),
            Text(
                f"{icon}  {label}",
                style=f"bold {palette.accent}" if active else palette.muted,
            ),
        )
    return Panel(
        table,
        title=Text(" WYNXO ", style=f"bold {palette.accent}"),
        border_style=palette.faint,
        box=self_box(palette),
        padding=(1, 0),
    )


def self_box(palette):
    # Keep one box style for the whole dashboard.  Rich's rounded box is
    # supported on the terminals WYNXO targets; ASCII fallback is handled by
    # the existing UI for restricted terminals.
    from rich.box import ROUNDED
    return ROUNDED


def _banner(self, model: str, endpoint: str, effort: str, workspace: str,
            pet=None, greeting: str = "") -> None:
    """Render the startup dashboard; the actual REPL remains unchanged."""
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

    hero_caption = Text()
    hero_caption.append("◈ companion", style=f"bold {p.accent}")
    hero_caption.append("  ·  sitting at the desk", style=p.muted)
    hero = Panel(
        Group(_pixel_art(p), hero_caption),
        title=Text(" companion ", style=p.muted),
        border_style=p.accent,
        box=self_box(p),
        padding=(0, 1),
    )

    session = Text()
    session.append("welcome back\n", style=f"bold {p.accent}")
    session.append("Your local AI workspace is ready.\n\n", style=p.text)
    session.append("model      ", style=p.muted)
    session.append(model_short + "\n", style=p.text)
    session.append("workspace  ", style=p.muted)
    session.append(path_short + "\n", style=p.text)
    session.append("effort     ", style=p.muted)
    session.append(effort, style=p.accent)

    session_panel = Panel(
        session,
        title=Text(" session ", style=p.muted),
        border_style=p.faint,
        box=self_box(p),
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
        box=self_box(p),
        padding=(1, 1),
    )

    right = Group(session_panel, command_panel)
    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(width=28)
    body.add_column(ratio=1)
    body.add_row(_nav(p), Group(hero, right))

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
    footer.append("Ctrl+C stop", style=p.muted)

    self.console.print()
    self.console.print(Panel(header, border_style=p.faint, box=self_box(p), padding=(0, 1)))
    self.console.print(body)
    self.console.print(Panel(footer, border_style=p.faint, box=self_box(p), padding=(0, 1)))
    self.console.print()


def install() -> None:
    """Install dashboard startup chrome without replacing the main renderer."""
    from .ui import UI
    UI.banner = _banner
