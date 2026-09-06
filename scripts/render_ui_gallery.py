"""Render five deterministic Wynxo UI review screens as SVG files.

This is intentionally a developer/review tool, not a second UI. It installs
the same product_ui + clean_ui layers as the real CLI, points the real Rich
console at a recorder, and exercises representative states. Run:

    python scripts/render_ui_gallery.py --out ui-gallery

The output is useful in PRs and bug reports because spacing, wrapping and
terminal-safe glyph decisions are visible without needing a live Ollama model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from wynxo import clean_ui, product_ui
from wynxo.ui import Glyphs, SafeConsole, UI

WIDTH = 112
HEIGHT = 34
WORKSPACE = "/home/elliot/wynxo-AI-Agent-Termianl-CLI"
MODEL = "qwen3-coder:30b"

_FONT_FACE = re.compile(r"\s*@font-face\s*\{.*?\}\s*", re.DOTALL)


class GalleryBar:
    """Tiny stand-in for the live bar so streamed text goes through Rich.

    CodeStreamer normally writes partial prose directly to console.file when
    there is no live region. That is correct for the real terminal, but Rich's
    SVG recorder only sees content that goes through Console.print. A live bar
    keeps the line in Rich until it is committed, which is exactly the path a
    normal interactive agent turn uses while it is answering.
    """

    def __init__(self):
        self.lead = None

    def set_lead(self, lead) -> None:
        self.lead = lead


def make_ui() -> UI:
    ui = UI()
    ui.console = SafeConsole(
        record=True,
        force_terminal=True,
        color_system="truecolor",
        width=WIDTH,
        height=HEIGHT,
        highlight=False,
    )
    ui.g = Glyphs(True)
    ui.width = WIDTH
    ui.narrow = False
    ui.bar = GalleryBar()
    # Home normally re-measures the real TTY. A gallery has a deliberate
    # viewport, so keep it fixed and deterministic.
    ui.refresh_size = lambda: None
    return ui


def _portable_svg(path: Path) -> None:
    """Make Rich's SVG self-contained enough for offline review.

    Rich references Fira Code from a CDN. CI artifacts and chat attachments
    are commonly viewed without network font access, where box drawing then
    falls back unpredictably. Keep the SVG text-based, remove the remote font
    faces, and ask for a broadly available monospace with line-drawing glyphs.
    """
    text = path.read_text(encoding="utf-8")
    text = _FONT_FACE.sub("\n", text)
    text = text.replace(
        "font-family: Fira Code, monospace;",
        'font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace;',
    )
    path.write_text(text, encoding="utf-8")


def save(ui: UI, out: Path, name: str, title: str) -> Path:
    target = out / f"{name}.svg"
    ui.console.save_svg(str(target), title=title, clear=False)
    _portable_svg(target)
    return target


def home_screen(out: Path) -> Path:
    ui = make_ui()
    ui.home(MODEL, WORKSPACE, mode="agent", version="0.1.0")
    return save(ui, out, "01-home", "Wynxo — Home")


def conversation_screen(out: Path) -> Path:
    ui = make_ui()
    ui.user_line(
        "find why the terminal tool crashes, fix it, and keep the input pinned"
    )
    ui.assistant_markdown(
        "I found the renderer bug and fixed the call path. The assistant body "
        "now shares the same baseline as the message rail, so streamed replies "
        "and finished replies line up consistently."
    )
    return save(ui, out, "02-conversation", "Wynxo — Conversation")


def tools_screen(out: Path) -> Path:
    ui = make_ui()
    ui.user_line("open terminal any")
    ui.tool_call(
        "list_applications",
        "terminal",
        "found Konsole and Kitty",
        True,
    )
    ui.tool_call(
        "launch_application",
        "Konsole",
        "launched installed terminal",
        True,
    )
    return save(ui, out, "03-tools", "Wynxo — Tool calls")


def plan_screen(out: Path) -> Path:
    ui = make_ui()
    ui.user_line("upgrade the CLI and keep checking for bugs")
    ui.todos(
        "[x] audit application launching\n"
        "[x] fix tool rendering\n"
        "[>] align streamed assistant replies\n"
        "[ ] run cross-platform tests\n"
        "[ ] merge and review screenshots"
    )
    return save(ui, out, "04-plan", "Wynxo — Plan")


def apps_screen(out: Path) -> Path:
    ui = make_ui()
    ui.user_line("what terminal apps are installed?")
    ui.table(
        ["application", "found in", "target"],
        [
            ("Konsole", "desktop entry", "/usr/share/applications/org.kde.konsole.desktop"),
            ("Kitty", "PATH", "/usr/bin/kitty"),
            ("Alacritty", "PATH", "/usr/bin/alacritty"),
        ],
        title="3 installed applications match 'terminal'",
    )
    return save(ui, out, "05-apps", "Wynxo — Installed applications")


def render(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    # Match bootstrap.py's real install order.
    product_ui.install()
    clean_ui.install()

    # CI is intentionally non-interactive. For a visual gallery we want to
    # exercise the interactive branch at a fixed viewport, not the compact
    # redirected-output fallback that CI itself normally uses.
    product_ui.is_dumb_terminal = lambda: False
    clean_ui.is_dumb_terminal = lambda: False

    # Keep screenshots stable across machines and review times.
    product_ui._clock = lambda: "2:28 AM"
    product_ui.terminal_height = lambda: HEIGHT
    clean_ui.terminal_height = lambda: HEIGHT

    return [
        home_screen(out),
        conversation_screen(out),
        tools_screen(out),
        plan_screen(out),
        apps_screen(out),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="render Wynxo UI review gallery")
    parser.add_argument("--out", type=Path, default=Path("ui-gallery"))
    args = parser.parse_args()

    paths = render(args.out)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
