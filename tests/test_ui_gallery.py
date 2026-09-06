"""The screenshot/gallery renderer must exercise the real UI without crashing."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET


def _visible_text(path) -> str:
    """Text a person sees in Rich's SVG, independent of tspans/layout tags."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    return " ".join("".join(root.itertext()).split())


def test_ui_gallery_renders_five_review_screens(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/render_ui_gallery.py", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    files = sorted(tmp_path.glob("*.svg"))
    assert [path.name for path in files] == [
        "01-home.svg",
        "02-conversation.svg",
        "03-tools.svg",
        "04-plan.svg",
        "05-apps.svg",
    ]
    assert all(path.stat().st_size > 500 for path in files)

    home_svg = files[0].read_text(encoding="utf-8")
    home = _visible_text(files[0])
    conversation = _visible_text(files[1])
    tools = _visible_text(files[2])
    plan = _visible_text(files[3])

    # The gallery must be the interactive product shell, not redirected output.
    assert "Ready to build." in home
    assert "/apps" in home
    assert "renderer bug" in conversation
    assert "[ok]" in tools
    assert "list applications" in tools
    assert "[>]" in plan

    # The redesign intentionally does not use giant box/card headings.
    assert "Tool done" not in tools
    assert "Plan 2/5 complete" not in plan

    # Review artifacts must not rely on a CDN font.
    assert "cdnjs.cloudflare.com" not in home_svg
    assert "DejaVu Sans Mono" in home_svg
