"""The screenshot/gallery renderer must exercise the real UI without crashing."""

from __future__ import annotations

import subprocess
import sys


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

    home = files[0].read_text(encoding="utf-8")
    conversation = files[1].read_text(encoding="utf-8")
    tools = files[2].read_text(encoding="utf-8")
    plan = files[3].read_text(encoding="utf-8")

    # The gallery must be the interactive product shell, not redirected output.
    assert "Ready to build." in home
    assert "/apps" in home
    assert "renderer bug" in conversation
    assert "[ok]" in tools
    assert "list applications" in tools
    assert "[&gt;]" in plan or "[>]" in plan

    # The redesign intentionally does not use giant box/card borders.
    assert "Tool  done" not in tools
    assert "Plan  2/5 complete" not in plan

    # Review artifacts must not rely on a CDN font.
    assert "cdnjs.cloudflare.com" not in home
    assert "DejaVu Sans Mono" in home
