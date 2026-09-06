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
