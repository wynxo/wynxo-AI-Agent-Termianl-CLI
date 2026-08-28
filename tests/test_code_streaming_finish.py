from __future__ import annotations

import re

from wynxo.tui import Transcript
from wynxo.ui import CodeStreamer, UI


def test_finish_flushes_partial_fenced_code_line() -> None:
    page = Transcript(width=60)
    ui = UI()
    ui.console = page.console
    ui.width = 60
    ui.live_ok = False

    streamer = CodeStreamer(ui, indent="  ")
    streamer.feed("```python\n")
    streamer.feed("def answer():\n")
    streamer.feed("    return 42")
    streamer.finish()
    page.drain()

    body = re.sub(r"\\x1b\\[[0-9;]*m", "", "\\n".join(page.lines))
    assert "def answer():" in body
    assert "    return 42" in body
