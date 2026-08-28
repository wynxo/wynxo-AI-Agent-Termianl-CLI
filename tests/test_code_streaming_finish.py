from __future__ import annotations

import re

from wynxo.tui import Transcript
from wynxo.ui import CodeStreamer, UI


def test_finish_flushes_partial_code_line() -> None:
    page = Transcript(width=60)
    ui = UI()
    ui.console = page.console
    ui.width = 60
    ui.live_ok = False

    streamer = CodeStreamer(ui, indent="  ", code=False, literal=True)
    streamer.feed("def answer():\\n")
    streamer.feed("    return 42")
    streamer.finish()
    page.drain()

    body = re.sub(r"\\x1b\\[[0-9;]*m", "", "\\n".join(page.lines))
    assert "    return 42" in body
