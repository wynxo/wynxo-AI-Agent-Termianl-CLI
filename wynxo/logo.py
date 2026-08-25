"""The start-up logo: art fitted to the terminal, lit by a moving gradient.

Three things decide whether a splash screen is a pleasure or an obstacle.

It has to fit. Art is drawn at whatever size its author had, routinely wider
than the terminal it lands in, and a logo that wraps is worse than no logo.
It is resampled rather than cropped, so the picture survives the shrink.

It has to be brief. Two thirds of a second, once, and never on the way to
somewhere else -- anything longer is a thing you sit through rather than
enjoy, every single time you start the program.

And it has to know when not to run. A pipe, a dumb terminal, `-p` for one
answer, animations turned off: in each case the right amount of logo is
none.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from rich.text import Text

from . import asciiart

ART_DIR = Path(__file__).resolve().parent / "art"

FRAME_TIME = 0.045
FRAMES = 16
"""Roughly two thirds of a second in total."""

BAND = 3
"""Columns per colour band. Per-character colour looks no better at this
size and costs a few thousand style objects a frame."""

# Pink through magenta and violet to red, and back. Sampled rather than
# computed so the ends meet: a hue that wraps through green on its way home
# would put a lime stripe across her face.
SWEEP = [
    (255, 120, 200), (255,  96, 190), (246,  74, 186), (228,  64, 190),
    (204,  62, 200), (176,  70, 214), (150,  84, 226), (132, 104, 236),
    (150,  84, 226), (176,  70, 214), (204,  62, 200), (228,  64, 190),
    (246,  74, 186), (255,  96, 190), (255, 120, 200), (255, 150, 205),
]


def available() -> list[str]:
    """The logos on disk, by name."""
    try:
        return sorted(p.stem for p in ART_DIR.glob("*.txt"))
    except OSError:
        return []


def read(name: str) -> str:
    try:
        return (ART_DIR / f"{name}.txt").read_text(encoding="utf-8",
                                                   errors="replace")
    except OSError:
        return ""


def fit(art: str, width: int, max_height: int) -> list[str]:
    """The art at the largest size that fits, as lines of text."""
    if not art.strip():
        return []
    rows = [r for r in art.split("\n")]
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    if not rows:
        return []

    source_w = max(len(r) for r in rows)
    source_h = len(rows)
    width = max(8, min(width, source_w))
    height = max(1, round(width * source_h / source_w))
    if height > max_height:
        # Height is the binding constraint on a short terminal, so work back
        # from it rather than letting the art run off the top.
        height = max(1, max_height)
        width = max(8, min(width, round(height * source_w / source_h)))

    if width >= source_w and height >= source_h:
        # It already fits. Returned untouched rather than round-tripped
        # through the ink ramp, which substitutes a character of similar
        # weight for every one -- fine for a photograph, ruinous for
        # hand-drawn line art, where it turns every / and \ into a +.
        return [r.rstrip() for r in rows]

    grid = asciiart.normalise(asciiart.from_text(art, width, height))
    return asciiart.render(grid, style="simple").split("\n")


def colour_at(row: int, column: int, phase: int) -> str:
    """The sweep colour for a cell, as a hex string.

    Diagonal: the row is folded into the offset, so the band travels across
    and down together rather than as a flat vertical wipe.
    """
    index = (column // BAND + row + phase) % len(SWEEP)
    r, g, b = SWEEP[index]
    return f"#{r:02x}{g:02x}{b:02x}"


def frame(lines: list[str], phase: int) -> Text:
    """One frame, coloured in bands."""
    out = Text()
    for row, line in enumerate(lines):
        column = 0
        while column < len(line):
            chunk = line[column:column + BAND]
            if chunk.strip():
                out.append(chunk, style=colour_at(row, column, phase))
            else:
                out.append(chunk)
            column += BAND
        out.append("\n")
    return out


def should_play(ui, animations: bool) -> bool:
    """Whether a logo is wanted here at all."""
    if not animations:
        return False
    try:
        if not ui.console.is_terminal:
            return False
    except Exception:
        return False
    return not getattr(ui, "narrow", False)


async def play(ui, name: str = "wyn", animations: bool = True) -> bool:
    """Show the logo, animated where the terminal allows it.

    Returns whether anything was drawn, so the caller can decide what else
    the start-up should say.
    """
    art = read(name)
    if not art:
        return False

    width = max(20, min(getattr(ui, "width", 80) - 2, 110))
    # Half the screen at most: the logo is a greeting, not the session.
    lines = fit(art, width, max_height=max(6, _rows(ui) // 2))
    if not lines:
        return False

    if not should_play(ui, animations) or not ui.live_ok:
        # Still coloured, just not moving. A static frame is the right
        # fallback everywhere a repainting widget cannot go -- which
        # includes the chat layout's transcript.
        ui.console.print(frame(lines, phase=0))
        return True

    from rich.live import Live

    try:
        with Live("", console=ui.console, refresh_per_second=30,
                  transient=True) as live:
            for step in range(FRAMES):
                live.update(frame(lines, phase=step))
                await asyncio.sleep(FRAME_TIME)
    except Exception:
        pass          # a logo is never worth failing a start-up over
    # Printed after the animation so it stays on screen: Live is transient,
    # and the point of the sweep is to arrive at the picture, not to erase it.
    ui.console.print(frame(lines, phase=FRAMES - 1))
    return True


def _rows(ui) -> int:
    """How many rows the screen actually has.

    Off the terminal rather than the console: under the chat layout the
    console is a buffer with a nominal height of ten thousand, so asking it
    reported a screen big enough for any logo and the cap never applied --
    the picture filled the window and pushed everything else off it.
    """
    import shutil

    try:
        return max(10, shutil.get_terminal_size((80, 24)).lines)
    except (OSError, ValueError):
        return 24
