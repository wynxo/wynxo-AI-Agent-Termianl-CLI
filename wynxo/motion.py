"""A small frame-based ASCII animation engine.

One scheduler drives every timed animation in the interface -- the speech
waveforms, the one-shot effects, the preview loops. One task, one timing
source, no per-widget timers, and everything stops with the app.

The mascot itself deliberately does NOT live here. Its face already
advances on every repaint (the chat layout repaints at a steady clip and
the rich bar redraws during turns), so a second timer would only fight the
renderer. The scheduler exists for the things that need explicit timing
independent of repaints: a waveform that must look like a waveform, an
effect that must play once and leave.

Every scene degrades the same way, in the same order: reduced-motion mode
keeps one static frame, a non-unicode terminal swaps in the ASCII set, and
a narrow terminal swaps in the compact set. Nothing here ever changes the
layout -- animations render into rows that already exist.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Scene:
    name: str
    frames: tuple[str, ...]
    label: str = ""
    fps: float = 6.0
    loops: bool = True
    ascii: tuple[str, ...] | None = None
    """: A plain-ASCII set for terminals without the unicode glyphs."""
    compact: tuple[str, ...] | None = None
    """: A shorter set for narrow terminals."""


# -- the scenes -------------------------------------------------------------
#
# All original. The face reuses the pet's glyphs (≽^•⩊•^≼) so the scenes
# read as the same character as the one in the status bar, which is what
# makes a showcase look like one product rather than a clip-art drawer.

def _wave(bars: tuple[str, ...]) -> str:
    return " ".join(bars)


WAVE_LOW = "▁▂▃▄▅▆▅▄▃▂"
WAVE_MID = "▂▄▆█▇▆▄▂▁▃"
WAVE_HIGH = "▄▆▇█▆▄▂▁▂▄"
WAVE_ASCII_LOW = "~_-_-_~"
WAVE_ASCII_MID = "-_~_~_-"
WAVE_ASCII_HIGH = "_~-~-~_"

SCENES: dict[str, Scene] = {
    "listening": Scene(
        "listening",
        (_wave(WAVE_LOW), _wave(WAVE_MID), _wave(WAVE_HIGH), _wave(WAVE_MID)),
        label="microphone picking up sound",
        fps=8.0,
        ascii=(WAVE_ASCII_LOW, WAVE_ASCII_MID, WAVE_ASCII_HIGH, WAVE_ASCII_MID),
        compact=("·", "·", "·"),
    ),
    "speaking": Scene(
        "speaking",
        (_wave(WAVE_HIGH), _wave(WAVE_MID), _wave(WAVE_LOW), _wave(WAVE_MID)),
        label="voice synthesised aloud",
        fps=8.0,
        ascii=(WAVE_ASCII_HIGH, WAVE_ASCII_MID, WAVE_ASCII_LOW, WAVE_ASCII_MID),
        compact=("·", "·", "·"),
    ),
    "transcribing": Scene(
        "transcribing",
        ("  ·", " · ", "·  ", " · "),
        label="words being turned into text",
        fps=10.0,
        ascii=("  .", " . ", ".  ", " . "),
        compact=("·", "·", "·"),
    ),
    # Small 3-line scenes for the showcase (/pet show, /animate). The face
    # is the same one the status bar uses, with a tiny body hint per state.
    "idle": Scene(
        "idle",
        (
            "   ≽^•⩊•^≼\n   ˘˘˘˘˘˘\n",
            "   ≽^•⩊•^≼\n   ˘˘˘˘˘˘\n",
            "   ≽^-⩊-^≼\n   ˘˘˘˘˘˘\n",
        ),
        label="waiting",
        fps=3.0,
        ascii=("   =^.^=\n   -----\n",) * 3,
    ),
    "thinking": Scene(
        "thinking",
        (
            " ≽^˘⩊•^≼  ·\n  ˘˘˘˘˘˘\n",
            " ≽^•⩊˘^≼   ·\n  ˘˘˘˘˘˘\n",
            " ≽^˘⩊•^≼  ·\n  ˘˘˘˘˘˘\n",
        ),
        label="working it out",
        fps=5.0,
        ascii=(" =^o.^=  .\n  -----\n", " =^.o^=   .\n  -----\n", " =^o.^=  .\n  -----\n"),
    ),
    "working": Scene(
        "working",
        (
            " ≽^•̀⩊•́^≼  ⌨\n  ˘˘˘˘˘˘\n",
            " ≽^•́⩊•̀^≼  ⌨\n  ˘˘˘˘˘˘\n",
        ),
        label="typing away",
        fps=6.0,
        ascii=(" =^>.<^=  >_\n  -----\n", " =^>.>^=  <_\n  -----\n"),
    ),
    "reading": Scene(
        "reading",
        (
            " ≽^◉⩊◉^≼  ▓\n  ˘˘˘˘˘˘\n",
            " ≽^◉⩊◉^≼  ▒\n  ˘˘˘˘˘˘\n",
            " ≽^-⩊-^≼  ▓\n  ˘˘˘˘˘˘\n",
        ),
        label="looking something up",
        fps=4.0,
        ascii=(" =^O.O^=  |\n  -----\n",) * 3,
    ),
    "running": Scene(
        "running",
        (
            " ≽^•⩊•^≼ฅ  /\\_/\n  ˘˘˘˘˘˘\n",
            " ≽^•⩊•^≼ﾉ  \\_/\\\n  ˘˘˘˘˘˘\n",
        ),
        label="running",
        fps=8.0,
        ascii=(" =^.^=/  /\\\n  -----\n", " =^.^=\\\\  \\/\n  -----\n"),
    ),
    "sleepy": Scene(
        "sleepy",
        (
            " ≽^-⩊-^≼  z\n  ˘˘˘˘˘˘\n",
            " ≽^-⩊-^≼   z\n  ˘˘˘˘˘˘\n",
        ),
        label="dozing off",
        fps=2.0,
        ascii=(" =^-.-^=  z\n  -----\n", " =^-.-^=   z\n  -----\n"),
    ),
    "happy": Scene(
        "happy",
        (
            " ≽^≧⩊≦^≼  ✦\n  ˘˘˘˘˘˘\n",
            " ≽^ᵕ⩊ᵕ^≼   ✦\n  ˘˘˘˘˘˘\n",
        ),
        label="pleased with how it went",
        fps=6.0,
        ascii=(" =^_^=  *\n  -----\n", " =^v^=   *\n  -----\n"),
    ),
    "error": Scene(
        "error",
        (
            " ≽^×⩊×^≼  !\n  ˘˘˘˘˘˘\n",
            " ≽^╥⩊╥^≼  !\n  ˘˘˘˘˘˘\n",
        ),
        label="hit a wall",
        fps=4.0,
        ascii=(" =^@.@^=  !\n  -----\n", " =^x.x^=  !\n  -----\n"),
    ),
    "sparkle": Scene(
        "sparkle",
        ("      ✦", "   ·  ✦  ·", "  ·  ✦  ·", "   ·  ✦  ·", "      ✦"),
        label="a little confetti",
        fps=8.0,
        loops=False,
        ascii=("      *", "   .  *  .", "  .  *  .", "   .  *  .", "      *"),
        compact=("*", "*", "*"),
    ),
}

# Speech states -> scene, and pet moods -> scene, so one lookup answers
# "what is the cat doing right now".
SPEECH_SCENES = {
    "listening": "listening",
    "transcribing": "transcribing",
    "speaking": "speaking",
}

MOOD_SCENES = {
    "idle": "idle", "thinking": "thinking", "working": "working",
    "reading": "reading", "running": "running", "happy": "happy",
    "sad": "error", "asking": "thinking", "sleepy": "sleepy",
}


def scene_for(name: str) -> Scene:
    """The scene for a state name, defaulting to the idle scene."""
    key = (name or "").strip().lower()
    if key in SCENES:
        return SCENES[key]
    if key in MOOD_SCENES:
        return SCENES[MOOD_SCENES[key]]
    if key in SPEECH_SCENES:
        return SCENES[SPEECH_SCENES[key]]
    return SCENES["idle"]


def select(scene: Scene, *, unicode: bool = True, width: int = 80,
           reduced: bool = False) -> tuple[str, ...]:
    """The frames that fit here: static under reduced motion, ASCII on a
    non-unicode terminal, compact on a narrow one, full set otherwise."""
    if reduced:
        return (scene.frames[0],)
    if not unicode and scene.ascii:
        return scene.ascii
    if width < 46 and scene.compact:
        return scene.compact
    return scene.frames


def preview(name: str, n: int = 3, *, unicode: bool = True,
            width: int = 80, reduced: bool = False) -> str:
    """A few frames of a scene side by side, for /animate and /pet.

    Deterministic: no timers, just the frame sequence laid out as a strip.
    A looping scene shows the first ``n`` frames; a one-shot shows the
    frames it has, padded with blanks so the strip does not shrink.
    """
    scene = scene_for(name)
    frames = select(scene, unicode=unicode, width=width, reduced=reduced)
    if scene.loops:
        cycle = [frames[i % len(frames)] for i in range(n)]
    else:
        cycle = list(frames) + [""] * max(0, n - len(frames))
    rows = [frame.split("\n") for frame in cycle]
    height = max(len(r) for r in rows) if rows else 1
    out = []
    for row in range(height):
        out.append("   ".join((r[row] if row < len(r) else "").rstrip()
                              for r in rows).rstrip())
    return "\n".join(out).rstrip()


# -- the scheduler ----------------------------------------------------------

@dataclass
class _Anim:
    name: str
    frames: tuple[str, ...]
    fps: float
    loops: bool
    callback: Callable[[str], None]
    index: int = 0
    next_at: float = 0.0


class MotionScheduler:
    """One timing loop for every timed animation.

    Register an animation with a callback that receives the current frame;
    the loop fires it at the scene's fps. One-shot scenes unregister
    themselves after their last frame. ``close()`` stops everything and can
    be called twice. Reduced-motion mode delivers a single static frame at
    registration and never starts the loop.

    ``step()`` advances frames synchronously, which is what the tests use --
    the engine is fully deterministic without an event loop.
    """

    def __init__(self, reduced: bool = False):
        self.reduced = reduced
        self._animations: dict[str, _Anim] = {}
        self._task: asyncio.Task | None = None
        self._closed = False

    @property
    def active(self) -> list[str]:
        return list(self._animations)

    def register(self, name: str, scene: Scene, callback,
                 *, fps: float | None = None, unicode: bool = True,
                 width: int = 80) -> _Anim | None:
        """Start (or replace) an animation. Returns the handle, or None in
        reduced-motion mode after the static frame has been delivered."""
        frames = select(scene, unicode=unicode, width=width, reduced=self.reduced)
        if not frames:
            return None
        if self.reduced:
            try:
                callback(frames[0])
            except Exception:
                pass
            self._animations.pop(name, None)
            return None
        anim = _Anim(name, frames, fps or scene.fps, scene.loops, callback)
        anim.next_at = time.monotonic()
        self._animations[name] = anim
        self._ensure_task()
        return anim

    def unregister(self, name: str) -> None:
        self._animations.pop(name, None)

    def stop_all(self) -> None:
        self._animations.clear()

    def step(self, name: str | None = None) -> None:
        """Advance one frame synchronously (tests, non-async callers)."""
        targets = (list(self._animations.values()) if name is None
                   else [self._animations[name]] if name in self._animations
                   else [])
        for anim in targets:
            try:
                anim.callback(anim.frames[anim.index])
            except Exception:
                pass
            if not anim.loops and anim.index >= len(anim.frames) - 1:
                self.unregister(anim.name)
                continue
            anim.index = (anim.index + 1) % len(anim.frames)
            anim.next_at = time.monotonic() + 1.0 / anim.fps

    # -- lifecycle ---------------------------------------------------------

    def _ensure_task(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return        # no running loop: step() drives it instead
        self._task = loop.create_task(self._run())

    async def _run(self) -> None:
        try:
            while not self._closed:
                now = time.monotonic()
                due = [a for a in self._animations.values()
                       if a.next_at <= now]
                if not due:
                    nexts = [a.next_at for a in self._animations.values()]
                    if not nexts:
                        await asyncio.sleep(3600)
                        continue
                    await asyncio.sleep(max(0.0, min(nexts) - now))
                    continue
                for anim in due:
                    if anim.name not in self._animations:
                        continue
                    try:
                        anim.callback(anim.frames[anim.index])
                    except Exception:
                        pass
                    if not anim.loops and anim.index >= len(anim.frames) - 1:
                        self.unregister(anim.name)
                        continue
                    anim.index = (anim.index + 1) % len(anim.frames)
                    anim.next_at = time.monotonic() + 1.0 / anim.fps
        except asyncio.CancelledError:
            pass

    def close(self) -> None:
        """Stop the loop and drop every animation. Idempotent."""
        self._closed = True
        self._animations.clear()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def aclose(self) -> None:
        task = self._task
        self.close()
        if task is not None and not task.done():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
