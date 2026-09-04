"""Letting text reach the screen the way it was written: a character at a time.

A model does not produce its answer smoothly. Ollama sends one token per
chunk -- two to six characters -- and the terminal is repainted at a fixed
frame rate, so what lands on screen is whatever accumulated between frames:
``pri``, then ``nt(\"hel``, then ``lo\")``. The text is live, and it still
reads as a thing being pasted in lumps rather than a thing being typed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from math import ceil

TICK = 1.0 / 40
"""Seconds between pieces.

The stream and the live UI use the same cadence, fast enough to make short
model chunks look continuous instead of arriving in visible jumps. Forty
frames per second is still cheap for a terminal, while the live-region
throttle prevents a faster producer from forcing extra redraws.
"""

LAG = 0.25
"""The most the display may trail the model, in seconds."""

Write = Callable[[str], None]


class Typewriter:
    """Holds streamed text and lets it out at a steady rate.

    ``lock`` is the caller's output lock: the drain loop takes it around
    slicing *and* writing together, so a :meth:`flush` from another task
    cannot get between the two and put the tail of a sentence after the line
    that was supposed to follow it.
    """

    def __init__(self, lock: asyncio.Lock, *, tick: float = TICK,
                 lag: float = LAG):
        self._lock = lock
        self.tick = tick
        self.lag = lag
        self.runs: list[list] = []
        self._pace = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def pending(self) -> int:
        return sum(len(text) for _, text in self.runs)

    def feed(self, write: Write, text: str) -> None:
        """Take generated text for ``write``. The caller holds the lock."""
        if not text:
            return
        if self._task is None:
            write(text)
            return
        if self.runs and self.runs[-1][0] == write:
            self.runs[-1][1] += text
        else:
            self.runs.append([write, text])
        self._wake.set()

    def flush(self) -> None:
        """Show everything still held, now, in order. Caller holds the lock."""
        runs, self.runs, self._pace = self.runs, [], 0
        for write, text in runs:
            write(text)

    @property
    def frames(self) -> int:
        """Frames a backlog is allowed to take to clear."""
        return max(1, int(self.lag / self.tick))

    def step(self, waiting: int) -> int:
        """How many characters to show this frame, given what is waiting."""
        return max(1, ceil(waiting / self.frames))

    def take(self) -> tuple[Write, str] | None:
        """Slice off this frame's piece. The caller holds the lock."""
        if not self.runs:
            self._pace = 0
            return None
        self._pace = max(self._pace, self.step(self.pending))
        write, text = self.runs[0]
        piece, rest = text[:self._pace], text[self._pace:]
        if rest:
            self.runs[0][1] = rest
        else:
            self.runs.pop(0)
        if not self.runs:
            self._pace = 0
        return write, piece

    def start(self) -> None:
        if self._task is None:
            self._wake = asyncio.Event()
            self._task = asyncio.ensure_future(self._run())

    def stop(self) -> None:
        """End the drain loop without awaiting from cancellation teardown."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            task.add_done_callback(lambda t: t.cancelled() or t.exception())

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while True:
                async with self._lock:
                    taken = self.take()
                    if taken is not None:
                        write, piece = taken
                        write(piece)
                if taken is None:
                    break
                await asyncio.sleep(self.tick)
