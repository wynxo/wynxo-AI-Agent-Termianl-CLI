"""Letting text reach the screen the way it was written: a character at a time.

A model does not produce its answer smoothly. Ollama sends one token per
chunk -- two to six characters -- and the terminal is repainted at a fixed
frame rate, so what lands on screen is whatever accumulated between frames:
``pri``, then ``nt("hel``, then ``lo")``. The text is live, and it still
reads as a thing being pasted in lumps rather than a thing being typed.

The fix is a buffer with a rate on it. Text goes in as it arrives and comes
out in even pieces, one per frame. Two properties matter:

*Never fall behind.* The buffer is drained within ``lag`` seconds no matter
how much is in it -- the piece size is computed from what is waiting, not
fixed -- so a fast model is shown at a fast, even rate rather than being
queued up behind a pretty animation. Watching the answer must never be
slower than producing it.

*Never reorder.* Held text is text that has already been generated, so
anything else that wants the screen has to let it out first. There is one
buffer for all three streams a turn produces -- the reasoning, the answer,
and the file being written -- because they interleave, and a pacer per
stream would hold one of them back past text that came after it.
:meth:`flush` empties the whole thing in order, and the caller holds the
same lock the drain loop uses, so a piece can never land between a tool
line and the sentence that introduced it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from math import ceil

TICK = 1.0 / 30
"""Seconds between pieces. Matched to the repaint rate: releasing more often
than the screen is redrawn only re-lumps the text inside a single frame."""

LAG = 0.25
"""The most the display may trail the model, in seconds.

Seven frames at ``TICK``, which is what decides whether a five-character
token is shown one character at a time or two. Four frames was not enough:
``print`` came out as ``pr`` ``in`` ``t``. A quarter of a second behind is
not perceptible while text is still arriving, and the last of it is flushed
whole when the answer ends, so it costs nothing at the finish either.

At the rate a local model actually writes -- thirty tokens a second is
around a hundred characters -- this is one to three characters a frame. The
slower the model, the closer to one, which is the right way round: a slow
model is the one you sit and watch.
"""

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
        """Pending text as ``[write, text]`` pairs, oldest first. Consecutive
        text for the same destination is joined, so the common case -- one
        stream writing for a while -- is a single run."""
        self._pace = 0
        """Characters per frame for the drain now under way, reset when the
        buffer empties. Held rather than recomputed because recomputing it
        against a shrinking buffer shrinks the piece too: the backlog then
        decays geometrically instead of clearing, and five thousand
        characters took twenty-eight frames to show rather than seven. The
        rule that makes the bound true is that a pace, once set by a
        backlog, stands until that backlog is gone."""
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def pending(self) -> int:
        return sum(len(text) for _, text in self.runs)

    # -- what the stream puts in -------------------------------------------

    def feed(self, write: Write, text: str) -> None:
        """Take generated text for ``write``. The caller holds the lock."""
        if not text:
            return
        if self._task is None:
            # No drain running: no live region to pace against -- a pipe,
            # -p, a dumb terminal -- or a turn that has already been torn
            # down. Output being read by a program must arrive as fast as it
            # is produced.
            write(text)
            return
        if self.runs and self.runs[-1][0] == write:
            self.runs[-1][1] += text
        else:
            self.runs.append([write, text])
        self._wake.set()

    def flush(self) -> None:
        """Show everything still held, now, in order. Caller holds the lock.

        Called before anything else writes, and at the end of a turn. The
        last few characters appear whole; there is nothing still coming that
        they could be paced against.
        """
        runs, self.runs, self._pace = self.runs, [], 0
        for write, text in runs:
            write(text)

    # -- what comes out ----------------------------------------------------

    @property
    def frames(self) -> int:
        """Frames a backlog is allowed to take to clear."""
        return max(1, int(self.lag / self.tick))

    def step(self, waiting: int) -> int:
        """How many characters to show this frame, given what is waiting.

        Enough to clear the buffer within ``lag``. One character while the
        model is slow, more as it gets ahead -- which is the honest reading
        of "as fast as it is being written".
        """
        return max(1, ceil(waiting / self.frames))

    def take(self) -> tuple[Write, str] | None:
        """Slice off this frame's piece. The caller holds the lock.

        From the head of the queue only: a piece never spans two
        destinations, so a frame that would have crossed from the reasoning
        into the answer stops at the boundary and the answer starts on the
        next one.
        """
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

    # -- the loop ----------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._wake = asyncio.Event()
            self._task = asyncio.ensure_future(self._run())

    def stop(self) -> None:
        """End the drain loop. Sync on purpose.

        This is called from a turn's teardown, which also runs when the turn
        was cancelled by Ctrl-C; awaiting there is how a stop swallows the
        cancellation it was supposed to be cleaning up after. Cancelling is
        enough on its own: a cancelled task is resumed only to be raised
        into, so it can never write another piece, and there is nothing it
        holds that needs closing. Whatever it was still holding is flushed
        by the caller a line later.
        """
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            # Otherwise a drain that failed for its own reasons is reported
            # by the loop as an exception nobody retrieved, long after the
            # turn it belonged to.
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
