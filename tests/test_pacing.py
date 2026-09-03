"""Text reaching the screen at a rate you can watch.

A model emits its answer in whatever lumps its tokeniser produces --
``pri``, ``nt("hel``, ``lo")`` -- and shown as they arrive those lumps are
what you see. The pacer holds them and lets the text out in even pieces.

Two things it must never do, both worse than lumpy text: fall behind the
model, so watching an answer takes longer than writing it; and reorder
anything, so a tool line lands in the middle of the sentence that
introduced it, or the reasoning and the answer cross over each other.
"""

from __future__ import annotations

import asyncio

import pytest

from wynxo.pacing import Typewriter


class _Pen:
    """A destination that remembers what it was asked to show."""

    def __init__(self, name: str = ""):
        self.name = name
        self.pieces: list[str] = []

    def write(self, text: str) -> None:
        self.pieces.append(text)

    @property
    def shown(self) -> str:
        return "".join(self.pieces)


@pytest.fixture
def pen():
    return _Pen()


@pytest.fixture
def paper(pen):
    """A pacer with its drain "running", so feeds are held rather than shown."""
    made = Typewriter(asyncio.Lock())
    made._task = object()
    return made


class TestNothingIsHeldWhereNothingIsAnimated:
    def test_without_a_running_drain_text_goes_straight_out(self, pen):
        """-p, a pipe, a dumb terminal: no live region, so no pacing. Output
        being read by a program must arrive as fast as it is produced."""
        Typewriter(asyncio.Lock()).feed(pen.write, "print('hello')")
        assert pen.shown == "print('hello')"

    def test_empty_text_is_not_a_write(self, pen, paper):
        paper.feed(pen.write, "")
        assert pen.pieces == []
        assert paper.pending == 0


class TestTheTextComesOutAsItWasTyped:
    def test_a_slow_model_is_shown_one_character_at_a_time(self, pen, paper):
        paper.feed(pen.write, "print")
        while taken := paper.take():
            write, piece = taken
            write(piece)
        assert pen.pieces == ["p", "r", "i", "n", "t"]

    def test_what_the_screen_shows_is_the_answer_growing(self, pen, paper):
        shown = []
        for chunk in ["pri", 'nt("hel', 'lo")']:
            paper.feed(pen.write, chunk)
            while taken := paper.take():
                taken[0](taken[1])
                shown.append(pen.shown)
        assert shown[:5] == ["p", "pr", "pri", "prin", "print"]
        assert shown[-1] == 'print("hello")'

    def test_the_pieces_are_smaller_than_the_chunks_they_came_from(self, pen, paper):
        """The whole point: a three-character token is three frames, not one."""
        paper.feed(pen.write, "pri")
        assert paper.take() == (pen.write, "p")


class TestItNeverFallsBehindTheModel:
    def test_a_backlog_is_cleared_within_the_lag(self, pen, paper):
        """However much is waiting, it is gone in lag/tick pieces. A model
        faster than the display is shown fast, not queued."""
        paper.feed(pen.write, "x" * 5000)
        pieces = 0
        while paper.take():
            pieces += 1
        assert pieces <= paper.frames

    def test_the_piece_grows_with_the_backlog(self, paper):
        assert paper.step(1) == 1
        assert paper.step(4) <= paper.step(40) <= paper.step(400)

    def test_a_whole_answer_arriving_at_once_is_not_held_back(self, pen, paper):
        """A provider with no streaming hands over the finished answer. There
        is nothing left to pace it against, and typing it out would only make
        the reply later than it had to be."""
        paper.feed(pen.write, "a" * 4000)
        seconds = 0.0
        while paper.take():
            seconds += paper.tick
        assert seconds <= paper.lag

    def test_the_pace_does_not_shrink_with_the_backlog_it_is_clearing(self, pen, paper):
        """Recomputed against what is left, the piece shrinks as the buffer
        does and the backlog decays instead of clearing."""
        paper.feed(pen.write, "x" * 700)
        sizes = []
        while taken := paper.take():
            sizes.append(len(taken[1]))
        assert sizes[0] == max(sizes)
        assert len(set(sizes[:-1])) == 1, "one size per drain, until the tail"


class TestThreeStreamsShareOneQueue:
    def test_a_piece_never_spans_two_destinations(self, paper):
        thinking, answer = _Pen("think"), _Pen("answer")
        paper.feed(thinking.write, "so the")
        paper.feed(answer.write, "Here is")
        seen = []
        while taken := paper.take():
            taken[0](taken[1])
            seen.append(taken[0])
        assert thinking.shown == "so the"
        assert answer.shown == "Here is"
        assert seen.index(answer.write) > max(
            i for i, w in enumerate(seen) if w == thinking.write
        ), "the reasoning is finished before the answer starts"

    def test_consecutive_text_for_one_destination_is_one_run(self, pen, paper):
        paper.feed(pen.write, "ab")
        paper.feed(pen.write, "cd")
        assert len(paper.runs) == 1

    def test_flush_writes_every_stream_in_the_order_it_arrived(self, paper):
        order = []
        paper.feed(lambda t: order.append(("first", t)), "one")
        paper.feed(lambda t: order.append(("second", t)), "two")
        paper.flush()
        assert order == [("first", "one"), ("second", "two")]


class TestNothingOvertakesHeldText:
    def test_flush_empties_everything_at_once(self, pen, paper):
        paper.feed(pen.write, "the end of the sentence")
        paper.flush()
        assert pen.shown == "the end of the sentence"
        assert paper.pending == 0

    def test_flush_with_nothing_held_writes_nothing(self, pen, paper):
        paper.flush()
        assert pen.pieces == []

    def test_text_taken_before_a_flush_is_not_shown_twice(self, pen, paper):
        paper.feed(pen.write, "abcdef")
        taken = paper.take()
        taken[0](taken[1])
        paper.flush()
        assert pen.shown == "abcdef"


class TestTheDrainLoop:
    async def test_it_writes_everything_it_was_given(self, pen):
        paper = Typewriter(asyncio.Lock(), tick=0.001, lag=0.01)
        paper.start()
        paper.feed(pen.write, "print('hello')")
        for _ in range(200):
            await asyncio.sleep(0.001)
            if not paper.pending:
                break
        paper.stop()
        assert pen.shown == "print('hello')"
        assert len(pen.pieces) > 1, "shown in pieces, not in one go"

    async def test_stopping_does_not_need_awaiting(self, pen):
        """Teardown runs when a turn was cancelled too, and a stop that
        awaits there swallows the cancellation it is cleaning up after."""
        paper = Typewriter(asyncio.Lock(), tick=0.001)
        paper.start()
        paper.feed(pen.write, "held")
        paper.stop()
        assert paper._task is None
        paper.flush()
        assert pen.shown == "held"

    async def test_a_stopped_drain_writes_nothing_more(self, pen):
        paper = Typewriter(asyncio.Lock(), tick=0.001)
        paper.start()
        paper.feed(pen.write, "abcdefghij")
        paper.stop()
        paper.runs.clear()
        await asyncio.sleep(0.02)
        assert pen.pieces == []

    async def test_the_lock_is_released_between_pieces(self, pen):
        """Held across the whole drain, nothing else could ever write."""
        lock = asyncio.Lock()
        paper = Typewriter(lock, tick=0.001, lag=0.01)
        paper.start()
        paper.feed(pen.write, "abcdefghijklmnop")
        await asyncio.sleep(0.003)
        async with lock:
            pass                          # would hang if the loop kept it
        paper.stop()

    async def test_feeding_after_a_stop_writes_straight_out(self, pen):
        """The turn is over; there is no drain left to hold anything."""
        paper = Typewriter(asyncio.Lock(), tick=0.001)
        paper.start()
        paper.stop()
        paper.feed(pen.write, "late")
        assert pen.shown == "late"
