"""How fast the model is, as opposed to how long you waited.

A turn on a machine where most of the model is on the CPU begins with the
weights being read off disk and the prompt being read. That can be a minute
before a single token appears. A speed measured across all of it is not the
model's speed, and it is the number somebody reads to tell whether a change
they just made helped -- so it being a fraction of the truth is worse than
it not being there.

Measured: sixty seconds of waiting and five of generating two hundred
tokens showed 3.1 tok/s, where the model was doing 40.
"""

from __future__ import annotations

import time

from wynxo.session import Usage
from wynxo.ui import UI, ActivityBar


class TestTheLiveFigureExcludesTheWait:
    def _bar(self, waited: float, tokens: int, generating: float):
        bar = ActivityBar(UI(), effort="low")
        bar.started = time.monotonic() - (waited + generating)
        bar.tokens = 1
        bar._first_token = time.monotonic() - generating
        bar.tokens = tokens
        return bar

    def test_a_long_wait_does_not_drag_the_speed_down(self):
        fast = self._bar(waited=1.0, tokens=200, generating=5.0)
        slow = self._bar(waited=60.0, tokens=200, generating=5.0)
        assert abs(fast.rate() - slow.rate()) < 2, (fast.rate(), slow.rate())

    def test_it_reports_what_the_model_actually_did(self):
        bar = self._bar(waited=60.0, tokens=200, generating=5.0)
        assert 35 < bar.rate() < 45, bar.rate()

    def test_nothing_is_claimed_before_the_first_token(self):
        bar = ActivityBar(UI(), effort="low")
        bar.started -= 60
        assert bar.rate() == 0.0

    def test_nor_in_the_first_moment_after_it(self):
        """Two tokens in a tenth of a second is not twenty a second."""
        bar = ActivityBar(UI(), effort="low")
        bar.tokens = 2
        assert bar.rate() == 0.0

    def test_the_clock_starts_once_and_stays(self):
        bar = ActivityBar(UI(), effort="low")
        bar.tokens = 1
        first = bar._first_token
        bar.tokens = 50
        bar.tokens = 100
        assert bar._first_token == first


class TestTheRecordedFigureUsesTheServersOwnMeasurement:
    def test_generation_time_is_kept_apart_from_wall_time(self):
        usage = Usage()
        # A request that took 65 seconds, 5 of them generating.
        usage.add_chunk(prompt=5000, completion=200,
                        duration_ns=65_000_000_000,
                        eval_ns=5_000_000_000)
        assert abs(usage.generation_seconds - 5.0) < 0.01
        assert abs(usage.wall_seconds - 65.0) < 0.01
        assert 35 < usage.tokens_per_second() < 45

    def test_a_server_that_does_not_break_it_down_still_gets_a_number(self):
        """llama.cpp's shim and some compat servers report only the total.
        A rough figure is better than none; it is simply the old one."""
        usage = Usage()
        usage.add_chunk(prompt=100, completion=200,
                        duration_ns=10_000_000_000, eval_ns=0)
        assert abs(usage.generation_seconds - 10.0) < 0.01
        assert 15 < usage.tokens_per_second() < 25

    def test_several_requests_accumulate(self):
        usage = Usage()
        for _ in range(3):
            usage.add_chunk(prompt=100, completion=100,
                            duration_ns=20_000_000_000,
                            eval_ns=2_000_000_000)
        assert abs(usage.generation_seconds - 6.0) < 0.01
        assert abs(usage.wall_seconds - 60.0) < 0.01
        assert usage.requests == 3

    def test_no_requests_is_not_a_division(self):
        assert Usage().tokens_per_second() == 0.0


class TestTheProviderHandsItOver:
    def test_the_chunk_carries_the_generation_time(self):
        from wynxo.provider import OllamaClient

        chunk = OllamaClient._to_chunk({
            "message": {"content": "hi"}, "done": True,
            "eval_count": 200, "prompt_eval_count": 5000,
            "total_duration": 65_000_000_000,
            "load_duration": 40_000_000_000,
            "eval_duration": 5_000_000_000,
        })
        assert chunk.eval_duration_ns == 5_000_000_000
        assert chunk.total_duration_ns == 65_000_000_000

    def test_a_server_that_omits_it_reports_zero_not_a_crash(self):
        from wynxo.provider import OllamaClient

        chunk = OllamaClient._to_chunk({"message": {}, "done": True})
        assert chunk.eval_duration_ns == 0
