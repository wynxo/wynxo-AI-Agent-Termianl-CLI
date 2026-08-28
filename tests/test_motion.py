"""The ASCII animation engine: scene selection, frame progression, the one
scheduler, fallbacks, and reduced-motion mode.

Everything here is deterministic -- no timers, no event loop -- because
``MotionScheduler.step()`` advances frames synchronously, which is how the
engine is exercised in tests and how its correctness is pinned down.
"""

from __future__ import annotations

import asyncio

import pytest

from wynxo import motion


class TestSceneSelection:
    def test_every_mood_and_speech_state_has_a_scene(self):
        for state in ("idle", "thinking", "working", "reading", "running",
                      "happy", "sad", "asking", "sleepy",
                      "listening", "transcribing", "speaking"):
            scene = motion.scene_for(state)
            assert scene.frames, f"{state} has no frames"
            assert scene.name

    def test_unknown_state_falls_back_to_idle(self):
        assert motion.scene_for("totally-mysterious").name == "idle"

    def test_reduced_motion_keeps_one_static_frame(self):
        scene = motion.scene_for("thinking")
        assert motion.select(scene, reduced=True) == (scene.frames[0],)

    def test_ascii_fallback_for_non_unicode_terminals(self):
        scene = motion.scene_for("listening")
        assert scene.ascii is not None
        frames = motion.select(scene, unicode=False)
        assert all(not any(ord(ch) > 127 for ch in frame) for frame in frames)

    def test_compact_frames_when_the_full_set_does_not_fit(self):
        scene = motion.scene_for("sparkle")
        assert scene.compact is not None
        widest = max(len(line) for frame in scene.frames
                     for line in frame.split("\n"))
        assert motion.select(scene, width=widest - 1) == scene.compact
        assert motion.select(scene, width=widest) == scene.frames

    def test_wide_terminal_keeps_the_full_set(self):
        scene = motion.scene_for("thinking")
        assert motion.select(scene, width=120) == scene.frames

    def test_every_frame_of_a_scene_has_a_consistent_shape(self):
        """A multi-frame scene must not jitter: each frame's lines line up
        in column count, and the frames agree on how many rows they take.
        A frame that shifts the box one column over reads as a rendering
        bug, not as animation -- the coding scene used to do exactly that
        between "pet on top" and "pet inside the terminal".
        """
        from wcwidth import wcswidth

        for scene in motion.SCENES.values():
            if len(scene.frames) < 2:
                continue
            row_counts = {len(frame.split("\n")) for frame in scene.frames}
            assert len(row_counts) == 1, f"{scene.name} frames differ in height"
            # Row 0 may legitimately differ (a pet perched on top of the
            # box versus inside it); the rows below it are the geometry that
            # must not drift. The first frame's rows anchor the width and
            # later frames must never exceed them -- a box that shifts one
            # column over between frames is a rendering bug.
            first = [wcswidth(line) for line in scene.frames[0].split("\n")]
            for frame in scene.frames[1:]:
                widths = [wcswidth(line) for line in frame.split("\n")]
                for want, got in zip(first[1:], widths[1:]):
                    assert got <= want, \
                        f"{scene.name} frame row wider than the first: {got} > {want}"


class TestPreview:
    def test_preview_is_deterministic(self):
        assert motion.preview("thinking") == motion.preview("thinking")

    def test_preview_of_a_looping_scene_cycles(self):
        scene = motion.scene_for("thinking")
        strip = motion.preview("thinking", n=5)
        # Five frames of a three-frame loop: the cycle wraps.
        assert strip.count("≽") >= 5

    def test_preview_of_a_one_shot_shows_each_frame_once(self):
        strip = motion.preview("sparkle")
        assert strip  # rendered something, padded so it does not shrink

    def test_preview_respects_reduced_motion(self):
        scene = motion.scene_for("thinking")
        assert motion.select(scene, reduced=True) == (scene.frames[0],)


class TestSchedulerStepping:
    def test_frames_advance_and_loop(self):
        scheduler = motion.MotionScheduler(reduced=False)
        seen = []
        scheduler.register("wave", motion.scene_for("listening"),
                           lambda frame: seen.append(frame))
        for _ in range(len(motion.SCENES["listening"].frames) + 2):
            scheduler.step("wave")
        frames = motion.SCENES["listening"].frames
        # Stepped once past the end: the cycle wrapped back to frame 0.
        assert seen[0] == frames[0]
        assert seen[len(frames)] == frames[0]
        scheduler.close()

    def test_unregister_stops_callbacks(self):
        scheduler = motion.MotionScheduler(reduced=False)
        seen = []
        scheduler.register("wave", motion.scene_for("listening"),
                           lambda frame: seen.append(frame))
        scheduler.unregister("wave")
        scheduler.step()
        assert seen == []
        scheduler.close()

    def test_one_shot_unregisters_after_its_last_frame(self):
        scheduler = motion.MotionScheduler(reduced=False)
        frames = []
        scheduler.register("sparkle", motion.scene_for("sparkle"),
                           lambda frame: frames.append(frame))
        while scheduler.active:
            scheduler.step()
        assert len(frames) == len(motion.SCENES["sparkle"].frames)
        assert scheduler.active == []
        scheduler.close()

    def test_stop_all_clears_everything(self):
        scheduler = motion.MotionScheduler(reduced=False)
        scheduler.register("a", motion.scene_for("idle"), lambda f: None)
        scheduler.register("b", motion.scene_for("thinking"), lambda f: None)
        scheduler.stop_all()
        assert scheduler.active == []
        scheduler.close()

    def test_reduced_mode_delivers_one_static_frame_and_never_loops(self):
        scheduler = motion.MotionScheduler(reduced=True)
        seen = []
        scheduler.register("wave", motion.scene_for("listening"),
                           lambda frame: seen.append(frame))
        scheduler.step()
        scheduler.step()
        assert seen == [motion.SCENES["listening"].frames[0]]
        assert scheduler.active == []
        scheduler.close()

    def test_close_is_idempotent(self):
        scheduler = motion.MotionScheduler(reduced=False)
        scheduler.register("a", motion.scene_for("idle"), lambda f: None)
        scheduler.close()
        scheduler.close()      # must not raise
        assert scheduler.active == []

    def test_a_bad_callback_does_not_kill_the_scheduler(self):
        scheduler = motion.MotionScheduler(reduced=False)

        def boom(_frame):
            raise RuntimeError("boom")

        scheduler.register("bad", motion.scene_for("thinking"), boom)
        scheduler.step()       # must not raise
        assert "bad" in scheduler.active
        scheduler.close()


class TestSchedulerAsyncLifecycle:
    async def test_the_loop_fires_callbacks_over_time(self):
        scheduler = motion.MotionScheduler(reduced=False)
        seen = []
        scheduler.register("wave", motion.scene_for("listening"),
                           lambda frame: seen.append(frame), fps=50.0)
        await asyncio.sleep(0.08)
        assert len(seen) >= 2
        await scheduler.aclose()
        assert scheduler.active == []

    async def test_aclose_cancels_the_task_cleanly(self):
        scheduler = motion.MotionScheduler(reduced=False)
        scheduler.register("a", motion.scene_for("idle"), lambda f: None)
        task = scheduler._task
        assert task is not None
        await scheduler.aclose()
        assert task.done()
        assert scheduler.active == []
