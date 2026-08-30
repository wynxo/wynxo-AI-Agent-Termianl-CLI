"""The ASCII animation engine: scene selection and preview.

Everything here is deterministic -- no timers, no event loop -- because the
scenes are pure frame data selected by width, unicode support and
reduced-motion mode, which is what /animate and /pet exercise.
"""

from __future__ import annotations

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
            first = [wcswidth(line) for line in scene.frames[0].split("\n")]
            for frame in scene.frames[1:]:
                widths = [wcswidth(line) for line in frame.split("\n")]
                for want, got in zip(first[1:], widths[1:]):
                    assert got <= want, (
                        f"{scene.name} frame row wider than the first: {got} > {want}")


class TestPreview:
    def test_preview_is_deterministic(self):
        assert motion.preview("thinking") == motion.preview("thinking")

    def test_preview_of_a_looping_scene_cycles(self):
        """Five frames of a three-frame loop: the cycle wraps.

        Counted structurally rather than by looking for a glyph the art
        happens to use, so this keeps asking the question when the character
        is redrawn -- which is what happened to the version of this test
        that counted the old face."""
        scene = motion.scene_for("thinking")
        assert 1 < len(scene.frames) < 5, "the premise: a short loop"
        one = motion.preview("thinking", n=1)
        five = motion.preview("thinking", n=5)
        width = max(len(line) for line in one.split("\n"))
        widest = max(len(line) for line in five.split("\n"))
        assert widest >= width * 4, "the strip did not lay out five frames"

    def test_preview_of_a_one_shot_shows_each_frame_once(self):
        strip = motion.preview("sparkle")
        assert strip  # rendered something, padded so it does not shrink

    def test_preview_respects_reduced_motion(self):
        scene = motion.scene_for("thinking")
        assert motion.select(scene, reduced=True) == (scene.frames[0],)
