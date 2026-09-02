"""Wyn: the companion, its staging, and where its state comes from.

What this replaced was a face in the status bar -- two lines and a pair of
eyes -- which could not read as a character doing anything because there was
nowhere for it to be. The tests that matter here are therefore about
staging and about provenance: that the frames hold their shape so the panel
cannot tear, and that every state comes from something the agent actually
did rather than from a clock.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len

from wynxo import companion
from wynxo.companion import HEIGHT, WIDTH, SCENES, State


class TestTheStagingHoldsItsShape:
    """A frame of the wrong size tears the box it is drawn in, and it does
    so on somebody else's terminal rather than here."""

    def test_every_state_has_a_scene(self):
        for state in State:
            assert state in SCENES, state

    @pytest.mark.parametrize("state", list(State))
    def test_every_frame_is_the_same_height(self, state):
        for frame in SCENES[state].frames:
            assert len(frame.split("\n")) == HEIGHT

    @pytest.mark.parametrize("state", list(State))
    def test_no_row_is_wider_than_the_panel(self, state):
        for frame in SCENES[state].frames + SCENES[state].ascii:
            for row in frame.split("\n"):
                assert cell_len(row) <= WIDTH, repr(row)

    @pytest.mark.parametrize("state", list(State))
    def test_nothing_uses_a_double_width_glyph(self, state):
        """A two-cell character inside a bordered panel pushes the right
        border out by one and the box stops being a box. CJK, kana and
        emoji are all excluded for this reason."""
        for frame in SCENES[state].frames:
            for char in frame:
                if char == "\n":
                    continue
                assert cell_len(char) == 1, f"{char!r} is {cell_len(char)} cells"

    @pytest.mark.parametrize("state", list(State))
    def test_the_ascii_tier_is_actually_ascii(self, state):
        """The fallback exists for a console that cannot render the rest --
        a Windows code page that is not UTF-8, most obviously. One stray
        box-drawing character in it defeats the whole point."""
        for frame in SCENES[state].ascii:
            assert frame.isascii(), repr(
                [c for c in frame if not c.isascii()])

    def test_every_state_has_an_ascii_tier(self):
        for state in State:
            assert SCENES[state].ascii, f"{state.value} has no ASCII frames"

    @pytest.mark.parametrize("state", list(State))
    def test_the_character_keeps_its_proportions(self, state):
        """One character, not several. The head is the same width and sits
        on the same row in every scene -- a companion whose proportions
        change between states looks like a different companion each time."""
        head = SCENES[state].frames[0].split("\n")[1]
        assert head.count("(") == 1 and head.count(")") == 1, head
        assert 8 <= len(head.strip()) <= 14, f"head row {head!r}"

    @pytest.mark.parametrize("state", list(State))
    def test_the_desk_is_the_floor_and_does_not_move(self, state):
        """It is what makes the character look seated rather than floating,
        so it is identical across the frames of a scene."""
        rows = [frame.split("\n")[4] for frame in SCENES[state].frames]
        assert len(set(rows)) == 1, f"the desk moves in {state.value}: {rows}"

    @pytest.mark.parametrize("state", list(State))
    def test_only_one_thing_moves_between_frames(self, state):
        """Two is a twitch; one is a breath."""
        frames = SCENES[state].frames
        for before, after in zip(frames, frames[1:]):
            rows_before = before.split("\n")
            rows_after = after.split("\n")
            changed = sum(1 for a, b in zip(rows_before, rows_after) if a != b)
            assert changed <= 2, (
                f"{state.value}: {changed} rows change at once")


class TestThereIsOnlyOneCharacter:
    def test_the_showcase_draws_the_same_art(self):
        """/animate used to show a different drawing from the one the
        running application displayed, because the character existed twice
        and the two drifted."""
        from wynxo import motion

        assert motion.scene_for("coding").frames == \
            companion.frames_for(State.CODING)
        assert motion.scene_for("idle").frames == \
            companion.frames_for(State.IDLE)

    def test_the_running_session_draws_the_companion_exactly_once(self):
        """Three renderings of the same character were on screen at once:
        the greeting line, the status strip, and a panel in the corner. Two
        faces for one companion reads as two companions.

        The live rendering is the activity bar's face, and nothing else in
        a running session draws one."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli)
        assert "companion.panel" not in source, (
            "a second live rendering of the companion is back")
        assert "companion.state_for" not in source

    def test_the_bar_is_where_the_face_lives(self):
        import inspect

        from wynxo.ui import ActivityBar

        source = inspect.getsource(ActivityBar._render)
        assert "self.pet.mark(" in source

    def test_nothing_advances_the_frame_but_a_repaint(self):
        """A scheduler would keep animating through a stall, showing typing
        while nothing is being written. The frame is stepped by the draw
        itself -- ``face(advance=True)`` -- so it moves exactly while the
        bar repaints and stops the instant it stops."""
        import inspect

        from wynxo.ui import ActivityBar

        source = inspect.getsource(ActivityBar._render)
        for clock in ("time.monotonic", "time.time", "asyncio.sleep",
                      "time.sleep", "perf_counter", "Thread", "Timer"):
            assert clock not in source, f"{clock} drives the companion"

    def test_the_companion_module_has_no_clock_at_all(self):
        source = (companion.__file__ and
                  __import__("pathlib").Path(companion.__file__).read_text())
        for forbidden in ("import time", "import threading", "asyncio",
                          "Thread(", "Timer("):
            assert forbidden not in source, f"{forbidden} in companion.py"

