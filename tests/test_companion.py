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


class TestTheStateComesFromTheAgent:
    """Not from a timer, and not from a mood."""

    @pytest.mark.parametrize("task,expected", [
        ("idle", State.IDLE),
        ("thinking", State.THINKING),
        ("planning", State.THINKING),
        ("executing", State.CODING),
        ("testing", State.TESTING),
        ("recovering", State.RECOVERING),
        ("completed", State.SUCCESS),
        ("failed", State.ERROR),
        ("cancelled", State.CANCELLED),
    ])
    def test_each_task_state_maps(self, task, expected):
        assert companion.state_for(task) is expected

    @pytest.mark.parametrize("tool,expected", [
        ("grep", State.SEARCHING), ("glob", State.SEARCHING),
        ("read_file", State.READING), ("github_read", State.READING),
        ("edit_file", State.CODING), ("write_file", State.CODING),
        ("github_write", State.CODING),
        ("run_tests", State.TESTING),
    ])
    def test_the_running_tool_says_more_than_executing_does(self, tool,
                                                            expected):
        """"executing" is equally true of reading a file and of writing one.
        Watching the companion should say which."""
        assert companion.state_for("executing", tool) is expected

    def test_an_unknown_tool_falls_back_to_the_task(self):
        assert companion.state_for("executing", "some_new_tool") is State.CODING

    def test_an_unknown_state_is_survivable(self):
        """A companion that raises inside a repaint is worse than a dull
        one."""
        assert companion.state_for("nonsense") is State.IDLE
        assert companion.state_for("") is State.IDLE

    @pytest.mark.parametrize("task", ["completed", "failed", "cancelled",
                                      "idle"])
    @pytest.mark.parametrize("tool", ["edit_file", "run_tests", "grep"])
    def test_a_leftover_tool_cannot_animate_a_finished_task(self, task, tool):
        """The tool is still set between a cancellation and the turn's
        teardown, and a companion that keeps typing under the word
        "Interrupted" is worse than one that does nothing."""
        assert companion.is_over(companion.state_for(task, tool))

    def test_listening_and_speaking_win(self):
        """Both are about the person rather than the work, and are the thing
        they are waiting on."""
        assert companion.state_for("executing", "edit_file",
                                   listening=True) is State.LISTENING
        assert companion.state_for("executing", "edit_file",
                                   speaking=True) is State.SPEAKING

    def test_is_over_agrees_with_the_task_state_machine(self):
        from wynxo.task_state import TaskState

        over = {TaskState.IDLE, TaskState.COMPLETED, TaskState.FAILED,
                TaskState.CANCELLED}
        for state in TaskState:
            assert companion.is_over(companion.state_for(state.value)) \
                is (state in over), state


class TestDrawingIt:
    def test_the_panel_is_a_closed_box(self):
        rows = companion.panel(State.CODING, 0)
        assert rows[0].startswith("╭") and rows[0].endswith("╮")
        assert rows[-1].startswith("╰") and rows[-1].endswith("╯")
        for row in rows[1:-1]:
            assert row.startswith("│") and row.endswith("│")

    def test_every_row_of_the_panel_is_the_same_width(self):
        for state in State:
            rows = companion.panel(state, 0)
            widths = {cell_len(row) for row in rows}
            assert len(widths) == 1, f"{state.value}: {widths}"

    def test_the_panel_is_the_same_size_in_every_state(self):
        """It is drawn over the conversation. A box that resized between
        states would make the text underneath it jump."""
        sizes = {len(companion.panel(state, 0)) for state in State}
        assert len(sizes) == 1, sizes

    def test_an_ascii_panel_is_all_ascii(self):
        rows = companion.panel(State.CODING, 0, unicode=False)
        assert "\n".join(rows).isascii()

    def test_reduced_motion_is_one_still_frame(self):
        """Not a slower animation -- none of one. It is asked for by people
        who find movement in the corner of the eye actively unpleasant, and
        a gentler version of the thing is still the thing."""
        for state in State:
            assert len(companion.frames_for(state, reduced=True)) == 1

    def test_the_frame_index_wraps_rather_than_raising(self):
        for i in (0, 1, 7, 999):
            assert companion.panel(State.CODING, i)

    def test_the_label_says_what_is_happening(self):
        assert companion.label_for(State.CODING) == "writing code"
        assert companion.label_for(State.TESTING) == "running tests"
        assert companion.label_for(State.SUCCESS) == "done"

    def test_a_narrow_terminal_still_gets_a_closed_box(self):
        for width in (20, 26, 34, 80):
            rows = companion.panel(State.IDLE, 0, width=width)
            widths = {cell_len(row) for row in rows}
            assert len(widths) == 1, f"width={width}: {widths}"


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
        assert "self.pet.padded()" in source

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


class TestThePanelNeverExceedsWhatItIsGiven:
    """It is drawn into a Float that is capped by the screen. A box wider
    than that gets its right border clipped off, which reads as a broken
    box rather than a narrow one -- the panel used to floor its width at
    the scene's own 30 columns and do exactly that on a 20-column
    terminal."""

    @pytest.mark.parametrize("width", [12, 16, 20, 26, 30, 34, 48, 60, 100])
    def test_the_box_closes_at_every_width(self, width):
        rows = companion.panel(State.CODING, 0, width=width)
        widths = {cell_len(row) for row in rows}
        assert len(widths) == 1, f"width={width}: rows measure {widths}"
        assert widths.pop() <= max(width, 10)

    @pytest.mark.parametrize("width", [16, 20, 34])
    def test_the_character_survives_the_crop(self, width):
        """Cropping loses the mug and the thought bubble before it loses the
        face, because the staging is left-aligned."""
        rows = companion.panel(State.IDLE, 0, width=width)
        assert any("(" in row and ")" in row for row in rows), rows

    def test_a_wide_terminal_does_not_stretch_it(self):
        """The conversation is the product; this sits beside it."""
        rows = companion.panel(State.IDLE, 0, width=200)
        assert cell_len(rows[0]) <= 60
