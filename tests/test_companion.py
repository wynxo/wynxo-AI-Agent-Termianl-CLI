"""What the companion is doing, and where that answer comes from.

The state is not a thing the presentation layer decides. It is read from
the agent -- which tool is running, what the task state machine says -- so
there is one answer to "what is happening" and the strip, the mark and the
character all draw from it. Keeping a second opinion in the UI is how a
companion ends up cheerfully typing through a turn that failed a minute
ago, and it is what these pin shut.

The pictures live in ``sprite.py`` and are tested in ``test_mascot.py``.
This module used to hold a set of staged ASCII scenes as well, which no
session ever rendered while a smaller face from ``pet.py`` was what
actually appeared; both are gone.
"""
from __future__ import annotations

import inspect
import pathlib

import pytest

from wynxo import companion
from wynxo.companion import State, state_for


class TestTheStateComesFromTheWork:
    @pytest.mark.parametrize("tool,expected", [
        ("read_file", State.READING),
        ("list_dir", State.READING),
        ("edit_file", State.CODING),
        ("write_file", State.CODING),
        ("grep", State.SEARCHING),
        ("web_search", State.SEARCHING),
        ("run_tests", State.TESTING),
    ])
    def test_the_running_tool_decides_while_a_task_runs(self, tool, expected):
        """"executing" is equally true of reading a file and of writing one,
        and those must not look alike: watching the companion should tell
        you which is happening without reading the transcript."""
        assert state_for(tool, "executing") is expected

    @pytest.mark.parametrize("task,expected", [
        ("completed", State.SUCCESS),
        ("failed", State.ERROR),
        ("cancelled", State.CANCELLED),
        ("idle", State.IDLE),
    ])
    def test_a_finished_turn_beats_a_leftover_tool(self, task, expected):
        """A tool name lingers after its call returns. A finished turn that
        went on drawing "reading" would be the companion claiming work that
        had stopped."""
        assert state_for("read_file", task) is expected

    def test_an_unknown_tool_falls_back_to_the_task(self):
        assert state_for("something_new", "testing") is State.TESTING

    def test_nothing_at_all_is_idle(self):
        assert state_for("", "") is State.IDLE

    def test_every_mapped_tool_lands_on_a_drawable_state(self):
        from wynxo import sprite

        for state in companion._BY_TOOL.values():
            assert sprite.FRAMES[state], state
        for state in companion._BY_TASK.values():
            assert sprite.FRAMES[state], state


class TestThereIsOnlyOneCharacter:
    def test_the_gallery_draws_what_the_session_draws(self):
        """/animate used to show a different drawing from the one a running
        session displayed, because the character existed twice and the two
        drifted. The gallery calls the same renderer now."""
        from wynxo.cli import Repl

        source = inspect.getsource(Repl._show_states)
        assert "sprite.rows(" in source
        assert "from .motion" not in source

    def test_the_old_presentation_layers_are_gone(self):
        for name in ("motion", "logo"):
            with pytest.raises(ImportError):
                __import__(f"wynxo.{name}")

    def test_the_voice_module_does_not_draw(self):
        """pet.py is the name and the lines it says. Nothing else.

        Matched on whole words: REMARKS contains MARKS, and a substring
        check here reported the voice tables as a leftover drawing table.
        """
        import re

        from wynxo import pet

        source = pathlib.Path(pet.__file__).read_text()
        for gone in ("FRAMES", "MARKS", "MOOD_ROLES", "Mood",
                     "ACTIVITY_MOODS"):
            assert not re.search(rf"\b{gone}\b", source), \
                f"{gone} is back in pet.py"
        for gone in ("def rows", "def mark", "def block", "def style",
                     "def react", "def set_activity"):
            assert gone not in source, f"{gone} is back in pet.py"

    def test_only_one_thing_draws_the_companion_during_a_turn(self):
        """The gallery may call the renderer -- that is the point of it,
        and drawing the same pixels is what stops it from drifting. What
        must not come back is a second live rendering: two companions on
        screen at once was the state this replaced.
        """
        from wynxo import cli, ui

        session = inspect.getsource(cli.TerminalCallbacks)
        assert "sprite." not in session, \
            "a second live rendering of the companion is back"
        assert hasattr(ui.ActivityBar, "_scene")


class TestNothingAnimatesOnAClockOfItsOwn:
    def test_the_state_module_has_no_clock_at_all(self):
        source = pathlib.Path(companion.__file__).read_text()
        for forbidden in ("import time", "import threading", "asyncio",
                          "Thread(", "Timer("):
            assert forbidden not in source, f"{forbidden} in companion.py"

    def test_the_sprite_module_has_no_clock_at_all(self):
        from wynxo import sprite

        source = pathlib.Path(sprite.__file__).read_text()
        for forbidden in ("import time", "import threading", "asyncio",
                          "Thread(", "Timer(", "sleep("):
            assert forbidden not in source, f"{forbidden} in sprite.py"

    def test_the_frame_is_the_callers_to_advance(self):
        """A scheduler would keep animating through a stall, showing typing
        while nothing is being written. The frame is a parameter, so it
        moves exactly while the bar repaints and stops when it stops."""
        from wynxo import sprite

        signature = inspect.signature(sprite.rows)
        assert "frame" in signature.parameters


class TestEveryStateLooksLikeADifferentState:
    """A state indicator whose states look alike indicates nothing.

    Thinking moved the eyes by one pixel row -- half a terminal row, the
    smallest change this sprite can make -- so at the size anyone actually
    sees it, thinking and idle were the same picture.
    """

    def _rendered(self, state, frame=1):
        from wynxo import sprite
        from wynxo.theme import resolve

        palette = resolve("purple")
        return tuple(
            (row.plain, tuple(sorted({str(s.style) for s in row.spans})))
            for row in sprite.rows(state, frame, palette))

    def test_no_two_states_draw_the_same_picture(self):
        """Checked on every frame, not just the first.

        listening and speaking each animated back onto the base body, so
        for half of every cycle they were pixel-for-pixel the idle picture
        -- and the state gallery, which draws frame one, showed idle for
        both of them."""
        from wynxo.companion import State
        from wynxo.sprite import FRAMES

        clashes = []
        for frame in range(4):
            seen = {}
            for state in State:
                picture = self._rendered(state, frame)
                if picture in seen and seen[picture] != state.value:
                    clashes.append(
                        f"frame {frame}: {state.value} == {seen[picture]}")
                seen.setdefault(picture, state.value)
        assert clashes == [], clashes
        assert FRAMES, "no frames at all"

    def test_thinking_differs_from_idle_in_shape_not_only_in_colour(self):
        """Colour alone is lost on a terminal without truecolour, and to
        anyone reading shape before hue."""
        from wynxo.companion import State

        idle = [row for row, _ in self._rendered(State.IDLE)]
        thinking = [row for row, _ in self._rendered(State.THINKING)]
        assert idle != thinking

    def test_success_and_error_never_share_a_silhouette(self):
        """The pair that must not be confused at a glance: one of them
        means stop reading and look."""
        from wynxo.companion import State

        good = [row for row, _ in self._rendered(State.SUCCESS)]
        bad = [row for row, _ in self._rendered(State.ERROR)]
        assert good != bad


class TestTheInkMapIsAllUsed:
    def test_every_declared_colour_is_drawn_somewhere(self):
        """A role in the map that no frame uses is a colour the character
        does not have. Two were sitting there: the good green, which meant
        success and a warning shared the amber of the mug, and a second
        screen tone nothing ever drew."""
        from wynxo.sprite import FRAMES, INK

        used = set()
        for frames in FRAMES.values():
            for frame in frames:
                for row in frame:
                    used |= set(row)
        assert sorted(set(INK) - used - {"."}) == []

    def test_every_pixel_drawn_has_a_colour(self):
        from wynxo.sprite import FRAMES, INK

        for state, frames in FRAMES.items():
            for frame in frames:
                for row in frame:
                    unknown = set(row) - set(INK)
                    assert not unknown, f"{state.value}: {unknown}"
