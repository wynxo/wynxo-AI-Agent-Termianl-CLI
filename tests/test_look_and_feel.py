"""How the interface reads: the wordmark, the capability row, the marks.

The one thing that has to be true of every animation here is that it
leaves behind exactly what the static version would. A flourish that
scrolls or that strands a half-drawn frame is not a flourish, it is
damage to a transcript somebody scrolls back through.
"""

from __future__ import annotations

import io
import os
import pty
import sys

import pytest

from wynxo.ui import (TOUCHES_THE_MACHINE, UI, blend, gradient,
                      verb)


def _captured(width=88, terminal=False):
    ui = UI()
    sink = io.StringIO()
    ui.console = type(ui.console)(
        file=sink, width=width, force_terminal=terminal,
        color_system="truecolor" if terminal else None, highlight=False)
    ui.width = width
    return ui, sink


class TestTheGradient:
    def test_the_ends_are_the_colours_asked_for(self):
        assert blend("#000000", "#ffffff", 0.0) == "#000000"
        assert blend("#000000", "#ffffff", 1.0) == "#ffffff"

    def test_the_middle_is_between_them(self):
        assert blend("#000000", "#ffffff", 0.5) == "#808080"

    def test_it_is_clamped(self):
        assert blend("#000000", "#ffffff", 5.0) == "#ffffff"
        assert blend("#000000", "#ffffff", -3.0) == "#000000"

    def test_a_theme_with_one_colour_gets_a_flat_wordmark(self):
        """A deliberately plain theme sets accent and bar_accent the same,
        and gets no gradient without anything special-casing it."""
        swept = gradient("wynxo", "#c77dff", "#c77dff")
        assert len({str(span.style) for span in swept.spans}) <= 1

    def test_named_ansi_colours_work_too(self):
        """The plain theme uses bright_white rather than a hex triplet."""
        assert gradient("wynxo", "bright_white", "bright_magenta").plain == "wynxo"

    def test_the_text_survives_being_coloured(self):
        assert gradient("wynxo", "#b47cff", "#ff7cc8").plain == "wynxo"


class TestTheCapabilityRow:
    def test_it_says_what_this_machine_can_do(self):
        ui, sink = _captured()
        ui.banner("m", "e", "high", "~/w",
                  capabilities=["drives your desktop", "sound"])
        assert "drives your desktop" in sink.getvalue()
        assert "sound" in sink.getvalue()

    def test_nothing_to_say_means_no_row(self):
        """On a server the honest row is no row. A line of crosses at
        startup is a list of things wynxo is not, which is not what a
        header is for."""
        ui, sink = _captured()
        ui.banner("m", "e", "high", "~/w", capabilities=[])
        assert len([ln for ln in sink.getvalue().splitlines() if ln.strip()]) == 1

    def test_the_identity_line_is_still_one_row(self):
        ui, sink = _captured()
        ui.banner("qwen3:27b", "http://localhost:11434", "high", "~/code/wynxo")
        rows = [ln for ln in sink.getvalue().splitlines() if ln.strip()]
        assert len(rows) == 1
        assert "wynxo" in rows[0] and "qwen3:27b" in rows[0]


class TestTheRevealLeavesNothingBehind:
    def test_a_pipe_gets_the_finished_line_and_no_frames(self):
        """Redirected, in a test, into `head`: one write, no animation."""
        ui, sink = _captured(terminal=False)
        ui.banner("qwen3:27b", "e", "high", "~/w")
        assert "\x1b[1G" not in sink.getvalue()
        assert "wynxo" in sink.getvalue()

    def test_a_very_narrow_terminal_skips_it(self):
        ui, sink = _captured(width=24, terminal=True)
        ui.banner("qwen3:27b", "e", "high", "~/w")
        assert "\x1b[1G" not in sink.getvalue()

    @pytest.mark.skipif(sys.platform == "win32", reason="no pty on Windows")
    def test_a_terminal_is_left_showing_the_whole_line(self):
        """The check that matters, through a real VT emulator: the frames
        are drawn over each other, so what is left is the finished line
        and nothing else."""
        pyte = pytest.importorskip("pyte")

        pid, fd = pty.fork()
        if pid == 0:                                   # pragma: no cover
            os.environ["COLUMNS"] = "88"
            try:
                # pytest replaces sys.stdout with its own capture object,
                # and rich holds whatever sys.stdout was when the Console
                # was built -- so without this the child writes into
                # pytest's buffer and the pty sees nothing at all.
                sys.stdout = os.fdopen(1, "w", buffering=1)
                from wynxo.ui import UI as _UI

                bar = _UI()
                bar.banner("qwen3:27b", "http://x", "high", "~/w",
                           capabilities=["drives your desktop"])
                sys.stdout.flush()
            finally:
                os._exit(0)
        raw = b""
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            raw += chunk
        os.waitpid(pid, 0)

        screen = pyte.Screen(88, 10)
        pyte.Stream(screen).feed(raw.decode("utf-8", "replace"))
        drawn = [ln.rstrip() for ln in screen.display if ln.strip()]
        assert drawn[0].startswith("wynxo · qwen3:27b")
        assert "drives your desktop" in drawn[1]
        # And the wipe left no smear: the wordmark appears once, not as
        # "wwywynwynxwynxo" -- which is what it drew when SafeConsole
        # scrubbed the carriage returns out of the frames.
        assert drawn[0].count("wynxo") == 1


class TestTheTwoKindsOfCall:
    """Reading a file and typing into somebody's browser are not the same
    event, and a transcript where they look identical is one you have to
    read word by word to find out what wynxo did to your machine."""

    def _drawn(self, name):
        ui, sink = _captured(terminal=True)
        ui.tool_call(name, "something", "a detail")
        return sink.getvalue()

    def test_file_work_keeps_the_arrow(self):
        assert self._drawn("read_file").lstrip().startswith("\x1b")
        assert "→" in self._drawn("edit_file")

    @pytest.mark.parametrize("name", sorted(TOUCHES_THE_MACHINE))
    def test_reaching_outside_the_project_is_marked_differently(self, name):
        assert "✦" in self._drawn(name)
        assert "→" not in self._drawn(name)

    def test_a_failure_still_reads_as_a_failure_either_way(self):
        for name in ("read_file", "control_computer"):
            ui, sink = _captured(terminal=True)
            ui.tool_call(name, "x", "went wrong", ok=False)
            assert "✗" in sink.getvalue()


class TestEveryToolHasAVerb:
    def test_the_machine_tools_are_not_shown_as_dispatch_names(self):
        """"control_computer" on a line somebody reads while waiting is
        three syllables that say nothing about what happened. "look"
        already is a word and is left alone -- the rule is that the column
        reads as verbs, not that every name is rewritten."""
        assert verb("control_computer") == "desktop"
        assert verb("system_control") == "system"
        assert verb("system_status") == "check"
        assert verb("look") == "look"

    def test_every_registered_tool_reads_as_something(self):
        from pathlib import Path

        from wynxo.tools import build_registry

        for name in build_registry(Path(".")).names():
            shown = verb(name)
            assert "_" not in shown, f"{name} shows as {shown!r}"


class TestTheCompanionKnowsAboutTheMachine:
    def test_driving_the_desktop_animates_as_typing(self):
        """The hands on the deck are not a metaphor there -- that is what
        is happening."""
        from wynxo.companion import State, state_for

        assert state_for(tool="control_computer") is State.CODING

    def test_looking_animates_as_a_scan(self):
        from wynxo.companion import State, state_for

        assert state_for(tool="look") is State.SEARCHING
