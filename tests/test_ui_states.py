"""Every state the live region can be in, rendered and inspected.

There is one live region now -- the activity bar -- and everything
provisional is inside it: the streamed edit card, the plan, the half-written
line, the recent tool events, and the status strip itself. That makes the
set of things that can be on screen at once small enough to enumerate, which
is the point: the bugs worth catching here are combinations, not components.

A DONE state with a spinner still turning, a cancelled turn still saying
"running", an answer hidden behind a card, a strip one cell too wide for the
terminal it is drawn on -- each is invisible in a test of any single piece
and obvious in a render of the whole.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.cells import cell_len

from wynxo import livediff
from wynxo.ui import ActivityBar, UI


WIDTHS = [40, 60, 80, 100, 140]


def _ui(width: int = 100) -> UI:
    ui = UI()
    ui.live_ok = False
    ui.width = width
    return ui


def _bar(state: str, width: int = 100) -> ActivityBar:
    """One deterministic bar per named state."""
    ui = _ui(width)
    bar = ActivityBar(ui, "medium", model="qwen3-coder:30b")
    bar.animate = False          # no sweep: the frames must not decide a test
    bar.started -= 4             # a plausible elapsed time

    if state == "idle":
        bar.activity = "idle"
    elif state == "thinking":
        bar.activity, bar.tokens = "thinking", 120
    elif state == "planning":
        bar.activity = "planning"
        bar.plan = "[x] read the parser\n[>] add the retry path\n[ ] run the tests"
    elif state == "tool_running":
        bar.activity, bar.detail = "running", "pytest -q"
    elif state == "tool_success":
        bar.activity, bar.detail = "idle", ""
        bar.tokens = 300
    elif state == "tool_failure":
        bar.activity, bar.detail = "idle", "exit code 1"
    elif state == "answer_streaming":
        from rich.text import Text

        bar.activity, bar.tokens = "writing", 512
        bar.lead = Text("  the answer so far")
    elif state == "editing":
        bar.activity = "editing"
        bar.card = livediff.DiffCard(tool="write_file", path="calc.py",
                                     before="a = 1\n")
        bar.card.feed("a = 2\nb = 3\n")
    elif state == "queued":
        bar.activity, bar.queued = "thinking", "and then fix the tests"
    elif state == "plan_complete":
        bar.activity = "verifying"
        bar.plan = "[x] one\n[x] two"
        bar.plan_done_frame = 1
    elif state == "narrow_everything":
        from rich.text import Text

        bar.activity, bar.detail = "running", "a/very/long/path/to/a/file.py"
        bar.queued = "another message entirely"
        bar.tokens, bar.context_pct = 4096, 87.5
        bar.lead = Text("  half a line of the answer")
        bar.plan = "[>] one\n[ ] two"
        bar.card = livediff.DiffCard(tool="edit_file", path="deep/nested.py")
        bar.card.feed("x = 1\n")
    else:                                        # pragma: no cover
        raise AssertionError(f"unknown state {state}")
    return bar


STATES = ["idle", "thinking", "planning", "tool_running", "tool_success",
          "tool_failure", "answer_streaming", "editing", "queued",
          "plan_complete", "narrow_everything"]


def _drawn(bar: ActivityBar, width: int) -> str:
    console = Console(file=io.StringIO(), width=width, force_terminal=False,
                      soft_wrap=False)
    console.print(bar)
    return console.file.getvalue()


def _card_rows(bar: ActivityBar, width: int) -> list[str]:
    """The rows of the edit card, exactly as the bar builds them.

    Everything else in the region is a rich renderable -- a Panel, a Text --
    and rich fits those to the console it is given. The card is the one part
    wynxo lays out itself, row by row, so it is the one that can be built
    too wide for the terminal it lands on. Measuring the printed output
    instead proves nothing: rich wraps an over-wide row onto two, so every
    line comes back within the width and the fault is hidden. On the real
    screen that wrap makes the region a line taller and the whole display
    shifts on the next repaint.
    """
    if bar.card is None:
        return []
    return bar.card.render(bar.ui.g, min(width, 100))


class TestTheStripIsAlwaysExactlyOneRowOfTheTerminal:
    """It is drawn on the bottom row. One cell too wide and the terminal
    wraps it, which scrolls the whole display by a line on every repaint."""

    @pytest.mark.parametrize("state", STATES)
    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_strip_fits(self, state, width):
        bar = _bar(state, width)
        assert cell_len(bar._render().plain) == width, state

    @pytest.mark.parametrize("state", STATES)
    def test_the_strip_is_a_single_line(self, state):
        assert "\n" not in _bar(state)._render().plain

    @pytest.mark.parametrize("state", ["editing", "narrow_everything"])
    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_edit_card_is_built_for_the_terminal_it_lands_on(self, state,
                                                                 width):
        bar = _bar(state, width)
        rows = _card_rows(bar, width)
        assert rows, f"{state} was supposed to have a card"
        for line in rows:
            assert cell_len(line) <= width, f"{state}@{width}: {line!r}"

    @pytest.mark.parametrize("state", STATES)
    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_whole_region_renders_within_the_terminal(self, state, width):
        """rich fits its own renderables, so this is a check that nothing
        escapes rich rather than a check on the layout."""
        drawn = _drawn(_bar(state, width), width)
        for line in drawn.rstrip("\n").split("\n"):
            assert cell_len(line) <= width, f"{state}@{width}: {line!r}"


class TestImpossibleCombinations:
    def test_a_finished_edit_is_not_still_streaming(self):
        bar = _bar("editing")
        bar.card.finish(ok=True)
        assert "streaming" not in _drawn(bar, 100), (
            "the card says it is still going after the edit ended")

    def test_a_card_taken_out_of_the_region_leaves_nothing_behind(self):
        bar = _bar("editing")
        assert "streaming" in _drawn(bar, 100)
        bar.card = None
        drawn = _drawn(bar, 100)
        assert "streaming" not in drawn and "calc.py" not in drawn

    def test_a_completed_plan_does_not_read_as_half_done(self):
        bar = _bar("plan_complete")
        assert bar.plan_is_complete()
        assert bar._plan_panel().title == "plan  2/2"

    def test_the_plan_counts_steps_not_wrapped_lines(self):
        bar = _bar("idle")
        bar.plan = ("[x] one\n"
                    "    a continuation line that is not a step\n"
                    "[ ] two")
        assert bar._plan_panel().title == "plan  1/2"

    def test_typing_wins_over_the_tool_detail(self):
        """Your own keystrokes beat a description of what the agent is up
        to: the detail is still one line up, and you cannot see what you are
        typing anywhere else."""
        bar = _bar("queued")
        bar.detail = "some/file.py"
        plain = bar._render().plain
        assert "and then fix the tests" in plain
        assert "some/file.py" not in plain

    def test_the_token_count_is_never_the_thing_dropped(self):
        """It is the reason the strip exists."""
        for width in WIDTHS:
            bar = _bar("narrow_everything", width)
            assert "4096 tok" in bar._render().plain, width

    def test_an_idle_bar_claims_no_work(self):
        plain = _bar("idle")._render().plain
        for busy in ("thinking", "writing", "running", "editing"):
            assert busy not in plain


class TestTheRegionIsALayerNotARecord:
    def test_nothing_it_draws_is_committed(self):
        """Transient is what makes the difference: the region erases its own
        rows on stop, so a card, a plan or a half-written line can never end
        up in the scrollback."""
        import inspect

        assert "transient=True" in inspect.getsource(ActivityBar.start)

    def test_stopping_twice_is_harmless(self):
        bar = _bar("editing")
        bar.stop()
        bar.stop()

    def test_a_bar_with_no_terminal_never_starts_a_live(self):
        bar = _bar("thinking")
        bar.ui.live_ok = False
        bar.start()
        assert bar._live is None
