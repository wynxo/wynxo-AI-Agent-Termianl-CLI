"""Rendering bugs found by reading and exercising the UI layer.

Each class here is one defect that was visible on screen: colours that only
half-changed, a paragraph that came out as a column of bare indents, a width
that stopped following the window, a plan that miscounted itself.
"""

from __future__ import annotations

import io
import re

from rich.cells import cell_len

from wynxo import ui as ui_module
from wynxo.theme import resolve
from wynxo.ui import UI, ActivityBar, CodeStreamer

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> list[str]:
    return [ANSI.sub("", line) for line in text.split("\n")]


class Pen:
    """A stand-in for the pinned bar. _rewritable only asks whether one is
    there, and set_lead is all the streamer calls on it."""

    def __init__(self):
        self.lead = None

    def set_lead(self, line) -> None:
        self.lead = line


def streamed(text: str, width: int = 40, indent: str = "  ",
             bar: bool = False, ui: UI | None = None) -> list[str]:
    ui = ui or UI()
    ui.width = width
    ui.live_ok = False
    out = io.StringIO()
    ui.console.file = out
    if bar:
        ui.bar = Pen()
    streamer = CodeStreamer(ui, indent=indent)
    streamer.feed(text)
    streamer.finish()
    return plain(out.getvalue())


class TestWideCharacterWrapping:
    """Japanese and Chinese have no spaces to break at, and their characters
    are two cells wide. The carry-down guard measured characters, so a word
    never looked too long for a line it had already overflowed: it was
    lifted onto a new line, overflowed that one, and was lifted again."""

    def test_a_run_of_wide_characters_does_not_shred_into_blank_lines(self):
        lines = streamed("日本語" * 30, width=40, bar=True)
        blank = [ln for ln in lines if not ln.strip()]
        assert len(blank) <= 3, (
            f"{len(blank)} blank lines: the word was carried down repeatedly")

    def test_no_line_runs_past_the_terminal(self):
        for text in ("日本語" * 30, "🙂" * 60, "中文字符" * 25):
            for bar in (False, True):
                lines = streamed(text, width=40, bar=bar)
                widest = max(cell_len(ln) for ln in lines)
                assert widest <= 40, f"{widest} cells at width 40: {text[:6]!r}"

    def test_the_text_survives_intact(self):
        """Wrapping must not eat or duplicate characters."""
        lines = streamed("日本語" * 20, width=40, bar=True)
        assert "".join(ln.strip() for ln in lines) == "日本語" * 20

    def test_ascii_prose_still_wraps_at_words(self):
        lines = [ln for ln in streamed(
            "the quick brown fox jumps over the lazy dog " * 4,
            width=44, bar=True) if ln.strip()]
        for line in lines:
            assert cell_len(line) <= 44
        assert len(lines) > 1
        # No word split across a line break.
        assert "".join(ln.strip() + " " for ln in lines).split() == (
            "the quick brown fox jumps over the lazy dog " * 4).split()

    def test_a_word_longer_than_the_line_is_broken_at_the_edge(self):
        lines = [ln for ln in streamed("a" * 200, width=40, bar=True)
                 if ln.strip()]
        assert len(lines) > 1
        assert "".join(ln.strip() for ln in lines) == "a" * 200


class TestLiteralWrapping:
    """File contents wrap at the edge rather than at words, but still by
    cells: a two-cell character written at the last column ran over."""

    def test_wide_characters_stay_inside_the_width(self):
        ui = UI()
        ui.width = 40
        ui.live_ok = False
        out = io.StringIO()
        ui.console.file = out
        streamer = CodeStreamer(ui, indent="  ", literal=True, code=False)
        streamer.feed("表" * 60)
        streamer.finish()
        for line in plain(out.getvalue()):
            assert cell_len(line) <= 40


class TestTheStreamerFollowsAResize:
    """A streamer lives for a whole turn. Its width was captured in the
    constructor, so widening the window mid-answer left the rest of the
    reply wrapped to the old column."""

    def test_the_wrap_column_tracks_the_ui(self):
        ui = UI()
        ui.width = 40
        streamer = CodeStreamer(ui, indent="  ")
        narrow = streamer.width
        ui.width = 120
        assert streamer.width > narrow


class TestWidthFollowsTheTerminal:
    """Everything that wraps -- the streamers, the activity bar, the diff
    cards -- reads ``ui.width``. A resize is one re-measure, from the
    SIGWINCH handler, and every consumer picks it up on its next draw."""

    def test_refresh_size_re_reads_the_terminal(self, monkeypatch):
        ui = UI()
        monkeypatch.setattr("wynxo.ui.terminal_width", lambda: 52)
        ui.refresh_size()
        assert ui.width == 52, "the UI never heard the window change size"

    def test_narrow_follows_the_new_width(self, monkeypatch):
        ui = UI()
        monkeypatch.setattr("wynxo.ui.terminal_width", lambda: 100)
        ui.refresh_size()
        assert ui.narrow is False
        monkeypatch.setattr("wynxo.ui.terminal_width", lambda: 40)
        ui.refresh_size()
        assert ui.narrow is True

    def test_a_streamer_widens_with_the_terminal(self, monkeypatch):
        ui = UI()
        monkeypatch.setattr("wynxo.ui.terminal_width", lambda: 40)
        ui.refresh_size()
        streamer = CodeStreamer(ui, indent="  ")
        narrow = streamer.width
        monkeypatch.setattr("wynxo.ui.terminal_width", lambda: 120)
        ui.refresh_size()
        assert streamer.width > narrow


class TestThePlanCountsItsOwnSteps:
    def bar(self, plan: str) -> ActivityBar:
        ui = UI()
        ui.live_ok = False
        bar = ActivityBar(ui, "medium")
        bar.plan = plan
        return bar

    def test_a_finished_plan_reads_as_finished(self):
        bar = self.bar("[x] one\n[x] two\n[x] three")
        assert "plan  3/3" in bar._plan_panel().plain
        assert bar.plan_is_complete()

    def test_a_wrapped_task_does_not_inflate_the_total(self):
        """`total` counted every non-empty line, so a task carrying a
        newline made a finished plan read as unfinished."""
        bar = self.bar("[x] one\n    continued\n[x] two")
        assert bar.plan_is_complete(), "the fixture is not a finished plan"
        drawn = bar._plan_panel().plain
        assert "plan  2/2" in drawn, f"{drawn!r} disagrees with plan_is_complete"
        assert "one continued" in drawn, "the wrapped half of the task was lost"


class TestATruncatedDiffSaysSo:
    def render(self, text: str) -> str:
        ui = UI()
        ui.live_ok = False
        out = io.StringIO()
        ui.console.file = out
        ui.console.width = 100
        ui.diff(text)
        return ANSI.sub("", out.getvalue())

    def test_a_long_diff_reports_what_it_left_out(self):
        """Cut off at exactly 120 lines with no mark, a diff reads as a
        diff that ended there -- a different claim."""
        drawn = self.render("\n".join(f"+line {i}" for i in range(300)))
        assert "180 more lines" in drawn

    def test_a_short_diff_says_nothing_extra(self):
        drawn = self.render("+one\n-two\n")
        assert "more line" not in drawn

    def test_the_singular_reads_properly(self):
        drawn = self.render("\n".join(
            f"+line {i}" for i in range(ui_module.UI.MAX_DIFF_LINES + 1)))
        assert "1 more line" in drawn and "1 more lines" not in drawn


class TestEveryColourFollowsTheTheme:
    def test_the_edit_card_and_the_surge_change_with_the_theme(self):
        """cli.py imports GOOD, BAD and BAR_ACCENT from ui, and the
        hand-kept sweep list named none of them: after /theme the edit
        card's done and failed marks kept the palette before last."""
        from wynxo import cli

        try:
            ui_module.apply_palette(resolve("sakura"))
            sakura = resolve("sakura")
            assert cli.GOOD == sakura.good
            assert cli.BAD == sakura.bad
            assert cli.BAR_ACCENT == sakura.bar_accent
        finally:
            ui_module.apply_palette(resolve("purple"))

    def test_a_second_change_still_lands(self):
        """The sweep matches on the previous value, so it has to keep
        working after the first change has moved it."""
        from wynxo import cli

        try:
            ui_module.apply_palette(resolve("sakura"))
            ui_module.apply_palette(resolve("ember"))
            assert cli.GOOD == resolve("ember").good
        finally:
            ui_module.apply_palette(resolve("purple"))

    def test_a_name_that_is_not_a_colour_is_left_alone(self):
        from wynxo import cli

        before = cli.WARN
        try:
            ui_module.apply_palette(resolve("ember"))
            assert cli.WARN == before
        finally:
            ui_module.apply_palette(resolve("purple"))


class TestMessagesKeepTheirColumn:
    """warn() and error() hang their continuation lines under the first
    word, so the marker stays the only thing in its column. textwrap
    measures with len(), so a message in a two-cell script came out twice
    as wide as it asked for and the console re-wrapped it at column zero --
    throwing away the indent the wrapping existed to produce."""

    def warned(self, message: str, width: int = 40) -> list[str]:
        ui = UI()
        ui.width = width
        ui.live_ok = False
        out = io.StringIO()
        ui.console.file = out
        ui.console.width = width
        ui.warn(message)
        return [ln for ln in plain(out.getvalue()) if ln.strip()]

    def test_wide_characters_stay_inside_the_terminal(self):
        for line in self.warned("設定ファイルが見つかりませんでした。" * 4):
            assert cell_len(line) <= 40

    def test_the_continuation_stays_under_the_first_word(self):
        lines = self.warned("設定ファイルが見つかりませんでした。" * 4)
        assert len(lines) > 1, "the fixture did not wrap"
        for line in lines[1:]:
            assert line.startswith("    "), f"lost its indent: {line!r}"

    def test_ascii_messages_are_unchanged(self):
        lines = self.warned("a normal long warning message " * 4)
        assert lines[0].startswith("  ! ")
        for line in lines[1:]:
            assert line.startswith("    ")

    def test_a_message_with_its_own_lines_keeps_them(self):
        lines = self.warned("first\nsecond")
        assert lines == ["  ! first", "    second"]


class TestWrapCells:
    def test_it_measures_in_cells(self):
        from wynxo.ui import wrap_cells

        for line in wrap_cells("日本語" * 10, 20):
            assert cell_len(line) <= 20

    def test_it_keeps_ascii_words_whole(self):
        from wynxo.ui import wrap_cells

        got = wrap_cells("the quick brown fox jumps over it", 12)
        assert all(cell_len(line) <= 12 for line in got)
        assert " ".join(got).split() == "the quick brown fox jumps over it".split()

    def test_a_word_wider_than_the_line_is_cut(self):
        from wynxo.ui import wrap_cells

        got = wrap_cells("x" * 50, 10)
        assert all(cell_len(line) <= 10 for line in got)
        assert "".join(got) == "x" * 50

    def test_empty_text_is_one_empty_line(self):
        from wynxo.ui import wrap_cells

        assert wrap_cells("", 10) == [""]


class TestTheBannerKeepsWhatMatters:
    """"Assembled by priority rather than truncated: on a narrow terminal
    the server address goes before the project path does, because the path
    is the one you actually need to see." It did the opposite."""

    def drawn(self, width: int, path: str) -> str:
        ui = UI()
        ui.width = width
        ui.live_ok = False
        out = io.StringIO()
        ui.console.file = out
        ui.console.width = width
        ui.banner("qwen2.5-coder:32b", "http://homelab:11434", "high", path)
        return self.rows(out.getvalue())

    @staticmethod
    def rows(written: str) -> list[str]:
        """The identity block's own rows: the name and model on one, the
        settings that give way on the next. The rule is not one of them."""
        return [ln.rstrip() for ln in plain(written)
                if ln.strip() and set(ln.strip()) != {"\u2500"}]

    def test_the_server_is_never_shown_without_the_path(self):
        """The property, at every width: dropping is by priority, so the
        server cannot survive something more important than itself."""
        path = "/home/you/code/some-project-with-a-long-name"
        squeezed = False
        for width in range(24, 130):
            line = "\n".join(self.drawn(width, path))
            has_path = "long-name" in line
            if "homelab" in line:
                assert has_path, (
                    f"width {width}: the server was kept and the path "
                    f"dropped, which is backwards -- {line!r}")
            elif has_path:
                squeezed = True
        assert squeezed, "no width actually exercised the choice"

    def test_both_fit_when_there_is_room(self):
        block = "\n".join(self.drawn(120, "/home/you/code/proj"))
        assert "proj" in block and "homelab" in block

    def test_no_row_of_the_banner_runs_past_the_terminal(self):
        for width in range(24, 130, 7):
            for row in self.drawn(
                    width, "/home/you/code/some-project-with-a-long-name"):
                assert cell_len(row) <= width, f"{width}: {row!r}"


class TestShortenPathHonoursItsBudget:
    """"never the full thing when it would push everything else off the
    line" -- but the fallback returned the last two components at whatever
    length they happened to be."""

    def test_a_long_directory_is_brought_inside_the_budget(self):
        ui = UI()
        for width in (40, 60, 80, 100, 120):
            ui.width = width
            got = ui.shorten_path(
                "/home/you/code/some-project-with-a-really-long-name")
            assert cell_len(got) <= max(18, width // 3), (
                f"{got!r} is over budget at width {width}")

    def test_a_short_path_is_left_exactly_as_it_is(self):
        ui = UI()
        ui.width = 120
        assert ui.shorten_path("/a/b/c") == "/a/b/c"

    def test_the_home_directory_still_becomes_a_tilde(self):
        import os

        ui = UI()
        ui.width = 200
        home = os.path.expanduser("~")
        assert ui.shorten_path(f"{home}/code/proj") == "~/code/proj"

    def test_the_last_component_is_what_survives(self):
        ui = UI()
        ui.width = 60
        assert "long-name" in ui.shorten_path(
            "/very/deep/tree/of/directories/here/some-project-long-name")
