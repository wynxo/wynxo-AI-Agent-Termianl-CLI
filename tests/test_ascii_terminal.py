"""A terminal that cannot draw box characters must still look right.

Under an ASCII locale (a Linux console, an old PuTTY, a stripped-down
Termux) every Unicode glyph we print comes out as a question mark. These
tests pin the fallbacks so the chrome degrades instead of turning into
noise.
"""

import io
import types

from rich.box import ASCII as ASCII_BOX, ROUNDED
from rich.cells import cell_len

from wynxo import cli
from wynxo.queue import Pending
from wynxo.select import HINT, HINT_ASCII
from wynxo.ui import UI, Glyphs, to_ascii


def ascii_ui() -> UI:
    ui = UI()
    ui.g = Glyphs(False)
    ui.box = ASCII_BOX
    return ui


def capture(ui: UI) -> io.StringIO:
    """Point the console at a buffer so we can read what was drawn."""
    stream = io.StringIO()
    ui.console.file = stream
    ui.console._width = ui.width
    return stream


class TestGlyphs:
    def test_box_characters_have_an_ascii_form(self):
        g = Glyphs(False)
        drawn = g.tl + g.tr + g.bl + g.br + g.hbar + g.vbar + g.ellipsis
        assert drawn.isascii()

    def test_unicode_terminals_keep_the_rounded_box(self):
        g = Glyphs(True)
        assert (g.tl, g.tr, g.bl, g.br) == ("╭", "╮", "╰", "╯")


class TestToAscii:
    def test_known_glyphs_are_translated(self):
        assert to_ascii("a · b … c") == "a . b ... c"
        assert to_ascii("↑↓") == "updown"

    def test_unknown_glyphs_are_dropped_not_mangled(self):
        # A question mark reads as a rendering bug; a gap does not.
        assert to_ascii("hi ☃ there") == "hi  there"


class TestInputBox:
    def _repl(self, ui: UI):
        """A stand-in with only what the border methods reach for."""
        repl = types.SimpleNamespace(ui=ui)
        repl._status_line = lambda: "medium . 0 tok . ctx 0%"
        repl._open_box = cli.Repl._open_box.__get__(repl, type(repl))
        repl._bottom_toolbar = cli.Repl._bottom_toolbar.__get__(repl, type(repl))
        repl._prompt_message = cli.Repl._prompt_message.__get__(repl, type(repl))
        repl._border_plain = cli.Repl._border_plain.__get__(repl, type(repl))
        return repl

    def test_top_edge_is_ascii(self, monkeypatch):
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        ui = ascii_ui()
        ui.width = 40
        stream = capture(ui)
        self._repl(ui)._open_box()
        line = stream.getvalue().strip()
        assert line.isascii()
        assert line == "+" + "-" * 38 + "+"

    def test_dumb_terminals_get_no_frame_at_all(self, monkeypatch):
        """prompt_toolkit draws no toolbar there, so an opening edge would
        be left hanging with nothing to close it."""
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: True)
        ui = ascii_ui()
        ui.width = 40
        stream = capture(ui)
        repl = self._repl(ui)
        repl._open_box()
        assert stream.getvalue() == ""
        assert repl._prompt_message().value == "<b>&gt;</b> "

    def test_bottom_edge_is_ascii_and_exactly_one_width(self):
        ui = ascii_ui()
        ui.width = 60
        border = self._repl(ui)._border_plain()
        assert border.isascii()
        assert len(border) == 60

    def test_border_fits_exactly_at_every_width(self):
        """One cell over and the bar wraps onto a second line."""
        for unicode_ok in (True, False):
            for width in (30, 40, 60, 80, 120, 200):
                ui = UI()
                ui.g = Glyphs(unicode_ok)
                ui.width = width
                border = self._repl(ui)._border_plain()
                assert cell_len(border) == width, (unicode_ok, width, border)

    def test_prompt_edge_is_ascii(self, monkeypatch):
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        ui = ascii_ui()
        assert "|" in self._repl(ui)._prompt_message().value

    def test_unicode_terminals_still_get_the_rounded_box(self):
        ui = UI()
        ui.g = Glyphs(True)
        ui.width = 60
        border = self._repl(ui)._border_plain()
        assert border.startswith("╰") and border.endswith("╯")


class TestPanelsAndRules:
    def test_box_style_follows_the_encoding(self):
        assert UI().box in (ROUNDED, ASCII_BOX)

    def test_rule_uses_the_glyph(self):
        ui = ascii_ui()
        ui.width = 30
        stream = capture(ui)
        ui.banner("m", "http://127.0.0.1:11434", "medium", "/tmp/p")
        assert stream.getvalue().isascii()

    def test_error_panel_is_ascii(self):
        ui = ascii_ui()
        ui.width = 40
        stream = capture(ui)
        ui.error("it broke")
        assert stream.getvalue().isascii()


class TestBanner:
    def test_header_never_wraps(self):
        for width in (30, 40, 60, 80, 100):
            ui = UI()
            ui.width = width
            stream = capture(ui)
            ui.banner("qwen3-coder:30b", "http://192.168.1.20:11434",
                      "medium", "/home/me/projects/wynxo")
            head = stream.getvalue().splitlines()[1]
            assert len(head) <= width, (width, head)


class TestStatsLine:
    class _Usage:
        completion_tokens = 83

        def tokens_per_second(self):
            return 14.0

    def test_narrow_terminals_drop_fields_instead_of_wrapping(self):
        for width in (24, 30, 36, 44, 80):
            ui = UI()
            ui.width = width
            stream = capture(ui)
            ui.stats(self._Usage(), 3.4, "medium", 6.0)
            line = stream.getvalue().strip("\n")
            assert len(line) <= width, (width, line)
            # The token count is the one thing that always survives.
            assert "83 tok" in line

    def test_wide_terminals_keep_everything(self):
        ui = UI()
        ui.width = 80
        stream = capture(ui)
        ui.stats(self._Usage(), 3.4, "medium", 6.0)
        line = stream.getvalue()
        for bit in ("medium", "83 tok", "14 tok/s", "3.4s", "ctx 6%"):
            assert bit in line


class TestPicker:
    def test_both_hints_exist_and_the_fallback_is_ascii(self):
        assert not HINT.isascii()
        assert HINT_ASCII.isascii()

    def test_labels_are_down_converted_for_ascii_terminals(self):
        import inspect

        from wynxo import select

        source = inspect.getsource(select.choose)
        assert "to_ascii" in source


class TestQueuePreview:
    def test_ellipsis_can_be_ascii(self):
        pending = Pending()
        pending.draft = "x" * 100
        assert pending.preview(width=20, ellipsis="...").isascii()

    def test_preview_respects_the_width(self):
        pending = Pending()
        pending.draft = "x" * 100
        assert len(pending.preview(width=20, ellipsis="...")) == 20


class TestCprWarning:
    """Terminals that never answer a cursor position request get a
    prompt_toolkit warning printed over the session. The bar is one line so
    that CPR is never needed, so the warning is noise."""

    def test_silencing_reaches_the_renderer(self):
        from wynxo.select import silence_cpr_warning

        class App:
            class renderer:
                cpr_not_supported_callback = object()

        app = App()
        silence_cpr_warning(app)
        assert app.renderer.cpr_not_supported_callback is None

    def test_a_prompt_toolkit_without_it_is_survivable(self):
        from wynxo.select import silence_cpr_warning

        class App:
            __slots__ = ()

        silence_cpr_warning(App())  # must not raise

    def test_the_repl_silences_its_own_session(self):
        import inspect

        # The classic prompt is built lazily now (its eager construction
        # crashed chat mode under Git Bash), so the CPR silencing moved
        # into the factory that actually builds it.
        source = inspect.getsource(cli.Repl._make_prompt_session)
        assert "silence_cpr_warning(session.app)" in source


class TestPinnedPlan:
    """The plan is one thing that changes, not a stream of panels. It used
    to print a fresh one on every update, so a five-step plan left five in
    the scrollback and the current one was whichever scrolled past last."""

    def bar(self, width=80, unicode_ok=True):
        from wynxo.ui import ActivityBar, Glyphs

        ui = UI()
        ui.g = Glyphs(unicode_ok)
        ui.box = ROUNDED if unicode_ok else ASCII_BOX
        ui.width = width
        return ActivityBar(ui, "medium")

    PLAN = "[x] read the config\n[>] add the retry\n[ ] cover it with a test"

    def test_no_plan_means_no_panel(self):
        assert self.bar()._plan_panel() is None

    def test_the_panel_counts_what_is_done(self):
        bar = self.bar()
        bar.set_plan(self.PLAN)
        assert "1/3" in bar._plan_panel().title

    def test_completion_is_recognised_only_when_every_step_is_ticked(self):
        bar = self.bar()
        bar.set_plan(self.PLAN)
        assert bar.plan_is_complete() is False
        bar.set_plan("[x] one\n[x] two")
        assert bar.plan_is_complete() is True

    def test_an_empty_plan_is_not_complete(self):
        """Otherwise it would 'finish' the moment it appeared."""
        bar = self.bar()
        bar.set_plan("")
        assert bar.plan_is_complete() is False
        bar.set_plan("   \n  ")
        assert bar.plan_is_complete() is False

    async def test_finishing_ticks_everything_then_clears(self):
        bar = self.bar()
        bar.set_plan(self.PLAN)
        assert bar._plan_panel() is not None
        await bar.finish_plan()
        assert bar.plan == ""
        assert bar._plan_panel() is None, "it must leave when the work is done"

    async def test_finishing_an_empty_plan_is_a_no_op(self):
        bar = self.bar()
        await bar.finish_plan()      # must not raise
        assert bar.plan == ""

    def test_setting_a_new_plan_cancels_a_stale_animation(self):
        bar = self.bar()
        bar.plan_done_frame = 4
        bar.set_plan(self.PLAN)
        assert bar.plan_done_frame == 0

    def test_the_plan_sits_above_the_status_strip(self):
        """Order matters: the strip is the thing anchored to the prompt."""
        from rich.console import Group

        bar = self.bar()
        bar.set_plan(self.PLAN)
        rendered = bar._renderable()
        assert isinstance(rendered, Group)
        assert rendered.renderables[0] is not None
        assert len(rendered.renderables) == 2

    def test_it_draws_in_ascii_too(self):
        bar = self.bar(unicode_ok=False)
        bar.set_plan(self.PLAN)
        stream = capture(bar.ui)
        bar.ui.console.print(bar._plan_panel())
        assert stream.getvalue().isascii()


class TestEffortSurge:
    """Stepping up to the top two levels should feel like it costs
    something, because it does."""

    async def _fires(self, previous, current, animations=True):
        import types

        from wynxo import cli
        from wynxo.ui import UI

        ui = UI()
        ui.width = 80
        played = []

        async def fake_surge(_ui, label, style, width=34):
            played.append(label)

        repl = types.SimpleNamespace(
            ui=ui, config=types.SimpleNamespace(animations=animations))
        import wynxo.ui as ui_module

        real, ui_module.surge = ui_module.surge, fake_surge
        try:
            await cli.Repl._effort_surge(repl, previous, current)
        finally:
            ui_module.surge = real
        return played

    async def test_it_fires_stepping_up_into_ultra(self):
        assert await self._fires("high", "ultra") == ["ULTRA"]

    async def test_it_fires_stepping_up_into_max(self):
        assert await self._fires("medium", "max") == ["MAX EFFORT"]

    async def test_it_does_not_fire_for_the_ordinary_levels(self):
        """An animation on every change would just be noise."""
        for level in ("low", "medium", "high", "xhigh"):
            assert await self._fires("low", level) == []

    async def test_it_does_not_fire_on_the_way_down(self):
        """Celebrating a step down is celebrating the wrong direction."""
        assert await self._fires("ultra", "max") == []

    async def test_it_does_not_fire_when_already_there(self):
        assert await self._fires("ultra", "ultra") == []

    async def test_animations_off_means_off(self):
        assert await self._fires("low", "ultra", animations=False) == []

    async def test_nothing_is_drawn_without_a_terminal(self):
        """A pipe would otherwise collect every frame as separate output."""
        from wynxo.ui import UI, surge

        ui = UI()
        assert ui.console.is_terminal is False   # pytest captures stdout
        stream = capture(ui)
        await surge(ui, "ULTRA", "bold")
        assert stream.getvalue() == ""


class TestActivityAnimation:
    """A static word next to a spinner still reads as stalled -- the spinner
    turns whether or not anything is happening."""

    def bar(self, activity="thinking", unicode_ok=True, animate=True):
        from wynxo.ui import ActivityBar, Glyphs

        ui = UI()
        ui.g = Glyphs(unicode_ok)
        ui.width = 70
        b = ActivityBar(ui, "medium")
        b.activity = activity
        b.animate = animate
        return b

    def test_the_label_changes_between_frames(self):
        bar = self.bar()
        seen = set()
        for frame in range(12):
            bar._frame = frame
            seen.add(bar._activity_text().plain)
        assert len(seen) > 1, "the label never moves"

    def test_the_word_itself_is_always_intact(self):
        """Animating must not eat characters out of the word."""
        bar = self.bar()
        for frame in range(20):
            bar._frame = frame
            assert bar._activity_text().plain.startswith("thinking")

    def test_the_label_is_a_fixed_width_so_the_bar_does_not_jitter(self):
        from rich.cells import cell_len

        bar = self.bar()
        widths = set()
        for frame in range(20):
            bar._frame = frame
            widths.add(cell_len(bar._activity_text().plain))
        assert len(widths) == 1, f"width wobbles: {widths}"

    def test_only_thinking_gets_the_dots(self):
        bar = self.bar(activity="writing")
        bar._frame = 9
        assert bar._activity_text().plain == "writing"

    def test_animations_off_means_a_still_label(self):
        bar = self.bar(animate=False)
        frames = {bar._activity_text().plain for bar._frame in range(12)}
        assert frames == {"thinking"}

    def test_ascii_terminals_get_the_plain_word(self):
        bar = self.bar(unicode_ok=False)
        for frame in range(12):
            bar._frame = frame
            assert bar._activity_text().plain == "thinking"

    def test_an_empty_activity_is_survivable(self):
        bar = self.bar(activity="")
        assert bar._activity_text().plain == ""


class TestCodeStreamsLive:
    """Watching a function appear a whole line at a time is the thing this
    exists to avoid."""

    def setup_streamer(self):
        from wynxo.ui import ActivityBar, CodeStreamer

        ui = UI()
        ui.width = 70
        bar = ActivityBar(ui, "medium")
        ui.bar = bar
        leads = []
        bar.set_lead = lambda line: leads.append(line.plain if line else None)
        return CodeStreamer(ui), leads

    ANSWER = "Here:\n\n```python\ndef check(t):\n    return len(t) > 10\n```\n\nDone."

    def test_a_code_line_grows_a_character_at_a_time(self):
        streamer, leads = self.setup_streamer()
        for char in self.ANSWER:
            streamer.feed(char)
        streamer.finish()

        growing = [l for l in leads if l and "def check" in l]
        assert len(growing) > 3, f"only {len(growing)} states: {growing}"
        assert any(l.strip() == "def c" for l in leads if l)

    def test_the_finished_block_is_still_correct(self):
        streamer, _ = self.setup_streamer()
        stream = capture(streamer.ui)
        for char in self.ANSWER:
            streamer.feed(char)
        streamer.finish()
        out = stream.getvalue()
        assert "def check(t):" in out
        assert "return len(t) > 10" in out
        assert "```" not in out, "the fence must not reach the screen"

    def test_the_fence_is_never_shown_half_written(self):
        streamer, leads = self.setup_streamer()
        for char in self.ANSWER:
            streamer.feed(char)
        streamer.finish()
        for lead in leads:
            if lead:
                assert "`" not in lead, f"a backtick leaked: {lead!r}"

    def test_prose_still_streams_by_word(self):
        """Code goes character by character; prose by word, so a word never
        appears split in half."""
        streamer, _ = self.setup_streamer()
        stream = capture(streamer.ui)
        for char in "The quick brown fox jumps over it.\n":
            streamer.feed(char)
        streamer.finish()
        assert "quick brown fox" in stream.getvalue()

    def test_a_half_written_code_line_is_not_lost_at_the_end(self):
        """A response cut off mid-line must still show what arrived."""
        streamer, _ = self.setup_streamer()
        stream = capture(streamer.ui)
        for char in "```python\ndef check(t):\n    return len(":
            streamer.feed(char)
        streamer.finish()
        assert "return len(" in stream.getvalue()

    def test_no_bar_means_the_old_line_at_a_time_behaviour(self):
        """Nothing pinned means nowhere to redraw, so waiting for the
        newline is correct rather than a regression."""
        from wynxo.ui import CodeStreamer

        ui = UI()
        ui.width = 70
        ui.bar = None
        streamer = CodeStreamer(ui)
        stream = capture(ui)
        for char in self.ANSWER:
            streamer.feed(char)
        streamer.finish()
        assert "def check(t):" in stream.getvalue()
