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
        repl._prompt_note = None
        repl._echo_prompt = cli.Repl._echo_prompt.__get__(repl, type(repl))
        repl._bottom_toolbar = cli.Repl._bottom_toolbar.__get__(repl, type(repl))
        repl._prompt_message = cli.Repl._prompt_message.__get__(repl, type(repl))
        repl._border_plain = cli.Repl._border_plain.__get__(repl, type(repl))
        return repl

    def test_there_is_no_top_edge_to_strand(self, monkeypatch):
        """The composer is one row, so nothing can drift away from it.

        There was a top border, printed separately with console.print while
        prompt_toolkit drew the closing toolbar. prompt_toolkit reaches the
        bottom row by emitting newlines, so the two drifted apart and the
        border was stranded halfway up the screen with a void of blank rows
        under it. Folding it into the prompt fixed that; deleting it is
        better still, because a full width of ─ above the input was chrome
        answering chrome -- the edge below already closes the region and
        carries the status as well.
        """
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        ui = ascii_ui()
        ui.width = 40
        message = self._repl(ui)._prompt_message().value
        assert "\n" not in message
        assert "-" not in message and "+" not in message
        assert message.isascii()

    def test_the_caret_is_the_one_the_transcript_uses(self, monkeypatch):
        """What you type and what it becomes are the same shape. The
        composer drew a bare ">" while the echoed line drew "❯"."""
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        ui = UI()
        ui.g = Glyphs(True)
        assert ui.g.caret in self._repl(ui)._prompt_message().value

    def test_the_box_is_never_drawn_into_the_transcript(self, monkeypatch):
        """Nothing prints a border any more, so no frame can be stranded."""
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        ui = ascii_ui()
        ui.width = 40
        stream = capture(ui)
        self._repl(ui)._echo_prompt("hello")
        drawn = stream.getvalue()
        assert "+--" not in drawn and "\u256d" not in drawn
        assert "> hello" in drawn

    def test_no_cursor_arithmetic_is_emitted(self, monkeypatch):
        """The old seating wrote raw cursor jumps and guessed the row from
        how much the turn had printed. It guessed wrong whenever the screen
        had scrolled, which is most of the time."""
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        ui = ascii_ui()
        ui.width = 40
        ui._lines_since_prompt = 7
        stream = capture(ui)
        self._repl(ui)._echo_prompt("hi")
        assert "\x1b[" not in stream.getvalue().replace("\x1b[0m", "")

    def test_dumb_terminals_get_no_frame_at_all(self, monkeypatch):
        """prompt_toolkit draws no toolbar there, so an opening edge would
        be left hanging with nothing to close it."""
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: True)
        ui = ascii_ui()
        ui.width = 40
        stream = capture(ui)
        repl = self._repl(ui)
        assert ui.g.caret in repl._prompt_message().value
        repl._echo_prompt("hello")
        assert stream.getvalue() == ""

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

    def test_the_prompt_is_the_caret_and_nothing_else(self, monkeypatch):
        """At column zero, which is where the echoed line puts it too, so
        what you type sits in the column it will land in."""
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        ui = ascii_ui()
        value = self._repl(ui)._prompt_message().value
        assert "|" not in value and "-" not in value
        assert ui.g.caret in value
        assert value.isascii()

    def test_prompt_edge_uses_the_palette_accent(self, monkeypatch):
        """The caret matches the frame around it: one hue, not a cyan
        caret inside a violet box."""
        monkeypatch.setattr(cli, "is_dumb_terminal", lambda: False)
        from wynxo.ui import ACCENT

        ui = UI()
        ui.g = Glyphs(True)
        value = self._repl(ui)._prompt_message().value
        assert f'fg="{ACCENT}"' in value
        assert "ansicyan" not in value

    def test_toolbar_frame_follows_the_theme(self):
        """/theme recolours the whole box -- top edge, bottom edge and
        caret all move together now."""
        from wynxo.theme import resolve
        from wynxo.ui import apply_palette, _ansi_of

        ui = UI()
        ui.g = Glyphs(True)
        ui.width = 100
        repl = self._repl(ui)
        value = repl._bottom_toolbar().value
        assert value.startswith(_ansi_of("#b47cff"))     # purple accent
        apply_palette(resolve("midnight"))
        value = repl._bottom_toolbar().value
        assert value.startswith(_ansi_of("#6ec7ff"))     # midnight accent

    def test_the_hint_trims_rather_than_vanishing(self):
        """The binding hints give way gradually, and the stop hint is the
        last to go: it is the one you most need to remember."""
        ui = UI()
        ui.g = Glyphs(True)
        repl = self._repl(ui)
        ui.width = 100
        full = repl._border_plain()
        assert "^O think" in full and "^R talk" in full and "^C stop" in full
        ui.width = 60
        mid = repl._border_plain()
        assert "^O think" in mid and "^R talk" not in mid
        assert "^C stop" in mid
        ui.width = 40
        tight = repl._border_plain()
        assert "^O think" not in tight and "^C stop" in tight
        ui.width = 30
        assert "^C stop" not in repl._border_plain()

    def test_the_bottom_edge_is_a_rule_with_no_corners(self):
        """There is no box left for it to be the bottom of.

        It closed a left edge coming down the composer once. That edge went
        -- a single │ beside the caret with nothing above it reads as a
        stray mark -- so the corner had nothing to turn from. What is left
        is a rule: the seam between the transcript and what you are
        typing, with the status set into it."""
        ui = UI()
        ui.g = Glyphs(True)
        ui.width = 60
        border = self._repl(ui)._border_plain()
        assert border.startswith("─") and border.endswith("─")
        assert not set(border) & set("╰╯╭╮│")


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

    def test_the_plan_counts_what_is_done(self):
        bar = self.bar()
        bar.set_plan(self.PLAN)
        assert "1/3" in bar._plan_panel().plain

    def test_every_step_carries_a_mark(self):
        """A step not started used to be bare indented text, which reads as
        a wrapped continuation of the step above rather than as a step."""
        for unicode_ok in (True, False):
            bar = self.bar(unicode_ok=unicode_ok)
            bar.set_plan(self.PLAN)
            g = bar.ui.g
            drawn = bar._plan_panel().plain
            for mark in (g.step_done, g.step_now, g.step_todo):
                assert mark in drawn, (unicode_ok, mark)

    def test_it_is_a_list_and_not_a_box(self):
        """A hundred-column box spent four cells of border and eighty of
        trailing whitespace per row to say three short things."""
        bar = self.bar()
        bar.set_plan(self.PLAN)
        plain = bar._plan_panel().plain
        for border in ("╭", "╮", "╰", "╯", "│"):
            assert border not in plain, border

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
        # The strip is the thing anchored to the prompt, so it is last.
        # What sits above it grew from one plan panel to the scene rows, so
        # the count is not the invariant -- the order is.
        assert len(rendered.renderables) >= 2
        assert "0.0s" in rendered.renderables[-1].plain
        above = " ".join(r.plain for r in rendered.renderables[:-1])
        assert "1/3" in above, above

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


class TestOnlyOneThingOnTheStripMoves:
    """The activity word is an answer, and answers hold still.

    It used to run a highlight through itself a character at a time, with
    cycling dots after it and -- on the kawaii theme -- a sparkle in front,
    all while the mascot animated beside them. Four moving things at twelve
    frames a second in an eighty-cell strip, and the word saying what the
    agent was doing was the hardest of the four to read.

    So: the mascot is the sign of life, the word is the answer, and the
    strip has exactly one animation at a time -- the mascot when it is on,
    the spinner when it is off, a still mark when neither may move.
    """

    def bar(self, activity="thinking", unicode_ok=True, animate=True,
            pet=None):
        from wynxo.ui import ActivityBar, Glyphs

        ui = UI()
        ui.g = Glyphs(unicode_ok)
        ui.width = 70
        b = ActivityBar(ui, "medium", pet=pet)
        b.activity = activity
        b.animate = animate
        return b

    def _frames(self, bar, count=24):
        """The whole pinned region per frame, not just the strip.

        The one animation moved: it is the companion in the scene above the
        strip now, and the strip holds still underneath it. Sampling only
        the strip answers "nothing moves" for a region that is moving."""
        out = []
        for frame in range(count):
            bar._frame = frame
            out.append("\n".join([t.plain for t in bar._scene()]
                                  + [bar._render().plain]))
        return out

    def test_the_word_never_moves(self):
        bar = self.bar()
        seen = {bar._activity_text().plain for bar._frame in range(24)}
        assert seen == {"thinking"}

    def test_the_word_is_a_fixed_width(self):
        from rich.cells import cell_len

        bar = self.bar()
        widths = {cell_len(bar._activity_text().plain)
                  for bar._frame in range(24)}
        assert len(widths) == 1, widths

    def test_no_trailing_dots_are_appended(self):
        for activity in ("thinking", "writing", "running"):
            bar = self.bar(activity=activity)
            assert bar._activity_text().plain == activity

    def test_an_empty_activity_is_survivable(self):
        assert self.bar(activity="")._activity_text().plain == ""

    def test_something_always_says_the_session_is_alive(self):
        bar = self.bar()
        bar.state = "searching"
        assert len(set(self._frames(bar))) > 1

    def test_the_strip_itself_holds_still_under_the_companion(self):
        """One animation at a time. With the companion moving above it, a
        spinner on the row beneath is the second.

        The companion is opt-in, so it has to be asked for here -- with it
        off, which is the default, the spinner is the only sign of life
        and the strip is where it belongs."""
        from wynxo.pet import Pet

        bar = self.bar(pet=Pet(enabled=True))
        bar.state = "searching"
        strips = set()
        for frame in range(24):
            bar._frame = frame
            strips.add(bar._render().plain)
        assert len(strips) == 1, strips

    def test_the_strip_animates_when_it_is_the_only_row(self):
        """Below the scene threshold the strip is all there is, so the
        spinner comes back to it."""
        bar = self.bar()
        bar.ui.width = 40
        assert len(set(self._frames(bar))) > 1

    def test_no_second_animation_rides_along_on_any_theme(self):
        """The kawaii theme used to add a cycling sparkle of its own, so on
        that one theme the strip had two animations instead of one."""
        from wynxo.theme import names, resolve

        for name in names():
            bar = self.bar()
            bar.ui.palette = resolve(name)
            # With the pet off and motion off, nothing at all may change.
            bar.animate = False
            assert len(set(self._frames(bar))) == 1, name

    def test_reduced_motion_holds_everything_still(self):
        bar = self.bar(animate=False)
        bar.state = "searching"
        assert len(set(self._frames(bar))) == 1

    def test_reduced_motion_still_says_what_is_happening(self):
        """Nothing moving must not mean nothing shown.

        Reduced motion drops the companion -- it is the moving part, and a
        still picture of a cat is decoration -- and keeps the words, which
        carry the same fact."""
        bar = self.bar(animate=False)
        drawn = "\n".join(t.plain for t in bar._scene())
        assert "thinking" in drawn
        assert bar.ui.g.busy in drawn
        assert not set(drawn) & set("▀▄█"), "the sprite is still moving"

    def test_ascii_terminals_get_a_mark_they_can_draw(self):
        bar = self.bar(unicode_ok=False, animate=False)
        drawn = "\n".join(t.plain for t in bar._scene())
        assert bar.ui.g.busy in drawn
        drawn.encode("ascii")
        bar._render().plain.encode("ascii")



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
