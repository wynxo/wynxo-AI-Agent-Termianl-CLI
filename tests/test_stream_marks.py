"""Markdown a model actually writes, rendered a character at a time.

The streamer draws each character as it arrives, so it has to decide what a
mark is before it can see what follows -- and every wrong guess it makes is
text that never reaches the screen. Content loss, not a style glitch: an
answer about `a**b` that renders as `ab` is an answer about something else.
"""

from __future__ import annotations

import io
import re

import pytest

from rich.console import COLOR_SYSTEMS as _COLOR_SYSTEMS

from wynxo.ui import UI, CodeStreamer


def render(text: str, chunk: int) -> str:
    """What reaches the screen, with the styling stripped off."""
    ui = UI()
    ui.console.file = io.StringIO()
    ui.console.width = ui.width = 200
    streamer = CodeStreamer(ui)
    for i in range(0, len(text), chunk):
        streamer.feed(text[i:i + chunk])
    streamer.finish()
    return " ".join(ui.console.file.getvalue().split())


# One character at a time is the interesting case -- it is how a local model
# actually streams -- and the whole string at once must agree with it.
CHUNKS = (1, 2, 3, 7, 1000)


@pytest.mark.parametrize("chunk", CHUNKS)
class TestNothingIsLost:
    def test_a_power_keeps_its_operator(self, chunk):
        """"2 ** 8 == 256" arrived as "2 8 == 256". Bold cannot open on a
        space, and a model writing arithmetic depends on that rule."""
        assert render("math: 2 ** 8 == 256", chunk) == "math: 2 ** 8 == 256"

    def test_asterisks_inside_a_code_span_are_characters(self, chunk):
        assert render("use `a**b` for powers", chunk) == "use a**b for powers"

    def test_a_code_span_full_of_marks_survives_whole(self, chunk):
        assert render("`**stars** and ## hashes`", chunk) \
            == "**stars** and ## hashes"

    def test_a_lone_asterisk_is_a_lone_asterisk(self, chunk):
        assert render("a * b * c", chunk) == "a * b * c"

    def test_a_trailing_asterisk_is_not_swallowed(self, chunk):
        assert render("trailing star *", chunk) == "trailing star *"

    def test_a_dangling_double_asterisk_is_not_swallowed(self, chunk):
        assert render("half open ** at the end", chunk) \
            == "half open ** at the end"

    def test_a_hash_mid_word_is_not_a_heading(self, chunk):
        assert render("not#a#heading", chunk) == "not#a#heading"


@pytest.mark.parametrize("chunk", CHUNKS)
class TestMarksStillDoTheirJob:
    def test_bold_markers_are_consumed(self, chunk):
        assert render("**bold** then plain", chunk) == "bold then plain"

    def test_backticks_are_consumed(self, chunk):
        assert render("a `code span` here", chunk) == "a code span here"

    def test_a_heading_loses_its_hashes(self, chunk):
        assert render("## heading", chunk) == "heading"

    def test_spans_and_bold_do_not_bleed_into_each_other(self, chunk):
        assert render("`x` then **y** then `z**w`", chunk) \
            == "x then y then z**w"


class TestStyling:
    def test_bold_is_actually_bold(self):
        assert "\x1b[1m" in render_styled("**loud**")

    def test_an_expression_in_a_span_is_not_bold(self):
        """The regression: `a**b` came out bold *and* two characters short."""
        assert "\x1b[1m" not in render_styled("`a**b`")

    def test_a_power_in_prose_is_not_bold(self):
        assert "\x1b[1m" not in render_styled("2 ** 8")


def render_styled(text: str) -> str:
    ui = UI()
    ui.console.file = io.StringIO()
    ui.console._force_terminal = True
    ui.console.width = ui.width = 200
    streamer = CodeStreamer(ui)
    for char in text:
        streamer.feed(char)
    streamer.finish()
    return ui.console.file.getvalue()


class TestFencedCodeIsActuallyHighlighted:
    """It was not, on any current rich, and nothing said so.

    ``highlight`` asked rich for the lexer through ``Syntax.get_lexer``,
    which rich 14 removed. The bare ``except`` around the call caught that
    AttributeError exactly as it would catch an unknown language, so every
    fenced block in every streamed answer came out with no colour -- and
    falling back to plain text is what the guard is *supposed* to do when it
    cannot recognise a language, so it looked deliberate.
    """

    def test_python_is_coloured(self):
        ui = UI()
        rendered = ui.highlight("for attempt in range(N):", "python")
        assert rendered.spans, "no styling applied to a plain Python line"
        assert rendered.plain == "for attempt in range(N):"

    def test_another_language_is_coloured_too(self):
        """Not just a Python special case."""
        assert UI().highlight("let x: u32 = 3;", "rust").spans

    def test_an_unknown_language_is_plain_rather_than_an_error(self):
        rendered = UI().highlight("whatever this is", "nonsense-lang")
        assert rendered.plain == "whatever this is"
        assert not rendered.spans

    def test_plain_text_asks_for_no_lexer_at_all(self):
        for language in ("", "text", "plain"):
            assert not UI().highlight("x = 1", language).spans

    def test_a_half_written_line_never_raises(self):
        """Code is drawn as it arrives, so pygments sees broken input."""
        for fragment in ('x = "unterminat', "def f(", "'''", "\\", "# "):
            assert UI().highlight(fragment, "python").plain == fragment

    def test_a_streamed_fence_reaches_the_screen_highlighted(self):
        """End to end: through the streamer, not just the helper."""
        import wynxo.ui as ui_module

        # Named explicitly. The palette is process-global, so whichever test
        # ran last decides it -- and under "minimal" every syntax role is
        # deliberately the same colour, which would make this pass or fail
        # on test ordering rather than on the code.
        ui = UI(theme="purple")
        ui.console.file = io.StringIO()
        # A console that will actually emit colour. Setting _force_terminal
        # alone leaves color_system None, so every style is dropped on the
        # way out and the assertion below could never pass however well the
        # highlighter worked.
        ui.console._force_terminal = True
        ui.console._color_system = _COLOR_SYSTEMS["truecolor"]
        ui.console.width = ui.width = 80
        streamer = CodeStreamer(ui)
        for char in "```python\nfor x in range(3):\n    pass\n```\n":
            streamer.feed(char)
        streamer.finish()
        out = ui.console.file.getvalue()
        assert "for x in range(3):" in re.sub(r"\x1b\[[0-9;]*m", "", out)
        red, green, blue = (int(ui_module.KEYWORD[i:i + 2], 16)
                            for i in (1, 3, 5))
        assert f"\x1b[38;2;{red};{green};{blue}m" in out, \
            "the keyword arrived uncoloured"


class TestCodeFollowsTheTheme:
    """Code was the one thing on screen /theme could not reach.

    Inline spans were a hardcoded amber and highlighted blocks used raw ANSI
    names, so an answer containing code was coloured by a scheme unrelated to
    everything around it -- and the same expression in prose and in a fenced
    block came out two different colours.
    """

    THEMES = ("purple", "midnight", "sakura", "kawaii", "ember", "catboy")

    def test_switching_theme_switches_the_syntax_colours(self):
        import wynxo.ui as ui_module

        seen = set()
        for theme in self.THEMES:
            UI(theme=theme)
            seen.add((ui_module.CODE_SPAN, ui_module.KEYWORD,
                      ui_module.LITERAL, ui_module.SYMBOL))
        assert len(seen) == len(self.THEMES), \
            "two themes share a syntax palette, so /theme does nothing here"

    def test_a_keyword_is_the_themes_keyword_colour(self):
        import wynxo.ui as ui_module

        for theme in self.THEMES:
            ui = UI(theme=theme)
            rendered = ui.highlight("for x in y:", "python")
            styles = {str(span.style) for span in rendered.spans}
            assert ui_module.KEYWORD in styles, theme

    def test_comments_are_the_ignorable_colour(self):
        """They are the one part of a program you may skip, which is what
        FAINT is for. They were MUTED, the same weight as the operators."""
        import wynxo.ui as ui_module

        ui = UI(theme="purple")
        rendered = ui.highlight("# just a note", "python")
        assert {str(s.style) for s in rendered.spans} == {ui_module.FAINT}


def styled_runs(text: str, width: int, theme: str = "purple"):
    """What reached the screen, grouped by the colour it arrived in."""
    from rich.console import COLOR_SYSTEMS

    ui = UI(theme=theme)
    ui.console.file = io.StringIO()
    ui.console._force_terminal = True
    ui.console._color_system = COLOR_SYSTEMS["truecolor"]
    ui.console.width = ui.width = width
    streamer = CodeStreamer(ui)
    for char in text:
        streamer.feed(char)
    streamer.finish()

    runs, pen = {}, ""
    for part in re.split(r"(\x1b\[[0-9;]*m)", ui.console.file.getvalue()):
        if part.startswith("\x1b"):
            pen = part
            continue
        runs[pen] = runs.get(pen, "") + part
    return {k: v.replace("\n", "") for k, v in runs.items()}


def _pen(colour: str) -> str:
    red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"\x1b[38;2;{red};{green};{blue}m"


class TestAWrapIsNotTheEndOfALine:
    """A wrap closed an open code span, so the backtick that really closed
    it *opened* one -- and the rest of the sentence came out in the code
    colour until the next wrap turned it off again. Narrower terminals wrap
    more, so the answer was most wrong exactly where there was least room.
    """

    PROSE = ("The retry loop sleeps a fixed `RETRY_BACKOFF` between attempts, "
             "so three failures cost about the same as one.")

    @pytest.mark.parametrize("width", [30, 34, 42, 55, 70, 100])
    def test_a_code_span_colours_itself_and_nothing_else(self, width):
        import wynxo.ui as ui_module

        UI(theme="purple")           # pin the palette: it is process-global
        runs = styled_runs(self.PROSE, width)
        assert runs.get(_pen(ui_module.CODE_SPAN), "") == "RETRY_BACKOFF"

    @pytest.mark.parametrize("width", [30, 34, 42, 55, 70])
    def test_bold_ends_where_it_was_closed(self, width):
        runs = styled_runs(
            "Some words before the **emphasis here** and a good deal of "
            "ordinary sentence after it.", width)
        assert runs.get("\x1b[1m", "") == "emphasis here"

    @pytest.mark.parametrize("width", [30, 42, 70])
    def test_nothing_is_lost_to_the_wrap(self, width):
        runs = styled_runs(self.PROSE, width)
        recovered = "".join(runs.values()).replace(" ", "")
        wanted = self.PROSE.replace("`", "").replace(" ", "")
        assert recovered == wanted

    def test_a_real_line_end_still_closes_a_stray_backtick(self):
        """The rule this preserves: one unpaired mark must not colour the
        rest of the answer."""
        import wynxo.ui as ui_module

        UI(theme="purple")
        runs = styled_runs("an `unclosed span\nand the next line entirely", 200)
        assert "next line" not in runs.get(_pen(ui_module.CODE_SPAN), "")


class TestStylingSurvivesTheBar:
    """Every real turn streams through the activity bar, and the bar takes a
    different path: the colour lives in the line's spans rather than going
    out as escapes. Only the second path was ever looked at.

    Carrying a word down to the next line rebuilt the part that stayed
    behind out of ``line.plain`` -- a fresh Text made from the characters,
    which is every span thrown away. So a sentence with a code span and a
    bold word came out entirely plain as soon as it wrapped, and it looked
    deliberate because the wrap was the only visible difference. The first
    paragraph of an answer is usually plain up to its first wrap, which is
    how this survived being looked at.
    """

    PROSE = ("The **total** wait becomes `RETRY_BACKOFF * (2**n - 1)`, which "
             "is worth checking against `request_timeout` before raising it.")

    def _plain(self, text, width, with_bar):
        """Everything that reached the screen, escapes stripped."""
        return re.sub(r"\x1b\[[0-9;]*m", "",
                      self._written(text, width, with_bar))

    def _written(self, text, width, with_bar):
        from rich.console import COLOR_SYSTEMS

        from wynxo.ui import ActivityBar

        ui = UI(theme="purple")
        ui.console.file = io.StringIO()
        ui.console._force_terminal = True
        ui.console._color_system = COLOR_SYSTEMS["truecolor"]
        ui.console.width = ui.width = width
        if with_bar:
            ui.bar = ActivityBar(ui, effort="low")
        streamer = CodeStreamer(ui)
        for char in text:
            streamer.feed(char)
        streamer.finish()
        return ui.console.file.getvalue()

    def _runs(self, text, width, with_bar):
        runs = {}
        for line in self._written(text, width, with_bar).split("\n"):
            pen = ""
            for part in re.split(r"(\x1b\[[0-9;]*m)", line):
                if part.startswith("\x1b"):
                    pen = part
                    continue
                runs[pen] = runs.get(pen, "") + part
        return runs

    @pytest.mark.parametrize("width", [30, 34, 42, 60, 84])
    @pytest.mark.parametrize("with_bar", [False, True])
    def test_a_wrap_never_costs_the_colour_before_it(self, width, with_bar):
        import wynxo.ui as ui_module

        UI(theme="purple")
        runs = self._runs(self.PROSE, width, with_bar)
        red, green, blue = (int(ui_module.CODE_SPAN[i:i + 2], 16)
                            for i in (1, 3, 5))
        code = runs.get(f"\x1b[38;2;{red};{green};{blue}m", "")
        assert code == "RETRY_BACKOFF * (2**n - 1)request_timeout"
        assert runs.get("\x1b[1m", "") == "total"

    @pytest.mark.parametrize("width", [30, 34, 42, 60, 84])
    @pytest.mark.parametrize("with_bar", [False, True])
    def test_neither_path_loses_a_character(self, width, with_bar):
        """Where they wrap is allowed to differ -- with a live region the
        line is still provisional and a word can be carried down whole,
        while without one it has already been written and cannot be taken
        back. What reaches the screen may not differ."""
        # Spelled out rather than derived: stripping "**" from the source
        # would take the one inside (2**n) with it, which is exactly the
        # character this whole file exists to protect.
        wanted = ("The total wait becomes RETRY_BACKOFF * (2**n - 1), which "
                  "is worth checking against request_timeout before raising "
                  "it.")
        shown = self._plain(self.PROSE, width, with_bar)
        assert "".join(shown.split()) == "".join(wanted.split())

    @pytest.mark.parametrize("width", [30, 34, 42, 60, 84])
    def test_the_two_paths_emphasise_the_same_words(self, width):
        import wynxo.ui as ui_module

        UI(theme="purple")
        red, green, blue = (int(ui_module.CODE_SPAN[i:i + 2], 16)
                            for i in (1, 3, 5))
        code_pen = f"\x1b[38;2;{red};{green};{blue}m"

        def emphasis(with_bar):
            runs = self._runs(self.PROSE, width, with_bar)
            return (runs.get(code_pen, ""), runs.get("\x1b[1m", ""))

        assert emphasis(False) == emphasis(True)
