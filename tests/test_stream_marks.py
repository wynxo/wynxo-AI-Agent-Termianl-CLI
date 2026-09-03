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
        ui = UI()
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
        # magenta is what _token_style gives a keyword.
        assert "\x1b[35m" in out, "the keyword arrived uncoloured"
