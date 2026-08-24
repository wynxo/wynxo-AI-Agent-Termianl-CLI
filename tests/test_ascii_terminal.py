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

        source = inspect.getsource(cli.Repl.__init__)
        assert "silence_cpr_warning(self.prompt_session.app)" in source
