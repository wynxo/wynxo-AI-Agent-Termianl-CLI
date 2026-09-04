"""Driving the desktop: key spelling, backend selection, and the guards.

There is no display server in CI, so the backends are driven through an
injected runner and checked on the argv they build. That is the right level
anyway: the bugs in this layer are spelling bugs -- a modifier sent as a
literal character, a wheel notch sent as a button that does not exist, a
grabber invoked in a mode that opens an interactive selector and waits.
"""

from __future__ import annotations


import pytest

from wynxo import desktop as d
from wynxo.desktop import (MacOS, Unavailable, Wayland, Windows, X11,
                           DesktopError, detect, parse_chord)


class Recorder:
    """Stands in for subprocess, keeping every argv it was handed."""

    def __init__(self, answers=None):
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, argv, timeout=d.DEFAULT_TIMEOUT, input_text=None):
        self.calls.append(list(argv))
        for key, answer in self.answers.items():
            if key in argv:
                return answer
        return ""

    @property
    def last(self) -> list[str]:
        return self.calls[-1]

    def flat(self) -> str:
        return " | ".join(" ".join(c) for c in self.calls)


class TestSpellingAKeystroke:
    @pytest.mark.parametrize("chord,mods,key", [
        ("ctrl+s", ["ctrl"], "s"),
        ("Ctrl-Shift-S", ["ctrl", "shift"], "S"),
        ("alt+f4", ["alt"], "f4"),
        ("super+space", ["super"], "space"),
        ("enter", [], "enter"),
        ("Return", [], "enter"),
        ("a", [], "a"),
    ])
    def test_the_spellings_a_model_reaches_for(self, chord, mods, key):
        assert parse_chord(chord) == (mods, key)

    def test_plus_is_both_separator_and_key(self):
        """`ctrl++` is zoom in. Splitting naively loses the key entirely."""
        assert parse_chord("ctrl++") == (["ctrl"], "+")

    def test_an_unknown_modifier_is_refused(self):
        with pytest.raises(DesktopError, match="not a modifier"):
            parse_chord("hyper+x")

    def test_an_unknown_key_is_refused(self):
        """Not guessed at. A guess here presses something nobody asked for,
        in whatever window is in front."""
        with pytest.raises(DesktopError, match="not a key"):
            parse_chord("wibble")

    def test_a_space_separated_chord_says_how_to_spell_it(self):
        with pytest.raises(DesktopError, match=r"join with '\+'"):
            parse_chord("ctrl s")

    def test_nothing_is_refused(self):
        with pytest.raises(DesktopError):
            parse_chord("")


class TestX11:
    def _x11(self, **kw):
        run = Recorder(**kw)
        return X11(run, grabber="scrot"), run

    def test_modifiers_reach_xdotool_by_name(self):
        x11, run = self._x11()
        x11.press("ctrl+shift+s")
        assert run.last == ["xdotool", "key", "--clearmodifiers",
                            "ctrl+shift+s"]
        # The letter's case is passed through rather than normalised: both
        # "ctrl+shift+s" and "ctrl+shift+S" are things xdotool understands,
        # and rewriting one into the other would be changing a keystroke on
        # the model's behalf for no gain.
        x11.press("Ctrl-Shift-S")
        assert run.last[-1] == "ctrl+shift+S"

    def test_named_keys_become_keysyms(self):
        x11, run = self._x11()
        x11.press("enter")
        assert run.last[-1] == "Return"
        x11.press("pageup")
        assert run.last[-1] == "Prior"

    def test_typing_clears_modifiers(self):
        """A modifier the user is physically holding turns every letter
        into a shortcut. Typing "s" into a held Ctrl is a save dialog in
        whatever window is in front."""
        x11, run = self._x11()
        x11.type_text("hello")
        assert "--clearmodifiers" in run.last

    def test_typing_stops_option_parsing_before_the_text(self):
        """Text beginning with a dash is text, not a flag."""
        x11, run = self._x11()
        x11.type_text("--force")
        assert run.last[-2:] == ["--", "--force"]

    def test_the_wheel_is_buttons_not_an_axis(self):
        """X11 has no scroll axis: up and down are buttons 4 and 5."""
        x11, run = self._x11()
        x11.scroll(3)
        assert run.last[-1] == "4" and "--repeat" in run.last
        x11.scroll(-2)
        assert run.last[-1] == "5"

    def test_scrolling_nothing_does_nothing(self):
        x11, run = self._x11()
        x11.scroll(0)
        assert run.calls == []

    def test_an_invented_button_is_refused(self):
        x11, _run = self._x11()
        with pytest.raises(DesktopError, match="not a mouse button"):
            x11.click("scroll-wheel-left")

    def test_the_pointer_is_read_from_the_shell_form(self):
        x11, _run = self._x11(answers={"getmouselocation":
                                       "X=100\nY=250\nSCREEN=0\n"})
        assert x11.pointer() == (100, 250)

    def test_a_window_that_closed_mid_query_is_dropped(self):
        """search then getwindowname is two calls, and a window can go away
        between them. That is ordinary, not an error to report."""
        def run(argv, timeout=d.DEFAULT_TIMEOUT, input_text=None):
            if "search" in argv:
                return "111\n222\n"
            if "getwindowname" in argv and "222" in argv:
                raise DesktopError("no such window")
            if "getwindowname" in argv:
                return "Konsole\n"
            return "X=0\nY=0\nWIDTH=800\nHEIGHT=600\n"

        titles = [w.title for w in X11(run).windows()]
        assert titles == ["Konsole"]

    def test_no_screenshot_without_a_grabber(self):
        with pytest.raises(DesktopError, match="scrot"):
            X11(Recorder(), grabber="").screenshot("/tmp/x.png")


class TestWayland:
    def test_wtype_holds_and_releases_each_modifier(self):
        """A modifier pressed and never released leaves the desktop stuck
        in it long after the batch is over."""
        run = Recorder()
        Wayland(run, typer="wtype").press("ctrl+shift+s")
        argv = run.last
        assert argv.count("-M") == 2 and argv.count("-m") == 2
        assert argv[argv.index("-k") + 1] == "s"

    def test_super_is_the_logo_key(self):
        run = Recorder()
        Wayland(run, typer="ydotool").press("super+space")
        assert "logo+space" in run.last

    def test_wtype_cannot_move_the_pointer_and_says_so(self):
        wayland = Wayland(Recorder(), typer="wtype")
        assert not wayland.can("move")
        with pytest.raises(DesktopError, match="ydotool"):
            wayland.move(10, 10)

    def test_windows_cannot_be_listed_and_the_reason_is_the_design(self):
        """Not a missing package. Wayland deliberately stops one
        application enumerating another's windows, and a refusal that
        suggests installing something would send somebody hunting."""
        wayland = Wayland(Recorder(), typer="ydotool")
        assert not wayland.can("windows")
        with pytest.raises(DesktopError, match="does not let one application"):
            wayland.windows()


class TestWindows:
    def test_sendkeys_syntax_in_ordinary_text_is_escaped(self):
        """SendKeys reads + as shift and ^ as ctrl. Unescaped, typing an
        email address sends modifiers: "a+b" arrives as "aB"."""
        assert Windows._sendkeys_escape("a+b^c%d") == "a{+}b{^}c{%}d"
        assert Windows._sendkeys_escape("plain") == "plain"

    def test_the_windows_key_is_refused_rather_than_dropped(self):
        with pytest.raises(DesktopError, match="cannot send the Windows key"):
            Windows(Recorder()).press("super+r")

    def test_a_chord_becomes_a_sendkeys_prefix(self):
        run = Recorder()
        Windows(run).press("ctrl+shift+s")
        assert "'^+s'" in " ".join(run.last)

    def test_a_literal_plus_survives_being_a_key(self):
        """`ctrl++` is zoom in, and `+` is also SendKeys' own shift prefix."""
        run = Recorder()
        Windows(run).press("ctrl++")
        assert "'^{+}'" in " ".join(run.last)


class TestMacOS:
    def test_cmd_and_super_are_the_same_key_here(self):
        run = Recorder(answers={"p": "100,200"})
        MacOS(run, cliclick=True).press("cmd+c")
        assert "kd:cmd" in run.last

    def test_without_cliclick_it_names_the_permission_too(self):
        """The failure shape on macOS is silent success: without
        Accessibility permission the command works and nothing moves."""
        mac = MacOS(Recorder(), cliclick=False)
        with pytest.raises(DesktopError, match="Accessibility"):
            mac.type_text("hi")


class TestChoosingABackend:
    def test_a_wayland_session_never_gets_xdotool(self, monkeypatch):
        """xdotool talks to XWayland, where it drives X applications and
        silently does nothing to native ones -- which is exactly the
        half-working case this refuses to be."""
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(d.sys, "platform", "linux")
        monkeypatch.setattr(d.shutil, "which",
                            lambda n: f"/usr/bin/{n}" if n in
                            ("xdotool", "ydotool", "grim") else None)
        assert detect().name == "wayland"

    def test_x11_with_xdotool(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(d.sys, "platform", "linux")
        monkeypatch.setattr(d.shutil, "which",
                            lambda n: "/usr/bin/x" if n in ("xdotool", "scrot")
                            else None)
        backend = detect()
        assert backend.name == "x11" and backend.can("screenshot")

    def test_x11_without_xdotool_names_the_package(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(d.sys, "platform", "linux")
        monkeypatch.setattr(d.shutil, "which", lambda n: None)
        backend = detect()
        assert isinstance(backend, Unavailable)
        assert "apt install xdotool" in backend.reason

    def test_headless_says_there_is_no_desktop(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("XDG_SESSION_TYPE", "tty")
        monkeypatch.setattr(d.sys, "platform", "linux")
        backend = detect()
        assert isinstance(backend, Unavailable)
        assert "no graphical session" in backend.reason
        # And it points at what does still work.
        assert "shell tool" in backend.reason


class TestScreenGrabbers:
    @pytest.mark.parametrize("grabber", d.GRABBERS)
    def test_every_grabber_is_invoked_non_interactively(self, grabber):
        """A grabber invoked wrong does not fail -- it opens a region
        selector and waits, which from a background agent is a hang."""
        argv = d._grab_argv(grabber, "/tmp/shot.png")
        assert argv[0] == grabber
        assert "/tmp/shot.png" in argv
        assert len(argv) > 1, "no bare invocation writes to a path"
