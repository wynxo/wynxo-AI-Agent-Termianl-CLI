"""Driving the computer itself: the pointer, the keyboard, and the windows.

Everything above this file talks about *what* to do -- click that, type
this, bring the editor forward. This file is the whole of *how*, and the
how is different on every desktop: X11 has xdotool, Wayland has ydotool and
wtype, Windows has SendInput through user32, macOS has cliclick and
AppleScript. None of them spell a keystroke the same way and none of them
can do quite the same set of things.

Two rules hold everywhere in here.

**Refuse rather than guess.** A backend that cannot do something says so and
does nothing. Half-working automation is worse than none: a click that
lands in the wrong place, or a keystroke that goes to whatever happens to
be in front, is not a failure anybody can see in a log -- it is a sentence
in the transcript claiming something happened, and a desktop that quietly
does not match it.

**Nothing here decides whether an action is allowed.** That is the
permission layer's job, above. This module knows how to press a key; it
does not know whether it should.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_TIMEOUT = 10.0
"""Long enough for a compositor to answer, short enough that a wedged
helper does not hang a turn. Nothing here is a long-running operation:
these are single input events and single queries."""


class DesktopError(Exception):
    """Something about the desktop could not be done; the message says what
    to install or what to do instead."""


@dataclass
class Window:
    """One window, as much as the backend can say about it."""

    id: str
    title: str = ""
    app: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def describe(self) -> str:
        where = (f"  {self.width}x{self.height}+{self.x}+{self.y}"
                 if self.width or self.height else "")
        app = f"  [{self.app}]" if self.app else ""
        return f"{self.title or '(untitled)'}{app}{where}"


# -- key names ---------------------------------------------------------------

MODIFIERS = {
    "ctrl": "ctrl", "control": "ctrl", "ctl": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt",
    "shift": "shift",
    "super": "super", "win": "super", "windows": "super", "meta": "super",
    "cmd": "cmd", "command": "cmd",
}
"""Every spelling a model reasonably reaches for, mapped to one name.

``cmd`` stays separate from ``super`` rather than folding into it: on macOS
they are different keys with different bindings, and a Ctrl-C sent as a
Cmd-C is a copy where an interrupt was meant."""

KEYS = {
    "enter": "enter", "return": "enter", "\n": "enter",
    "esc": "esc", "escape": "esc",
    "tab": "tab", "backspace": "backspace", "bs": "backspace",
    "delete": "delete", "del": "delete",
    "space": "space", " ": "space",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end",
    "pageup": "pageup", "pgup": "pageup",
    "pagedown": "pagedown", "pgdn": "pagedown",
    "insert": "insert", "ins": "insert",
    "menu": "menu", "printscreen": "printscreen", "prtsc": "printscreen",
    **{f"f{n}": f"f{n}" for n in range(1, 25)},
}


def parse_chord(chord: str) -> tuple[list[str], str]:
    """Split "ctrl+shift+s" into (["ctrl", "shift"], "s").

    Raises DesktopError on anything that is not a chord, because the
    alternative is pressing something nobody asked for. A model that writes
    "ctrl s" or "Ctrl-S" gets told the spelling rather than having it
    guessed at -- guessing here means an unrelated key going into whatever
    window is in front.
    """
    raw = (chord or "").strip()
    if not raw:
        raise DesktopError("no key was given.")
    # "+" is both the separator and a key. A trailing one is the key.
    parts = [p for p in raw.replace("-", "+").split("+") if p] or ["+"]
    if raw.endswith("+") and len(parts) >= 1:
        parts.append("+")
    *mods, key = parts
    names: list[str] = []
    for mod in mods:
        canonical = MODIFIERS.get(mod.strip().lower())
        if canonical is None:
            raise DesktopError(
                f"{mod!r} is not a modifier. Use ctrl, alt, shift, super or "
                f"cmd, as in 'ctrl+s'.")
        if canonical not in names:
            names.append(canonical)
    key = key.strip()
    low = key.lower()
    if low in KEYS:
        return names, KEYS[low]
    if len(key) == 1:
        return names, key
    hint = (" Modifiers join with '+', as in 'ctrl+s'."
            if " " in key.strip() else "")
    raise DesktopError(
        f"{key!r} is not a key this recognises.{hint} Single characters work "
        "as themselves; named keys are enter, esc, tab, backspace, delete, "
        "space, the arrows, home, end, pageup, pagedown, insert and f1-f24.")


# -- backends ----------------------------------------------------------------

def _run(argv: list[str], timeout: float = DEFAULT_TIMEOUT,
         input_text: str | None = None) -> str:
    """One helper invocation. Returns stdout; raises DesktopError on failure."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, input=input_text)
    except FileNotFoundError:
        raise DesktopError(f"{argv[0]} is not installed.") from None
    except subprocess.TimeoutExpired:
        raise DesktopError(
            f"{argv[0]} did not answer within {timeout:.0f}s. The desktop "
            "may be locked or not responding.") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        why = detail[0][:200] if detail else f"exit {proc.returncode}"
        raise DesktopError(f"{argv[0]} failed: {why}")
    return proc.stdout


class Backend:
    """What every desktop can be asked to do.

    A backend implements what it can and leaves the rest raising
    DesktopError from here, which is the refusal: a caller gets a sentence
    naming the missing piece rather than a no-op that reads as success.
    """

    name = "none"
    display = ""

    def __init__(self, run=_run):
        self.run = run

    # -- what this one can actually do --
    def can(self, action: str) -> bool:
        return action in self.actions()

    def actions(self) -> set[str]:
        return set()

    def _no(self, action: str) -> "DesktopError":
        return DesktopError(
            f"the {self.name} backend cannot {action}. "
            f"{self.missing(action)}")

    def missing(self, action: str) -> str:
        return ""

    # -- input --
    def type_text(self, text: str) -> None:
        raise self._no("type text")

    def press(self, chord: str) -> None:
        raise self._no("press keys")

    def move(self, x: int, y: int) -> None:
        raise self._no("move the pointer")

    def click(self, button: str = "left", count: int = 1) -> None:
        raise self._no("click")

    def scroll(self, amount: int) -> None:
        raise self._no("scroll")

    # -- looking --
    def pointer(self) -> tuple[int, int]:
        raise self._no("report the pointer position")

    def screen(self) -> tuple[int, int]:
        raise self._no("report the screen size")

    def windows(self) -> list[Window]:
        raise self._no("list windows")

    def focused(self) -> Window | None:
        raise self._no("say which window has focus")

    def focus(self, window: Window) -> None:
        raise self._no("change which window has focus")

    def screenshot(self, path: str) -> None:
        raise self._no("take a screenshot")


class Unavailable(Backend):
    """No usable backend. Carries why, and what to install."""

    name = "unavailable"

    def __init__(self, reason: str, run=_run):
        super().__init__(run)
        self.reason = reason

    def missing(self, action: str) -> str:
        return self.reason

    def _no(self, action: str) -> DesktopError:
        return DesktopError(self.reason)


class X11(Backend):
    """X11, through xdotool -- and xdotool alone.

    It does every one of these operations, which is the reason to prefer it
    over a pile of single-purpose helpers: one binary either present or
    absent, rather than five capabilities each independently missing.
    Screenshots are the exception, since xdotool takes none; whichever
    grabber is installed does that.
    """

    name = "x11"

    #: xdotool's spelling of the keys this normalises. Anything not here is
    #: passed through as itself, which is right for single characters and
    #: for X keysym names that already match.
    KEYS = {
        "enter": "Return", "esc": "Escape", "tab": "Tab",
        "backspace": "BackSpace", "delete": "Delete", "space": "space",
        "up": "Up", "down": "Down", "left": "Left", "right": "Right",
        "home": "Home", "end": "End", "pageup": "Prior",
        "pagedown": "Next", "insert": "Insert", "menu": "Menu",
        "printscreen": "Print",
        **{f"f{n}": f"F{n}" for n in range(1, 25)},
    }
    MODS = {"ctrl": "ctrl", "alt": "alt", "shift": "shift",
            "super": "super", "cmd": "super"}
    BUTTONS = {"left": "1", "middle": "2", "right": "3"}

    def __init__(self, run=_run, grabber: str = ""):
        super().__init__(run)
        self.grabber = grabber
        self.display = os.environ.get("DISPLAY", "")

    def actions(self) -> set[str]:
        doable = {"type", "press", "move", "click", "scroll", "pointer",
                  "screen", "windows", "focused", "focus"}
        if self.grabber:
            doable.add("screenshot")
        return doable

    def missing(self, action: str) -> str:
        if action == "take a screenshot":
            return ("Install one of scrot, maim, import (imagemagick) or "
                    "gnome-screenshot.")
        return "Install xdotool."

    def _spell(self, chord: str) -> str:
        mods, key = parse_chord(chord)
        parts = [self.MODS[m] for m in mods]
        parts.append(self.KEYS.get(key, key))
        return "+".join(parts)

    def type_text(self, text: str) -> None:
        # --clearmodifiers so a modifier the user is physically holding --
        # or one left down by a previous chord -- does not turn every
        # letter into a shortcut. Typing "s" into a held Ctrl is a save
        # dialog, or worse, in whatever window is in front.
        self.run(["xdotool", "type", "--clearmodifiers", "--delay", "12",
                  "--", text])

    def press(self, chord: str) -> None:
        self.run(["xdotool", "key", "--clearmodifiers", self._spell(chord)])

    def move(self, x: int, y: int) -> None:
        self.run(["xdotool", "mousemove", str(int(x)), str(int(y))])

    def click(self, button: str = "left", count: int = 1) -> None:
        code = self.BUTTONS.get(button)
        if code is None:
            raise DesktopError(
                f"{button!r} is not a mouse button. Use left, middle or right.")
        self.run(["xdotool", "click", "--repeat", str(max(1, int(count))),
                  "--delay", "80", code])

    def scroll(self, amount: int) -> None:
        # X11 has no scroll axis: wheel up and down are buttons 4 and 5.
        steps = abs(int(amount))
        if not steps:
            return
        self.run(["xdotool", "click", "--repeat", str(steps), "--delay", "40",
                  "4" if amount > 0 else "5"])

    def pointer(self) -> tuple[int, int]:
        out = self.run(["xdotool", "getmouselocation", "--shell"])
        found = dict(line.split("=", 1) for line in out.splitlines()
                     if "=" in line)
        return int(found.get("X", 0)), int(found.get("Y", 0))

    def screen(self) -> tuple[int, int]:
        out = self.run(["xdotool", "getdisplaygeometry"])
        width, _, height = out.strip().partition(" ")
        return int(width or 0), int(height or 0)

    def windows(self) -> list[Window]:
        out = self.run(["xdotool", "search", "--onlyvisible", "--name", "."])
        found: list[Window] = []
        for wid in out.split():
            window = self._window(wid)
            if window is not None and window.title:
                found.append(window)
        return found

    def _window(self, wid: str) -> Window | None:
        try:
            title = self.run(["xdotool", "getwindowname", wid]).strip()
        except DesktopError:
            return None      # it closed between the search and the query
        geometry = {}
        try:
            for line in self.run(["xdotool", "getwindowgeometry", "--shell",
                                  wid]).splitlines():
                if "=" in line:
                    key, _, value = line.partition("=")
                    geometry[key.strip()] = value.strip()
        except DesktopError:
            pass

        def number(key: str) -> int:
            try:
                return int(geometry.get(key, 0))
            except ValueError:
                return 0

        return Window(id=wid, title=title, x=number("X"), y=number("Y"),
                      width=number("WIDTH"), height=number("HEIGHT"))

    def focused(self) -> Window | None:
        wid = self.run(["xdotool", "getactivewindow"]).strip()
        return self._window(wid) if wid else None

    def focus(self, window: Window) -> None:
        self.run(["xdotool", "windowactivate", "--sync", window.id])

    def screenshot(self, path: str) -> None:
        if not self.grabber:
            raise self._no("take a screenshot")
        self.run(_grab_argv(self.grabber, path), timeout=30.0)


class Wayland(Backend):
    """Wayland, which deliberately does not let one application drive
    another.

    That is the security model working as designed, and it is why this
    backend is assembled from separate pieces rather than one xdotool:
    ydotool goes through the kernel's uinput device (so it needs its daemon
    running and the user in the right group), wtype speaks the virtual
    keyboard protocol, and grim takes pictures on wlroots compositors but
    not on GNOME or KDE, which each have their own.

    Every piece is optional and each is reported by name, because "wayland
    is not supported" is not a thing anybody can act on, and "ydotool is
    installed but its daemon is not running" is.
    """

    name = "wayland"

    KEYS = {
        "enter": "Return", "esc": "Escape", "tab": "Tab",
        "backspace": "BackSpace", "delete": "Delete", "space": "space",
        "up": "Up", "down": "Down", "left": "Left", "right": "Right",
        "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next",
        "insert": "Insert", "menu": "Menu", "printscreen": "Print",
        **{f"f{n}": f"F{n}" for n in range(1, 25)},
    }
    MODS = {"ctrl": "ctrl", "alt": "alt", "shift": "shift",
            "super": "logo", "cmd": "logo"}

    def __init__(self, run=_run, typer: str = "", grabber: str = ""):
        super().__init__(run)
        self.typer = typer          # "ydotool" or "wtype"
        self.grabber = grabber
        self.display = os.environ.get("WAYLAND_DISPLAY", "")

    def actions(self) -> set[str]:
        doable: set[str] = set()
        if self.typer:
            doable |= {"type", "press"}
        if self.typer == "ydotool":
            # Only ydotool drives the pointer: wtype is a keyboard.
            doable |= {"move", "click", "scroll"}
        if self.grabber:
            doable.add("screenshot")
        return doable

    def missing(self, action: str) -> str:
        if action == "take a screenshot":
            return ("Install grim (wlroots), or use your desktop's own tool "
                    "-- spectacle on KDE, gnome-screenshot on GNOME.")
        if action in ("move the pointer", "click", "scroll"):
            return ("Install ydotool and start ydotoold; wtype is a keyboard "
                    "only and cannot move the pointer.")
        if action in ("list windows", "say which window has focus",
                      "change which window has focus"):
            return ("Wayland does not let one application enumerate or focus "
                    "another's windows. Ask the user to bring the window "
                    "forward, or launch the application instead.")
        return ("Install ydotool (and start ydotoold) or wtype to send input "
                "on Wayland.")

    def _spell(self, chord: str) -> str:
        mods, key = parse_chord(chord)
        return "+".join([*(self.MODS[m] for m in mods),
                         self.KEYS.get(key, key)])

    def type_text(self, text: str) -> None:
        if self.typer == "wtype":
            self.run(["wtype", "--", text])
        elif self.typer == "ydotool":
            self.run(["ydotool", "type", "--key-delay", "12", "--", text])
        else:
            raise self._no("type text")

    def press(self, chord: str) -> None:
        spelled = self._spell(chord)
        if self.typer == "wtype":
            mods, key = parse_chord(chord)
            argv = ["wtype"]
            for mod in mods:
                argv += ["-M", self.MODS[mod]]
            argv += ["-k", self.KEYS.get(key, key)]
            for mod in mods:
                argv += ["-m", self.MODS[mod]]
            self.run(argv)
        elif self.typer == "ydotool":
            self.run(["ydotool", "key", spelled])
        else:
            raise self._no("press keys")

    def move(self, x: int, y: int) -> None:
        if self.typer != "ydotool":
            raise self._no("move the pointer")
        self.run(["ydotool", "mousemove", "--absolute", "-x", str(int(x)),
                  "-y", str(int(y))])

    def click(self, button: str = "left", count: int = 1) -> None:
        if self.typer != "ydotool":
            raise self._no("click")
        code = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}.get(button)
        if code is None:
            raise DesktopError(
                f"{button!r} is not a mouse button. Use left, middle or right.")
        for _ in range(max(1, int(count))):
            self.run(["ydotool", "click", code])

    def screenshot(self, path: str) -> None:
        if not self.grabber:
            raise self._no("take a screenshot")
        self.run(_grab_argv(self.grabber, path), timeout=30.0)


class MacOS(Backend):
    """macOS, through cliclick for input and screencapture for pictures.

    ``screencapture`` ships with the system; cliclick does not, and there is
    no built-in substitute -- osascript can send keystrokes but not move the
    pointer, so the two are reported separately rather than as one
    "unsupported".

    Every one of these needs Accessibility permission for the terminal
    running wynxo, granted once in System Settings. Without it the
    commands succeed and nothing happens, which is the worst failure shape
    there is, so it is named in the refusal rather than left to be
    discovered.
    """

    name = "macos"

    KEYS = {"enter": "return", "esc": "esc", "tab": "tab",
            "backspace": "delete", "delete": "fwd-delete", "space": "space",
            "up": "arrow-up", "down": "arrow-down", "left": "arrow-left",
            "right": "arrow-right", "home": "home", "end": "end",
            "pageup": "page-up", "pagedown": "page-down",
            **{f"f{n}": f"f{n}" for n in range(1, 17)}}
    MODS = {"ctrl": "ctrl", "alt": "alt", "shift": "shift",
            "cmd": "cmd", "super": "cmd"}
    BUTTONS = {"left": "c", "right": "rc", "middle": "tc"}

    def __init__(self, run=_run, cliclick: bool = False):
        super().__init__(run)
        self.cliclick = cliclick

    def actions(self) -> set[str]:
        doable = {"screenshot", "windows", "focused"}
        if self.cliclick:
            doable |= {"type", "press", "move", "click", "pointer"}
        return doable

    def missing(self, action: str) -> str:
        if action in ("list windows", "say which window has focus"):
            return ""
        return ("Install cliclick (`brew install cliclick`), then grant the "
                "terminal Accessibility permission in System Settings > "
                "Privacy & Security > Accessibility -- without it these "
                "commands report success and nothing moves.")

    def type_text(self, text: str) -> None:
        if not self.cliclick:
            raise self._no("type text")
        self.run(["cliclick", "-w", "12", f"t:{text}"])

    def press(self, chord: str) -> None:
        if not self.cliclick:
            raise self._no("press keys")
        mods, key = parse_chord(chord)
        held = ",".join(self.MODS[m] for m in mods)
        spelled = self.KEYS.get(key, key)
        argv = ["cliclick"]
        if held:
            argv.append(f"kd:{held}")
        argv.append(f"kp:{spelled}" if key in self.KEYS else f"t:{spelled}")
        if held:
            argv.append(f"ku:{held}")
        self.run(argv)

    def move(self, x: int, y: int) -> None:
        if not self.cliclick:
            raise self._no("move the pointer")
        self.run(["cliclick", f"m:{int(x)},{int(y)}"])

    def click(self, button: str = "left", count: int = 1) -> None:
        if not self.cliclick:
            raise self._no("click")
        verb = self.BUTTONS.get(button)
        if verb is None:
            raise DesktopError(
                f"{button!r} is not a mouse button. Use left, middle or right.")
        where = self.pointer()
        for _ in range(max(1, int(count))):
            self.run(["cliclick", f"{verb}:{where[0]},{where[1]}"])

    def pointer(self) -> tuple[int, int]:
        if not self.cliclick:
            raise self._no("report the pointer position")
        out = self.run(["cliclick", "p"]).strip()
        _, _, coords = out.partition(":")
        x, _, y = (coords or out).partition(",")
        try:
            return int(x.strip()), int(y.strip())
        except ValueError:
            return 0, 0

    def windows(self) -> list[Window]:
        script = ('tell application "System Events" to get the name of every '
                  'process whose background only is false')
        out = self.run(["osascript", "-e", script])
        return [Window(id=n.strip(), title=n.strip(), app=n.strip())
                for n in out.split(",") if n.strip()]

    def focused(self) -> Window | None:
        script = ('tell application "System Events" to get the name of the '
                  'first process whose frontmost is true')
        name = self.run(["osascript", "-e", script]).strip()
        return Window(id=name, title=name, app=name) if name else None

    def focus(self, window: Window) -> None:
        self.run(["osascript", "-e",
                  f'tell application "{window.app or window.title}" to activate'])

    def screenshot(self, path: str) -> None:
        # -x so the shutter does not make a noise every time the agent looks.
        self.run(["screencapture", "-x", path], timeout=30.0)


class Windows(Backend):
    """Windows, through PowerShell rather than a helper to install.

    SendKeys and the Win32 cursor calls are both reachable from PowerShell,
    which is on every Windows since 7 -- so this needs nothing installed,
    which matters more here than the extra hundred milliseconds a
    PowerShell start costs. SendKeys has real limits (it cannot hold a
    modifier across a pointer action, and it has no middle button), and
    those are refusals rather than approximations.
    """

    name = "windows"

    KEYS = {"enter": "{ENTER}", "esc": "{ESC}", "tab": "{TAB}",
            "backspace": "{BACKSPACE}", "delete": "{DELETE}", "space": " ",
            "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}",
            "right": "{RIGHT}", "home": "{HOME}", "end": "{END}",
            "pageup": "{PGUP}", "pagedown": "{PGDN}", "insert": "{INSERT}",
            **{f"f{n}": f"{{F{n}}}" for n in range(1, 17)}}
    MODS = {"ctrl": "^", "alt": "%", "shift": "+", "super": "", "cmd": ""}

    def actions(self) -> set[str]:
        return {"type", "press", "move", "click", "screen", "pointer",
                "screenshot", "windows", "focused", "focus"}

    def missing(self, action: str) -> str:
        return "PowerShell is required and was not found."

    def _powershell(self, script: str) -> str:
        return self.run(["powershell", "-NoProfile", "-NonInteractive",
                         "-Command", script])

    @staticmethod
    def _sendkeys_escape(text: str) -> str:
        """SendKeys reads these as syntax, so a literal one must be braced.

        Without this, typing an email address sends a modifier: "+" is
        shift and "^" is ctrl, so "a+b" arrives as "aB"."""
        out = []
        for char in text:
            out.append("{" + char + "}" if char in "+^%~(){}[]" else char)
        return "".join(out)

    def type_text(self, text: str) -> None:
        body = self._sendkeys_escape(text).replace("'", "''")
        self._powershell(
            "Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.SendKeys]::SendWait('{body}')")

    def press(self, chord: str) -> None:
        mods, key = parse_chord(chord)
        if any(m in ("super", "cmd") for m in mods):
            raise DesktopError(
                "SendKeys cannot send the Windows key. Use the application's "
                "own shortcut, or ask the user to press it.")
        prefix = "".join(self.MODS[m] for m in mods)
        spelled = self.KEYS.get(key, self._sendkeys_escape(key))
        body = (prefix + spelled).replace("'", "''")
        self._powershell(
            "Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.SendKeys]::SendWait('{body}')")

    def move(self, x: int, y: int) -> None:
        self._powershell(
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.Cursor]::Position = "
            f"New-Object System.Drawing.Point({int(x)},{int(y)})")

    def click(self, button: str = "left", count: int = 1) -> None:
        down, up = {"left": (0x02, 0x04), "right": (0x08, 0x10)}.get(
            button, (0, 0))
        if not down:
            raise DesktopError(
                f"{button!r} cannot be clicked here. Use left or right.")
        self._powershell(
            'Add-Type -MemberDefinition \'[DllImport("user32.dll")]'
            'public static extern void mouse_event(int f,int x,int y,'
            'int d,int e);\' -Name U -Namespace W; '
            + " ".join(f"[W.U]::mouse_event({down},0,0,0,0); "
                       f"[W.U]::mouse_event({up},0,0,0,0);"
                       for _ in range(max(1, int(count)))))

    def pointer(self) -> tuple[int, int]:
        out = self._powershell(
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$p=[System.Windows.Forms.Cursor]::Position; \"$($p.X) $($p.Y)\"")
        x, _, y = out.strip().partition(" ")
        try:
            return int(x), int(y)
        except ValueError:
            return 0, 0

    def screen(self) -> tuple[int, int]:
        out = self._powershell(
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$s=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "\"$($s.Width) $($s.Height)\"")
        w, _, h = out.strip().partition(" ")
        try:
            return int(w), int(h)
        except ValueError:
            return 0, 0

    def windows(self) -> list[Window]:
        out = self._powershell(
            "Get-Process | Where-Object {$_.MainWindowTitle} | "
            "ForEach-Object { \"$($_.Id)`t$($_.ProcessName)`t"
            "$($_.MainWindowTitle)\" }")
        found = []
        for line in out.splitlines():
            bits = line.split("\t")
            if len(bits) >= 3:
                found.append(Window(id=bits[0].strip(), app=bits[1].strip(),
                                    title=bits[2].strip()))
        return found

    def focused(self) -> Window | None:
        out = self._powershell(
            'Add-Type -MemberDefinition \'[DllImport("user32.dll")]'
            'public static extern IntPtr GetForegroundWindow();\' '
            '-Name F -Namespace W; '
            "$h=[W.F]::GetForegroundWindow(); "
            "Get-Process | Where-Object {$_.MainWindowHandle -eq $h} | "
            "ForEach-Object { \"$($_.Id)`t$($_.ProcessName)`t"
            "$($_.MainWindowTitle)\" }")
        for line in out.splitlines():
            bits = line.split("\t")
            if len(bits) >= 3:
                return Window(id=bits[0].strip(), app=bits[1].strip(),
                              title=bits[2].strip())
        return None

    def focus(self, window: Window) -> None:
        self._powershell(
            "Add-Type -AssemblyName Microsoft.VisualBasic; "
            f"[Microsoft.VisualBasic.Interaction]::AppActivate({window.id})")

    def screenshot(self, path: str) -> None:
        target = str(path).replace("'", "''")
        self._powershell(
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
            "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$i=New-Object System.Drawing.Bitmap($b.Width,$b.Height); "
            "$g=[System.Drawing.Graphics]::FromImage($i); "
            "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,"
            "$b.Size); "
            f"$i.Save('{target}');", )


# -- screenshots -------------------------------------------------------------

GRABBERS = ("grim", "spectacle", "gnome-screenshot", "scrot", "maim", "import")
"""Screen grabbers, most specific first. grim before the desktop ones so a
wlroots session does not start a KDE dialog; ``import`` last because
imagemagick's is the slowest and the most likely to be there for unrelated
reasons."""


def _grab_argv(grabber: str, path: str) -> list[str]:
    """How to make ``grabber`` write a full-screen PNG to ``path``.

    Each one spells "no dialog, no sound, this file" differently, and
    getting it wrong does not fail -- it opens an interactive region
    selector and waits, which from a background agent is a hang.
    """
    return {
        "grim": ["grim", path],
        "scrot": ["scrot", "--overwrite", path],
        "maim": ["maim", "--format=png", path],
        "import": ["import", "-window", "root", path],
        "gnome-screenshot": ["gnome-screenshot", "-f", path],
        "spectacle": ["spectacle", "-b", "-n", "-f", "-o", path],
    }[grabber]


def _first_present(names) -> str:
    return next((n for n in names if shutil.which(n)), "")


# -- detection ---------------------------------------------------------------

def detect(run=_run) -> Backend:
    """The best backend this machine can offer, or an Unavailable saying why.

    Session type first, binaries second. A machine with both xdotool and
    ydotool installed running a Wayland session must not get X11: xdotool
    talks to XWayland, where it can drive X applications and silently does
    nothing to native ones -- which is exactly the half-working case this
    module refuses to be.
    """
    if sys.platform == "win32":
        return Windows(run)
    if sys.platform == "darwin":
        return MacOS(run, cliclick=bool(shutil.which("cliclick")))

    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland = os.environ.get("WAYLAND_DISPLAY", "")
    display = os.environ.get("DISPLAY", "")
    grabber = _first_present(GRABBERS)

    if session == "wayland" or (wayland and not display):
        typer = _first_present(("ydotool", "wtype"))
        if not typer and not grabber:
            return Unavailable(
                "This is a Wayland session and no input helper is installed. "
                "Wayland deliberately stops one application driving another, "
                "so it needs ydotool (with ydotoold running) or wtype. "
                "Install one, or log into an X11 session where xdotool "
                "works on its own.", run)
        return Wayland(run, typer=typer, grabber=grabber)

    if display:
        if shutil.which("xdotool"):
            return X11(run, grabber=grabber)
        return Unavailable(
            "xdotool is not installed, and on X11 it is what moves the "
            "pointer and sends keys. Install it with your package manager "
            "(apt install xdotool, pacman -S xdotool, dnf install xdotool).",
            run)

    return Unavailable(
        "There is no graphical session here: neither DISPLAY nor "
        "WAYLAND_DISPLAY is set. wynxo is running on a terminal-only "
        "machine -- over SSH, in a container, or on a server -- so there is "
        "no desktop to drive. The shell tool still runs commands.", run)
