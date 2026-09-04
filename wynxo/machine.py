"""What this machine is, worked out once, before anything is attempted.

An assistant that tries things to find out what it can do is an assistant
that fails in front of you. Asked to close a window on a server it should
say there is no desktop -- not send a keystroke into nothing and report
what happened. Asked to commit, it should know whether `gh` is there before
it plans around it.

So the answer is settled at startup, once, and three things read it:

  the model     a few lines in the system prompt, so it does not offer what
                this machine cannot do
  the runtime   checked before a desktop action, so a request that cannot
                work is refused with the fix rather than attempted
  the user      /desktop and the startup line

It goes in the system prompt rather than in front of each turn -- unlike
the window list, none of this changes while wynxo is running, and the
system prompt is the part the server keeps between turns. Facts that hold
all session belong in the half that is only paid for once.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field


@dataclass
class Machine:
    """The shape of this computer, as far as wynxo is concerned."""

    os: str = ""
    session: str = ""
    """x11, wayland, windows, macos, or "headless"."""
    desktop_env: str = ""
    """KDE, GNOME, and so on -- what the user would call it."""
    backend: str = "unavailable"
    can: frozenset = field(default_factory=frozenset)
    """What can actually be driven. Empty means nothing can."""
    blocked: str = ""
    """Why nothing can be driven, and what to install."""
    tools: dict = field(default_factory=dict)
    """Command-line things worth knowing about, present or not."""
    controls: frozenset = field(default_factory=frozenset)
    """The parts of the machine beyond its windows that can be operated --
    sound, media, clipboard, notifications, power."""
    sighted: bool = False
    """Whether the model in use accepts pictures. Not a fact about the
    computer, but it belongs beside them: what wynxo can do here is the
    intersection of what the desktop allows and what the model can take."""

    @property
    def has_desktop(self) -> bool:
        return bool(self.can)

    def why_not(self, action: str) -> str:
        """Why ``action`` cannot happen here, or "" if it can.

        The whole point of probing first: this answers before anything is
        attempted, so a request that cannot work is refused with the fix
        instead of being tried and reported on.
        """
        if action in self.can:
            return ""
        if not self.can:
            return self.blocked or "There is no desktop to drive here."
        from .desktop import detect

        return detect().missing(_PHRASE.get(action, action))

    def summary(self) -> str:
        """One line, for the startup header."""
        if not self.has_desktop:
            return f"{self.os} · no desktop"
        env = f" {self.desktop_env}" if self.desktop_env else ""
        return f"{self.os} ·{env} {self.session} · {len(self.can)} desktop actions"

    def prompt_block(self) -> str:
        """What the model is told, once, in the system prompt."""
        lines = [f"This machine: {self.os}."]
        if self.has_desktop:
            env = f" ({self.desktop_env})" if self.desktop_env else ""
            doable = sorted(_ALL & set(self.can))
            lines.append(
                f"It has a {self.session} desktop{env}, and you can drive it: "
                + ", ".join(doable) + ". Use control_computer for those and "
                "`look` to see what is open.")
            if self.sighted and "screenshot" in self.can:
                lines.append(
                    "You can see: call `look` and the screen is put in front "
                    "of you. Do that before clicking anywhere -- a "
                    "coordinate you have not looked at is a guess.")
            elif "screenshot" in self.can:
                lines.append(
                    "You cannot see images, so `look` saves a screenshot for "
                    "the user rather than for you. Work from the window list "
                    "and keyboard shortcuts; do not guess at coordinates.")
            missing = sorted(_ALL - set(self.can))
            if missing:
                lines.append(
                    "You cannot " + ", ".join(missing) + " here. Do not offer "
                    "to -- say what is missing instead.")
        else:
            lines.append(
                "There is no desktop: nothing can be clicked, typed into or "
                "focused. Do not offer to. Commands still run in the shell.")
        if self.controls:
            lines.append(
                "You can also operate: " + ", ".join(sorted(self.controls))
                + ". Use system_control for those and system_status to read "
                "how the machine is doing -- memory, disk, battery, what is "
                "slowing it down. Both beat composing shell commands: the "
                "right command differs by machine and they pick it.")
        absent = [name for name, there in self.tools.items() if not there]
        if absent:
            lines.append("Not installed: " + ", ".join(absent)
                         + ". Do not plan around them.")
        return "\n".join(lines)


_ALL = {"type", "press", "move", "click", "scroll", "windows", "focus",
        "screenshot"}
"""The things worth telling the model about.

Deliberately not everything a backend reports. "pointer", "focused" and
"screen" are questions it can answer, not things it can do, and listing
them under "you can drive it" reads as capability the model then offers."""

_PHRASE = {
    "type": "type text", "press": "press keys", "move": "move the pointer",
    "click": "click", "scroll": "scroll", "windows": "list windows",
    "focus": "change which window has focus", "screenshot": "take a screenshot",
}

_WATCHED = ("git", "gh", "docker", "tesseract", "rg", "fd")
"""Worth telling the model about, because it plans differently without
them. Not a survey of the machine -- every name here costs prompt on every
turn, so it is the short list of things wynxo's own tools reach for."""


def probe(backend=None, model_info=None) -> Machine:
    """Work out what this machine is. Cheap, and done once."""
    from .desktop import detect
    from .system import Audio, Clipboard, Media, Notifier, Power
    from .vision import can_see

    backend = backend if backend is not None else detect()
    controls = {name for name, there in (
        ("sound", Audio().available), ("media playback", Media().available),
        ("the clipboard", Clipboard().available),
        ("notifications", Notifier().available),
        ("power", bool(Power().actions())),
    ) if there}
    session = backend.name if backend.name != "unavailable" else "headless"
    return Machine(
        os=_describe_os(),
        session=session,
        desktop_env=_desktop_env(),
        backend=backend.name,
        can=frozenset(backend.actions()),
        blocked=getattr(backend, "reason", ""),
        tools={name: bool(shutil.which(name)) for name in _WATCHED},
        controls=frozenset(controls),
        sighted=can_see(model_info),
    )


def _describe_os() -> str:
    if sys.platform == "win32":
        return f"Windows {platform.release()}"
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0] or platform.release()}"
    return f"{_distro()} ({platform.system()} {platform.release()})"


def _distro() -> str:
    """The name a Linux user would use for their own system.

    platform.system() says "Linux" on every one of them, which is true and
    useless: the thing that decides how a package is installed is whether
    this is Arch, Debian or NixOS.
    """
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            fields = dict(
                line.rstrip("\n").split("=", 1) for line in handle
                if "=" in line)
    except OSError:
        return "Linux"
    name = (fields.get("PRETTY_NAME") or fields.get("NAME") or "Linux")
    return name.strip().strip('"') or "Linux"


def _desktop_env() -> str:
    for key in ("XDG_CURRENT_DESKTOP", "DESKTOP_SESSION", "XDG_SESSION_DESKTOP"):
        if value := os.environ.get(key, "").strip():
            return value.split(":")[0]
    return ""
