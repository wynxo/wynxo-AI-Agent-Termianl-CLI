"""Colour, in one place.

Every colour in the interface comes from here, so the whole thing can change
character by swapping a palette rather than by hunting for style strings.

Colours are hex, which rich degrades to 256- or 16-colour automatically on
terminals that cannot do truecolour -- including the older Windows console.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str

    accent: str          # headings, the cursor, the box
    accent_dim: str      # the same hue, quieter
    text: str
    muted: str           # secondary detail
    faint: str           # things you should be able to ignore
    """Ignorable, not unreadable.

    Every one of these was mixed to sit just under 4.5:1 against a black
    terminal -- the threshold below which body text stops being reliably
    legible -- because "quiet" and "invisible" are one nudge apart and the
    first draft landed on the wrong side of it. They stay well under muted,
    so the hierarchy is unchanged; they are simply readable when you go
    looking."""

    good: str
    warn: str
    bad: str

    bar_bg: str          # the pinned status strip
    bar_text: str
    bar_dim: str
    bar_accent: str

    code_theme: str      # pygments theme for fenced blocks

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if k != "name"}

    def role(self, name: str) -> str:
        """A colour by the job it does, falling back to the body text.

        The mascot asks for "resting" or "busy" rather than for cyan, so a
        theme can be a theme rather than a recolour of everything except the
        cat. Before this, MOOD_STYLES named literal colours -- grey62,
        bright_cyan, bright_magenta -- and the mascot was the one thing on
        screen that /theme could not touch.
        """
        return getattr(self, name, self.text)


# Default. A deep violet that stays legible on a black terminal and does not
# collide with the green/yellow/red the status lines need to keep meaning.
PURPLE = Palette(
    name="purple",
    accent="#b47cff",
    accent_dim="#7c5cbf",
    text="#e6e0f0",
    muted="#9a8fb0",
    faint="#7b7192",
    good="#7ee081",
    warn="#f0c674",
    bad="#ff6b7a",
    bar_bg="#2a1f3d",
    bar_text="#e6e0f0",
    bar_dim="#a99cc4",
    bar_accent="#c9a6ff",
    code_theme="material",
)

MIDNIGHT = Palette(
    name="midnight",
    accent="#6ec7ff",
    accent_dim="#4a86b8",
    text="#dfe7ef",
    muted="#8fa3b8",
    faint="#65788b",
    good="#7ee081",
    warn="#f0c674",
    bad="#ff6b7a",
    bar_bg="#182430",
    bar_text="#dfe7ef",
    bar_dim="#9fb3c6",
    bar_accent="#8fd4ff",
    code_theme="monokai",
)

# Pink and violet, turned up. Same legibility rules as PURPLE -- the accent
# still has to survive on a black background and must not drift into the
# red the `bad` status uses, which is why the pinks stay on the magenta side.
SAKURA = Palette(
    name="sakura",
    accent="#ff8ad8",
    accent_dim="#c264a8",
    text="#fbe9f6",
    muted="#c9a2c8",
    faint="#8f6f92",
    good="#8ef0a6",
    warn="#ffd479",
    bad="#ff5d7a",
    bar_bg="#3b1d3d",
    bar_text="#fbe9f6",
    bar_dim="#d0a8d4",
    bar_accent="#ffb3e6",
    code_theme="material",
)

# Soft candy pink with lavender highlights. Unlike sakura's saturated neon,
# this keeps the low-contrast, cosy look of a handheld game UI while retaining
# enough separation for errors and tool output to stay legible.
KAWAII = Palette(
    name="kawaii",
    accent="#ff9fce",
    accent_dim="#c77aa9",
    text="#fff2fa",
    muted="#d7b4cf",
    faint="#936f8c",
    good="#a8f0c0",
    warn="#ffe18a",
    bad="#ff7898",
    bar_bg="#42233e",
    bar_text="#fff2fa",
    bar_dim="#e3bddd",
    bar_accent="#ffc0e3",
    code_theme="material",
)

EMBER = Palette(
    name="ember",
    accent="#ff9d5c",
    accent_dim="#c26a33",
    text="#f0e6de",
    muted="#b39d8c",
    faint="#867263",
    good="#9ad17a",
    warn="#f0c674",
    bad="#ff6b6b",
    bar_bg="#2f2018",
    bar_text="#f0e6de",
    bar_dim="#c4ab98",
    bar_accent="#ffb87a",
    code_theme="gruvbox-dark",
)

# A 16-colour fallback for terminals that cannot do more, and for anyone who
# wants their own terminal palette respected rather than overridden.
PLAIN = Palette(
    name="plain",
    accent="bright_magenta",
    accent_dim="magenta",
    text="default",
    muted="bright_black",
    faint="bright_black",
    good="green",
    warn="yellow",
    bad="red",
    bar_bg="black",
    bar_text="default",
    bar_dim="bright_black",
    bar_accent="bright_magenta",
    code_theme="ansi_dark",
)

# Catboy heaven: the premium personality theme. Violet and pink over a
# near-black ground, with a pastel cyan holding the cooler end so the whole
# thing is not one hue.
#
# Rebuilt rather than tuned. The first version made `bad` #ff69b4 against an
# accent of #ff6ec7 -- two hot pinks four steps apart, so an error was the
# same colour as a heading and stopped reading as an error at all. The body
# text was lemon chiffon, a yellow, which belongs to no part of this palette
# and made ordinary prose look like a warning. Errors are the one thing a
# theme may never make pretty at the cost of legibility.
CATBOY = Palette(
    name="catboy",
    accent="#c77dff",        # violet, the identity colour
    accent_dim="#8e5bc4",
    text="#f2e9ff",          # near-white with a violet cast
    muted="#b9a3d4",
    faint="#7e6f96",
    good="#8ff0c4",          # soft mint
    warn="#ffcf7a",          # warm amber
    bad="#ff7a9c",           # soft red, clearly not the accent
    bar_bg="#1b1226",
    bar_text="#f2e9ff",
    bar_dim="#b9a3d4",
    bar_accent="#ff9fd6",    # pink, against the violet accent
    code_theme="material",
)

# Reduced-motion theme: the same plain grey palette with no animation
# ambitions. `/theme minimal` also switches the animation engine to its
# static mode, so this is the palette for the accessibility option.
MINIMAL = Palette(
    name="minimal",
    accent="bright_white",
    accent_dim="bright_black",
    text="default",
    muted="bright_black",
    faint="bright_black",
    good="bright_green",
    warn="yellow",
    bad="bright_red",
    bar_bg="black",
    bar_text="default",
    bar_dim="bright_black",
    bar_accent="bright_white",
    code_theme="ansi_dark",
)

PALETTES: dict[str, Palette] = {
    p.name: p for p in (PURPLE, SAKURA, KAWAII, MIDNIGHT, EMBER, CATBOY, PLAIN, MINIMAL)
}
DEFAULT = "purple"


_active: Palette | None = None


def use(palette: Palette) -> None:
    """Remember which palette is in force.

    Colours are pushed into the modules that imported them, which works for
    module-level constants and not for anything that has to choose a colour
    per draw -- the mascot's, which depends on its mood. One place to ask
    keeps that from becoming a second palette.
    """
    global _active
    _active = palette


def active() -> Palette:
    return _active if _active is not None else PALETTES[DEFAULT]


def resolve(name: str) -> Palette:
    """Look up a palette, falling back to the default rather than failing.

    A bad theme name in a config file should not stop the agent starting.
    """
    return PALETTES.get((name or "").strip().lower(), PALETTES[DEFAULT])


def names() -> list[str]:
    return list(PALETTES)
