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

    Every one of these was mixed to sit at or above 4.5:1 against a black
    terminal -- the threshold below which body text stops being reliably
    legible. They still stay below muted, so the hierarchy remains quiet.
    """

    good: str
    warn: str
    bad: str

    bar_bg: str          # the pinned status strip
    bar_text: str
    bar_dim: str
    bar_accent: str

    code: str            # `inline code` in the model's prose
    keyword: str         # a language's structure words
    literal: str         # strings and numbers
    symbol: str          # the names a program defines
    """Syntax colour, as four roles rather than as a pygments theme.

    Code was the one thing on screen /theme could not reach: inline spans
    were a hardcoded amber and highlighted blocks used raw ANSI names --
    magenta keywords, cyan numbers -- so an answer containing code was
    coloured by a scheme unrelated to the rest of the interface.

    Four roles are enough for streaming code: language structure, literals,
    program-defined names, and inline code. The default palette deliberately
    keeps those roles close together so code is readable without becoming a
    rainbow inside an otherwise calm terminal.
    """

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if k != "name"}

    def role(self, name: str) -> str:
        """A colour by the job it does, falling back to the body text."""
        return getattr(self, name, self.text)


# Default. Purple remains Wynxo's identity, but the old default spent a bright
# violet on almost every focal element and paired it with very pale body text.
# This version is graphite first and violet second: neutral body colours,
# restrained syntax roles, and one violet accent. Status colours retain their
# semantic meaning without turning ordinary turns into a light show.
PURPLE = Palette(
    name="purple",
    accent="#b47cff",
    accent_dim="#75689a",
    text="#d6d3dc",
    muted="#918b9b",
    faint="#787281",
    good="#82b58a",
    warn="#c9a96e",
    bad="#cf7d86",
    bar_bg="#18161c",
    bar_text="#d6d3dc",
    bar_dim="#8d8797",
    bar_accent="#a894d2",
    code="#c8c3cf",
    keyword="#ad9bcf",
    literal="#9eae9b",
    symbol="#9faeb9",
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
    code="#e6c07b",
    keyword="#79c7ff",
    literal="#8fd9a8",
    symbol="#a9b7f5",
)

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
    code="#ffd08a",
    keyword="#ff9fe0",
    literal="#9df0b8",
    symbol="#a9c8ff",
)

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
    code="#ffd8a0",
    keyword="#ffaad6",
    literal="#a8f0c0",
    symbol="#a9d4ff",
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
    code="#f0c674",
    keyword="#ffb072",
    literal="#a8d98a",
    symbol="#8fc6d9",
)

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
    code="yellow",
    keyword="bright_magenta",
    literal="green",
    symbol="bright_cyan",
)

CATBOY = Palette(
    name="catboy",
    accent="#c77dff",
    accent_dim="#8e5bc4",
    text="#f2e9ff",
    muted="#b9a3d4",
    faint="#7e6f96",
    good="#8ff0c4",
    warn="#ffcf7a",
    bad="#ff7a9c",
    bar_bg="#1b1226",
    bar_text="#f2e9ff",
    bar_dim="#b9a3d4",
    bar_accent="#ff9fd6",
    code="#ffcf7a",
    keyword="#d49aff",
    literal="#8ff0c4",
    symbol="#9ad0ff",
)

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
    code="bright_white",
    keyword="bright_white",
    literal="white",
    symbol="white",
)

PALETTES: dict[str, Palette] = {
    p.name: p
    for p in (PURPLE, SAKURA, KAWAII, MIDNIGHT, EMBER, CATBOY, PLAIN, MINIMAL)
}
DEFAULT = "purple"

_active: Palette | None = None


def use(palette: Palette) -> None:
    """Remember which palette is in force."""
    global _active
    _active = palette


def active() -> Palette:
    return _active if _active is not None else PALETTES[DEFAULT]


def resolve(name: str) -> Palette:
    """Look up a palette, falling back to the default rather than failing."""
    return PALETTES.get((name or "").strip().lower(), PALETTES[DEFAULT])


def names() -> list[str]:
    return list(PALETTES)
