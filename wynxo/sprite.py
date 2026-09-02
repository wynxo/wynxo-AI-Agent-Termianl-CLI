"""The Wynxo companion, drawn as a half-block sprite.

Two pixels to a cell. ``▀`` is drawn with the top pixel's colour as the
foreground and the bottom pixel's as the background, so a terminal row
carries two rows of picture and a 14x10 sprite fits in five rows of five-
teen columns. That is the whole trick, and it is what makes the difference
between a character and a face made of punctuation: at 14x10 there is room
for a silhouette -- ears, a head that tapers, shoulders, a laptop in front
-- and a silhouette is what makes something recognisable across states.

What came before was three systems at once. ``pet.py`` drew a line-art cat
in the header and a single mark in the status strip. ``companion.py`` held
a second character, a full set of ASCII scenes with a laptop and a coffee
cup, that no session ever rendered -- its state machine was wired up and
its pictures were not. ``motion.py`` wrapped those same scenes a third time
for a preview command. Three drawings of one animal, of which the one the
user actually saw was the smallest.

This is the only one now. It keeps the part of ``companion.py`` that was
doing real work -- the states, and the mapping from tool and task to state
-- and replaces the pictures.

Transparency matters more than it sounds. A pixel with nothing in it is
left as the terminal's own background rather than painted, so the sprite is
a shape on the conversation instead of a coloured rectangle sitting on it:

    both pixels empty      a space
    top empty              ▄ in the bottom pixel's colour
    bottom empty           ▀ in the top pixel's colour
    both the same colour   █
    two different colours  ▀ over a background

Only the last case sets a background at all, and only for the cells where
two different opaque colours genuinely stack.
"""
from __future__ import annotations

from rich.text import Text

from .companion import State

WIDTH = 14
"""Columns. Also the pixel width -- one pixel per column."""

HEIGHT = 5
"""Terminal rows. Ten pixel rows, two to a row."""

MIN_COLUMNS = 60
"""Below this the sprite is not drawn at all.

The companion is the first thing to go when the terminal gets tight: it is
seventh in the hierarchy and the status beside it is fourth. A narrow
terminal gets the words."""

# -- ink --------------------------------------------------------------------

INK: dict[str, str] = {
    "F": "accent",       # fur
    "f": "accent_dim",   # fur in shadow: paws, the underside of the jaw
    "E": "bar_bg",       # eye -- the palette's darkest, whatever the theme
    "S": "bar_accent",   # what is on the laptop screen
    "s": "accent_dim",   # the screen, dimmer
    "L": "muted",        # the laptop case
    "C": "warn",         # the mug
    "W": "faint",        # steam, and the question mark while thinking
    "G": "good",
    "R": "bad",
    ".": "",             # nothing: leave the terminal's own background
}
"""Pixel character to palette role. Roles rather than colours, so /theme
reaches the companion -- it was the one thing on screen a theme could not
touch for most of this project's life."""


# -- the character ----------------------------------------------------------

_BODY = [
    "...F......F...",   # ear tips
    "..FFF....FFF..",   # ears
    "...FFFFFFFF...",   # the crown, narrower than the head
    "..FFFFFFFFFF..",
    "..FFEEFFEEFF..",   # eyes
    "...FFFFFFFF...",   # jaw
    ".FFFFFFFFFFFF.",   # shoulders
    "..SSSSSSSSSS..",   # the screen, in front of the chest
    "..SSSSSSSSSS..",
    ".LLLLLLLLLLLL.",   # the case
]
"""Every frame starts here and edits rows.

Written as one base rather than as twelve independent pictures because the
silhouette is the character: if the ears move a pixel between idle and
thinking, it stops reading as the same animal and starts reading as an
animation glitch. Only the rows a state has a reason to change are
replaced, so nothing can drift by accident."""

EYES_OPEN = "FFEEFFEEFF"
EYES_SHUT = "FFFFFFFFFF"     # nothing: at two pixels an eye either is or isn't
EYES_GLAD = "FffFFFFffF"     # the corners lift, in the shadow colour
EYES_LEFT = "FEEFFEEFFF"     # both eyes together -- one eye moving is a squint
EYES_RIGHT = "FFFEEFFEEF"
EYES_SHUT_R = "FRRFFFFRRF"   # shut, and the wrong colour

_EYE_NOTE = """Why a blink closes the eyes rather than dimming them.

Two pixels is the whole eye, so "half shut" is not available: the only
changes that survive are present, absent, and moved. Dimming the pixel
looked right in a screenshot and vanished in motion, because it left the
cell's glyph identical and only the colour moved -- so on a terminal
without truecolour, or to anyone reading shape before hue, the character
never blinked at all. Closing them changes ▀█▀▀██▀▀█▀ to ▀████████▀,
which reads at a glance and reads without colour."""


def _frame(**rows: str) -> list[str]:
    """The base with some rows replaced. Keyword is r0..r9."""
    out = list(_BODY)
    for key, value in rows.items():
        index = int(key[1:])
        assert len(value) == WIDTH, f"{key}: {len(value)} columns"
        out[index] = value
    return out


def _eyes(pattern: str, row: int = 4) -> str:
    assert len(pattern) == 10, pattern
    return ".." + pattern + ".."


def _screen(pattern: str) -> str:
    assert len(pattern) == 10, pattern
    return ".." + pattern + ".."


# The screen's contents, as two rows of ten pixels. Bright cells are the
# "text" on it; they move while something is actually being written.
_DARK = "ssssssssss"


def _typing(step: int) -> list[str]:
    """A caret running along a line of text, and a line already written.

    Four frames, and the only thing that moves. The eye is drawn to motion,
    so the motion is put where the meaning is: while the agent is editing a
    file, the thing that moves is the writing on the screen."""
    top = list("SSSSSsssss")
    caret = 5 + step % 5
    row = list(_DARK)
    row[:caret - 5] = "S" * (caret - 5)
    return [_screen("".join(top)), _screen("".join(row))]


FRAMES: dict[State, list[list[str]]] = {
    # Breathing, and a blink every fourth beat. Nothing else moves.
    State.IDLE: [
        _frame(r4=_eyes(EYES_OPEN)),
        _frame(r4=_eyes(EYES_OPEN)),
        _frame(r4=_eyes(EYES_OPEN)),
        _frame(r4=_eyes(EYES_SHUT)),
    ],
    # Eyes up and away from the screen, and a mark floating beside the ear.
    State.THINKING: [
        _frame(r3=_eyes(EYES_OPEN), r4=_eyes("FFFFFFFFFF")),
        _frame(r3=_eyes(EYES_OPEN), r4=_eyes("FFFFFFFFFF")),
        _frame(r3=_eyes(EYES_LEFT), r4=_eyes("FFFFFFFFFF")),
        _frame(r3=_eyes(EYES_OPEN), r4=_eyes("FFFFFFFFFF")),
    ],
    # Looking left, then right. Searching is the one state where the eyes
    # sweep rather than settle.
    State.SEARCHING: [
        _frame(r4=_eyes(EYES_LEFT)),
        _frame(r4=_eyes(EYES_LEFT)),
        _frame(r4=_eyes(EYES_RIGHT)),
        _frame(r4=_eyes(EYES_RIGHT)),
    ],
    # Eyes down on the screen, which is lit but still.
    State.READING: [
        _frame(r4=_eyes("FFEEFFEEFF"), r5="...FFFFFFFF...",
               r7=_screen("SSSSSSssss")),
        _frame(r4=_eyes(EYES_SHUT), r7=_screen("SSSSSSssss")),
        _frame(r4=_eyes("FFEEFFEEFF"), r7=_screen("SSSSSSssss")),
        _frame(r4=_eyes("FFEEFFEEFF"), r7=_screen("SSSSSSssss")),
    ],
    # Paws up at the keyboard and the caret running. The flagship state.
    State.CODING: [
        _frame(r4=_eyes(EYES_OPEN), r6=".fFFFFFFFFFFf.",
               r7=_typing(0)[0], r8=_typing(0)[1]),
        _frame(r4=_eyes(EYES_OPEN), r6="ffFFFFFFFFFFff",
               r7=_typing(1)[0], r8=_typing(1)[1]),
        _frame(r4=_eyes(EYES_OPEN), r6=".fFFFFFFFFFFf.",
               r7=_typing(2)[0], r8=_typing(2)[1]),
        _frame(r4=_eyes(EYES_OPEN), r6="ffFFFFFFFFFFff",
               r7=_typing(3)[0], r8=_typing(3)[1]),
    ],
    # Watching. Hands off the keyboard, a bar filling on the screen.
    State.TESTING: [
        _frame(r4=_eyes(EYES_OPEN), r7=_screen("SSssssssss")),
        _frame(r4=_eyes(EYES_OPEN), r7=_screen("SSSSssssss")),
        _frame(r4=_eyes(EYES_OPEN), r7=_screen("SSSSSSssss")),
        _frame(r4=_eyes(EYES_OPEN), r7=_screen("SSSSSSSSss")),
    ],
    # Something failed and is being worked out: thinking, with the screen
    # still showing the damage.
    State.RECOVERING: [
        _frame(r3=_eyes(EYES_OPEN), r4=_eyes("FFFFFFFFFF"),
               r7=_screen("RRssssssss")),
        _frame(r3=_eyes(EYES_LEFT), r4=_eyes("FFFFFFFFFF"),
               r7=_screen("RRssssssss")),
    ],
    # Done, and it is the mug that says so. The one state with a prop.
    State.SUCCESS: [
        _frame(r4=_eyes(EYES_GLAD), r6=".FFFFFFFFFF.W.",
               r7="..SSSSSSSS.CC.", r8="..SSSSSSSS.CC.",
               r9=".LLLLLLLLLL.CC"),
        _frame(r4=_eyes(EYES_GLAD), r6=".FFFFFFFFFF.W.",
               r7="..SSSSSSSS.CC.", r8="..SSSSSSSS.CC.",
               r9=".LLLLLLLLLL.CC"),
        _frame(r4=_eyes(EYES_SHUT), r6=".FFFFFFFFFF.W.",
               r7="..SSSSSSSS.CC.", r8="..SSSSSSSS.CC.",
               r9=".LLLLLLLLLL.CC"),
    ],
    # The ears go down. Colour alone had error and success drawing the
    # same silhouette -- both are "eyes not plainly open" -- and those two
    # are the pair that must never be confused at a glance, since one of
    # them means stop reading and look. The ears are the biggest shape the
    # character has, so they are what changes.
    State.ERROR: [
        _frame(r0="..............", r1="..FF......FF..",
               r4=_eyes(EYES_SHUT_R), r7=_screen("RRRRssssss")),
    ],
    State.CANCELLED: [
        _frame(r0="..............", r1="..FF......FF..",
               r4=_eyes(EYES_SHUT), r7=_screen(_DARK), r8=_screen(_DARK)),
    ],
    # One ear up. Listening is the only thing an ear can say that a face
    # this size cannot.
    State.LISTENING: [
        _frame(r0="...F.....FFF..", r1="..FFF....FFF..",
               r4=_eyes(EYES_OPEN)),
        _frame(r0="...F......F...", r1="..FFF....FFF..",
               r4=_eyes(EYES_OPEN)),
    ],
    State.SPEAKING: [
        _frame(r4=_eyes(EYES_OPEN), r5="...FFFWWFFFF.."),
        _frame(r4=_eyes(EYES_OPEN), r5="...FFFFFFFF..."),
    ],
}


# -- drawing ----------------------------------------------------------------

def _pack(top: str, bottom: str, colour) -> tuple[str, str]:
    """One cell from two stacked pixels: the glyph, and the style for it."""
    if top == "." and bottom == ".":
        return " ", ""
    if top == ".":
        return "▄", colour(bottom)
    if bottom == ".":
        return "▀", colour(top)
    if top == bottom:
        return "█", colour(top)
    return "▀", f"{colour(top)} on {colour(bottom)}"


def rows(state, frame: int, palette) -> list[Text]:
    """The sprite, as HEIGHT rich Texts of WIDTH cells.

    Pure: the caller owns the frame counter, so the companion cannot animate
    on a clock of its own. That is the rule the whole UI is built on -- one
    clock, driven by drawing -- and it is what keeps a finished turn from
    leaving something twitching on the screen.
    """
    frames = FRAMES.get(_state(state)) or FRAMES[State.IDLE]
    pixels = frames[frame % len(frames)]

    def colour(char: str) -> str:
        role = INK.get(char, "")
        return palette.role(role) if role else ""

    out = []
    # strict: an odd number of pixel rows would otherwise drop the last
    # one and draw a shorter character, silently.
    for top, bottom in zip(pixels[0::2], pixels[1::2], strict=True):
        line = Text()
        for a, b in zip(top, bottom, strict=True):
            glyph, style = _pack(a, b, colour)
            line.append(glyph, style=style)
        out.append(line)
    return out


def _state(value) -> State:
    if isinstance(value, State):
        return value
    try:
        return State(str(value))
    except ValueError:
        return State.IDLE


GLYPHS = " ▀▄█"
"""Every character the sprite can draw. All four are one cell wide."""


def _ambiguous_is_wide() -> bool:
    """Whether this terminal is likely to draw ▀ as two cells.

    The half-blocks are East Asian Width "Ambiguous", and there is no
    Neutral alternative: ▀ and ▄ are the only glyphs that split a cell
    horizontally, and both are Ambiguous, as is █. That is a property of
    the technique rather than a choice between codepoints -- half-block
    rendering is not available in a locale that draws Ambiguous wide.

    In a Western locale Ambiguous is one cell and everything is fine. In a
    CJK locale a terminal may draw it as two, which would double the
    sprite's width and shift the text beside it by fourteen columns on
    every frame. So the sprite is declined there, and the status lines --
    which say the same thing in words -- are drawn on their own.
    """
    import os

    locale = " ".join(os.environ.get(name, "") for name in
                      ("LC_ALL", "LC_CTYPE", "LANG")).lower()
    return any(tag in locale for tag in ("ja", "ko", "zh", "cjk"))


def fits(width: int, unicode_ok: bool) -> bool:
    """Whether to draw the companion at all.

    Half-blocks are the whole technique, so a terminal that cannot render
    ▀ gets no sprite rather than a worse drawing of the same character in
    punctuation -- that fallback is exactly what this replaced. The status
    line beside it carries the state in words either way, so nothing is
    lost but the picture.
    """
    return unicode_ok and width >= MIN_COLUMNS and not _ambiguous_is_wide()
