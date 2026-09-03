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

WIDTH = 18
"""Columns. Also the pixel width -- one pixel per column."""

HEIGHT = 6
"""Terminal rows. Twelve pixel rows, two to a row."""

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
    "L": "muted",        # the laptop case
    "K": "bar_bg",       # the screen panel, dark, so it reads as a screen
    "C": "warn",         # the mug
    "W": "faint",        # steam, and the thought drifting off the ear
    "G": "good",         # the screen when the work came out right
    "R": "bad",          # the screen when it did not
    ".": "",             # nothing: leave the terminal's own background
}
"""Pixel character to palette role. Roles rather than colours, so /theme
reaches the companion -- it was the one thing on screen a theme could not
touch for most of this project's life."""


# -- the character ----------------------------------------------------------

_BODY = [
    "....F........F....",   # ear tips
    "....FF......FF....",   # ears, rising from the head's corners
    "....FFFFFFFFFF....",   # crown, drawing in under the ears
    "...FFFFFFFFFFFF...",   # brow -- also where the eyes go when they look up
    "...FFEEFFFFEEFF...",   # eyes, set wide
    "....FFFFffFFFF....",   # muzzle, and the jaw drawing in
    "......FFFFFF......",   # neck
    "..fFFFFFFFFFFFFf..",   # shoulders, and the tops of both arms
    "..ffLKKKKKKKKLff..",   # upper arms, and the lid: pale case, dark screen
    "...fLKKKKKKKKLf...",   # forearms coming in toward the keyboard
    "...LLLLLLLLLLLL...",   # the deck, in front of the lid
    "....ff......ff....",   # both paws, resting against the near edge
]
"""Every frame starts here and edits rows.

Written as one base rather than as a dozen independent pictures because the
silhouette is the character: if the ears move a pixel between idle and
thinking, it stops reading as the same animal and starts reading as an
animation glitch. Only the rows a state has a reason to change are
replaced, so nothing can drift by accident.

Three things in this drawing were arrived at by drawing them wrong first.

The head has to come to a *neck*. Drawn as jaw straight into shoulders --
a narrow row over a wider one, both in fur -- head and body fused into one
blob and the laptop under it read as a plinth the face was sitting on. Six
pixels at r6, narrower than either, is the whole separation, and it is what
makes this a character at a desk rather than a mask on a keyboard.

The paws have to break the silhouette. On the deck they were two dim
pixels inside a pale bar: invisible at a glance, and a hand you cannot see
cannot be seen to type. In front of it, against the terminal's own
background, they are the two lowest shapes on the character -- and lifting
one onto the deck is a full-contrast change from purple-on-nothing to
purple-on-pale.

And it grew from fourteen by ten to eighteen by twelve to have any of this
at all. At the old size the character was a head with a laptop under its
chin and no body: there was nowhere to put a hand, so "coding" was two dim
pixels at the edges of the jaw, and the pose everybody draws for thinking
-- eyes up, a paw at the face -- could not be drawn. Four columns and one
terminal row buy shoulders, two arms, and two paws that can be somewhere
specific.

The screen faces us, which is not what you would see standing in front of
somebody at a laptop. What is on it is half of what the companion is for,
and the paws are stylised the same way, in front of the deck rather than
hidden behind the lid.
"""

# Eye rows are twelve pixels, columns 3..14 -- the width of the head.
EYES_OPEN = "FFEEFFFFEEFF"
EYES_SHUT = "FFFFFFFFFFFF"   # nothing: at two pixels an eye either is or isn't
EYES_GLAD = "FFfFFFFFFfFF"   # the corners lift, in the shadow colour
EYES_LEFT = "FEEFFFFEEFFF"   # both eyes together -- one eye moving is a squint
EYES_RIGHT = "FFFEEFFFFEEF"
EYES_SHUT_R = "FFRRFFFFRRFF"  # shut, and the wrong colour

_EYE_NOTE = """Why a blink closes the eyes rather than dimming them.

Two pixels is the whole eye, so "half shut" is not available: the only
changes that survive are present, absent, and moved. Dimming the pixel
looked right in a screenshot and vanished in motion, because it left the
cell's glyph identical and only the colour moved -- so on a terminal
without truecolour, or to anyone reading shape before hue, the character
never blinked at all."""

_UP_NOTE = """Looking up is a row, not a shade.

The eyes live on r4 and the brow above them on r3 is plain fur, so raising
the gaze means drawing the eyes on r3 and leaving r4 blank. That is half a
terminal row of movement, and it is the largest change this sprite can make
to a face without moving the head -- which is why thinking is legible at a
glance now, and used to need a mark beside the ear to be legible at all."""


def _frame(**rows: str) -> list[str]:
    """The base with some rows replaced. Keyword is r0..r11."""
    out = list(_BODY)
    for key, value in rows.items():
        index = int(key[1:])
        assert len(value) == WIDTH, f"{key}: {len(value)} columns"
        out[index] = value
    return out


def _face(pattern: str) -> str:
    """A head row: twelve pixels at columns 3..14."""
    assert len(pattern) == 12, pattern
    return "..." + pattern + "..."


def _look(up: bool = False, eyes: str = EYES_OPEN) -> dict[str, str]:
    """The two rows a gaze occupies, as _frame keywords."""
    if up:
        return {"r3": _face(eyes), "r4": _face(EYES_SHUT)}
    return {"r4": _face(eyes)}


def _lid(cells: str) -> str:
    """The upper screen row: eight inner pixels, arms either side."""
    assert len(cells) == 8, cells
    return "..ffL" + cells + "Lff.."


def _panel(cells: str) -> str:
    """The lower screen row: the same, with the forearms drawn in."""
    assert len(cells) == 8, cells
    return "...fL" + cells + "Lf..."


_DARK = _lid("KKKKKKKK")

# The paws. Down is r11, resting in front of the deck against nothing; up
# is r10, the same two columns one row higher and drawn over the deck. Both
# positions are full contrast, which is what makes one pixel of movement
# read as a hand working rather than as a rendering artefact.
_DECK = "...LLLLLLLLLLLL..."
_DECK_LEFT = "...LffLLLLLLLLL..."
_DECK_RIGHT = "...LLLLLLLLLffL..."
_REST = "....ff......ff...."
_REST_LEFT = "....ff............"
_REST_RIGHT = "............ff...."
_REST_NONE = ".................."

# The paw that comes up to the face when the character is thinking, and the
# arm holding it there: elbow on the desk, forearm vertical, paw against
# the jaw. Drawn outside the head's own columns so it has a silhouette of
# its own rather than reading as a smudge on the chin.
_CHIN_PAW = "....FFFFffFFFFff.."
_CHIN_ARM = "......FFFFFF..ff.."


FRAMES: dict[State, list[list[str]]] = {
    # Breathing, and a blink every fourth beat. Nothing else moves.
    State.IDLE: [
        _frame(**_look()),
        _frame(**_look()),
        _frame(**_look()),
        _frame(**_look(eyes=EYES_SHUT)),
    ],
    # The pose everybody draws for thinking: eyes up and away from the
    # screen, one paw off the keys and up against the jaw, and a thought
    # coming off the ear. Three signals, because this is the state somebody
    # glances at to answer "is it stuck?" -- and because at the old sprite
    # size none of them could be drawn. Thinking was the eyes moved one
    # pixel row, which at the size anyone actually sees this was the same
    # picture as idle.
    State.THINKING: [
        _frame(**_look(up=True), r5=_CHIN_PAW, r6=_CHIN_ARM, r11=_REST_LEFT),
        _frame(**_look(up=True), r5=_CHIN_PAW, r6=_CHIN_ARM, r11=_REST_LEFT,
               r1="....FF......FF.W.."),
        _frame(**_look(up=True, eyes=EYES_LEFT), r5=_CHIN_PAW, r6=_CHIN_ARM,
               r11=_REST_LEFT, r0="....F........F.W..",
               r1="....FF......FF..W."),
        _frame(**_look(up=True), r5=_CHIN_PAW, r6=_CHIN_ARM, r11=_REST_LEFT,
               r0="....F........F..W."),
    ],
    # Looking left, back through centre, right, back -- a sweep needs the
    # middle position or it is a flick between two stares -- with a single
    # lit pixel travelling across the screen under it.
    #
    # The marker is not decoration. There are three eye positions and one
    # of them is the one idle uses, so a sweep that passes through centre
    # spends half its frames drawn exactly as idle. A state has to be
    # legible in every frame it is in; it is the motion that may vary. The
    # marker is also the honest picture of what searching is: something
    # running along, looking at each thing in turn.
    State.SEARCHING: [
        _frame(**_look(eyes=EYES_LEFT), r9=_panel("SKKKKKKK")),
        _frame(**_look(), r9=_panel("KKSKKKKK")),
        _frame(**_look(eyes=EYES_RIGHT), r9=_panel("KKKKSKKK")),
        _frame(**_look(), r9=_panel("KKKKKKSK")),
    ],
    # Eyes down on a screen filling with text, and a blink. The character
    # is still and the page is not, which is what reading looks like. Both
    # paws stay on the near edge: nothing is being written.
    State.READING: [
        _frame(r8=_lid("SSKKKKKK")),
        _frame(r8=_lid("SSSSSKKK"), r9=_panel("SSKKKKKK")),
        _frame(r8=_lid("SSSSSSSS"), r9=_panel("SSSSSKKK"),
               **_look(eyes=EYES_SHUT)),
        _frame(r8=_lid("SSSSSSSS"), r9=_panel("SSSSSSSK")),
    ],
    # The flagship state: the paws alternate onto the keys and a line of
    # code grows on the screen under them, filling the first row and
    # starting a second. The alternation is the whole thing -- a hand that
    # only ever rests is a hand on a keyboard, and a hand that leaves it
    # and comes back is a hand typing -- and six frames rather than four
    # because a line that fills and wraps reads as writing, where a bar
    # that fills and snaps back reads as a progress meter.
    #
    # A paw is up on the second frame because that is the one the gallery
    # draws, and a gallery that shows the flagship state with both hands
    # resting is showing the one frame where nothing is happening.
    State.CODING: [
        _frame(r8=_lid("SKKKKKKK")),
        _frame(r10=_DECK_LEFT, r11=_REST_RIGHT, r8=_lid("SSSKKKKK")),
        _frame(r8=_lid("SSSSSKKK")),
        _frame(r10=_DECK_RIGHT, r11=_REST_LEFT, r8=_lid("SSSSSSSS")),
        _frame(r8=_lid("SSSSSSSS"), r9=_panel("SSKKKKKK")),
        _frame(r10=_DECK_LEFT, r11=_REST_RIGHT, r8=_lid("SSSSSSSS"),
               r9=_panel("SSSSSKKK")),
    ],
    # Watching it run. Both paws come up onto the deck and stay there --
    # there is nothing to type while a test decides -- and a bar fills.
    State.TESTING: [
        _frame(r10="...LffLLLLLLffL...", r11=_REST_NONE, r9=_panel("SKKKKKKK")),
        _frame(r10="...LffLLLLLLffL...", r11=_REST_NONE, r9=_panel("SSSKKKKK")),
        _frame(r10="...LffLLLLLLffL...", r11=_REST_NONE, r9=_panel("SSSSSKKK")),
        _frame(r10="...LffLLLLLLffL...", r11=_REST_NONE, r9=_panel("SSSSSSSK")),
    ],
    # Something failed and is being worked out: the thinking pose, with the
    # damage still on the screen.
    State.RECOVERING: [
        _frame(**_look(up=True), r5=_CHIN_PAW, r6=_CHIN_ARM, r11=_REST_LEFT,
               r8=_lid("RRKKKKKK")),
        _frame(**_look(up=True, eyes=EYES_LEFT), r5=_CHIN_PAW, r6=_CHIN_ARM,
               r11=_REST_LEFT, r8=_lid("RRKKKKKK")),
    ],
    # Done, and it is the mug that says so. The one state with a prop --
    # and the screen goes green, because "it worked" is the one thing on
    # this character worth spending the good colour on.
    State.SUCCESS: [
        _frame(**_look(eyes=EYES_GLAD), r7="..fFFFFFFFFFFFFf.W",
               r8=_lid("GGGGGGGG"), r9="...fLKKKKKKKKLf.CC",
               r10="...LLLLLLLLLLLL.CC", r11="....ff......ff..CC"),
        _frame(**_look(eyes=EYES_GLAD), r8="..ffLGGGGGGGGLff.W",
               r9="...fLKKKKKKKKLf.CC",
               r10="...LLLLLLLLLLLL.CC", r11="....ff......ff..CC"),
        _frame(**_look(eyes=EYES_SHUT), r8=_lid("GGGGGGGG"),
               r9="...fLKKKKKKKKLf.CC",
               r10="...LLLLLLLLLLLL.CC", r11="....ff......ff..CC"),
    ],
    # The ears go down. Colour alone had error and success drawing the same
    # silhouette -- both are "eyes not plainly open" -- and those two are
    # the pair that must never be confused at a glance, since one of them
    # means stop reading and look. The ears are the biggest shape the
    # character has, so they are what changes. The paws come off the deck
    # as well: nothing is being written while this is true.
    State.ERROR: [
        _frame(r0="..................", r1="...FF......FF.....",
               **_look(eyes=EYES_SHUT_R), r8=_lid("RRRRKKKK"),
               r11=_REST_NONE),
    ],
    State.CANCELLED: [
        _frame(r0="..................", r1="...FF......FF.....",
               **_look(eyes=EYES_SHUT), r8=_DARK, r11=_REST_NONE),
    ],
    # One ear up, and something arriving at it. The ear alone was one pixel
    # of difference from the base body -- technically distinct, and not
    # legible at the size anyone sees this. So listening borrows the device
    # that made thinking readable, mirrored: a mark, on the left, moving
    # toward the ear rather than drifting away from it. The paws come off
    # the keys, because a character still typing is not listening.
    State.LISTENING: [
        _frame(r0="..W.F........F....", r11=_REST_NONE),
        _frame(r0="W...F........F....", r1="..W.FF......FF....",
               r11=_REST_NONE),
    ],
    # The mouth opens and closes, and neither frame is the closed mouth of
    # the base body -- a state has to be legible in every frame it is in,
    # and a frame that is pixel-for-pixel the idle picture is not.
    State.SPEAKING: [
        _frame(r5="....FFFFWWFFFF....", r11=_REST_NONE),
        _frame(r5="....FFFWWWWFFF....", r11=_REST_NONE),
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
