"""The Wynxo companion, drawn as a larger human-like catboy sprite.

The mascot is deliberately a person first and a cat second: a human head,
hair, small cat ears, neck, shoulders, torso, arms reaching a laptop, and
hands moving over the keyboard. The previous compact drawing was mostly a
cat face with a laptop attached to it, so every state read like a different
cat icon instead of the same character actually doing work.

Two pixels are packed into one terminal cell with half-block glyphs. The
sprite is intentionally larger than the old 18x12 pixel drawing so the
silhouette has enough room to show a human pose and readable state changes.
"""
from __future__ import annotations

from rich.text import Text

from .companion import State

WIDTH = 28
HEIGHT = 9
MIN_COLUMNS = 60

INK: dict[str, str] = {
    "F": "accent",       # hair / cat ears
    "f": "accent_dim",   # hoodie / clothing / shadows
    "P": "text",         # skin
    "E": "bar_bg",       # eyes
    "W": "faint",        # mouth / small motion marks
    "L": "muted",        # laptop body
    "K": "bar_bg",       # laptop bezel
    "S": "bar_accent",   # screen contents
    "G": "good",         # successful screen
    "R": "bad",          # failed screen
    "C": "warn",         # small mug / warm accent
    ".": "",
}

_BODY = [
    "........FFF......FFF.........",
    ".......FFFFF....FFFFF........",
    "......FFFFFFFFFFFFFFFF......",
    ".....FFFFFFFFFFFFFFFFFF.....",
    ".....FFFEEFFFFFFFFEEFFF.....",
    ".....FFFFFFFFFFFFFFFFFF.....",
    "......FFFFFFPPPPFFFFFF......",
    "........FFFFWWWWFFFF........",
    "..........PPPPPPPP..........",
    ".......ffffffffffffff.......",
    ".....ffffLLLLLLLLLLffff.....",
    "....fffffLKKKKKKKKLfffff....",
    "...ffffffLSSSSSSSSLffffff...",
    ".......LLLLLLLLLLLLLL.......",
    "......LLfffLLLLLLfffLL......",
    ".......LLffffffffffLL.......",
    "........ffffffffffff........",
    "......ffffffffffffffff......",
]

EYES_OPEN = "FFEEFFFFFFFFEEFF"
EYES_SHUT = "FFFFFFFFFFFFFFFF"
EYES_LEFT = "FEEFFFFFFFFFEEFF"
EYES_RIGHT = "FFEFFFFFFFFFEEEF"
EYES_GLAD = "FFfFFFFFFFfFFFFF"


def _frame(**changes: str) -> list[str]:
    """Build a frame and safely normalize hand/prop overlays to the canvas.

    A few animation overlays are intentionally shorter than the canvas so
    they can be authored as compact shapes. They are padded on the right
    rather than rejected, while the base character remains exactly WIDTH
    cells wide.
    """
    out = list(_BODY)
    for key, value in changes.items():
        index = int(key[1:])
        if len(value) < WIDTH:
            value = value.ljust(WIDTH, ".")
        elif len(value) > WIDTH:
            value = value[:WIDTH]
        out[index] = value
    assert len(out) == HEIGHT * 2
    assert all(len(row) == WIDTH for row in out)
    return out


def _face(eyes: str) -> str:
    """A 16-pixel human face centered in the 28-pixel canvas."""
    assert len(eyes) == 16
    return "....." + eyes + "......."


def _screen(cells: str) -> tuple[str, str]:
    """Two laptop rows, with the bezel and keyboard framing the display."""
    assert len(cells) == 8
    return ".....ffffL" + cells + "Lffff.....", "....fffffL" + cells + "Lfffff...."


def _with_screen(top: str, bottom: str, **changes: str) -> list[str]:
    a, b = _screen(top)
    return _frame(r10=a, r11=b, **changes)


FRAMES: dict[State, list[list[str]]] = {
    State.IDLE: [
        _frame(r4=_face(EYES_OPEN)),
        _frame(r4=_face(EYES_OPEN)),
        _frame(r4=_face(EYES_OPEN)),
        _frame(r4=_face(EYES_SHUT)),
    ],
    State.THINKING: [
        _frame(r3=_face(EYES_OPEN), r4=_face(EYES_SHUT),
               r9=".......ffffffffff.......", r10=".....ffffLLLLLLffff....."),
        _frame(r3=_face(EYES_LEFT), r4=_face(EYES_SHUT),
               r9="......ffffPPffff........", r10=".....ffffLLLLLLffff....."),
        _frame(r3=_face(EYES_RIGHT), r4=_face(EYES_SHUT),
               r9="......ffffPPffff........", r10=".....ffffLLLLLLffff.....",
               r7="........FFFFWWWWFFFF...."),
        _frame(r3=_face(EYES_OPEN), r4=_face(EYES_SHUT),
               r9=".......ffffffffff.......", r10=".....ffffLLLLLLffff....."),
    ],
    State.SEARCHING: [
        _with_screen("SKKKKKKK", "KKKKKKKK", r4=_face(EYES_LEFT)),
        _with_screen("KSKKKKKK", "KKKKKKKK", r4=_face(EYES_OPEN)),
        _with_screen("KKKKSKKK", "KKKKKKKK", r4=_face(EYES_RIGHT)),
        _with_screen("KKKKKKSK", "KKKKKKKK", r4=_face(EYES_OPEN)),
    ],
    State.READING: [
        _with_screen("SSKKKKKK", "KKKKKKKK", r4=_face(EYES_SHUT)),
        _with_screen("SSSSSKKK", "SSKKKKKK", r4=_face(EYES_SHUT)),
        _with_screen("SSSSSSSS", "SSSSSKKK", r4=_face(EYES_SHUT)),
        _with_screen("SSSSSSSS", "SSSSSSSK", r4=_face(EYES_SHUT)),
    ],
    State.CODING: [
        _with_screen("SKKKKKKK", "KKKKKKKK",
                     r14="......LLfffLLLLfffLL......",
                     r15=".......LLfffffffLL......."),
        _with_screen("SSSKKKKK", "KKSSKKKK",
                     r14="......LLfffLLLLLLffLL.....",
                     r15=".......LLffffffffffLL....."),
        _with_screen("SSSSSKKK", "SSKKKKKK",
                     r14="......LLffffLLLLfffLL......",
                     r15=".......LLfffffffLL........"),
        _with_screen("SSSSSSSS", "SSSSSKKK",
                     r14="......LLfffLLLLLLffLL.....",
                     r15=".......LLffffffffffLL....."),
        _with_screen("SSSSSSSS", "SSSSSSSK",
                     r14="......LLffffLLLLfffLL......",
                     r15=".......LLfffffffLL........"),
    ],
    State.TESTING: [
        _with_screen("SKKKKKKK", "KKKKKKKK", r14="......LLffffLLLLffffLL...."),
        _with_screen("SSSKKKKK", "KKSSKKKK", r14="......LLffffLLLLffffLL...."),
        _with_screen("SSSSSKKK", "SSKKKKKK", r14="......LLffffLLLLffffLL...."),
        _with_screen("SSSSSSSK", "SSSSKKKK", r14="......LLffffLLLLffffLL...."),
    ],
    State.RECOVERING: [
        _with_screen("RRKKKKKK", "RRKKKKKK", r3=_face(EYES_LEFT), r4=_face(EYES_SHUT)),
        _with_screen("RRKKKKKK", "KKRRKKKK", r3=_face(EYES_RIGHT), r4=_face(EYES_SHUT)),
    ],
    State.SUCCESS: [
        _with_screen("GGGGGGGG", "GGGGGGGG", r4=_face(EYES_GLAD),
                     r12="...ffffffLGGGGGGGGLffffff..."),
        _with_screen("GGGGGGGG", "GGGGGGGG", r4=_face(EYES_GLAD),
                     r7="........FFFFCCFFFF........"),
        _with_screen("GGGGGGGG", "GGGGGGGG", r4=_face(EYES_SHUT)),
    ],
    State.ERROR: [
        _with_screen("RRRRKKKK", "RRKKKKKK", r3=_face(EYES_SHUT), r4=_face(EYES_SHUT)),
        _with_screen("RRRRKKKK", "KKRRKKKK", r3=_face(EYES_LEFT), r4=_face(EYES_SHUT)),
    ],
    State.CANCELLED: [
        _frame(r0="............................", r1=".......FFFFF....FFFFF.......",
               r3=_face(EYES_SHUT), r4=_face(EYES_SHUT)),
    ],
    State.LISTENING: [
        _frame(r0="........FFF......FFF..W....", r11="....fffffLKKKKKKKKLfffff..."),
        _frame(r0=".....W..FFF......FFF.......", r1="......FFFFF....FFFFF.......",
               r11="....fffffLKKKKKKKKLfffff..."),
    ],
    State.SPEAKING: [
        _frame(r7="........FFFFWWWWFFFF........", r14="......LLffffLLLLffffLL......"),
        _frame(r7="........FFFFWWWWWWFFFF......", r14="......LLffffLLLLffffLL......"),
    ],
}


def _pack(top: str, bottom: str, colour) -> tuple[str, str]:
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
    frames = FRAMES.get(_state(state)) or FRAMES[State.IDLE]
    pixels = frames[frame % len(frames)]

    def colour(char: str) -> str:
        role = INK.get(char, "")
        return palette.role(role) if role else ""

    out: list[Text] = []
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


def _ambiguous_is_wide() -> bool:
    import os
    locale = " ".join(os.environ.get(name, "") for name in
                      ("LC_ALL", "LC_CTYPE", "LANG")).lower()
    return any(tag in locale for tag in ("ja", "ko", "zh", "cjk"))


def fits(width: int, unicode_ok: bool) -> bool:
    return unicode_ok and width >= MIN_COLUMNS and not _ambiguous_is_wide()
