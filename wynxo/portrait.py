"""The companion, drawn large: a catboy at his desk.

This is the picture the interface is built around rather than a decoration
dropped into a corner of it, so it is drawn rather than sampled -- placed in
normalised coordinates and rasterised at whatever size the layout can give
it, so the same illustration is crisp on a tall terminal and crisp on a
short one.

Deliberately a person, not an animal. A human head with human proportions,
messy hair over the eyes, a hoodie, one hand at the keyboard and the other
holding his face up; the ears are the only cat about him. The small animated
companion in ``sprite.py`` is the same character at a size where a
silhouette is all that survives -- this is the one you can actually see.

The aspect is fixed. Normalised coordinates are only isotropic if the canvas
keeps the proportions the drawing was composed at, and the first version did
not: at a wider box the head came out an oval and the ears came out horns.
``draw()`` therefore takes a width and derives the height, and the layout
picks a width that fits the rows it has rather than passing a box.

Everything is near-black. That is the point and it is also the difficulty:
hair, hoodie, wall and desk are four different blacks, and four blacks with
no light between them is one black. What separates them is a one-pixel lit
rim on the edges the laptop screen would actually reach, and ordered
dithering for everything that falls off gradually -- the wall behind him,
the glow on his face. No gradients: a terminal has none, and this design has
ruled them out anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

from .pixelart import Canvas, darken, is_hex, mix

NATIVE_CELLS = 46
NATIVE_PIXELS = 64
"""The grid the composition below was laid out on: 46 pixels across, 64
down. Every coordinate in this module is one of those two divided by its
extent, which is why they can be read as proportions of the figure."""

ASPECT = NATIVE_PIXELS / NATIVE_CELLS
"""Pixel rows per pixel column. A half-block pixel is very nearly square --
a cell is ~2.15 times taller than wide and holds two of them -- so this is
a true aspect ratio and not a fudge factor."""

MIN_CELLS = 32
"""Below this the face stops being a face and the ears stop being ears.

Measured, not guessed: at thirty columns the eyes merge into the fringe and
the ear triangles round away to two bumps on a black block. The layout drops
the picture there rather than draw a smudge -- a character you cannot read
says less than giving the conversation the whole width."""

PX = 1.0 / (NATIVE_CELLS - 1)
PY = 1.0 / (NATIVE_PIXELS - 1)
"""One pixel of the native grid, in normalised units.

Faces are built out of these. An eye is three pixels across, and writing
that as ``cu - 1.5 * PX`` to ``cu + 1.5 * PX`` says so; writing it as a
radius of 0.033 says nothing, and the first attempt rounded into a
five-pixel blob with a three-pixel highlight sitting on it -- which at this
size is not an eye, it is a hole.
"""


def rows_for(cells: int) -> int:
    """Terminal rows the picture occupies at a given width."""
    return max(4, int(round(cells * ASPECT)) // 2)


def cells_for(rows: int) -> int:
    """The widest picture that fits in a given number of rows."""
    return max(0, int(rows * 2 / ASPECT))


# -- ink ---------------------------------------------------------------------

@dataclass(frozen=True)
class Ink:
    """Every shade in the picture, mixed from the palette in force.

    A palette carries roles, not shades, and this needs four separable
    near-blacks. Mixing them from the palette's own accent and bar
    background keeps the artwork inside /theme: under `ember` the catboy
    sits in a warm room and under `midnight` a cold one, without a second
    palette to keep in step with the first.
    """

    wall: str
    wall_lit: str
    shelf: str
    book: str
    book_lit: str
    hair: str
    hair_lit: str
    ear: str
    skin: str
    skin_lit: str
    skin_dark: str
    eye: str
    iris: str
    mouth: str
    hood: str
    hood_lit: str
    hood_dark: str
    laptop: str
    laptop_lit: str
    glow: str
    desk: str
    desk_lit: str

    @classmethod
    def of(cls, palette) -> "Ink":
        accent, ground = palette.accent, palette.bar_bg
        if not (is_hex(accent) and is_hex(ground)):
            # A 16-colour palette has nothing to blend. Flat roles: fewer
            # steps, but every one of them a colour the terminal has.
            return cls(
                wall="", wall_lit=palette.faint, shelf=palette.faint,
                book=palette.faint, book_lit=palette.muted,
                hair=palette.accent_dim, hair_lit=palette.accent,
                ear=palette.accent, skin=palette.muted,
                skin_lit=palette.text, skin_dark=palette.faint,
                eye=palette.faint, iris=palette.accent,
                mouth=palette.faint, hood=palette.accent_dim,
                hood_lit=palette.muted, hood_dark=palette.faint,
                laptop=palette.faint, laptop_lit=palette.accent,
                glow=palette.accent, desk=palette.faint,
                desk_lit=palette.muted,
            )
        # One near-black with the accent's hue in it is the room's base
        # note; every other shade is a step away from that.
        base = darken(mix(ground, accent, 0.20), 0.66)
        return cls(
            # The room. Two steps below anything in front of it: drawn any
            # brighter the shelf wins, and the picture reads as a bookcase
            # with somebody standing in the way of it.
            wall=base,
            wall_lit=mix(base, accent, 0.10),
            shelf=mix(base, accent, 0.24),
            book=mix(base, accent, 0.12),
            book_lit=mix(base, accent, 0.28),
            # The figure. The hair is the darkest thing in the frame and the
            # hood is a clear step above the wall, which is the separation
            # that lets a black character sit against a black room.
            hair=darken(base, 0.55),
            hair_lit=mix(darken(base, 0.40), accent, 0.34),
            ear=mix(darken(base, 0.30), accent, 0.42),
            hood=mix(base, accent, 0.20),
            hood_lit=mix(base, accent, 0.42),
            hood_dark=darken(base, 0.45),
            skin=mix(palette.muted, ground, 0.30),
            skin_lit=mix(palette.text, accent, 0.30),
            skin_dark=mix(palette.faint, ground, 0.50),
            eye=darken(base, 0.75),
            iris=mix(accent, palette.text, 0.30),
            mouth=mix(darken(base, 0.35), accent, 0.20),
            # The laptop's back is unlit, so it is the darkest large shape
            # after the hair; only its top edge carries any light at all.
            laptop=darken(mix(base, accent, 0.10), 0.30),
            laptop_lit=mix(accent, palette.text, 0.22),
            glow=mix(accent, ground, 0.55),
            desk=mix(base, accent, 0.06),
            desk_lit=mix(base, accent, 0.30),
        )

    def style(self, code: str) -> str:
        return "" if code == " " else getattr(self, _CODES.get(code, ""), "")


_CODES = {
    "w": "wall", "W": "wall_lit", "s": "shelf", "b": "book", "B": "book_lit",
    "k": "hair", "K": "hair_lit", "e": "ear",
    "n": "skin", "N": "skin_lit", "d": "skin_dark",
    "y": "eye", "i": "iris", "m": "mouth",
    "h": "hood", "H": "hood_lit", "j": "hood_dark",
    "l": "laptop", "L": "laptop_lit", "g": "glow",
    "t": "desk", "T": "desk_lit",
}
"""One letter per shade. The drawing below reads as a picture in these, and
the codes are what the tests assert on -- a silhouette is a fact about the
art, colour is a fact about the theme, and they are checked separately."""


# -- the drawing -------------------------------------------------------------
#
# Composed on the native grid and written in its pixels, not in fractions.
# ``Scene`` is the only reason that is possible: it takes coordinates in
# 46-by-64 and hands the canvas normalised ones, so the artwork below can be
# read as a drawing -- the eyes are four pixels wide and eight apart, the
# ears are eight tall -- while still rasterising at any size the layout can
# afford. The first version of this file was written in fractions and every
# proportion in it was a number nobody could check.

class Scene:
    """The canvas, addressed in the pixels the composition was drawn on."""

    def __init__(self, art: Canvas) -> None:
        self.art = art

    @staticmethod
    def _u(x: float) -> float:
        return x / (NATIVE_CELLS - 1)

    @staticmethod
    def _v(y: float) -> float:
        return y / (NATIVE_PIXELS - 1)

    def rect(self, x0, y0, x1, y1, ink) -> None:
        self.art.rect(self._u(x0), self._v(y0), self._u(x1), self._v(y1), ink)

    def ellipse(self, cx, cy, rx, ry, ink, **kw) -> None:
        self.art.ellipse(self._u(cx), self._v(cy), self._u(rx), self._v(ry),
                         ink, **kw)

    def poly(self, points, ink) -> None:
        self.art.poly([(self._u(x), self._v(y)) for x, y in points], ink)

    def stroke(self, x0, y0, x1, y1, ink, *, thickness, taper=1.0) -> None:
        self.art.stroke(self._u(x0), self._v(y0), self._u(x1), self._v(y1),
                        ink, thickness=self._u(thickness), taper=taper)

    def dither(self, x0, y0, x1, y1, ink, density, *, over=None) -> None:
        self.art.dither(self._u(x0), self._v(y0), self._u(x1), self._v(y1),
                        ink, density, over=over)

    def outline(self, ink, edge, *, sides="t") -> None:
        self.art.outline(ink, edge, sides=sides)


def draw(cells: int) -> Canvas:
    """Rasterise the scene at ``cells`` wide, keeping the fixed aspect."""
    cells = max(MIN_CELLS, cells)
    art = Canvas(cells, rows_for(cells) * 2)
    scene = Scene(art)
    _room(scene)
    _shoulders(scene)
    _arms(scene)
    _head(scene)
    _cheek_hand(scene)
    _laptop(scene)
    _desk(scene)
    _light(scene)
    return art


def _room(s: Scene) -> None:
    """A dark room with a shelf of books behind him.

    The background exists to give the figure an edge to be seen against and
    for no other reason, so it is kept two steps below everything in front
    of it. Drawn brighter, the shelf wins: the first pass had books the same
    value as the character's hair and the picture read as a bookcase with
    someone in the way of it.
    """
    s.rect(0, 0, 45, 61, "w")
    s.dither(0, 0, 45, 61, "W", lambda u, v: 0.26 * (1 - v) ** 1.4)
    for board in (14, 31, 48):
        _books(s, board)
        s.rect(1, board, 44, board + 0.8, "s")


def _books(s: Scene, board: int) -> None:
    """Spines standing on a shelf board.

    Deterministic, not random: the picture is redrawn on every resize, and a
    bookshelf that reshuffles itself when the window changes width is a
    distraction rather than a room.
    """
    # width, how far short of the board's full height it stands, whether it
    # is one of the few catching light
    spines = ((1.4, 1.0, 0), (1.0, 0.0, 1), (1.2, 1.8, 0), (0.8, 0.5, 0),
              (1.6, 0.0, 0), (1.0, 2.4, 0), (1.3, 0.7, 1), (0.9, 0.0, 0),
              (1.5, 1.5, 0), (1.1, 0.3, 0), (1.4, 2.1, 0), (1.0, 0.0, 1),
              (1.2, 1.1, 0), (1.6, 0.4, 0), (0.9, 1.9, 0), (1.3, 0.0, 0))
    x = 1.5
    index = board % len(spines)
    while x < 43:
        width, short, lit = spines[index % len(spines)]
        s.rect(x, board - 6 + short, min(43.5, x + width), board,
               "B" if lit else "b")
        x += width + 0.4
        index += 1


def _shoulders(s: Scene) -> None:
    """The hoodie he is sitting in. Drawn first, so the head sits in it."""
    s.ellipse(23, 74, 19.5, 33, "h")             # shoulders, from y 41
    s.ellipse(23, 42, 12.5, 5, "h")              # the collar round the neck


def _arms(s: Scene) -> None:
    """One arm up to the face, one down to the keys.

    The pose is most of the character -- it is the difference between
    someone working and someone posing for a portrait. The hand at the face
    is drawn later, after the head, because it is in front of it.
    """
    s.stroke(12, 50, 7, 58, "h", thickness=8)        # upper arm, to the desk
    s.stroke(7.5, 57, 16, 33, "h", thickness=7, taper=0.8)       # forearm
    s.stroke(36, 46, 33, 58, "h", thickness=8)       # the arm at the keys


def _head(s: Scene) -> None:
    """Ears, hair, and a face with a forehead and a jaw.

    Two things keep this a person at terminal resolution. The eyes sit below
    a forehead rather than in the middle of a circle, and the face narrows
    to a jaw rather than ending in the curve it started with. Drop either
    and it reads as an animal's head however human the rest is.
    """
    s.rect(20.5, 32, 25.5, 42, "d")              # the neck, behind the jaw

    # The hair mass, and the locks falling past it. Tapered to a point
    # rather than squared off: the first pass ended them in a straight line
    # at the shoulders and each one read as a black brick beside his face.
    s.ellipse(23, 21, 11, 12, "k", squash_top=0.95)
    s.poly([(12, 18), (17.5, 18), (17.5, 33), (14.5, 37), (12.5, 33)], "k")
    s.poly([(34, 18), (28.5, 18), (28.5, 32), (31.5, 36), (33.5, 32)], "k")

    # Ears last of the three, so they stand on the hair rather than behind
    # it. Behind it, all that showed was the tip -- two thin spikes over the
    # crown, which read as antennae.
    s.poly([(13.5, 15), (21, 12.5), (14.8, 3)], "k")
    s.poly([(32.5, 15), (25, 12.5), (31.2, 3)], "k")
    s.poly([(15.6, 12.8), (19.6, 11.4), (16.2, 5.5)], "e")
    s.poly([(30.4, 12.8), (26.4, 11.4), (29.8, 5.5)], "e")

    # The face, cut out of the hair: a cranium, a narrower jaw, and a chin
    # that comes to a point rather than to a curve.
    #
    # Drawn twice. Once whole in the shadow tone, then again smaller and
    # lower in the lit one, which puts the light on the jaw and leaves the
    # forehead dark -- correct, because the only light in this room is the
    # screen under his chin. Shading it by scattering dark pixels over a
    # flat face instead, which is what the pass before this did, gave him
    # freckles.
    s.ellipse(23, 23, 8.2, 8.5, "d")
    s.ellipse(23, 28, 6.8, 6, "d")
    s.poly([(18.5, 30), (27.5, 30), (23, 35)], "d")
    s.ellipse(23, 25, 7.0, 7.2, "n")
    s.ellipse(23, 28.5, 6.0, 5.2, "n")
    s.poly([(19.2, 30), (26.8, 30), (23, 33.8)], "n")

    # Messy fringe: uneven strands over the forehead, the longest stopping
    # just above the eyes. Even strands are a helmet; this is the difference.
    # The strands overlap. Left with gaps between them the forehead showed
    # through in pale vertical slots, and hair with grey stripes in it is a
    # highlight, not a fringe -- so the raggedness is all in how far down
    # each one reaches.
    for x, depth in ((12.6, 4), (15.2, 8), (17.8, 3), (20.4, 6), (23, 9),
                     (25.6, 4), (28.2, 7), (30.8, 3)):
        s.rect(x, 12, x + 3.0, 12 + depth, "k")

    _face(s)


def _face(s: Scene) -> None:
    """Eyes, nose, mouth. Four pixels of eye and one of everything else."""
    for cx, flip in ((19.4, False), (26.6, True)):
        s.rect(cx - 2, 22, cx + 2, 25.5, "y")        # the eye
        s.rect(cx - 1.4, 23, cx + 1.4, 25.5, "i")    # the iris
        s.rect(cx - 0.4, 24.2, cx + 0.4, 25.5, "y")  # the pupil
        spark = cx - 1.6 if not flip else cx + 0.8
        s.rect(spark, 22.2, spark + 0.8, 23, "N")    # one catchlight each
    s.rect(22.8, 28.6, 23.4, 29.2, "d")              # nose, one pixel
    s.rect(22, 31, 24, 31.5, "m")                    # mouth


def _cheek_hand(s: Scene) -> None:
    """The hand his face is resting on -- in front of the hair, not under it.

    Drawn after the head for that reason alone. On the first pass it went
    with the rest of the arm, the side lock painted straight over it, and
    the forearm ran up to the jaw and stopped: the pose read as an arm that
    had been cut off rather than as someone leaning on his hand.
    """
    s.ellipse(16, 32, 3.6, 3.8, "n")                 # the fist
    s.ellipse(16, 29.4, 3.4, 1.2, "d")               # knuckles
    for x in (14, 15.8, 17.6):                       # fingers, folded
        s.rect(x, 30.4, x + 0.6, 34.6, "d")
    # A shadow down the side of it, or the hand and the cheek are one
    # skin-coloured shape and the pose disappears into the face.
    s.rect(19.4, 29.5, 20, 35.5, "d")


def _laptop(s: Scene) -> None:
    """Open, seen from behind, its screen the only light in the room.

    The lid crosses in front of his chest -- it is between him and the
    viewer, which is what makes this a desk he is sitting at rather than a
    prop standing beside him. Its back is unlit, so it is the darkest large
    shape in the picture and only its top edge carries any light at all.
    """
    s.poly([(15, 47), (37, 47), (40, 57), (12, 57)], "l")
    # The screen's light on the wall behind him, and on him. Two passes
    # rather than one over everything: a single wash put the same speckle on
    # the hood and the room, so the halo landed on top of the character
    # instead of behind him.
    s.dither(6, 38, 44, 47, "g",
             lambda u, v: 0.55 * v ** 2.0 * (1 - abs(u - 0.5) * 1.05),
             over="w")
    s.dither(6, 38, 44, 47, "g",
             lambda u, v: 0.30 * v ** 2.0 * (1 - abs(u - 0.5) * 1.05),
             over="W")
    s.rect(15.5, 47, 36.5, 47.8, "L")                # the lid's lit edge
    s.poly([(26, 51), (27.6, 52.4), (26, 53.8), (24.4, 52.4)], "L")


def _desk(s: Scene) -> None:
    """The deck, the hand resting on it, and the edge the picture ends at."""
    s.poly([(10, 57), (42, 57), (44, 61), (8, 61)], "t")
    s.rect(8, 56.6, 44, 57.4, "T")                   # the hinge line
    s.ellipse(32, 59, 4, 2, "n")                     # the hand on the keys
    for x in (30, 32, 34):
        s.rect(x, 58.5, x + 0.6, 60.4, "d")          # fingers over the keys
    s.ellipse(7, 58.5, 4, 2.2, "h")                  # the other elbow
    # Below the desk is the room's darkest value: the picture ends in
    # shadow rather than at a border.
    s.rect(0, 61, 45, 63, "t")
    s.dither(0, 61, 45, 63, "j", lambda u, v: 0.5 * v)


def _light(s: Scene) -> None:
    """What the screen reaches, and what it does not.

    There is one light in the picture and it is below him, so what it lands
    on is the underside of the jaw, the front of the hood, and the tops of
    his hands. Nothing above the eyes is lit, which is what makes the room
    dark rather than merely grey -- and the rim goes on the hood alone,
    because outlining every shape in the frame is how a drawing turns into
    a diagram of itself.
    """
    s.dither(17, 30, 29, 35, "N",
             lambda u, v: 0.34 * v * (1 - abs(u - 0.5) * 1.4), over="n")
    s.dither(10, 44, 36, 56, "H",
             lambda u, v: 0.22 * (1 - v) ** 1.4 * (1 - abs(u - 0.5) * 1.2),
             over="h")
    s.dither(4, 52, 42, 63, "j", lambda u, v: 0.22 * v, over="h")
    s.rect(21, 42, 21.8, 49, "H")                    # drawstrings
    s.rect(25.2, 42, 26, 47.5, "H")


def rows(cells: int, palette) -> list[Text]:
    """The scene as terminal rows, coloured by the palette in force."""
    return draw(cells).rows(Ink.of(palette).style)


def fits(width: int, unicode_ok: bool) -> bool:
    """Whether the illustration can be drawn honestly at this width.

    The same locale gate the small sprite uses: the half-block glyphs are
    East Asian Width Ambiguous, so a CJK locale may draw them two cells wide
    and every row of the picture would be twice the width the layout
    measured for it.
    """
    from .sprite import _ambiguous_is_wide

    return unicode_ok and width >= MIN_CELLS and not _ambiguous_is_wide()
