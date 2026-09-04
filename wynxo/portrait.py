"""The companion, painted: a catboy at his desk, lit by his own screen.

This is the picture the interface is built around rather than a decoration
dropped into a corner of it, so it is painted rather than assembled out of
flat shapes. Everything here is placed in the coordinates of a 46-by-64 grid
-- the size the composition was drawn at -- and rasterised through
``raster.Raster`` at twice the resolution it will be shown at, so the same
illustration is crisp on a tall terminal and crisp on a short one, and detail
finer than a terminal pixel survives as tone instead of disappearing.

The version before this one filled about twenty flat colours into geometric
shapes. It read as a diagram of a catboy. What changed is not the shapes, it
is the light: there is one source in this room, it is the laptop, and every
surface in the frame is painted according to whether it can see it. That is
what the chin, the collarbone, the inner forearms and the tops of the hands
being bright is doing, and it is why the top of his head is nearly black.

It is also what the drawing is written in. Nearly every shape is laid down
twice or three times -- a form shadow, then the light, then a highlight --
because at this size an edge is one pixel and a tonal ramp is four, so the
ramp is what the eye has to read the shape from.

Deliberately a person, not an animal. Human proportions, a skull with a jaw
under it, eyes below a forehead, messy hair with individual strands in it,
a hoodie with folds; the ears are the only cat about him. The small animated
companion in ``sprite.py`` is the same character at a size where a
silhouette is all that survives -- this is the one you can actually see.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from math import cos, sin

from rich.text import Text

from . import raster
from .raster import Colour, Raster, SUPERSAMPLE, mix, rgb, saturate, shade

NATIVE_CELLS = 46
NATIVE_PIXELS = 64
"""The grid the composition below was laid out on: 46 across, 64 down. Every
coordinate in this module is in those units, which is why the drawing can be
read as proportions of the figure rather than as magic numbers."""

ASPECT = NATIVE_PIXELS / NATIVE_CELLS
"""How tall the picture stands for a given width, in terminal rows per two
cells. Fixed, and it has to be: the composition is only isotropic on the
shape it was painted at, and a picture free to take any box it is handed
comes out with an oval head and horns for ears."""

MIN_CELLS = 32
"""Below this the face stops being a face. The layout drops the picture
rather than draw a smudge -- a character you cannot read says less than
giving the conversation the whole width."""

MAX_CELLS = 58
"""And where more columns stop buying anything.

The composition is fixed, so past this the picture is the same drawing with
larger pixels: it stops gaining detail and starts taking width off the
conversation, which is the thing it is there to introduce."""


def rows_for(cells: int) -> int:
    """Terminal rows the picture occupies at a given width."""
    return max(4, int(round(cells * ASPECT)) // 2)


def cells_for(rows: int) -> int:
    """The widest picture that fits in a given number of rows."""
    return max(0, int(rows * 2 / ASPECT))


# -- the palette the picture is painted from ---------------------------------

WARM_SKIN: Colour = (232.0, 196.0, 178.0)
"""The tone every complexion in the picture starts from, before the room's
hue and the room's exposure are applied to it."""

REFERENCE = ("#c77dff", "#1b1226", "#f2e9ff", "#b9a3d4")
"""Accent, ground, text and muted, for a palette that has no numbers in it.

The plain and minimal themes name ANSI colours, which cannot be mixed. The
choice is between not drawing at all and drawing in a fixed violet that rich
then quantises to whatever the terminal has. Drawing wins: a rough picture
is still the character, and a missing one is a hole in the layout."""


@dataclass(frozen=True)
class Ink:
    """Every colour in the painting, mixed from the palette in force.

    A palette carries four or five roles and a painting needs fifty shades,
    so these are derived rather than listed: one near-black with the accent's
    hue in it is the room's base note, the skin is the accent warmed and
    lifted, and everything else is a step from one of those. Under `ember`
    the catboy sits in a warm room and under `midnight` a cold one, without a
    second palette to keep in step with the first.
    """

    key: Colour            # what the screen throws
    ambient: Colour        # what the room gives back
    wall: Colour
    wall_far: Colour
    shelf: Colour
    book: Colour
    book_lit: Colour
    hair: Colour
    hair_lit: Colour
    hair_rim: Colour
    ear: Colour
    skin: Colour
    skin_shadow: Colour
    skin_deep: Colour
    blush: Colour
    sclera: Colour
    iris: Colour
    iris_lit: Colour
    pupil: Colour
    lash: Colour
    mouth: Colour
    hood: Colour
    hood_lit: Colour
    hood_deep: Colour
    seam: Colour
    laptop: Colour
    laptop_edge: Colour
    screen: Colour
    desk: Colour
    desk_lit: Colour

    @classmethod
    def of(cls, palette) -> "Ink":
        accent, ground, text, muted = (
            (palette.accent, palette.bar_bg, palette.text, palette.muted)
            if raster.is_hex(palette.accent) and raster.is_hex(palette.bar_bg)
            else REFERENCE)
        acc, gnd = rgb(accent), rgb(ground)
        pale = rgb(text)
        base = shade(mix(gnd, acc, 0.22), -0.72)

        # The screen. Everything lit in this picture is lit by this and
        # nothing else, so it is the one colour that is allowed to be bright.
        key = mix(acc, pale, 0.42)
        ambient = shade(mix(gnd, acc, 0.35), -0.55)

        # Skin is the one thing in the picture that cannot be derived from
        # the palette alone: every role in a violet theme is violet, and a
        # face mixed only from those comes out lavender and, worse, pale --
        # which in a room with one dim light reads as a mask floating in the
        # dark. So it starts from a real skin tone, takes the room's hue,
        # and is then pushed back down into the room's exposure.
        skin = mix(mix(WARM_SKIN, acc, 0.26), gnd, 0.30)
        return cls(
            key=key,
            ambient=ambient,
            wall=shade(base, -0.30),
            wall_far=shade(base, -0.58),
            shelf=mix(base, acc, 0.22),
            book=shade(mix(base, acc, 0.14), -0.10),
            book_lit=mix(base, acc, 0.34),
            hair=shade(base, -0.72),
            hair_lit=shade(mix(base, acc, 0.28), -0.30),
            hair_rim=mix(acc, pale, 0.30),
            ear=mix(shade(base, -0.35), mix(acc, pale, 0.35), 0.45),
            skin=shade(skin, 0.06),
            skin_shadow=shade(mix(skin, gnd, 0.42), -0.22),
            skin_deep=shade(mix(skin, gnd, 0.70), -0.40),
            blush=mix(skin, saturate(mix(acc, (255, 120, 150), 0.5), 0.2),
                      0.35),
            sclera=saturate(mix(mix(WARM_SKIN, acc, 0.3), gnd, 0.16), -0.35),
            iris=mix(acc, gnd, 0.35),
            iris_lit=mix(acc, pale, 0.45),
            pupil=shade(base, -0.85),
            lash=shade(base, -0.80),
            mouth=shade(mix(skin, gnd, 0.55), -0.35),
            hood=shade(mix(base, acc, 0.16), -0.12),
            hood_lit=mix(base, acc, 0.42),
            hood_deep=shade(base, -0.60),
            seam=mix(base, acc, 0.34),
            laptop=shade(mix(base, acc, 0.08), -0.42),
            laptop_edge=mix(acc, pale, 0.35),
            screen=mix(acc, pale, 0.55),
            desk=shade(mix(base, acc, 0.10), -0.30),
            desk_lit=mix(base, acc, 0.36),
        )


# -- the drawing -------------------------------------------------------------

class Scene:
    """The canvas, addressed in the coordinates the composition was drawn in.

    Every method takes 46-by-64 units and hands the raster supersampled ones.
    That is the whole reason the painting below can be read: a strand of hair
    is "from the crown to just past the eyebrow", not "from 0.24 to 0.41".
    """

    __slots__ = ("art", "k", "ink")

    def __init__(self, art: Raster, scale: float, ink: Ink) -> None:
        self.art, self.k, self.ink = art, scale, ink

    @contextmanager
    def part(self, name: str):
        """Name what is about to be painted, for the drawing's own record.

        Costs nothing unless the canvas was asked to keep one. See
        ``Raster.label``.
        """
        was, self.art.pen = self.art.pen, name
        try:
            yield
        finally:
            self.art.pen = was

    def rect(self, x0, y0, x1, y1, colour, alpha=1.0):
        k = self.k
        self.art.rect(x0 * k, y0 * k, x1 * k, y1 * k, colour, alpha)

    def ellipse(self, cx, cy, rx, ry, colour, alpha=1.0, **kw):
        k = self.k
        self.art.ellipse(cx * k, cy * k, rx * k, ry * k, colour, alpha, **kw)

    def poly(self, points, colour, alpha=1.0):
        k = self.k
        self.art.poly([(x * k, y * k) for x, y in points], colour, alpha)

    def capsule(self, x0, y0, x1, y1, r0, r1, colour, alpha=1.0):
        k = self.k
        self.art.capsule(x0 * k, y0 * k, x1 * k, y1 * k,
                         r0 * k, r1 * k, colour, alpha)

    def glow(self, cx, cy, rx, ry, colour, strength, falloff=2.0):
        k = self.k
        self.art.glow(cx * k, cy * k, rx * k, ry * k, colour, strength,
                      falloff)

    def strand(self, x, y, angle, length, width, colour, alpha=1.0,
               curve=0.0):
        """One tapered stroke that bends: a lock of hair, a finger, a fold.

        Hair is the difference between a head and a helmet, and hair is not a
        shape -- it is forty of these, each a little different. Drawn in two
        segments so it can curve, which is what stops a fringe looking like a
        row of matchsticks.
        """
        mid = (x + cos(angle) * length * 0.5,
               y + sin(angle) * length * 0.5)
        bend = angle + curve
        end = (mid[0] + cos(bend) * length * 0.5,
               mid[1] + sin(bend) * length * 0.5)
        self.capsule(x, y, mid[0], mid[1], width, width * 0.72, colour, alpha)
        self.capsule(mid[0], mid[1], end[0], end[1], width * 0.72,
                     width * 0.12, colour, alpha)


def paint(cells: int, ink: Ink, *, labelled: bool = False) -> Raster:
    """The whole picture, back to front, on square pixels.

    Painted larger than it will be shown and on a square grid, then squeezed
    to the shape of a terminal cell's quarter at the very end. Composing on
    the stretched grid instead would mean writing every circle in the drawing
    as an ellipse and every angle as a lie.
    """
    cells = max(MIN_CELLS, min(MAX_CELLS, cells))
    width = cells * 2 * SUPERSAMPLE
    art = Raster(width, rows_for(cells) * 4 * SUPERSAMPLE, labelled=labelled)
    scene = Scene(art, width / NATIVE_CELLS, ink)

    with scene.part("room"):
        _room(scene)
    # The room is behind him, and a camera would not have both in focus.
    art.blur(max(1, int(scene.k * 0.8)))
    with scene.part("glow"):
        _screen_light(scene)
    with scene.part("hood"):
        _torso(scene)
    with scene.part("arms"):
        _arms(scene)
    with scene.part("head"):
        _head(scene)
    with scene.part("cheek hand"):
        _cheek_hand(scene)
    with scene.part("laptop"):
        _laptop(scene)
    with scene.part("bloom"):
        _bloom(scene)
    # A dark picture with one light in it, not a grey picture. The gamma
    # pushes the midtones down so the only thing that reads as bright is the
    # thing that is actually emitting.
    art.grade(lift=(3.0, 1.0, 6.0), gain=0.98, gamma=1.12)
    art.vignette(strength=0.45, fade=scene.k * 2.2)
    return art


# -- the room ----------------------------------------------------------------

def _room(s: Scene) -> None:
    """A dark room with a shelf of books, two steps below everything in
    front of it. Drawn any brighter the shelf wins and the picture reads as
    a bookcase with somebody in the way of it."""
    ink = s.ink
    s.rect(0, 0, NATIVE_CELLS, NATIVE_PIXELS,
           lambda v: mix(ink.wall_far, ink.wall, v ** 0.6))
    for board in (15, 32, 49):
        _books(s, board)
        s.rect(0, board, NATIVE_CELLS, board + 1.1, ink.shelf)
        s.rect(0, board + 1.1, NATIVE_CELLS, board + 2.2,
               shade(ink.shelf, -0.55))
    _shelf_things(s)


def _books(s: Scene, board: int) -> None:
    """Spines standing on a shelf board.

    Deterministic, not random: the picture is redrawn on every resize, and a
    bookshelf that reshuffles itself when the window changes width is a
    distraction rather than a room.
    """
    ink = s.ink
    # width, height, and how much light the spine catches
    spines = ((1.6, 8.0, 0.0), (1.1, 9.2, 0.5), (1.4, 7.2, 0.0),
              (0.9, 8.8, 0.0), (1.8, 9.5, 0.2), (1.2, 6.6, 0.0),
              (1.5, 8.4, 0.7), (1.0, 9.0, 0.0), (1.7, 7.0, 0.0),
              (1.3, 8.6, 0.3), (1.6, 9.3, 0.0), (1.1, 7.6, 0.0),
              (1.4, 8.9, 0.6), (1.9, 6.9, 0.0), (1.0, 9.1, 0.0),
              (1.5, 7.8, 0.2), (1.2, 8.3, 0.0), (1.7, 9.4, 0.0))
    x, index = -0.8, board % len(spines)
    while x < NATIVE_CELLS:
        width, height, lit = spines[index % len(spines)]
        top = board - height
        tone = mix(ink.book, ink.book_lit, lit)
        # Spines are lit from the front, so each one is a shade darker down
        # its own depth -- which is what makes a shelf read as objects with
        # gaps between them rather than as a striped rectangle.
        s.rect(x, top, x + width, board,
               lambda v, t=tone: mix(shade(t, -0.35), t, min(1.0, v * 1.6)))
        s.rect(x + width - 0.35, top, x + width, board, shade(tone, -0.5))
        s.rect(x, top, x + width, top + 0.5, shade(tone, 0.12))
        x += width + 0.28
        index += 1


def _shelf_things(s: Scene) -> None:
    """A plant and a leaning stack, because a shelf of nothing but spines is
    a texture and a shelf with two objects on it is a room."""
    ink = s.ink
    leaf = mix(ink.book_lit, ink.key, 0.25)
    s.ellipse(38.5, 12.6, 1.5, 1.1, shade(ink.shelf, -0.2))    # the pot
    for angle, length in ((-2.5, 4.2), (-1.9, 5.0), (-1.3, 4.4), (-0.8, 3.4)):
        s.strand(38.5, 11.6, angle, length, 0.42, leaf, 0.85, curve=0.4)
    for i, (width, lean) in enumerate(((3.4, 0.0), (3.0, 0.25), (2.6, -0.2))):
        y = 30.4 - i * 1.15
        s.poly([(5.0, y), (5.0 + width, y - lean), (5.0 + width, y - lean + 1.0),
                (5.0, y + 1.0)], mix(ink.book, ink.book_lit, 0.2 + i * 0.2))


def _screen_light(s: Scene) -> None:
    """What the laptop throws into the room before anything is in the way.

    Additive and wide: the wall behind him is not lit evenly, it has a
    lantern in front of it, and that single gradient does most of the work of
    separating a black character from a black room.
    """
    ink = s.ink
    s.glow(23, 52, 30, 26, ink.key, 0.20, falloff=2.4)
    s.glow(23, 49, 15, 12, ink.key, 0.14, falloff=1.8)


# -- the figure --------------------------------------------------------------

def _torso(s: Scene) -> None:
    """The hoodie he is sitting in: shoulders, hood, collar, folds.

    Painted rather than filled. The chest can see the screen and the sides
    cannot, so it is a body in a room and not a silhouette pasted on one.
    """
    ink = s.ink
    # A rim first, then the shoulders over it, so the light is left as a ring
    # along the outer contour. Without it the hoodie and the wall are two
    # near-blacks meeting and the figure has no silhouette -- the difference
    # between a character in a room and a hole in the middle of one.
    #
    # Three soft passes rather than one bright one. A single offset ring is a
    # stroke, and a stroke round a limb is a wireframe: the first attempt
    # traced both arms in a hard violet line.
    for grow, alpha in ((2.4, 0.10), (1.5, 0.14), (0.7, 0.18)):
        s.ellipse(23, 84, 23.5 + grow, 43 + grow,
                  mix(ink.hood, ink.key, 0.30), alpha)
    s.ellipse(23, 84, 23.5, 43, ink.hood)                   # from about y 41
    s.ellipse(23, 43.2, 9.2, 6.0, shade(ink.hood, -0.25))   # the hood, bunched
    s.ellipse(14.4, 43.0, 6.2, 4.2, shade(ink.hood, -0.35), angle=-0.5)
    s.ellipse(31.6, 43.0, 6.2, 4.2, shade(ink.hood, -0.35), angle=0.5)

    # The screen is below him, so the front of the chest is lit and the
    # shoulders are not. Three soft passes, widest and dimmest first.
    s.ellipse(23, 60, 15, 15, ink.hood_lit, 0.20)
    s.ellipse(23, 57, 10, 10, ink.hood_lit, 0.18)
    s.ellipse(23, 54, 6, 6, mix(ink.hood_lit, ink.key, 0.3), 0.16)

    # The collar, and the shadow the jaw drops onto it.
    s.ellipse(23, 40.4, 7.6, 2.5, mix(ink.hood, ink.seam, 0.45))
    s.ellipse(23, 39.4, 6.8, 1.9, shade(ink.hood, -0.45))
    s.ellipse(23, 37.9, 5.0, 2.2, ink.hood_deep, 0.75)

    # Folds. A hoodie without them is a bag; with them it is worn.
    for x0, y0, x1, y1, wide, tone in (
            (16.0, 50.0, 14.2, 62.0, 0.55, -0.40),
            (30.0, 50.0, 31.8, 62.0, 0.55, -0.40),
            (19.5, 52.0, 18.8, 63.0, 0.40, -0.28),
            (26.5, 52.0, 27.2, 63.0, 0.40, -0.28),
            (21.5, 55.0, 21.2, 63.0, 0.32, 0.16),
            (24.5, 55.0, 24.8, 63.0, 0.32, 0.16)):
        s.capsule(x0, y0, x1, y1, wide, wide * 0.7,
                  shade(ink.hood, tone) if tone < 0
                  else mix(ink.hood, ink.hood_lit, tone), 0.55)

    # Drawstrings, with an aglet on each, so the hood reads as clothing.
    for x, drop in ((20.4, 51.0), (25.6, 48.6)):
        s.capsule(x, 41.2, x + (0.4 if x > 23 else -0.4), drop,
                  0.30, 0.26, mix(ink.seam, ink.hood_lit, 0.4))
        s.ellipse(x + (0.4 if x > 23 else -0.4), drop, 0.34, 0.7,
                  shade(ink.seam, 0.2))


def _arms(s: Scene) -> None:
    """One arm up to the face, one down to the keys.

    The pose is most of the character -- it is the difference between someone
    working and someone posing for a portrait. The hand at the face is
    painted later, after the head, because it is in front of it.
    """
    ink = s.ink
    sleeve = shade(ink.hood, -0.14)
    # Left: shoulder, elbow on the desk, forearm rising back to the cheek.
    # No rim on the arms: they cross the hoodie rather than the wall, so an
    # edge on them has nothing to separate them from and reads as a drawn
    # outline. What tells them from the body is the shadow under them.
    s.capsule(12.7, 50.4, 8.3, 58.9, 3.9, 3.4, ink.hood_deep, 0.55)
    s.capsule(8.5, 58.4, 16.7, 35.8, 3.4, 2.6, ink.hood_deep, 0.55)
    s.capsule(12.5, 50.0, 8.0, 58.5, 3.7, 3.2, sleeve)
    s.capsule(8.2, 58.0, 16.4, 35.4, 3.2, 2.4, sleeve)
    # No highlight down the forearm at all. Every version of one -- a
    # capsule, then a long thin ellipse -- came out as a bright wire lying
    # across the picture, because a narrow light shape on a dark ground is a
    # line whatever it was meant to be. The arm is dark, and reads.
    s.ellipse(9.6, 56.6, 3.2, 2.2, shade(ink.hood, -0.30), angle=0.5)
    # The cuff, where the sleeve stops and the wrist starts.
    s.ellipse(16.2, 35.8, 2.3, 1.5, mix(ink.hood, ink.seam, 0.45),
              angle=-1.25)

    # Right: over the shoulder and down behind the laptop.
    s.capsule(33.3, 49.9, 37.0, 57.9, 3.8, 3.2, ink.hood_deep, 0.55)
    s.capsule(33.5, 49.5, 37.2, 57.5, 3.6, 3.0, sleeve)
    s.capsule(35.0, 50.5, 37.6, 57.0, 1.1, 0.8,
              mix(ink.hood, ink.hood_lit, 0.45), 0.5)
    s.capsule(37.0, 57.0, 32.5, 59.4, 2.6, 2.0, sleeve)


def _head(s: Scene) -> None:
    """Skull, jaw, ears, hair and face -- in that order, front to back.

    Two things keep this a person at terminal resolution. The eyes sit below
    a forehead rather than in the middle of a circle, and the face narrows to
    a jaw rather than ending in the curve it started with. Drop either and it
    reads as an animal's head however human the rest of the drawing is.
    """
    ink = s.ink
    with s.part("neck"):
        s.ellipse(23, 36.6, 3.0, 3.2, ink.skin_shadow)      # the neck
        s.ellipse(23, 38.4, 3.4, 1.7, ink.skin_deep, 0.8)   # under the jaw

    with s.part("hair"):
        _hair_back(s)
    with s.part("ears"):
        _ears(s)

    # The face: a cranium and a jaw that tapers to a chin.
    with s.part("face"):
        s.ellipse(23, 22.5, 8.0, 9.0, ink.skin, squash_top=0.92)
        s.ellipse(23, 27.0, 6.8, 6.6, ink.skin)
        s.poly([(16.8, 28.2), (29.2, 28.2), (26.4, 32.8), (23, 34.0),
                (19.6, 32.8)], ink.skin)

    with s.part("face"):
        _face_light(s)
    with s.part("features"):
        _face(s)
    with s.part("fringe"):
        _hair_front(s)


def _hair_back(s: Scene) -> None:
    """The mass behind the head, and the locks falling past the jaw."""
    ink = s.ink
    s.ellipse(23, 21.4, 12.3, 11.6, ink.hair, squash_top=0.95)
    s.poly([(10.9, 18), (16.6, 18), (18.2, 33), (16.0, 40), (11.3, 32)],
           ink.hair)
    s.poly([(35.1, 18), (29.4, 18), (27.8, 32), (30.0, 39), (34.7, 31)],
           ink.hair)
    # A cool edge along the crown, from whatever is behind him. One rim, and
    # a narrow one: outlining every shape is how a painting turns into a
    # diagram of itself.
    for offset, alpha in ((0.0, 0.20), (0.5, 0.10)):
        s.ellipse(23, 21.4 + offset, 12.3, 11.6, ink.hair_rim, alpha,
                  squash_top=0.95)
        s.ellipse(23, 22.5 + offset, 11.9, 11.4, ink.hair, 1.0,
                  squash_top=0.95)
    for x, angle, length in ((12.9, 1.42, 15.0), (33.1, 1.72, 14.0),
                             (11.8, 1.50, 11.0), (34.2, 1.64, 10.5)):
        s.strand(x, 22, angle, length, 1.5, ink.hair, 1.0, curve=0.12)
        s.strand(x + (0.5 if x > 23 else -0.5), 22, angle, length * 0.7,
                 0.42, ink.hair_lit, 0.35, curve=0.12)


def _ears(s: Scene) -> None:
    """Cat ears: short, wide, set on the skull, with fur in them.

    Long thin ones read as horns, which is what the pass before this drew.
    """
    ink = s.ink
    for flip in (-1, 1):
        cx = 23 + flip * 8.0
        tip = (cx + flip * 1.7, 5.2)
        s.poly([(cx - flip * 4.4, 15.4), (cx + flip * 3.6, 13.6), tip],
               ink.hair)
        s.poly([(cx - flip * 2.5, 13.6), (cx + flip * 2.1, 12.6),
                (tip[0] - flip * 0.2, 8.0)], ink.ear)
        s.poly([(cx - flip * 1.6, 13.0), (cx + flip * 1.2, 12.2),
                (tip[0] - flip * 0.3, 9.2)], mix(ink.ear, ink.hair_rim, 0.35),
               0.6)
        # Tufts, out of the fold and along the leading edge.
        # Two tufts, short, lying along the fold rather than standing off
        # it. Three long ones stood up over the crown and read as whiskers
        # growing out of the top of his head.
        for t, length in ((0.35, 1.7), (0.62, 1.3)):
            s.strand(cx - flip * 2.6 + flip * 3.8 * t, 14.4 - 7.2 * t,
                     -1.25 + flip * 0.55, length, 0.26, ink.hair_lit, 0.40,
                     curve=flip * 0.25)
        s.strand(tip[0], tip[1] + 0.6, 1.6 - flip * 0.2, 2.6, 0.45, ink.hair)


def _face_light(s: Scene) -> None:
    """Where the screen reaches him, and where it does not.

    The whole reason the face reads as a head. There is one light and it is
    below him, so the jaw and the underside of the nose are lit and the
    forehead is not -- and between the two there has to be modelling, or the
    middle of the face is a flat pale slab with features drawn on it, which
    is what the pass before this one produced.
    """
    ink = s.ink
    # Form shadow: the top of the face, both sides, and the hollow under
    # each cheekbone. The hollows are what give a face a structure rather
    # than an outline.
    s.ellipse(23, 17.5, 8.0, 5.4, ink.skin_shadow, 0.60)
    s.ellipse(16.9, 25.0, 2.8, 6.5, ink.skin_shadow, 0.34)
    s.ellipse(29.1, 25.0, 2.8, 6.5, ink.skin_shadow, 0.34)
    s.ellipse(23, 20.5, 7.0, 3.4, ink.skin_deep, 0.24)      # under the fringe
    for flip in (-1, 1):
        s.ellipse(23 + flip * 5.4, 30.2, 2.2, 2.4, ink.skin_shadow, 0.34)
        s.ellipse(23 + flip * 6.4, 28.6, 1.6, 2.4, ink.skin_deep, 0.22)
    s.ellipse(23, 30.9, 3.2, 1.3, ink.skin_shadow, 0.26)    # under the nose

    # The shadow the jaw drops on the neck. Without it the chin and the
    # throat are one pale column and the head has no underside -- the single
    # thing that most stops a face reading as a mask.
    s.ellipse(23, 35.4, 4.6, 2.1, ink.skin_deep, 0.70)
    s.ellipse(23, 34.2, 3.6, 1.2, ink.skin_deep, 0.55)

    # Key light, from below. Widest and weakest first, so it ramps -- and
    # none of it strong enough to flatten what the shadows just built.
    s.ellipse(23, 32.4, 5.8, 3.8, mix(ink.skin, ink.key, 0.30), 0.24)
    s.ellipse(23, 32.8, 3.8, 2.4, mix(ink.skin, ink.key, 0.44), 0.22)
    s.ellipse(23, 33.2, 2.2, 1.3, mix(ink.skin, ink.key, 0.56), 0.20)
    for flip in (-1, 1):                                    # cheekbones
        s.ellipse(23 + flip * 4.2, 28.2, 2.3, 1.6,
                  mix(ink.skin, ink.key, 0.22), 0.24)
        s.ellipse(23 + flip * 4.4, 29.4, 1.7, 1.1, ink.blush, 0.20)


def _face(s: Scene) -> None:
    """Brows, eyes, nose and mouth."""
    ink = s.ink
    for flip in (-1, 1):
        with s.part("eye left" if flip < 0 else "eye right"):
            _eye(s, 23 + flip * 3.9, 25.4, flip)
        # A brow runs outward and lifts at its far end, and most of it is
        # under the fringe. Drawn from the inside out so the thick end is
        # the end nearest the nose, which is where a brow is thickest.
        s.strand(23 + flip * 1.7, 21.9, 0.16 * flip if flip > 0 else
                 3.1416 - 0.16, 4.1, 0.28, ink.lash, 0.50,
                 curve=-0.30 * flip)

    # Nose: one soft shadow down its far side and a lit tip. That is all a
    # nose is at this size, and it is the whole of what makes the face turn
    # towards you. Two dark dots for nostrils, which is what stood here
    # first, reads as a punctuation mark sitting on his face.
    s.ellipse(23.7, 28.7, 0.60, 1.45, ink.skin_shadow, 0.70, angle=0.18)
    s.ellipse(23.2, 29.7, 0.95, 0.48, shade(ink.skin_shadow, -0.25), 0.60)
    s.ellipse(22.7, 28.9, 0.62, 0.60, mix(ink.skin, ink.key, 0.60), 0.70)

    # Mouth: a soft line with a lift at one end, a shadow under the lower
    # lip, and a highlight on it. A straight dark bar reads as a slot.
    s.capsule(21.1, 31.7, 23.0, 31.95, 0.36, 0.40, shade(ink.mouth, -0.30))
    s.capsule(23.0, 31.95, 25.0, 31.5, 0.40, 0.28, shade(ink.mouth, -0.30))
    s.ellipse(23.1, 32.8, 1.7, 0.55, ink.skin_shadow, 0.50)
    s.ellipse(23.1, 32.5, 1.3, 0.36, mix(ink.skin, ink.key, 0.65), 0.65)


def _eye(s: Scene, cx: float, cy: float, flip: int) -> None:
    """One eye, in the order an eye is painted.

    Socket, sclera, iris, a limbal ring, light in the bottom of the iris
    because the light is below him, pupil, lash line, and two specular
    points. It is six pixels across when it reaches the terminal and every
    one of those steps still shows in it -- that is what supersampling is
    for, and it is the difference between a character who is looking at you
    and one with two dots on his face.
    """
    ink = s.ink
    s.ellipse(cx, cy + 0.5, 2.5, 2.0, ink.skin_deep, 0.45)      # the socket
    s.ellipse(cx, cy, 2.30, 1.95, ink.sclera)
    s.ellipse(cx, cy - 0.70, 2.30, 1.10, shade(ink.sclera, -0.34), 0.45)

    s.ellipse(cx + flip * 0.12, cy + 0.15, 1.46, 1.56, ink.iris)
    s.ellipse(cx + flip * 0.12, cy + 0.60, 1.32, 1.05, ink.iris_lit, 0.65)
    s.ellipse(cx + flip * 0.12, cy + 0.15, 1.46, 1.56,
              shade(ink.iris, -0.55), 0.40)                     # limbal ring
    s.ellipse(cx + flip * 0.12, cy + 0.15, 1.14, 1.24, ink.iris)
    s.ellipse(cx + flip * 0.12, cy + 0.66, 1.00, 0.68, ink.iris_lit, 0.55)
    s.ellipse(cx + flip * 0.12, cy + 0.20, 0.58, 0.68, ink.pupil)

    # Lids. The upper is a real line and the lower is barely there, which is
    # the whole difference between an open eye and a drawn circle.
    s.capsule(cx - 2.4, cy - 1.78, cx + 2.4, cy - 1.78, 0.46, 0.30,
              ink.lash, 0.92)
    s.capsule(cx + flip * 1.9, cy - 1.72, cx + flip * 2.9, cy - 1.24,
              0.32, 0.15, ink.lash, 0.80)
    s.capsule(cx - 1.9, cy + 1.66, cx + 1.9, cy + 1.66, 0.22, 0.16,
              mix(ink.skin, ink.key, 0.35), 0.45)

    s.ellipse(cx - 0.68, cy - 0.48, 0.46, 0.40, ink.screen, 0.95)
    s.ellipse(cx + 0.80, cy + 0.86, 0.26, 0.22, ink.screen, 0.55)


def _hair_front(s: Scene) -> None:
    """The fringe: strands, not a shape.

    Uneven lengths, two of them reaching the eyes, a couple catching the rim
    light. Even strands are a helmet, a solid block is a hat, and this is the
    single thing that most decides whether the drawing looks made or drawn.
    """
    ink = s.ink
    fringe = (
        (13.4, 16.0, 1.30, 9.0, 1.5, 0.30),
        (15.2, 13.6, 1.42, 11.5, 1.7, 0.22),
        (17.4, 12.0, 1.52, 13.0, 1.6, 0.14),
        (19.6, 11.2, 1.62, 10.0, 1.5, 0.10),
        (21.8, 10.9, 1.70, 12.5, 1.4, 0.04),
        (23.9, 10.9, 1.52, 9.5, 1.4, -0.10),
        (26.0, 11.4, 1.44, 12.0, 1.5, -0.16),
        (28.2, 12.4, 1.34, 10.0, 1.6, -0.22),
        (30.2, 14.2, 1.24, 12.0, 1.7, -0.30),
        (32.0, 16.4, 1.16, 8.5, 1.5, -0.34),
    )
    for x, y, angle, length, width, curve in fringe:
        s.strand(x, y, angle, length, width, ink.hair, 1.0, curve=curve)
    # Finer strands over the top of those. Kept short: run to full length
    # they cross the forehead and the eyes and read as scratches on the
    # face rather than as hair in front of it.
    for x, y, angle, length, width, curve in fringe:
        s.strand(x + 0.6, y - 0.4, angle - 0.06, length * 0.42, width * 0.32,
                 ink.hair_lit, 0.26, curve=curve)
    for x, y, angle, length in ((16.2, 12.6, 1.40, 4.0),
                                (22.8, 10.6, 1.62, 3.4),
                                (29.2, 13.0, 1.36, 3.8)):
        s.strand(x, y, angle, length, 0.28, ink.hair_rim, 0.34, curve=-0.1)
    # A cowlick, because hair described as messy has to be untidy somewhere.
    s.strand(24.6, 9.6, -1.1, 4.6, 0.75, ink.hair, 1.0, curve=0.9)
    s.strand(21.0, 9.4, -2.1, 3.6, 0.62, ink.hair, 1.0, curve=-0.8)


def _cheek_hand(s: Scene) -> None:
    """The hand his face is resting on -- in front of the hair, not under it.

    Painted after the head for that reason alone. Put with the rest of the
    arm, the side lock paints straight over it and the forearm runs up to the
    jaw and stops: the pose reads as an arm that has been cut off rather than
    as someone leaning on his hand.

    A loose fist under the cheekbone, seen from its side: a palm, three
    curled fingers with a crease between each, a thumb up the jaw. Four long
    strokes down the cheek was the first attempt, and at this size four
    parallel lines on a circle is a barcode, not a hand.
    """
    ink = s.ink
    lit = mix(ink.skin, ink.key, 0.30)
    # The shadow it throws on the cheek behind it, first: a hand painted the
    # same value as the face it is resting against disappears into it.
    s.ellipse(20.0, 32.4, 2.2, 3.4, ink.skin_deep, 0.55, angle=-0.20)
    s.ellipse(17.6, 32.4, 3.2, 3.5, shade(ink.skin_shadow, -0.20), angle=-0.30)
    s.ellipse(17.4, 33.2, 2.7, 2.7, mix(ink.skin, ink.skin_shadow, 0.60))

    # Curled fingers, stacked across the fist rather than combed down it.
    for i, (cx, cy, tilt) in enumerate(((16.6, 30.9, -0.34),
                                        (16.9, 32.4, -0.26),
                                        (17.3, 33.9, -0.18))):
        s.ellipse(cx, cy, 1.9, 0.86, mix(ink.skin, ink.skin_shadow, 0.45),
                  angle=tilt)
        s.ellipse(cx - 0.15, cy - 0.28, 1.6, 0.42, lit, 0.55, angle=tilt)
        s.capsule(cx - 1.9, cy + 0.72, cx + 1.7, cy + 0.60, 0.20, 0.16,
                  ink.skin_deep, 0.55)
        if not i:
            s.ellipse(cx + 1.5, cy + 0.1, 0.55, 0.62, lit, 0.40)   # knuckle
    # The thumb, lying up the jaw -- lighter than the fingers, because it is
    # the part of the hand turned towards the screen.
    s.capsule(18.9, 35.0, 19.9, 30.4, 0.88, 0.60,
              mix(ink.skin, ink.skin_shadow, 0.30))
    s.capsule(18.7, 34.6, 19.6, 31.0, 0.34, 0.24, lit, 0.60)
    s.ellipse(19.8, 30.5, 0.62, 0.50, lit, 0.45)
    # Where the cheek presses into it.
    s.ellipse(20.3, 31.4, 0.9, 2.4, ink.skin_deep, 0.45, angle=0.16)
    s.ellipse(17.4, 35.6, 2.3, 1.0, mix(ink.skin, ink.key, 0.40), 0.45)


# -- the desk ----------------------------------------------------------------

def _laptop(s: Scene) -> None:
    """Open, seen from behind, its screen the only light in the room.

    The lid crosses in front of his chest -- it is between him and the viewer,
    which is what makes this a desk he is sitting at rather than a prop
    standing beside him. Its back is unlit, so it is the darkest large shape
    in the frame after his hair, and only its top edge carries any light.
    """
    ink = s.ink
    s.poly([(6.5, 62.6), (39.5, 62.6), (41.5, 64), (4.5, 64)],
           shade(ink.desk, -0.40))                         # the desk's edge
    s.poly([(10.0, 59.0), (36.0, 59.0), (39.5, 62.8), (6.5, 62.8)],
           lambda v: mix(ink.desk_lit, ink.desk, v ** 0.6))
    s.poly([(13.0, 59.4), (33.0, 59.4), (35.4, 62.2), (10.6, 62.2)],
           shade(ink.laptop, 0.12))                        # the keyboard
    for i in range(4):                                     # rows of keys
        y = 59.8 + i * 0.62
        s.rect(13.6 + i * 0.4, y, 32.4 - i * 0.4, y + 0.26,
               shade(ink.laptop, -0.40), 0.55)
    s.ellipse(23, 62.4, 3.2, 0.5, shade(ink.laptop, 0.18), 0.7)   # trackpad

    s.poly([(12.0, 49.0), (34.0, 49.0), (36.0, 59.2), (10.0, 59.2)],
           lambda v: mix(shade(ink.laptop, 0.12), ink.laptop, v ** 0.45))
    # The lit edge is where light escapes past the lid, not a strip light:
    # two passes, the inner one narrow and the outer one soft and dim.
    s.poly([(12.0, 49.0), (34.0, 49.0), (34.1, 49.5), (11.9, 49.5)],
           ink.laptop_edge, 0.85)
    s.poly([(11.8, 49.5), (34.2, 49.5), (34.4, 50.4), (11.6, 50.4)],
           ink.laptop_edge, 0.22)
    s.ellipse(23, 54.4, 1.4, 1.5, mix(ink.laptop_edge, ink.laptop, 0.45), 0.7)
    s.ellipse(23, 54.4, 0.62, 0.68, shade(ink.laptop, -0.25), 0.9)

    # The hand on the keys.
    s.art.pen = "keyboard hand"
    s.ellipse(29.4, 60.2, 2.5, 1.2, ink.skin_shadow, angle=0.22)
    s.ellipse(29.2, 59.9, 2.2, 0.9, mix(ink.skin, ink.key, 0.22), angle=0.22)
    for x, length in ((28.0, 1.4), (29.0, 1.7), (30.0, 1.6), (30.9, 1.2)):
        s.strand(x, 60.4, 1.5, length, 0.34, ink.skin_shadow, 0.75, curve=0.12)
        s.capsule(x - 0.5, 60.5, x - 0.5, 60.5 + length * 0.7, 0.10, 0.08,
                  ink.skin_deep, 0.45)


def _bloom(s: Scene) -> None:
    """The light coming over the top of the lid, and what it lands on.

    Last, over everything, because bloom is what a lens does and not what a
    surface is. It is also the only thing in the frame allowed to be bright,
    which is what makes the rest of it read as dark rather than as grey.
    """
    ink = s.ink
    s.glow(23, 49.0, 15, 3.6, ink.screen, 0.26, falloff=1.8)
    s.glow(23, 49.0, 26, 9.0, ink.key, 0.13, falloff=2.4)
    s.glow(23, 61.4, 14, 2.6, ink.key, 0.08, falloff=2.0)


# -- what the layout asks for ------------------------------------------------

@lru_cache(maxsize=8)
def _painted(cells: int, palette_name: str, ink: Ink) -> tuple[Text, ...]:
    """One render per size and theme, kept.

    Painting is a couple of hundred milliseconds -- cheap once and expensive
    on every repaint, and this is redrawn whenever the terminal changes width.
    The palette's name is in the key as well as its colours because two
    themes could in principle mix to the same ink and should still be told
    apart in a cache the user can reason about.
    """
    art = paint(cells, ink).resampled(SUPERSAMPLE, SUPERSAMPLE * 2)
    return tuple(art.rows())


def parts(cells: int, palette) -> dict[str, list[tuple[int, int]]]:
    """Which pixels of the painting each named part ended up owning.

    For the tests that check the composition rather than the colours. The
    coordinates are the painting's own, before it is squeezed onto cells.
    """
    return paint(cells, Ink.of(palette), labelled=True).parts()


def rows(cells: int, palette) -> list[Text]:
    """The scene as terminal rows, painted in the palette in force."""
    return [row.copy() for row in
            _painted(max(MIN_CELLS, min(MAX_CELLS, cells)),
                     palette.name, Ink.of(palette))]


def fits(width: int, unicode_ok: bool) -> bool:
    """Whether the illustration can be drawn honestly at this width.

    The same locale gate the small sprite uses: the half-block glyphs are
    East Asian Width Ambiguous, so a CJK locale may draw them two cells wide
    and every row of the picture would be twice the width the layout measured
    for it.
    """
    from .sprite import _ambiguous_is_wide

    return unicode_ok and width >= MIN_CELLS and not _ambiguous_is_wide()
