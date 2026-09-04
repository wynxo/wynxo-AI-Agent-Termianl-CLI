"""Pixel art for a terminal, two pixels to a cell.

The technique is the one ``sprite.py`` already uses for the small animated
companion: a cell is split with ``▀``/``▄``, the top pixel drawn as the
foreground and the bottom as the background, so a terminal row holds two
rows of picture. What is new here is that the drawing is *resolution
independent*. Everything is placed in normalised 0..1 coordinates and
rasterised into whatever canvas the layout can afford, so the same artwork
is a crisp 24-row illustration on a tall terminal and a crisp 14-row one on
a laptop, rather than a fixed grid that has to be cropped or doubled.

Two rules the drawing side depends on:

* A pixel here is very nearly square. A terminal cell is about 2.15 times
  taller than it is wide and a half-block pixel is half a cell tall, so the
  pixel aspect is ~1.07. Normalised coordinates can therefore be treated as
  square without the figure coming out stretched -- which is the usual
  reason homemade terminal art looks melted.
* Nothing is painted where the picture is transparent. A cell whose two
  pixels are both empty stays a space with no style at all, so the art sits
  *on* the terminal background instead of on a coloured rectangle of its
  own.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

EMPTY = " "
"""The transparent ink. Never drawn, never styled."""

GLYPHS = " ▀▄█"
"""The whole vocabulary. All four are one cell wide."""


# -- colour ------------------------------------------------------------------
#
# The palette gives roles, not shades, and a dark illustration needs more
# steps than a palette has roles: hair, hoodie, wall and desk are four
# different near-blacks. Rather than invent a second palette, the shades are
# mixed from the palette's own colours, so the artwork follows /theme
# instead of only agreeing with the default one.


def is_hex(colour: str) -> bool:
    """Whether a style string is a colour we can do arithmetic on.

    The plain and minimal palettes name ANSI colours -- "bright_magenta",
    "default" -- which have no numeric value to mix. Blending has to be
    skipped there rather than guessed at, and the caller falls back to the
    role itself.
    """
    colour = (colour or "").strip()
    return len(colour) == 7 and colour.startswith("#") and \
        all(c in "0123456789abcdefABCDEF" for c in colour[1:])


def _rgb(colour: str) -> tuple[int, int, int]:
    return (int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16))


def mix(a: str, b: str, t: float) -> str:
    """``a`` blended ``t`` of the way towards ``b``.

    Returns ``a`` unchanged when either side is not a hex colour, so a
    16-colour palette degrades to flat roles rather than to an exception.
    """
    if not (is_hex(a) and is_hex(b)):
        return a
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    left, right = _rgb(a), _rgb(b)
    return "#" + "".join(
        f"{round(l + (r - l) * t):02x}" for l, r in zip(left, right, strict=True)
    )


BLACK = "#000000"
WHITE = "#ffffff"


def darken(colour: str, amount: float) -> str:
    return mix(colour, BLACK, amount)


def lighten(colour: str, amount: float) -> str:
    return mix(colour, WHITE, amount)


# -- the canvas --------------------------------------------------------------

# An ordered dither matrix. Dithering is what gives the artwork its texture
# -- a wall that fades, a glow that falls off -- without gradients, which a
# terminal cannot do and which this design has ruled out anyway.
#
# 8x8 rather than the usual 4x4: a 4x4 matrix has sixteen levels, and at the
# low densities this artwork uses that quantises to "every other pixel on
# this row and none on the next", which reads as banding rather than as
# texture. Sixty-four levels put the threshold somewhere useful.
def _bayer(order: int) -> tuple[tuple[int, ...], ...]:
    """The standard recursive construction, doubled ``order`` times.

    Written out rather than typed out. The hand-written 8x8 that stood here
    first was not a permutation of 0..63 at all -- it clustered into 4x4
    blocks, so every dithered surface came out in visible squares. A matrix
    is easier to derive than to proofread.
    """
    matrix: tuple[tuple[int, ...], ...] = ((0,),)
    for _ in range(order):
        n = len(matrix)
        matrix = tuple(
            tuple(
                4 * matrix[y % n][x % n] + (0, 2, 3, 1)[(y // n) * 2 + (x // n)]
                for x in range(n * 2)
            )
            for y in range(n * 2)
        )
    return matrix


_BAYER = _bayer(3)
_LEVELS = float(len(_BAYER) ** 2)
_SIZE = len(_BAYER)

@dataclass
class Canvas:
    """A grid of ink codes, addressed in normalised coordinates.

    ``width`` and ``height`` are pixels, not cells: a canvas is rendered as
    ``height // 2`` terminal rows.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        self.px: list[list[str]] = [[EMPTY] * self.width
                                    for _ in range(self.height)]

    # -- coordinates ---------------------------------------------------
    def x(self, u: float) -> int:
        return int(round(u * (self.width - 1)))

    def y(self, v: float) -> int:
        return int(round(v * (self.height - 1)))

    def _span(self, u0: float, u1: float) -> tuple[int, int]:
        a, b = self.x(u0), self.x(u1)
        return (a, b) if a <= b else (b, a)

    def _rows(self, v0: float, v1: float) -> tuple[int, int]:
        a, b = self.y(v0), self.y(v1)
        return (a, b) if a <= b else (b, a)

    # -- primitives ----------------------------------------------------
    def put(self, x: int, y: int, ink: str) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.px[y][x] = ink

    def rect(self, u0: float, v0: float, u1: float, v1: float,
             ink: str) -> None:
        x0, x1 = self._span(u0, u1)
        y0, y1 = self._rows(v0, v1)
        for y in range(max(0, y0), min(self.height, y1 + 1)):
            row = self.px[y]
            for x in range(max(0, x0), min(self.width, x1 + 1)):
                row[x] = ink

    def ellipse(self, cu: float, cv: float, ru: float, rv: float, ink: str,
                *, squash_top: float = 1.0) -> None:
        """A filled ellipse.

        ``squash_top`` scales the upper half's radius only, which is how a
        head gets a flatter crown than chin without becoming two shapes.
        """
        cx, cy = self.x(cu), self.y(cv)
        rx = max(1.0, ru * (self.width - 1))
        ry = max(1.0, rv * (self.height - 1))
        for y in range(max(0, int(cy - ry)), min(self.height, int(cy + ry) + 1)):
            radius = ry * (squash_top if y < cy else 1.0)
            dy = (y - cy) / radius
            if abs(dy) > 1:
                continue
            half = rx * (1 - dy * dy) ** 0.5
            for x in range(max(0, int(cx - half)), min(self.width, int(cx + half) + 1)):
                self.px[y][x] = ink

    def poly(self, points: list[tuple[float, float]], ink: str) -> None:
        """A filled polygon, by scanline. Used for the ears and the desk."""
        pts = [(self.x(u), self.y(v)) for u, v in points]
        if len(pts) < 3:
            return
        top = max(0, min(p[1] for p in pts))
        bottom = min(self.height - 1, max(p[1] for p in pts))
        for y in range(top, bottom + 1):
            crossings: list[float] = []
            for i, (x0, y0) in enumerate(pts):
                x1, y1 = pts[(i + 1) % len(pts)]
                if (y0 <= y < y1) or (y1 <= y < y0):
                    crossings.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            crossings.sort()
            for start, end in zip(crossings[0::2], crossings[1::2], strict=False):
                for x in range(max(0, int(round(start))),
                               min(self.width, int(round(end)) + 1)):
                    self.px[y][x] = ink

    def stroke(self, u0: float, v0: float, u1: float, v1: float, ink: str,
               *, thickness: float = 0.02, taper: float = 1.0) -> None:
        """A thick line, for limbs. ``taper`` thins it towards the end."""
        x0, y0 = self.x(u0), self.y(v0)
        x1, y1 = self.x(u1), self.y(v1)
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        wide = max(1.0, thickness * (self.width - 1))
        for i in range(steps + 1):
            t = i / steps
            cx = x0 + (x1 - x0) * t
            cy = y0 + (y1 - y0) * t
            radius = wide * (1 + (taper - 1) * t) / 2
            for y in range(int(cy - radius), int(cy + radius) + 1):
                span = radius * radius - (y - cy) ** 2
                if span < 0:
                    continue
                half = span ** 0.5
                for x in range(int(cx - half), int(cx + half) + 1):
                    self.put(x, y, ink)

    def dither(self, u0: float, v0: float, u1: float, v1: float, ink: str,
               density, *, over: str | None = None) -> None:
        """Scatter ``ink`` across a region at a varying density.

        ``density`` is either a number or a function of the normalised
        position within the region, returning 0..1. ``over`` restricts the
        scatter to pixels already holding that ink, which is how a fold or a
        glow lands on one surface without leaking onto the one behind it.
        """
        x0, x1 = self._span(u0, u1)
        y0, y1 = self._rows(v0, v1)
        span_x = max(1, x1 - x0)
        span_y = max(1, y1 - y0)
        constant = None if callable(density) else float(density)
        for y in range(max(0, y0), min(self.height, y1 + 1)):
            for x in range(max(0, x0), min(self.width, x1 + 1)):
                if over is not None and self.px[y][x] != over:
                    continue
                amount = constant if constant is not None else \
                    float(density((x - x0) / span_x, (y - y0) / span_y))
                if amount <= 0:
                    continue
                if amount >= 1 or _BAYER[y % _SIZE][x % _SIZE] / _LEVELS < amount:
                    self.px[y][x] = ink

    def outline(self, ink: str, edge: str, *, sides: str = "t") -> None:
        """Put ``edge`` on the lit boundary of every ``ink`` region.

        A one-pixel rim is what separates two near-blacks from each other in
        a picture that is mostly near-black; without it the hoodie and the
        wall are the same shape.
        """
        marks: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.px[y][x] != ink:
                    continue
                if "t" in sides and (y == 0 or self.px[y - 1][x] != ink):
                    marks.append((x, y))
                elif "l" in sides and (x == 0 or self.px[y][x - 1] != ink):
                    marks.append((x, y))
                elif "r" in sides and (x == self.width - 1
                                       or self.px[y][x + 1] != ink):
                    marks.append((x, y))
        for x, y in marks:
            self.px[y][x] = edge

    # -- output --------------------------------------------------------
    def rows(self, colour) -> list[Text]:
        """The canvas as terminal rows, two pixel rows to each.

        ``colour`` maps an ink code to a style string; an empty style means
        the ink is transparent.
        """
        out: list[Text] = []
        for top, bottom in zip(self.px[0::2], self.px[1::2], strict=True):
            line = Text(no_wrap=True)
            for a, b in zip(top, bottom, strict=True):
                glyph, style = _pack(colour(a), colour(b))
                line.append(glyph, style=style)
            out.append(line)
        return out


def _pack(top: str, bottom: str) -> tuple[str, str]:
    """One cell from two pixel colours. Empty styles are transparent."""
    if not top and not bottom:
        return " ", ""
    if not top:
        return "▄", bottom
    if not bottom:
        return "▀", top
    if top == bottom:
        return "█", top
    return "▀", f"{top} on {bottom}"
