"""A small software rasteriser, and the way a picture becomes terminal cells.

The interface draws its companion as an illustration rather than as an icon,
and an illustration at this size lives or dies on tone. A shape filled flat
reads as a shape; the same shape with light falling across it reads as an
object. So this is a painter's canvas -- float colour, alpha compositing,
soft-edged fills, blur -- and not a grid of coloured squares.

Two decisions carry the quality.

*Supersampling.* Everything is drawn at ``SUPERSAMPLE`` times the resolution
the terminal will show and averaged down at the end. That is what
anti-aliases every edge without a single line of edge-handling code, and it
is also what lets detail smaller than a terminal pixel -- a strand of hair, a
catchlight, the line of an eyelid -- survive as tone rather than vanishing.
Drawing at output resolution and rounding is what makes homemade terminal art
look like a mosaic.

*Rows, not pixels.* A fill is applied a scanline at a time with a slice
assignment, which is a memcpy rather than a Python loop, so a canvas can be
repainted a couple of hundred times and still be quick enough to draw at
start-up. Anything that has to vary along a row says so and pays for it.

The output is half-block cells: ``▀`` with the upper pixel as the foreground
and the lower as the background, so one terminal row carries two rows of
picture at full colour.
"""

from __future__ import annotations

from math import cos, sin

from rich.text import Text

SUPERSAMPLE = 2
"""Samples per device pixel in the narrow direction. Everything is drawn this
much larger and averaged down, which is what anti-aliases every edge without
a line of edge-handling code."""

QUADRANTS = {
    0b0001: "▘", 0b0010: "▝", 0b0011: "▀", 0b0100: "▖",
    0b0101: "▌", 0b0110: "▞", 0b0111: "▛", 0b1000: "▗",
    0b1001: "▚", 0b1010: "▐", 0b1011: "▜", 0b1100: "▄",
    0b1101: "▙", 0b1110: "▟", 0b1111: "█",
}
"""Which glyph lights which quarters of a cell: bit 0 top-left, then
top-right, bottom-left, bottom-right.

Half-blocks give a cell two pixels and two colours, which is exact but
coarse: at the width a terminal can spare, a face came out fifteen pixels
across and the eyes, the nose and the hand at the jaw were fighting over the
same four of them. Quadrants give a cell four pixels and still two colours,
so the horizontal resolution doubles at the cost of the two diagonal pixels
in a cell having to agree on a shade. On a picture this smooth that error is
invisible and the extra column of samples is not.

Every one of these lives in the same Block Elements range as ▀, so a font
that can draw the old output can draw this one."""

GLYPHS = " " + "".join(QUADRANTS.values())
"""The whole vocabulary. Every one is a single cell -- and every one is East
Asian Width Ambiguous, which is why the caller has to check the locale before
using any of them."""

Colour = tuple[float, float, float]


# -- colour ------------------------------------------------------------------
#
# Palettes give roles, not shades, and a painting needs a hundred shades. All
# of them are mixed from the palette in force, so the artwork follows /theme
# rather than merely agreeing with the default one.

def is_hex(colour: str) -> bool:
    """Whether a style string is a colour we can do arithmetic on.

    The plain and minimal palettes name ANSI colours -- "bright_magenta",
    "default" -- which have no numeric value to mix. Blending has to decline
    there rather than guess, and the caller falls back to flat roles.
    """
    colour = (colour or "").strip()
    return len(colour) == 7 and colour.startswith("#") and \
        all(c in "0123456789abcdefABCDEF" for c in colour[1:])


def rgb(colour: str) -> Colour:
    return (float(int(colour[1:3], 16)), float(int(colour[3:5], 16)),
            float(int(colour[5:7], 16)))


def hexed(colour: Colour) -> str:
    return "#" + "".join(f"{max(0, min(255, int(c + 0.5))):02x}"
                         for c in colour)


def mix(a: Colour, b: Colour, t: float) -> Colour:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


BLACK: Colour = (0.0, 0.0, 0.0)
WHITE: Colour = (255.0, 255.0, 255.0)


def shade(colour: Colour, amount: float) -> Colour:
    """Darker for a negative amount, lighter for a positive one.

    One call instead of two, because shading is a single axis in the artwork
    and writing it as two functions made every tonal ramp read as a list of
    unrelated colours.
    """
    return mix(colour, BLACK if amount < 0 else WHITE, abs(amount))


def saturate(colour: Colour, amount: float) -> Colour:
    grey = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
    return tuple(grey + (c - grey) * (1.0 + amount) for c in colour)


# -- the canvas --------------------------------------------------------------

class Raster:
    """A float RGBA image, painted a scanline at a time.

    Channels are held as four flat lists rather than as pixel tuples so a
    horizontal run of one colour is four slice assignments instead of a loop
    over pixels -- which is the difference between this drawing in a fifth of
    a second and in five.
    """

    __slots__ = ("w", "h", "r", "g", "b", "a", "label", "pen")

    def __init__(self, width: int, height: int, colour: Colour = BLACK,
                 alpha: float = 0.0, *, labelled: bool = False) -> None:
        self.w, self.h = width, height
        size = width * height
        self.r = [colour[0]] * size
        self.g = [colour[1]] * size
        self.b = [colour[2]] * size
        self.a = [alpha] * size
        self.label: list[str] | None = [""] * size if labelled else None
        """Which part of the drawing last painted each pixel, opaquely.

        Off by default and free when it is: a painting has no shapes left in
        it once it is painted, so there is nothing for a test to ask about
        except colours -- and "is the pixel at 19,25 the violet of an iris"
        is a test of the palette pretending to be a test of the composition.
        Turned on, the same paint calls that make the picture also record
        who made each pixel, so "he has two eyes and they are below the
        fringe" can be asked of the drawing rather than of a screenshot.
        """
        self.pen = ""

    # -- spans ---------------------------------------------------------
    def span(self, y: int, x0: float, x1: float, colour: Colour,
             alpha: float = 1.0) -> None:
        """One horizontal run, composited over what is there."""
        if y < 0 or y >= self.h or alpha <= 0.0:
            return
        left = max(0, int(x0 + 0.5))
        right = min(self.w, int(x1 + 0.5))
        if right <= left:
            return
        i, j = y * self.w + left, y * self.w + right
        n = right - left
        if self.label is not None and alpha >= 0.5:
            self.label[i:j] = [self.pen] * n
        if alpha >= 1.0:
            self.r[i:j] = [colour[0]] * n
            self.g[i:j] = [colour[1]] * n
            self.b[i:j] = [colour[2]] * n
            self.a[i:j] = [1.0] * n
            return
        keep = 1.0 - alpha
        cr, cg, cb = colour[0] * alpha, colour[1] * alpha, colour[2] * alpha
        self.r[i:j] = [v * keep + cr for v in self.r[i:j]]
        self.g[i:j] = [v * keep + cg for v in self.g[i:j]]
        self.b[i:j] = [v * keep + cb for v in self.b[i:j]]
        self.a[i:j] = [v * keep + alpha for v in self.a[i:j]]

    def point(self, x: float, y: float, colour: Colour,
              alpha: float = 1.0) -> None:
        self.span(int(y + 0.5), x, x + 1, colour, alpha)

    # -- shapes --------------------------------------------------------
    def rect(self, x0: float, y0: float, x1: float, y1: float,
             colour, alpha: float = 1.0) -> None:
        """A filled rectangle. ``colour`` may be a function of the vertical
        position within it, 0 at the top and 1 at the bottom."""
        top, bottom = int(y0 + 0.5), int(y1 + 0.5)
        height = max(1, bottom - top)
        varying = callable(colour)
        for y in range(top, bottom):
            tone = colour((y - top) / height) if varying else colour
            self.span(y, x0, x1, tone, alpha)

    def ellipse(self, cx: float, cy: float, rx: float, ry: float,
                colour, alpha: float = 1.0, *, squash_top: float = 1.0,
                angle: float = 0.0) -> None:
        """A filled ellipse, optionally flattened above its centre or turned.

        ``squash_top`` scales the upper radius alone, which is how a skull
        gets a flatter crown than jaw without becoming two shapes; ``angle``
        turns it, which is how a limb or a lock of hair gets a direction.
        """
        if angle:
            self._turned_ellipse(cx, cy, rx, ry, colour, alpha, angle)
            return
        varying = callable(colour)
        top = max(0, int(cy - ry * squash_top))
        bottom = min(self.h - 1, int(cy + ry + 1))
        for y in range(top, bottom + 1):
            radius = ry * (squash_top if y < cy else 1.0)
            if radius <= 0:
                continue
            dy = (y + 0.5 - cy) / radius
            if dy < -1.0 or dy > 1.0:
                continue
            half = rx * (1.0 - dy * dy) ** 0.5
            tone = colour((y + 0.5 - (cy - ry * squash_top))
                          / max(1e-6, ry * (1 + squash_top))) \
                if varying else colour
            self.span(y, cx - half, cx + half, tone, alpha)

    def _turned_ellipse(self, cx, cy, rx, ry, colour, alpha, angle) -> None:
        """The general case, evaluated per pixel because a turned ellipse has
        no closed-form span. Kept for limbs and locks, which are small."""
        tone = colour(0.5) if callable(colour) else colour
        ca, sa = cos(-angle), sin(-angle)
        reach = max(rx, ry) + 1
        for y in range(max(0, int(cy - reach)),
                       min(self.h, int(cy + reach) + 1)):
            dy = y + 0.5 - cy
            start = None
            for x in range(max(0, int(cx - reach)),
                           min(self.w, int(cx + reach) + 1)):
                dx = x + 0.5 - cx
                u = (dx * ca - dy * sa) / rx
                v = (dx * sa + dy * ca) / ry
                inside = u * u + v * v <= 1.0
                if inside and start is None:
                    start = x
                elif not inside and start is not None:
                    self.span(y, start, x, tone, alpha)
                    start = None
            if start is not None:
                self.span(y, start, self.w, tone, alpha)

    def capsule(self, x0: float, y0: float, x1: float, y1: float,
                r0: float, r1: float, colour, alpha: float = 1.0) -> None:
        """A thick line with round ends and a radius that may taper.

        Limbs, fingers, hair. Walked along its own length rather than
        rasterised as a rotated rectangle, which keeps the joint between two
        of them smooth instead of notched.
        """
        steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1)
        varying = callable(colour)
        for i in range(steps + 1):
            t = i / steps
            self.ellipse(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t,
                         r0 + (r1 - r0) * t, r0 + (r1 - r0) * t,
                         colour(t) if varying else colour, alpha)

    def poly(self, points, colour, alpha: float = 1.0) -> None:
        """A filled polygon, by scanline."""
        if len(points) < 3:
            return
        varying = callable(colour)
        top = max(0, int(min(p[1] for p in points)))
        bottom = min(self.h - 1, int(max(p[1] for p in points)) + 1)
        height = max(1, bottom - top)
        for y in range(top, bottom + 1):
            middle = y + 0.5
            crossings = []
            for index, (px, py) in enumerate(points):
                qx, qy = points[(index + 1) % len(points)]
                if (py <= middle < qy) or (qy <= middle < py):
                    crossings.append(px + (middle - py) * (qx - px) / (qy - py))
            crossings.sort()
            tone = colour((y - top) / height) if varying else colour
            # strict=False on purpose: a scanline through a vertex can
            # produce an odd number of crossings, and the honest answer to
            # a half-open span is to drop it rather than to raise.
            for start, end in zip(crossings[0::2], crossings[1::2],
                                  strict=False):
                self.span(y, start, end, tone, alpha)

    # -- light ---------------------------------------------------------
    def glow(self, cx: float, cy: float, rx: float, ry: float,
             colour: Colour, strength: float, falloff: float = 2.0) -> None:
        """Light added rather than painted, fading to nothing at the edge.

        Additive, so it brightens what is under it instead of tinting it a
        flat colour -- which is the difference between a lamp and a sticker
        of a lamp.
        """
        red, green, blue = colour
        top = max(0, int(cy - ry))
        bottom = min(self.h - 1, int(cy + ry))
        for y in range(top, bottom + 1):
            dy = (y + 0.5 - cy) / ry
            if dy < -1 or dy > 1:
                continue
            base = y * self.w
            left = max(0, int(cx - rx))
            right = min(self.w - 1, int(cx + rx))
            r, g, b, a = self.r, self.g, self.b, self.a
            for x in range(left, right + 1):
                dx = (x + 0.5 - cx) / rx
                d = dx * dx + dy * dy
                if d >= 1.0:
                    continue
                k = (1.0 - d) ** falloff * strength
                i = base + x
                r[i] += red * k
                g[i] += green * k
                b[i] += blue * k
                if a[i] < k:
                    a[i] = min(1.0, a[i] + k)

    def blur(self, radius: int, y0: int = 0, y1: int | None = None) -> None:
        """A separable box blur, twice, which is close enough to a gaussian.

        The background is behind the character and a camera would not have
        both in focus. Blurring it is the single cheapest thing that makes a
        flat drawing read as a room with depth in it.
        """
        if radius < 1:
            return
        y1 = self.h if y1 is None else y1
        for _ in range(2):
            self._blur_rows(radius, y0, y1)
            self._blur_columns(radius, y0, y1)

    def _blur_rows(self, radius: int, y0: int, y1: int) -> None:
        w = self.w
        for channel in (self.r, self.g, self.b, self.a):
            for y in range(y0, y1):
                base = y * w
                row = channel[base:base + w]
                out = []
                total = 0.0
                window = min(radius, w - 1)
                total = sum(row[:window + 1])
                count = window + 1
                for x in range(w):
                    out.append(total / count)
                    if x - radius >= 0:
                        total -= row[x - radius]
                        count -= 1
                    if x + radius + 1 < w:
                        total += row[x + radius + 1]
                        count += 1
                channel[base:base + w] = out

    def _blur_columns(self, radius: int, y0: int, y1: int) -> None:
        w = self.w
        for channel in (self.r, self.g, self.b, self.a):
            for x in range(w):
                column = channel[y0 * w + x:y1 * w + x:w]
                n = len(column)
                out = []
                window = min(radius, n - 1)
                total = sum(column[:window + 1])
                count = window + 1
                for y in range(n):
                    out.append(total / count)
                    if y - radius >= 0:
                        total -= column[y - radius]
                        count -= 1
                    if y + radius + 1 < n:
                        total += column[y + radius + 1]
                        count += 1
                channel[y0 * w + x:y1 * w + x:w] = out

    def parts(self) -> dict[str, list[tuple[int, int]]]:
        """Which pixels each named part of the drawing ended up owning."""
        found: dict[str, list[tuple[int, int]]] = {}
        if self.label is None:
            return found
        for i, name in enumerate(self.label):
            if name:
                found.setdefault(name, []).append((i % self.w, i // self.w))
        return found

    def over(self, other: "Raster") -> None:
        """Composite another canvas of the same size on top of this one."""
        r, g, b, a = self.r, self.g, self.b, self.a
        orr, og, ob, oa = other.r, other.g, other.b, other.a
        for i in range(len(a)):
            alpha = oa[i]
            if alpha <= 0.0:
                continue
            if alpha >= 1.0:
                r[i], g[i], b[i], a[i] = orr[i], og[i], ob[i], 1.0
                continue
            keep = 1.0 - alpha
            r[i] = r[i] * keep + orr[i] * alpha
            g[i] = g[i] * keep + og[i] * alpha
            b[i] = b[i] * keep + ob[i] * alpha
            a[i] = a[i] * keep + alpha

    def vignette(self, strength: float, fade: float) -> None:
        """Darken towards the frame, and dissolve the last of it entirely.

        The picture has to end somewhere, and a hard edge makes it read as a
        photograph pasted into the terminal. Falling away to nothing lets it
        sit in the conversation instead of on it.
        """
        w, h = self.w, self.h
        r, g, b, a = self.r, self.g, self.b, self.a
        edge = max(1.0, fade)
        for y in range(h):
            dy = min(y, h - 1 - y) / edge
            base = y * w
            for x in range(w):
                near = min(min(x, w - 1 - x) / edge, dy)
                if near >= 1.0:
                    continue
                k = near * near * (3 - 2 * near)          # smoothstep
                i = base + x
                dark = 1.0 - strength * (1.0 - k)
                r[i] *= dark
                g[i] *= dark
                b[i] *= dark
                a[i] *= k

    def grade(self, lift: Colour, gain: float, gamma: float) -> None:
        """The final pass every rendered picture gets: a little light in the
        shadows, a little contrast, and a curve."""
        r, g, b = self.r, self.g, self.b
        for channel, floor in ((r, lift[0]), (g, lift[1]), (b, lift[2])):
            for i, v in enumerate(channel):
                v = floor + v * gain
                v = 0.0 if v < 0.0 else 255.0 if v > 255.0 else v
                channel[i] = 255.0 * (v / 255.0) ** gamma

    # -- output --------------------------------------------------------
    def resampled(self, fx: int, fy: int) -> "Raster":
        """This canvas averaged down, which is where the anti-aliasing
        happens and where sub-pixel detail turns into tone.

        The two factors differ because the thing being drawn onto is not
        square: a quarter of a terminal cell is about twice as tall as it is
        wide, so the painting is done on square pixels and squeezed at the
        very end rather than composed on a stretched grid where every circle
        would have to be written as an ellipse.
        """
        w, h = self.w // fx, self.h // fy
        out = Raster(w, h)
        area = fx * fy
        sr, sg, sb, sa = self.r, self.g, self.b, self.a
        stride = self.w
        for y in range(h):
            for x in range(w):
                tr = tg = tb = ta = 0.0
                for dy in range(fy):
                    base = (y * fy + dy) * stride + x * fx
                    for dx in range(fx):
                        i = base + dx
                        alpha = sa[i]
                        tr += sr[i] * alpha
                        tg += sg[i] * alpha
                        tb += sb[i] * alpha
                        ta += alpha
                i = y * w + x
                if ta > 0.0:
                    out.r[i] = tr / ta
                    out.g[i] = tg / ta
                    out.b[i] = tb / ta
                out.a[i] = ta / area
        return out

    def rows(self, *, threshold: float = 0.12) -> list[Text]:
        """The picture as terminal rows, four pixels to a cell.

        A cell gets one foreground and one background, so its four pixels are
        split at their own mean brightness and each side is averaged. On a
        painting -- where four touching pixels are nearly the same colour --
        that is very close to lossless, and it buys twice the horizontal
        detail of a half-block.

        Where the picture has faded out, the transparent quarters are simply
        left unlit: the artwork dissolves into the conversation at its edges
        rather than ending in a rectangle of near-black that is not quite the
        terminal's own.
        """
        out: list[Text] = []
        r, g, b, a = self.r, self.g, self.b, self.a
        w = self.w
        for row in range(self.h // 2):
            line = Text(no_wrap=True)
            top, bottom = row * 2 * w, (row * 2 + 1) * w
            for cell in range(w // 2):
                x = cell * 2
                quarters = (top + x, top + x + 1, bottom + x, bottom + x + 1)
                lit = [i for i, q in enumerate(quarters) if a[q] >= threshold]
                if not lit:
                    line.append(" ")
                    continue
                if len(lit) < 4:
                    # Part of the cell is off the edge of the picture. One
                    # colour, and the rest left to the terminal.
                    pattern = sum(1 << i for i in lit)
                    line.append(QUADRANTS[pattern],
                                style=_mean(r, g, b,
                                            [quarters[i] for i in lit]))
                    continue
                tones = [0.299 * r[q] + 0.587 * g[q] + 0.114 * b[q]
                         for q in quarters]
                split = sum(tones) / 4.0
                pattern = sum(1 << i for i, t in enumerate(tones) if t >= split)
                if pattern in (0, 0b1111):
                    line.append("█", style=_mean(r, g, b, quarters))
                    continue
                fore = [q for i, q in enumerate(quarters) if pattern >> i & 1]
                back = [q for i, q in enumerate(quarters)
                        if not pattern >> i & 1]
                line.append(QUADRANTS[pattern],
                            style=f"{_mean(r, g, b, fore)} on "
                                  f"{_mean(r, g, b, back)}")
            out.append(line)
        return out


def _mean(r, g, b, indices) -> str:
    n = float(len(indices))
    return hexed((sum(r[i] for i in indices) / n,
                  sum(g[i] for i in indices) / n,
                  sum(b[i] for i in indices) / n))
