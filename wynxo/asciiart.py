"""Turning a picture into text.

Real photo-to-ASCII is a sampling job, not a drawing one: the image is
reduced to a grid, each cell's brightness is measured, and a character of
matching visual weight is put in its place. That is why it looks like the
photograph -- nothing is being interpreted, only measured.

Decoding and rendering are kept apart. Rendering works on a plain grid of
brightness values, so it is testable without an image library at all, and
the loaders are free to be as fussy as each format demands.

The one thing this must get right and is easy to get wrong: a terminal cell
is about twice as tall as it is wide. Sampling on a square grid gives a
picture stretched to twice its height, which is the usual reason homemade
ASCII art looks melted.
"""

from __future__ import annotations

from pathlib import Path

# Dark to light. The eye reads these as a smooth ramp at terminal sizes,
# which a naive "  .:-=+*#%@" does not -- its middle is too crowded.
RAMP = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
BLOCKS = " ░▒▓█"

CELL_ASPECT = 2.15
"""How much taller a terminal cell is than it is wide. Measured rather than
assumed to be 2: most terminal fonts are a little taller still, and the
error shows up as a face stretched lengthways."""


def ramp_for(style: str) -> str:
    return {"blocks": BLOCKS, "detail": RAMP,
            "simple": " .:-=+*#%@"}.get(style, RAMP)


def render(grid: list[list[float]], style: str = "detail",
           invert: bool = False) -> str:
    """A grid of 0..1 brightness values as lines of text."""
    ramp = ramp_for(style)
    if invert:
        ramp = ramp[::-1]
    last = len(ramp) - 1
    lines = []
    for row in grid:
        line = []
        for value in row:
            value = 0.0 if value < 0 else 1.0 if value > 1 else value
            line.append(ramp[int(value * last + 0.5)])
        lines.append("".join(line).rstrip())
    return "\n".join(lines)


def normalise(grid: list[list[float]]) -> list[list[float]]:
    """Stretch the brightness range to fill 0..1.

    A webcam photo of someone in a dim room uses a fraction of the available
    range, and mapped straight onto the ramp it comes out as an even wash of
    mid-tone characters. Stretching is what makes the features appear.
    """
    values = [v for row in grid for v in row]
    if not values:
        return grid
    low, high = min(values), max(values)
    if high - low < 1e-6:
        return grid
    span = high - low
    return [[(v - low) / span for v in row] for row in grid]


# -- loaders ---------------------------------------------------------------

def load(path: Path, width: int = 100) -> list[list[float]]:
    """Brightness grid for an image, sized for a terminal `width` wide."""
    path = Path(path)
    if path.suffix.lower() in (".pgm", ".ppm", ".pnm"):
        return _load_netpbm(path, width)
    return _load_pillow(path, width)


def _load_pillow(path: Path, width: int) -> list[list[float]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageSupportMissing(
            "Reading a JPEG or PNG needs Pillow, which wynxo does not "
            "install by default.\n"
            "  pip install pillow\n"
            "Then try again. (A .pgm or .ppm file works without it.)"
        ) from exc

    with Image.open(path) as image:
        image = image.convert("L")
        height = max(1, round(width * image.height / image.width / CELL_ASPECT))
        image = image.resize((width, height))
        pixels = list(image.getdata())

    return [[pixels[y * width + x] / 255.0 for x in range(width)]
            for y in range(height)]


def _load_netpbm(path: Path, width: int) -> list[list[float]]:
    """The one image format worth decoding by hand, for testing without
    an image library present."""
    data = path.read_bytes()
    fields: list[bytes] = []
    index = 0
    while len(fields) < 4 and index < len(data):
        while index < len(data) and data[index:index + 1].isspace():
            index += 1
        if data[index:index + 1] == b"#":
            while index < len(data) and data[index:index + 1] != b"\n":
                index += 1
            continue
        start = index
        while index < len(data) and not data[index:index + 1].isspace():
            index += 1
        fields.append(data[start:index])
    magic, source_w, source_h, _maxval = (f.decode("ascii", "replace")
                                          for f in fields[:4])
    index += 1
    source_w, source_h = int(source_w), int(source_h)
    body = data[index:]
    step = 3 if magic == "P6" else 1

    def brightness(x: int, y: int) -> float:
        at = (y * source_w + x) * step
        if at + step > len(body):
            return 0.0
        if step == 1:
            return body[at] / 255.0
        # Rec. 601 luma: the eye is far more sensitive to green than blue,
        # and a plain average turns a red shirt and a blue one into the
        # same grey.
        r, g, b = body[at], body[at + 1], body[at + 2]
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    height = max(1, round(width * source_h / source_w / CELL_ASPECT))
    return [[brightness(min(source_w - 1, x * source_w // width),
                        min(source_h - 1, y * source_h // height))
             for x in range(width)]
            for y in range(height)]


# Ink weight for the characters ASCII art is usually drawn with. Built from
# the ramp so it agrees with what render() produces, with the common art
# characters pinned explicitly -- a converter's output has to survive being
# read back in at a different size.
_WEIGHTS = {ch: i / (len(RAMP) - 1) for i, ch in enumerate(RAMP)}
_WEIGHTS.update({ch: i / 9 for i, ch in enumerate(" .:-=+*#%@")})
_WEIGHTS.update({ch: i / 4 for i, ch in enumerate(BLOCKS)})


def weigh(char: str) -> float:
    """How much ink a character puts on the page, from 0 to 1."""
    if char in _WEIGHTS:
        return _WEIGHTS[char]
    return 0.0 if char.isspace() else 0.55


def from_text(art: str, width: int, height: int | None = None) -> list[list[float]]:
    """A piece of ASCII art as a brightness grid, so it can be resized.

    Art arrives at whatever size it was drawn, which is routinely wider than
    the terminal it has to fit. Reading the characters back to ink and
    resampling is the only way to shrink it that keeps the picture; cropping
    loses half of it and dropping every other line loses the shading.

    No aspect correction here: the source is already in character cells, so
    its proportions are right and must simply be preserved.
    """
    rows = art.split("\n")
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    if not rows:
        return []
    source_w = max(len(r) for r in rows)
    source_h = len(rows)
    width = max(1, width)
    if height is None:
        height = max(1, round(width * source_h / source_w))

    padded = [r.ljust(source_w) for r in rows]
    grid = []
    for y in range(height):
        y0, y1 = y * source_h // height, max((y + 1) * source_h // height,
                                             y * source_h // height + 1)
        row = []
        for x in range(width):
            x0, x1 = x * source_w // width, max((x + 1) * source_w // width,
                                                x * source_w // width + 1)
            # Averaged over the block it stands for, so shading survives the
            # shrink rather than being decided by whichever pixel was landed on.
            total = count = 0.0
            for sy in range(y0, min(y1, source_h)):
                line = padded[sy]
                for sx in range(x0, min(x1, source_w)):
                    total += weigh(line[sx])
                    count += 1
            row.append(total / count if count else 0.0)
        grid.append(row)
    return grid


class ImageSupportMissing(RuntimeError):
    """Raised when a format needs a library that is not installed."""


def from_image(path, width: int = 100, style: str = "detail",
               invert: bool = False, stretch: bool = True) -> str:
    """An image file as ASCII art, ready to print."""
    grid = load(Path(path), width)
    if stretch:
        grid = normalise(grid)
    return render(grid, style=style, invert=invert)
