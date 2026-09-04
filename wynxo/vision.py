"""Putting a picture in front of the model.

A screenshot is the difference between an assistant that can be told where
to click and one that can look. wynxo could take them and could not send
them, which is the least useful half: the file landed on disk and the model
was told a path it had no way to open.

Two things decide whether this is worth doing on any given turn, and both
are checked rather than assumed: whether the model can see at all, and
whether the picture is small enough to be worth what it costs. A screenshot
is expensive -- a full desktop is worth something on the order of a
thousand tokens, every turn it stays in the conversation -- so it is
resized on the way in and never allowed to pile up.
"""

from __future__ import annotations

import base64
from pathlib import Path

MAX_EDGE = 1400
"""Longest side, in pixels, after resizing.

A 4K screenshot costs several times a 1400px one and reads no better: the
model is looking for buttons and panels, not fine print. Below about a
thousand, UI text stops being legible and the picture becomes a picture of
a blur -- which is worse than none, because it looks like information."""

MAX_BYTES = 4_000_000
"""A hard ceiling on the encoded image. Past this something is wrong --
a monitor wall, a corrupt file -- and sending it would hang the turn."""


class VisionError(Exception):
    """The picture could not be prepared; the message says why."""


def can_see(model_info) -> bool:
    """Whether this model accepts images.

    Asked of the server rather than guessed from the name. Model names are
    not a capability list -- families ship both vision and text-only builds
    under names that differ by a suffix, and guessing wrong in either
    direction is bad: refusing a model that can see wastes the feature,
    and sending a picture to one that cannot is an error the user reads as
    wynxo being broken.

    An older Ollama reports no capabilities at all. That is unknown, not
    no: it comes back False, so nothing is attempted, which is the safe
    direction of a guess nobody can make.
    """
    capabilities = getattr(model_info, "capabilities", None)
    return bool(capabilities) and "vision" in capabilities


def encode(path: str | Path) -> str:
    """A screenshot as base64, resized to something worth sending.

    Pillow is already a dependency for the mascot's sprite work, so this
    costs no new install. Without it the raw file goes as-is up to the
    ceiling -- a 4K PNG is wasteful rather than wrong.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise VisionError(f"could not read {path.name}: {exc}") from None
    if not raw:
        raise VisionError(f"{path.name} is empty.")

    shrunk = _resize(raw)
    if len(shrunk) > MAX_BYTES:
        raise VisionError(
            f"{path.name} is {len(shrunk) // 1_000_000}MB after resizing, "
            "which is too large to send.")
    return base64.b64encode(shrunk).decode("ascii")


def _resize(raw: bytes) -> bytes:
    """Down to MAX_EDGE on the longest side. The original if it cannot be."""
    try:
        import io

        from PIL import Image
    except ImportError:
        return raw
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if max(image.size) <= MAX_EDGE:
                return raw
            scale = MAX_EDGE / max(image.size)
            size = (max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)))
            # LANCZOS because the subject is text and edges. A cheaper
            # filter turns small UI labels into grey, and a label that
            # cannot be read is the whole reason the picture was taken.
            resized = image.convert("RGB").resize(size, Image.LANCZOS)
            out = io.BytesIO()
            resized.save(out, format="PNG", optimize=True)
            return out.getvalue()
    except Exception:
        # Anything Pillow can raise on a file it does not like. The
        # original is still a valid image to whoever asked for one.
        return raw


def describe(path: str | Path) -> str:
    """What the picture is, for the message it travels with."""
    path = Path(path)
    try:
        size = path.stat().st_size // 1024
    except OSError:
        size = 0
    return f"{path.name} ({size} KB)"
