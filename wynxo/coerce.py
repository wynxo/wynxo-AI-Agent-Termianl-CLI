"""Reading values that came from somewhere we do not control.

Two places feed wynxo data it did not create: the server on the other end of
/api/chat, and its own files on disk after a crash, a full disk, or a version
of wynxo that wrote a different shape. Both used to be taken on trust, and
both produced the same failure -- a mistyped field travelling several frames
inland before dying as `'dict' object has no attribute 'strip'`, which reads
to the user as wynxo crashing rather than as bad input.

So the rule is that data crosses into wynxo through one of these, at the
boundary, and everything past that point can rely on its types. They coerce
rather than reject wherever coercion has an obvious meaning: text that
arrived wrapped in an object is still text, and a count sent as "128" is
still 128. Losing it would be its own bug.
"""

from __future__ import annotations


def as_text(value: object) -> str:
    """A wire field as a string, whatever the server actually sent.

    ``None`` and ``""`` both mean absent, so both come back empty. Anything
    else is stringified rather than dropped: a server that wraps reasoning in
    an object is being odd, but the text inside is still the model's thought
    and the user would rather see it than lose it.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(as_text(item) for item in value)
    if isinstance(value, dict):
        # The shapes seen in the wild all park the text under one of these.
        for key in ("text", "content", "thinking", "reasoning", "value"):
            if key in value:
                return as_text(value[key])
        return ""
    return str(value)


def as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        # A single call sent unwrapped, which some shims do.
        return [value]
    return []


def as_int(value: object) -> int:
    """A count as an int. Strings and floats appear in compat servers."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value and value not in (
            float("inf"), float("-inf")) else 0
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (ValueError, OverflowError):
            return 0
    return 0


def as_float(value: object, default: float = 0.0) -> float:
    """A timestamp or duration as a float.

    Takes a default rather than always falling back to zero: a missing
    ``created_at`` means "now", and a session dated 1970 sorts to the bottom
    of the resume list and looks like it was lost.
    """
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (ValueError, OverflowError):
            return default
    else:
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default        # NaN and infinities break every comparison
    return number
