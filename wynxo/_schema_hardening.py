from __future__ import annotations

import math


def install() -> None:
    from .schema import Field

    original = Field._bounded
    if getattr(original, "_wynxo_finite", False):
        return

    def bounded(self, value, loc, errors):
        if self.type is float and not math.isfinite(float(value)):
            errors.append((loc, "must be a finite number"))
        return original(self, value, loc, errors)

    bounded._wynxo_finite = True
    Field._bounded = bounded


install()
