from __future__ import annotations

import math

import pytest

from wynxo.schema import Field, Schema, ValidationError


class NumberInput(Schema):
    value = Field(float, "finite number")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_float_is_rejected(value):
    with pytest.raises(ValidationError):
        NumberInput.validate({"value": value})
