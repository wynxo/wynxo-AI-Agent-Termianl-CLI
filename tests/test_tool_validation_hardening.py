from __future__ import annotations

import pytest

from wynxo.schema import Field, Schema, ValidationError
from wynxo.tools.base import Tool


class SampleInput(Schema):
    path = Field(str, "path")


class SampleTool(Tool):
    name = "sample"
    description = "sample"
    Input = SampleInput

    async def run(self, args: SampleInput):
        raise AssertionError("not executed")


@pytest.mark.asyncio
async def test_unknown_tool_argument_is_rejected(tmp_path):
    result = await SampleTool(tmp_path).invoke({"path": "x", "paht": "typo"})

    assert not result.ok
    assert "unexpected field" in result.error


def test_config_schema_can_still_ignore_unknown_fields():
    value = SampleInput.validate({"path": "x", "future_option": 1})
    assert value.path == "x"
