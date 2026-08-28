from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from wynxo._agent_hardening import _stream_test_output


@pytest.mark.asyncio
async def test_stream_test_output_awaits_async_callback() -> None:
    callback = AsyncMock()

    await _stream_test_output(callback, "pytest 1 passed")

    callback.assert_awaited_once_with("shell", "pytest 1 passed")


@pytest.mark.asyncio
async def test_stream_test_output_ignores_ui_callback_failures() -> None:
    callback = AsyncMock(side_effect=RuntimeError("ui closed"))

    await _stream_test_output(callback, "still running")

    callback.assert_awaited_once_with("shell", "still running")
