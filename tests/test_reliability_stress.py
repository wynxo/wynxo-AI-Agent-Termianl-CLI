from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from wynxo.events import ExecutionState, ToolEvent
from wynxo.tools.shell import Shell


@pytest.mark.asyncio
async def test_tool_event_cannot_change_terminal_state_after_cancel() -> None:
    event = ToolEvent("shell", "sleep")
    event.start()
    event.cancel("user interrupted")

    event.finish(True, output="late output")

    assert event.state is ExecutionState.CANCELLED
    assert event.error == "user interrupted"
    assert event.output == ""


@pytest.mark.asyncio
async def test_shell_success_has_predictable_structured_metadata(tmp_path: Path) -> None:
    result = await Shell(tmp_path).invoke({"command": "python -c \"print('ok')\""})

    assert result.ok
    assert result.metadata["command"]
    assert result.metadata["stdout"].strip() == "ok"
    assert result.metadata["stderr"] == ""
    assert result.metadata["exit_code"] == 0
    assert result.metadata["timed_out"] is False
    assert result.metadata["cancelled"] is False


@pytest.mark.asyncio
async def test_shell_timeout_preserves_output_and_marks_timeout(tmp_path: Path) -> None:
    result = await Shell(tmp_path).invoke(
        {"command": "python -c \"print('before', flush=True); import time; time.sleep(2)\"", "timeout": 1}
    )

    assert not result.ok
    assert result.metadata["timed_out"] is True
    assert result.metadata["exit_code"] is None
    assert "before" in result.metadata["stdout"]


@pytest.mark.asyncio
async def test_shell_cancellation_terminates_child(tmp_path: Path) -> None:
    task = asyncio.create_task(
        Shell(tmp_path).invoke(
            {"command": "python -c \"import time; time.sleep(30)\"", "timeout": 30}
        )
    )
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
