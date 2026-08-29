from __future__ import annotations

import asyncio

from wynxo.cli import TerminalCallbacks
from wynxo.ui import UI


def test_empty_stream_callbacks_are_noops():
    callbacks = TerminalCallbacks(UI())

    async def run():
        await callbacks.on_thinking("")
        await callbacks.on_content("")
        await callbacks.on_code("")
        await callbacks.on_tool_output("shell", "")

    asyncio.run(run())
    assert callbacks._thinking_buffer == []
    assert callbacks.streamer is None


def test_concurrent_status_callbacks_are_serialized():
    callbacks = TerminalCallbacks(UI())
    active = 0
    maximum = 0

    async def stage(name, detail=""):
        nonlocal active, maximum
        async with callbacks._status_lock:
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            callbacks._status_message = name
            active -= 1

    async def run():
        await asyncio.gather(stage("one"), stage("two"), stage("three"))

    asyncio.run(run())
    assert maximum == 1
