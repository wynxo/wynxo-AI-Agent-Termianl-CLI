from __future__ import annotations


async def _stream_test_output(callback, line: str) -> None:
    """Forward automatic verification output without creating orphaned coroutines."""
    try:
        result = callback("shell", line)
        if result is not None:
            await result
    except Exception:
        pass
