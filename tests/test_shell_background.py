"""Background shell jobs: start, poll, finish, kill.

Production runs the whole session on one asyncio loop, and the background
machinery spans calls -- a job started in one invoke() is polled by a later
invoke(). Each test therefore drives its whole flow inside a single
``asyncio.run``, the way the agent actually does.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from wynxo.tools import build_registry
from wynxo.tools.shell import _BACKGROUND


def _registry(tmp_path: Path):
    return build_registry(tmp_path, allow_shell=True)


def _python(code: str) -> str:
    # Single quotes inside the code, double around the whole -c argument:
    # the default shell removes double quotes as grouping when it launches
    # the child, so the code must carry its own single quotes.
    return f'{sys.executable} -c "{code}"'


def test_background_job_starts_and_finishes(tmp_path: Path):
    reg = _registry(tmp_path)
    shell = reg.get("shell")
    poll = reg.get("background_poll")

    async def go():
        started = await shell.invoke({
            "command": _python("import time; time.sleep(0.3); print('slow work done')"),
            "background": True,
        })
        assert started.ok
        job_id = started.metadata["job_id"]
        assert job_id in _BACKGROUND

        # Wait on the real reader state -- the poll tool below also drives
        # the loop, but checking done directly first keeps this deterministic.
        for _ in range(40):
            await asyncio.sleep(0.05)
            if _BACKGROUND[job_id]["done"].is_set():
                break
        else:
            raise AssertionError("job never finished")

        # Now the poll tool's own contract: it must report finished, the
        # exit code, and the captured output.
        result = await poll.invoke({"job_id": job_id})
        assert result.metadata.get("finished") is True
        assert result.metadata.get("exit_code") == 0
        assert "slow work done" in result.output

    asyncio.run(go())


def test_poll_sees_partial_output_while_running(tmp_path: Path):
    reg = _registry(tmp_path)
    shell = reg.get("shell")
    poll = reg.get("background_poll")

    async def go():
        started = await shell.invoke({
            "command": _python("import time; print('early'); time.sleep(2); print('late')"),
            "background": True,
        })
        job_id = started.metadata["job_id"]
        # Poll for up to ~1.5s (before the 2s command ends): a running job
        # must expose output it has already produced.
        saw_early = False
        for _ in range(30):
            result = await poll.invoke({"job_id": job_id})
            if result.metadata.get("finished"):
                break
            if "early" in (result.metadata.get("stdout", "") or ""):
                saw_early = True
                break
            await asyncio.sleep(0.05)
        assert saw_early, "a running job should report output it has produced"

    asyncio.run(go())


def test_kill_stops_a_background_job(tmp_path: Path):
    reg = _registry(tmp_path)
    shell = reg.get("shell")
    poll = reg.get("background_poll")

    async def go():
        started = await shell.invoke({
            "command": _python("import time; time.sleep(30)"),
            "background": True,
        })
        job_id = started.metadata["job_id"]
        assert _BACKGROUND[job_id]["process"].returncode is None

        result = await poll.invoke({"job_id": job_id, "kill": True})
        meta = result.metadata
        assert meta.get("finished") is True, "kill must mark the job finished"

    asyncio.run(go())


def test_poll_of_unknown_job_is_an_error(tmp_path: Path):
    reg = _registry(tmp_path)
    poll = reg.get("background_poll")
    result = asyncio.run(poll.invoke({"job_id": "nope"}))
    assert not result.ok
    assert "No background job" in result.output