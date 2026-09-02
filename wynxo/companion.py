"""What the companion is doing, as a state.

The state machine only. This module held the pictures too -- a full set of
staged ASCII scenes with a desk, a laptop and a mug -- and no session ever
drew them: the mapping below was wired up and the frames were not, while a
smaller face from ``pet.py`` was what actually appeared on screen. The
drawing is ``sprite.py`` now, once, and this is the part that was always
doing the work.

There is no clock here and no picture. A state is a fact about the agent --
which tool is running, what the task state machine says -- so it is derived
from those on demand rather than stored, and the companion cannot animate
through a stall or pretend work is happening.
"""

from __future__ import annotations

from enum import Enum

class State(str, Enum):
    """What the agent is actually doing. Not moods -- events."""

    IDLE = "idle"
    THINKING = "thinking"
    SEARCHING = "searching"
    READING = "reading"
    CODING = "coding"
    TESTING = "testing"
    RECOVERING = "recovering"
    SUCCESS = "success"
    ERROR = "error"
    LISTENING = "listening"
    SPEAKING = "speaking"
    CANCELLED = "cancelled"


# -- what the agent is doing, as a state ------------------------------------

_BY_TOOL = {
    "edit_file": State.CODING, "write_file": State.CODING,
    "multi_edit": State.CODING, "github_write": State.CODING,
    "read_file": State.READING, "list_dir": State.READING,
    "github_read": State.READING, "projectmap": State.READING,
    "grep": State.SEARCHING, "glob": State.SEARCHING,
    "web_search": State.SEARCHING, "search": State.SEARCHING,
    "run_tests": State.TESTING,
    "shell": State.THINKING,
}
"""The running tool, where it says more than the task state does.

"executing" is equally true of reading a file and of writing one, and those
must not look the same -- watching the companion should tell you which is
happening without reading the transcript."""

_BY_TASK = {
    "idle": State.IDLE, "thinking": State.THINKING,
    "planning": State.THINKING, "executing": State.CODING,
    "testing": State.TESTING, "recovering": State.RECOVERING,
    "completed": State.SUCCESS, "failed": State.ERROR,
    "cancelled": State.CANCELLED,
}

_OVER = frozenset({State.IDLE, State.SUCCESS, State.ERROR, State.CANCELLED})
"""States that mean no task is running. A tool left over from the turn that
just ended must not animate one of these into looking busy."""


def state_for(tool: str = "", task: str = "") -> State:
    """What the companion is doing, from what the agent is actually doing.

    The tool wins while a task is running, because the task state is too
    coarse to draw: "executing" is equally true of reading a file and of
    writing one, and those must not look alike -- watching the companion
    should tell you which is happening without reading the transcript.

    The task wins when it says the turn is over. A tool name lingers after
    its call returns, and a finished turn that went on drawing "reading"
    would be the companion claiming work that had stopped.
    """
    settled = _BY_TASK.get(str(task).strip().lower())
    if settled in _OVER:
        return settled
    if tool:
        by_tool = _BY_TOOL.get(str(tool).strip().lower())
        if by_tool is not None:
            return by_tool
    return settled or State.IDLE


def _parse(value) -> State:
    try:
        return State(str(value).strip().lower())
    except ValueError:
        return State.IDLE

