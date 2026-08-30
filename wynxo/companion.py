"""Wyn: the companion that sits beside the work.

What this replaces was a face in the status bar -- two lines, a pair of
eyes, a squiggle -- and it never read as a character doing anything. It
could not: there was nowhere for it to be. This gives it somewhere. Every
state is a small staged scene with the same furniture in the same places --
a desk, a laptop, a mug -- so what changes between frames is the character,
which is the only way movement reads as movement rather than as the picture
being swapped for a different picture.

The character is original to wynxo. The staging rules it follows:

* One character, drawn the same way in every scene: same head width, same
  ear angle, same eye row. A companion whose proportions change between
  states looks like several companions.
* The desk is the floor of the frame and never moves. It is what makes the
  character look seated rather than floating.
* Exactly one thing moves per frame pair. Two is a twitch; one is a breath.
* Every scene is the same height, so the panel never resizes underneath the
  conversation.

Three tiers, chosen by what the terminal can actually do rather than by a
setting: box-drawing where it exists, plain ASCII where it does not, and a
single still frame under reduced motion. Nothing here uses a codepoint
wider than one cell -- CJK, kana and emoji are all excluded on purpose,
because a two-cell glyph inside a bordered panel tears the border.

There is no clock in this module. Frames advance when the caller repaints,
and the caller repaints when something happens, so the companion cannot
animate through a stall and cannot pretend work is happening.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

WIDTH = 30
"""Inner width of the scene, in cells. The panel adds a border either side,
landing on the 32-34 the plan panel already occupies, so the two stack in
the same column without either having to move."""

HEIGHT = 6
"""Rows per scene. Fixed, so the panel is the same size in every state."""


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


@dataclass(frozen=True)
class Scene:
    """One state, as the frames that show it."""

    state: State
    frames: tuple[str, ...]
    label: str
    ascii: tuple[str, ...] = ()
    """Plain 7-bit frames, for a console that cannot draw the rest -- a
    Windows code page that is not UTF-8, most obviously."""
    compact: tuple[str, ...] = ()
    """A narrower staging for a terminal too thin for the desk."""

    def __post_init__(self) -> None:
        # A scene that is the wrong size tears the panel it is drawn in, and
        # it does so on somebody else's terminal rather than here. Cheap to
        # check once at import; impossible to notice otherwise.
        for name, frames in (("frames", self.frames), ("ascii", self.ascii)):
            for frame in frames:
                rows = frame.split("\n")
                if len(rows) != HEIGHT:
                    raise ValueError(
                        f"{self.state.value}.{name}: {len(rows)} rows, "
                        f"expected {HEIGHT}")
                for row in rows:
                    if len(row) > WIDTH:
                        raise ValueError(
                            f"{self.state.value}.{name}: row of {len(row)} "
                            f"cells, max {WIDTH}: {row!r}")


def _scene(state, label, frames, ascii_frames=(), compact=()):
    return Scene(state=state, label=label, frames=tuple(frames),
                 ascii=tuple(ascii_frames), compact=tuple(compact))


# -- the staging ------------------------------------------------------------
#
# Read these as a flip-book: each pair differs in one place. The desk row and
# the laptop are identical in every scene that has them, so the eye reads the
# character as moving rather than the picture as changing.

_IDLE = _scene(
    State.IDLE, "waiting",
    [
        "      ╱\\_╱\\\n"
        "     ( ˘   ˘ )\n"
        "      \\  ᵕ  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( -   - )\n"
        "      \\  ᵕ  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( ˘   ˘ )\n"
        "      \\  ᵕ  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( -   - )\n"
        "      \\  u  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
        "      /\\_/\\\n"
        "     ( o   o )\n"
        "      \\  u  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_THINKING = _scene(
    State.THINKING, "thinking",
    [
        "   ·  ╱\\_╱\\\n"
        "     ( ˘   • )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "  ·   ╱\\_╱\\\n"
        "     ( •   ˘ )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        " ∘    ╱\\_╱\\\n"
        "     ( ˘   • )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "   .  /\\_/\\\n"
        "     ( -   o )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
        "  .   /\\_/\\\n"
        "     ( o   - )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_SEARCHING = _scene(
    State.SEARCHING, "searching",
    [
        "      ╱\\_╱\\   ∘\n"
        "     ( •   ˘ )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\  ∘\n"
        "     ( ˘   • )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\   o\n"
        "     ( o   - )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
        "      /\\_/\\  o\n"
        "     ( -   o )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_READING = _scene(
    State.READING, "reading",
    [
        "      ╱\\_╱\\\n"
        "     ( •   • )\n"
        "     ╭┴─────┴╮\n"
        "     │░░░░░░░│\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( •   • )\n"
        "     ╭┴─────┴╮\n"
        "     │▒░░░░░░│\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( o   o )\n"
        "     ++-----++\n"
        "     |.......|\n"
        "    -+-------+-\n"
        "                ",
        "      /\\_/\\\n"
        "     ( o   o )\n"
        "     ++-----++\n"
        "     |:......|\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_CODING = _scene(
    State.CODING, "writing code",
    [
        "      ╱\\_╱\\\n"
        "     ( ˃   ˂ )\n"
        "     ╭┴─────┴╮\n"
        "     │▓▒▒▒▒▒▒│\n"
        "    ─┴─╥───╥─┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( ˃   ˂ )\n"
        "     ╭┴─────┴╮\n"
        "     │▒▒▓▒▒▒▒│\n"
        "    ─┴─╥───╥─┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( ˃   ˂ )\n"
        "     ╭┴─────┴╮\n"
        "     │▒▒▒▒▓▒▒│\n"
        "    ─┴─╥───╥─┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( ˃   ˂ )\n"
        "     ╭┴─────┴╮\n"
        "     │▒▒▒▒▒▒▓│\n"
        "    ─┴─╥───╥─┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( >   < )\n"
        "     ++-----++\n"
        "     |#......|\n"
        "    -+-|---|-+-\n"
        "                ",
        "      /\\_/\\\n"
        "     ( >   < )\n"
        "     ++-----++\n"
        "     |..#....|\n"
        "    -+-|---|-+-\n"
        "                ",
        "      /\\_/\\\n"
        "     ( >   < )\n"
        "     ++-----++\n"
        "     |.....#.|\n"
        "    -+-|---|-+-\n"
        "                ",
    ],
)

_TESTING = _scene(
    State.TESTING, "running tests",
    [
        "      ╱\\_╱\\\n"
        "     ( o   o )\n"
        "     ╭┴─────┴╮\n"
        "     │▪▫▫▫▫▫▫│\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( o   o )\n"
        "     ╭┴─────┴╮\n"
        "     │▪▪▪▫▫▫▫│\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( o   o )\n"
        "     ╭┴─────┴╮\n"
        "     │▪▪▪▪▪▫▫│\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( o   o )\n"
        "     ++-----++\n"
        "     |=......|\n"
        "    -+-------+-\n"
        "                ",
        "      /\\_/\\\n"
        "     ( o   o )\n"
        "     ++-----++\n"
        "     |====...|\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_RECOVERING = _scene(
    State.RECOVERING, "trying another way",
    [
        "      ╱\\_╱\\\n"
        "     ( ˘   ˘ )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "   ·  ╱\\_╱\\\n"
        "     ( •   ˘ )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "  ∘   ╱\\_╱\\\n"
        "     ( ˘   • )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( -   - )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
        "   .  /\\_/\\\n"
        "     ( o   - )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_SUCCESS = _scene(
    State.SUCCESS, "done",
    [
        "      ╱\\_╱\\    ∘\n"
        "     ( ˘   ˘ )\n"
        "      \\  ‿  /  ╭─╮\n"
        "     ╭───────╮ │ │\n"
        "    ─┴───────┴─╰─╯\n"
        "                ",
        "      ╱\\_╱\\   ∘\n"
        "     ( ˘   ˘ )\n"
        "      \\  ‿  /  ╭─╮\n"
        "     ╭───────╮ │ │\n"
        "    ─┴───────┴─╰─╯\n"
        "                ",
    ],
    [
        "      /\\_/\\    o\n"
        "     ( -   - )\n"
        "      \\  u  /  ,-.\n"
        "     +-------+ | |\n"
        "    -+-------+-`-'\n"
        "                ",
        "      /\\_/\\   o\n"
        "     ( -   - )\n"
        "      \\  u  /  ,-.\n"
        "     +-------+ | |\n"
        "    -+-------+-`-'\n"
        "                ",
    ],
)

_ERROR = _scene(
    State.ERROR, "that did not work",
    [
        "      ╱\\_╱\\\n"
        "     ( ×   × )\n"
        "      \\  ~  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( ×   × )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( x   x )\n"
        "      \\  ~  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
        "      /\\_/\\\n"
        "     ( x   x )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_LISTENING = _scene(
    State.LISTENING, "listening",
    [
        "      ╱\\_╱\\  ·\n"
        "     ( o   o )\n"
        "      \\  ᵕ  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\  ∘\n"
        "     ( o   o )\n"
        "      \\  ᵕ  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\  .\n"
        "     ( o   o )\n"
        "      \\  u  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
        "      /\\_/\\  o\n"
        "     ( o   o )\n"
        "      \\  u  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_SPEAKING = _scene(
    State.SPEAKING, "speaking",
    [
        "      ╱\\_╱\\\n"
        "     ( ˘   ˘ )\n"
        "      \\  ∪  /  ·\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
        "      ╱\\_╱\\\n"
        "     ( ˘   ˘ )\n"
        "      \\  ᵕ  /  ∘\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( -   - )\n"
        "      \\  o  /  .\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
        "      /\\_/\\\n"
        "     ( -   - )\n"
        "      \\  u  /  o\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

_CANCELLED = _scene(
    State.CANCELLED, "stopped",
    [
        "      ╱\\_╱\\\n"
        "     ( -   - )\n"
        "      \\  ω  /\n"
        "     ╭───────╮\n"
        "    ─┴───────┴─\n"
        "                ",
    ],
    [
        "      /\\_/\\\n"
        "     ( -   - )\n"
        "      \\  w  /\n"
        "     +-------+\n"
        "    -+-------+-\n"
        "                ",
    ],
)

SCENES: dict[State, Scene] = {
    s.state: s for s in (
        _IDLE, _THINKING, _SEARCHING, _READING, _CODING, _TESTING,
        _RECOVERING, _SUCCESS, _ERROR, _LISTENING, _SPEAKING, _CANCELLED,
    )
}


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


def is_over(state: "State | str") -> bool:
    """Whether this state means there is nothing in flight."""
    if isinstance(state, State):
        return state in _OVER
    try:
        return State(str(state).strip().lower()) in _OVER
    except ValueError:
        return True


def state_for(task_state: str, tool: str = "", *, listening: bool = False,
              speaking: bool = False) -> State:
    """The companion's state, from what the agent is really doing.

    Voice wins, because it is about the person rather than the work and is
    the one thing they are waiting on. Otherwise the running tool decides,
    but only while a task is running: between a cancellation and the turn's
    teardown the tool is still set, and a companion that keeps typing under
    the word "Interrupted" is worse than one that does nothing.
    """
    if listening:
        return State.LISTENING
    if speaking:
        return State.SPEAKING
    resolved = _BY_TASK.get(str(task_state).strip().lower(), State.IDLE)
    if tool and resolved not in _OVER:
        return _BY_TOOL.get(tool.strip().lower(), resolved)
    return resolved


# -- drawing ----------------------------------------------------------------

def frames_for(state: "State | str", *, unicode: bool = True,
               reduced: bool = False, width: int = 80) -> tuple[str, ...]:
    """The frames to cycle for a state, given what the terminal can do.

    Reduced motion is one still frame -- not a slower animation, none of
    one. It is asked for by people who find movement in the corner of the
    eye actively unpleasant, and a gentler version of the thing is still
    the thing.
    """
    scene = SCENES.get(state if isinstance(state, State)
                       else _parse(state), SCENES[State.IDLE])
    if not unicode and scene.ascii:
        chosen = scene.ascii
    elif width < WIDTH + 4 and scene.compact:
        chosen = scene.compact
    else:
        chosen = scene.frames
    return (chosen[0],) if reduced else chosen


def label_for(state: "State | str") -> str:
    scene = SCENES.get(state if isinstance(state, State) else _parse(state))
    return scene.label if scene else ""


def _parse(value) -> State:
    try:
        return State(str(value).strip().lower())
    except ValueError:
        return State.IDLE


def panel(state: "State | str", frame: int = 0, *, unicode: bool = True,
          reduced: bool = False, width: int = WIDTH + 4,
          title: str = "wyn") -> list[str]:
    """The companion, in its own bordered box, as rows ready to draw.

    Bounded on purpose and in both directions: it is drawn as a Float over
    the conversation, so a box that grew would cover the thing the person is
    actually reading. The conversation is the product; this sits beside it.
    """
    frames = frames_for(state, unicode=unicode, reduced=reduced, width=width)
    body = frames[frame % len(frames)].split("\n")
    # Never wider than what it was given. Flooring this at the scene's own
    # width made the panel 30 columns on a 20-column terminal, which the
    # float then clipped -- so the box lost its right border and stopped
    # being a box. Below the staging's width the rows crop instead, which
    # loses the mug and the thought before it loses the character.
    inner = max(10, min(width, 60)) - 2
    if unicode:
        tl, tr, bl, br, h, v = "╭", "╮", "╰", "╯", "─", "│"
    else:
        tl, tr, bl, br, h, v = "+", "+", "+", "+", "-", "|"
    label = label_for(state)
    head = f" {title} " + (f"{h} {label} " if label else "")
    head = head[:inner]
    rows = [tl + head + h * max(0, inner - len(head)) + tr]
    for line in body:
        rows.append(v + line[:inner].ljust(inner) + v)
    rows.append(bl + h * inner + br)
    return rows
