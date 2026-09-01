"""A small frame-based ASCII animation library.

The scenes here are the shared vocabulary of the companion's moods: the
pet face in the status bar maps a mood to a scene, and /animate and /pet
show print the frames as strips. The face itself advances on every repaint
of the live bar, so no scheduler lives here -- animations render into rows
that already exist, and the one-shot effects that need explicit timing
(a surge, a wake) are drawn by the UI directly.

Every scene degrades the same way, in the same order: reduced-motion mode
keeps one static frame, a non-unicode terminal swaps in the ASCII set, and
a narrow terminal swaps in the compact set. Nothing here ever changes the
layout -- animations render into rows that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    name: str
    frames: tuple[str, ...]
    label: str = ""
    fps: float = 6.0
    loops: bool = True
    ascii: tuple[str, ...] | None = None
    """: A plain-ASCII set for terminals without the unicode glyphs."""
    compact: tuple[str, ...] | None = None
    """: A shorter set for narrow terminals."""


# -- the scenes -------------------------------------------------------------
#
# All original. The face reuses the pet's glyphs (≽^•⩊•^≼) so the scenes
# read as the same character as the one in the status bar, which is what
# makes a showcase look like one product rather than a clip-art drawer.


def _from_companion(key: str, state) -> Scene:
    """A showcase Scene backed by the companion's own staging.

    The character used to be drawn twice -- a two-line face here, a
    different one in the status bar -- and the two drifted, so /animate
    showed something the running application never displayed. There is one
    set of frames now; this is the view of it that /animate and /pet want.
    """
    from . import companion

    return Scene(
        key,
        frames=companion.frames_for(state),
        label=companion.label_for(state),
        fps=5.0,
        ascii=companion.frames_for(state, unicode=False),
    )


SCENES: dict[str, Scene] = {
    # Small 3-line scenes for the showcase (/pet show, /animate). The face
    # is the same one the status bar uses, with a tiny body hint per state.
    "idle": Scene(
        "idle",
        (
            "   ≽^•⩊•^≼\n   ˘˘˘˘˘˘\n",
            "   ≽^•⩊•^≼\n   ˘˘˘˘˘˘\n",
            "   ≽^-⩊-^≼\n   ˘˘˘˘˘˘\n",
        ),
        label="waiting",
        fps=3.0,
        ascii=("   =^.^=\n   -----\n",) * 3,
    ),
    "thinking": Scene(
        "thinking",
        (
            " ≽^˘⩊•^≼  · \n  ˘˘˘˘˘˘\n",
            " ≽^•⩊˘^≼   ·\n  ˘˘˘˘˘˘\n",
        ),
        label="working it out",
        fps=5.0,
        ascii=(" =^o.^=  . \n  -----\n", " =^.o^=   .\n  -----\n"),
    ),
    "working": Scene(
        "working",
        (
            " ≽^•̀⩊•́^≼  ⌨\n  ˘˘˘˘˘˘\n",
            " ≽^•́⩊•̀^≼  ⌨\n  ˘˘˘˘˘˘\n",
        ),
        label="typing away",
        fps=6.0,
        ascii=(" =^>.<^=  >_\n  -----\n", " =^>.>^=  <_\n  -----\n"),
    ),
    "reading": Scene(
        "reading",
        (
            " ≽^◉⩊◉^≼  ▓\n  ˘˘˘˘˘˘\n",
            " ≽^◉⩊◉^≼  ▒\n  ˘˘˘˘˘˘\n",
            " ≽^-⩊-^≼  ▓\n  ˘˘˘˘˘˘\n",
        ),
        label="looking something up",
        fps=4.0,
        ascii=(" =^O.O^=  |\n  -----\n",) * 3,
    ),
    "searching": Scene(
        "searching",
        (
            " ≽^◉⩊◉^≼\n  ▓▓▓▓\n  ----\n",
            " ≽^◉⩊◉^≼\n  ----\n  ▓▓▓▓\n",
        ),
        label="scanning the project",
        fps=6.0,
        ascii=(" =^O.O^=\n  ++++\n  ----\n", " =^O.O^=\n  ----\n  ++++\n"),
    ),
    # The coding scene is the little story: the neko hops onto a tiny
    # terminal, gets inside it, and types until the screen fills. Every
    # frame is the same width and height so the box never shifts columns
    # between frames -- a jittering box reads as a rendering bug, not life.
    "coding": Scene(
        "coding",
        (
            "≽^•⩊•^≼\n┌───────────┐\n│ >_        │\n└───────────┘\n",
            "┌───────────┐\n│ ≽^•̀⩊•́^≼   │\n│ >_        │\n└───────────┘\n",
            "┌───────────┐\n│ ≽^•́⩊•̀^≼   │\n│ ████░░    │\n└───────────┘\n",
            "┌───────────┐\n│ ≽^≧⩊≦^≼   │\n│ ██████    │\n└───────────┘\n",
        ),
        label="typing at a tiny terminal",
        fps=5.0,
        ascii=(
            "=^.^=\n+-----------+\n| >_        |\n+-----------+\n",
            "+-----------+\n| =^>.>^=   |\n| >_        |\n+-----------+\n",
            "+-----------+\n| =^>.>^=   |\n| ####      |\n+-----------+\n",
            "+-----------+\n| =^_^=     |\n| ######    |\n+-----------+\n",
        ),
        compact=(" ≽^•̀⩊•́^≼ █", " ≽^•́⩊•̀^≼ █", " ≽^≧⩊≦^≼ █"),
    ),
    "testing": Scene(
        "testing",
        (
            " ≽^•⩊•^≼  ░░░\n  ──────\n",
            " ≽^•⩊•^≼  ▓░░\n  ──────\n",
            " ≽^•⩊•^≼  ▓▓░\n  ──────\n",
            " ≽^≧⩊≦^≼  ▓▓▓\n  ──────\n",
        ),
        label="watching the tests run",
        fps=6.0,
        ascii=(" =^.^=  ...\n  -----\n", " =^.^=  ..#\n  -----\n", " =^.^=  .##\n  -----\n", " =^_^=  ###\n  -----\n"),
        compact=(" ░░░", " ▓░░", " ▓▓░", " ▓▓▓"),
    ),
    "running": Scene(
        "running",
        (
            " ≽^•⩊•^≼ฅ  /\\_/\n  ˘˘˘˘˘˘\n",
            " ≽^•⩊•^≼ﾉ  \\_/\\\n  ˘˘˘˘˘˘\n",
        ),
        label="running",
        fps=8.0,
        ascii=(" =^.^=/  /\\\n  -----\n", " =^.^=\\\\  \\/\n  -----\n"),
    ),
    "sleepy": Scene(
        "sleepy",
        (
            " ≽^-⩊-^≼  z\n  ˘˘˘˘˘˘\n",
            " ≽^-⩊-^≼   z\n  ˘˘˘˘˘˘\n",
        ),
        label="dozing off",
        fps=2.0,
        ascii=(" =^-.-^=  z\n  -----\n", " =^-.-^=   z\n  -----\n"),
    ),
    "happy": Scene(
        "happy",
        (
            " ≽^≧⩊≦^≼  ✦ \n  ˘˘˘˘˘˘\n",
            " ≽^ᵕ⩊ᵕ^≼   ✦\n  ˘˘˘˘˘˘\n",
        ),
        label="pleased with how it went",
        fps=6.0,
        ascii=(" =^_^=  * \n  -----\n", " =^v^=   *\n  -----\n"),
    ),
    "error": Scene(
        "error",
        (
            " ≽^×⩊×^≼  !\n  ˘˘˘˘˘˘\n",
            " ≽^╥⩊╥^≼  !\n  ˘˘˘˘˘˘\n",
        ),
        label="hit a wall",
        fps=4.0,
        ascii=(" =^@.@^=  !\n  -----\n", " =^x.x^=  !\n  -----\n"),
    ),
    "sparkle": Scene(
        "sparkle",
        ("      ✦", "   ·  ✦  ·", "  ·  ✦  ·", "   ·  ✦  ·", "      ✦"),
        label="a little confetti",
        fps=8.0,
        loops=False,
        ascii=("      *", "   .  *  .", "  .  *  .", "   .  *  .", "      *"),
        compact=("*", "*", "*"),
    ),
}

# Pet moods -> scene, so one lookup answers "what is the cat doing right now".
# The character's own states come from companion.py, which is where the
# staging lives. Anything left in SCENES above is showcase decoration --
# sparkle and the like -- which is not the character and never was.
from .companion import State as _CompanionState  # noqa: E402

for _key, _state in (
    ("idle", _CompanionState.IDLE),
    ("thinking", _CompanionState.THINKING),
    ("searching", _CompanionState.SEARCHING),
    ("reading", _CompanionState.READING),
    ("coding", _CompanionState.CODING),
    ("working", _CompanionState.CODING),
    ("testing", _CompanionState.TESTING),
    ("happy", _CompanionState.SUCCESS),
    ("error", _CompanionState.ERROR),
    ("running", _CompanionState.TESTING),
):
    SCENES[_key] = _from_companion(_key, _state)
del _key, _state


MOOD_SCENES = {
    "idle": "idle", "thinking": "thinking", "working": "coding",
    "reading": "reading", "searching": "searching", "testing": "testing",
    "running": "running", "happy": "happy", "sad": "error",
    "asking": "thinking", "sleepy": "sleepy", "cancelled": "idle",
}


def scene_for(name: str) -> Scene:
    """The scene for a state name, defaulting to the idle scene.

    Moods resolve through MOOD_SCENES first -- the working mood is the
    coding scene, for instance -- so the pet's mood always picks the scene
    that tells the right story. A bare scene name still works for
    /animate."""
    key = (name or "").strip().lower()
    if key in MOOD_SCENES:
        return SCENES[MOOD_SCENES[key]]
    if key in SCENES:
        return SCENES[key]
    return SCENES["idle"]


def select(scene: Scene, *, unicode: bool = True, width: int = 80,
           reduced: bool = False) -> tuple[str, ...]:
    """The frames that fit here: static under reduced motion, ASCII on a
    non-unicode terminal, compact only when the full set is wider than the
    space available, full set otherwise."""
    if reduced:
        return (scene.frames[0],)
    if not unicode and scene.ascii:
        return scene.ascii
    widest = max(len(line) for frame in scene.frames
                 for line in frame.split("\n"))
    if scene.compact and width < widest:
        return scene.compact
    return scene.frames


def preview(name: str, n: int = 3, *, unicode: bool = True,
            width: int = 80, reduced: bool = False) -> str:
    """A few frames of a scene side by side, for /animate and /pet.

    Deterministic: no timers, just the frame sequence laid out as a strip.
    A looping scene shows the first ``n`` frames; a one-shot shows the
    frames it has, padded with blanks so the strip does not shrink.
    """
    scene = scene_for(name)
    frames = select(scene, unicode=unicode, width=width, reduced=reduced)
    if scene.loops:
        cycle = [frames[i % len(frames)] for i in range(n)]
    else:
        cycle = list(frames) + [""] * max(0, n - len(frames))
    rows = [frame.split("\n") for frame in cycle]
    height = max(len(r) for r in rows) if rows else 1
    out = []
    for row in range(height):
        out.append("   ".join((r[row] if row < len(r) else "").rstrip()
                              for r in rows).rstrip())
    return "\n".join(out).rstrip()
