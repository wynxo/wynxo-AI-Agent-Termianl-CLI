"""The pet: a small face that reacts to what the agent is doing.

It sits at the left of the status bar and changes with state, so a glance
tells you whether wynxo is thinking, reading, writing or stuck, without
reading the words. That is the actual point -- it is a status indicator with
a personality, not decoration.

Everything here degrades: an ASCII face where the terminal cannot do unicode,
a still face where animation is off, and nothing at all when the pet is
disabled. It never gates behaviour and never speaks for the model.
"""

from __future__ import annotations

import random

from rich.cells import cell_len
from dataclasses import dataclass, field
from enum import Enum


class Mood(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    READING = "reading"
    WORKING = "working"
    RUNNING = "running"
    ASKING = "asking"
    HAPPY = "happy"
    SAD = "sad"


# Frames per mood. The last frame of a cycle is usually a blink, which is what
# makes a still face look alive without moving anything else.
FACES: dict[Mood, list[str]] = {
    # ^...^ with the outer brackets angled is read as ears, which is what
    # makes these cats rather than faces with whiskers drawn on.
    Mood.IDLE:     ["≽^•⩊•^≼", "≽^•⩊•^≼", "≽^•⩊•^≼", "≽^-⩊-^≼"],
    Mood.THINKING: ["≽^˘⩊•^≼", "≽^•⩊˘^≼", "≽^˘⩊•^≼", "≽^•⩊•^≼"],
    Mood.READING:  ["≽^◉⩊◉^≼", "≽^◉⩊◉^≼", "≽^◉⩊◉^≼", "≽^-⩊-^≼"],
    Mood.WORKING:  ["≽^•̀⩊•́^≼", "≽^•́⩊•̀^≼"],
    Mood.RUNNING:  ["≽^•⩊•^≼ฅ", "≽^•⩊•^≼ﾉ", "≽^•⩊•^≼ฅ", "≽^•⩊•^≼ﾉ"],
    Mood.ASKING:   ["≽^•⩊•^≼?", "≽^•⩊•^≼ ", "≽^•⩊•^≼?", "≽^•⩊•^≼ "],
    Mood.HAPPY:    ["≽^≧⩊≦^≼", "≽^ᵕ⩊ᵕ^≼"],
    Mood.SAD:      ["≽^╥⩊╥^≼", "≽^╥⩊╥^≼", "≽^×⩊×^≼"],
}

FACES_ASCII: dict[Mood, list[str]] = {
    # =^.^= is the oldest cat in the book and the only one every font has.
    Mood.IDLE:     ["=^.^=", "=^.^=", "=^.^=", "=^-^="],
    Mood.THINKING: ["=^o.^=", "=^.o^=", "=^o.^=", "=^.o^="],
    Mood.READING:  ["=^O.O^=", "=^O.O^=", "=^O.O^=", "=^-.-^="],
    Mood.WORKING:  ["=^>.<^=", "=^>.>^="],
    Mood.RUNNING:  ["=^.^=/", "=^.^=-", "=^.^=\\", "=^.^=-"],
    Mood.ASKING:   ["=^.^=?", "=^.^= ", "=^.^=?", "=^.^= "],
    Mood.HAPPY:    ["=^_^=", "=^v^="],
    Mood.SAD:      ["=^x.x^=", "=^x.x^=", "=^@.@^="],
}

# Rounder and fluffier, for the kawaii voice. Paws out.
FACES_KAWAII: dict[Mood, list[str]] = {
    Mood.IDLE:     ["₍ᐢ•ﻌ•ᐢ₎", "₍ᐢ•ﻌ•ᐢ₎", "₍ᐢ•ﻌ•ᐢ₎", "₍ᐢ-ﻌ-ᐢ₎"],
    Mood.THINKING: ["₍ᐢ˘ﻌ•ᐢ₎", "₍ᐢ•ﻌ˘ᐢ₎", "₍ᐢ˘ﻌ•ᐢ₎", "₍ᐢ•ﻌ•ᐢ₎"],
    Mood.READING:  ["₍ᐢ◉ﻌ◉ᐢ₎", "₍ᐢ◉ﻌ◉ᐢ₎", "₍ᐢ◉ﻌ◉ᐢ₎", "₍ᐢ-ﻌ-ᐢ₎"],
    Mood.WORKING:  ["₍ᐢ•̀ﻌ•́ᐢ₎", "₍ᐢ•́ﻌ•̀ᐢ₎"],
    Mood.RUNNING:  ["ฅ₍ᐢ•ﻌ•ᐢ₎ฅ", "₍ᐢ•ﻌ•ᐢ₎ﾉ ", "ฅ₍ᐢ•ﻌ•ᐢ₎ฅ", "₍ᐢ•ﻌ•ᐢ₎ﾉ "],
    Mood.ASKING:   ["₍ᐢ•ﻌ•ᐢ₎?", "₍ᐢ•ﻌ•ᐢ₎ ", "₍ᐢ•ﻌ•ᐢ₎?", "₍ᐢ•ﻌ•ᐢ₎ "],
    Mood.HAPPY:    ["₍ᐢ≧ﻌ≦ᐢ₎", "₍ᐢᵕﻌᵕᐢ₎"],
    Mood.SAD:      ["₍ᐢ╥ﻌ╥ᐢ₎", "₍ᐢ╥ﻌ╥ᐢ₎", "₍ᐢ×ﻌ×ᐢ₎"],
}

MOOD_STYLES: dict[Mood, str] = {
    Mood.IDLE: "grey62",
    Mood.THINKING: "bright_cyan",
    Mood.READING: "bright_blue",
    Mood.WORKING: "bright_cyan",
    Mood.RUNNING: "bright_magenta",
    Mood.ASKING: "yellow",
    Mood.HAPPY: "green",
    Mood.SAD: "red",
}

# Which activity name maps to which mood. Anything unrecognised stays THINKING,
# which is the honest default: something is happening and we did not label it.
ACTIVITY_MOODS: dict[str, Mood] = {
    "thinking": Mood.THINKING,
    "planning": Mood.THINKING,
    "critiquing plan": Mood.THINKING,
    "reconciling": Mood.THINKING,
    "compacting context": Mood.THINKING,
    "reading": Mood.READING,
    "listing": Mood.READING,
    "finding": Mood.READING,
    "searching": Mood.READING,
    "writing": Mood.WORKING,
    "writing file": Mood.WORKING,
    "editing": Mood.WORKING,
    "planning steps": Mood.WORKING,
    "running": Mood.RUNNING,
    "verifying": Mood.THINKING,
    "executing": Mood.WORKING,
    "repairing tool call": Mood.SAD,
}


# Short remarks, chosen once per event. Deliberately few and dry: a companion
# that comments constantly stops being charming after ten minutes.
REMARKS: dict[str, list[str]] = {
    "greet": ["ready when you are", "what are we building?", "listening"],
    "done": ["done", "that's it", "finished"],
    "denied": ["fair enough", "leaving it"],
    "error": ["that didn't work", "hit a wall"],
    "long": ["still going", "this one's slow", "hang on"],
    "interrupted": ["stopped", "ok, dropping it"],
}

REMARKS_KAWAII: dict[str, list[str]] = {
    "greet": ["ready when you are~", "what are we making today?", "listening~", "let's do our best! ♡", "nya~ ready~"],
    "done": ["all done~", "there we go", "finished~", "yay! completed! ✨", "mission accomplished~ ♡"],
    "denied": ["okay, leaving it", "no worries~", "that's fine~ ♡", "understood~"],
    "error": ["ah, that didn't work", "hit a wall, sorry", "uh oh~ 😭", "let's try again~"],
    "long": ["still going~", "this one's slow", "almost...", "hanging in there~", "patience~ ✨"],
    "interrupted": ["stopped~", "okay, dropping it", "understood~ ♡"],
}


def face_width(text: str) -> int:
    """Display cells, not codepoints.

    Faces are full of characters where the two differ: a combining accent is
    a codepoint occupying no cell, and a CJK dot occupies two. Getting this
    wrong pads every face by the wrong amount and makes the bar jitter on
    each frame.
    """
    return cell_len(text)


@dataclass
class Pet:
    """State, face and voice of the companion."""

    name: str = "wyn"
    enabled: bool = True
    animate: bool = True
    unicode: bool = True
    mood: Mood = Mood.IDLE
    style_name: str = "default"
    """``kawaii`` swaps in the rounder face set."""
    pace: int = 3
    """Frames the counter must advance before the face changes. Lower is
    faster; set_pace() drives it from the effort level."""
    _frame: int = field(default=0, repr=False)

    # -- appearance --------------------------------------------------------

    def faces(self) -> dict[Mood, list[str]]:
        if not self.unicode:
            return FACES_ASCII
        return FACES_KAWAII if self.style_name == "kawaii" else FACES

    def face(self, advance: bool = True) -> str:
        """The current frame. ``advance`` steps the animation.

        ``pace`` divides the frame counter, so a lower value animates faster.
        It is driven by the effort level: choosing ultra should visibly cost
        something, and a companion working visibly harder is a cheaper way to
        show that than a number nobody reads.
        """
        frames = self.faces()[self.mood]
        if advance and self.animate:
            self._frame += 1
        index = (self._frame // max(1, self.pace)) % len(frames) if self.animate else 0
        return frames[index]

    def set_pace(self, effort: str) -> None:
        """Faster animation the harder it is working."""
        from .effort import ORDER

        try:
            rank = ORDER.index(effort)
        except ValueError:
            return
        # 4 frames per step at low, down to 1 at ultra.
        self.pace = max(1, 4 - (rank * 3) // max(1, len(ORDER) - 1))

    def style(self) -> str:
        return MOOD_STYLES[self.mood]

    def width(self) -> int:
        """Widest frame for this mood, in cells, so the bar does not jitter."""
        return max(face_width(f) for f in self.faces()[self.mood])

    def padded(self, advance: bool = True) -> str:
        face = self.face(advance)
        return face + " " * max(0, self.width() - face_width(face))

    # -- state -------------------------------------------------------------

    def set_activity(self, activity: str) -> None:
        mood = ACTIVITY_MOODS.get(activity.strip().lower())
        if mood is None and activity:
            mood = Mood.THINKING
        self.react(mood or Mood.IDLE)

    def react(self, mood: Mood) -> None:
        if mood is not self.mood:
            self.mood = mood
            self._frame = 0

    def rest(self) -> None:
        self.react(Mood.IDLE)

    # -- voice -------------------------------------------------------------

    def remark(self, event: str) -> str:
        """One short line for an event, or "" when the pet is off."""
        if not self.enabled:
            return ""
        table = REMARKS_KAWAII if self.style_name == "kawaii" else REMARKS
        options = table.get(event)
        return random.choice(options) if options else ""

    def greeting(self) -> str:
        if not self.enabled:
            return ""
        return f"{self.face(advance=False)}  {self.name} — {self.remark('greet')}"
