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
    Mood.IDLE:     ["(•ᴗ•)", "(•ᴗ•)", "(•ᴗ•)", "(-ᴗ-)"],
    Mood.THINKING: ["(◐ᴗ◐)", "(◓ᴗ◓)", "(◑ᴗ◑)", "(◒ᴗ◒)"],
    Mood.READING:  ["(◉ᴗ◉)", "(◉ᴗ◉)", "(◉ᴗ◉)", "(-ᴗ-)"],
    Mood.WORKING:  ["(¬ᴗ¬)", "(ºᴗº)"],
    Mood.RUNNING:  ["(•ᴗ•)╭", "(•ᴗ•)─", "(•ᴗ•)╰", "(•ᴗ•)─"],
    Mood.ASKING:   ["(•ᴗ•)?", "(•ᴗ•) ", "(•ᴗ•)?", "(•ᴗ•) "],
    Mood.HAPPY:    ["(≧ᴗ≦)", "(ˆᴗˆ)"],
    Mood.SAD:      ["(×ᴗ×)", "(×ᴗ×)", "(⊙ᴗ⊙)"],
}

FACES_ASCII: dict[Mood, list[str]] = {
    Mood.IDLE:     ["(o_o)", "(o_o)", "(o_o)", "(-_-)"],
    Mood.THINKING: ["(o_O)", "(O_o)", "(o_O)", "(O_o)"],
    Mood.READING:  ["(0_0)", "(0_0)", "(0_0)", "(-_-)"],
    Mood.WORKING:  ["(>_<)", "(>_>)"],
    Mood.RUNNING:  ["(o_o)/", "(o_o)-", "(o_o)\\", "(o_o)-"],
    Mood.ASKING:   ["(o_o)?", "(o_o) ", "(o_o)?", "(o_o) "],
    Mood.HAPPY:    ["(^_^)", "(^-^)"],
    Mood.SAD:      ["(x_x)", "(x_x)", "(@_@)"],
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
    _frame: int = field(default=0, repr=False)

    # -- appearance --------------------------------------------------------

    def faces(self) -> dict[Mood, list[str]]:
        return FACES if self.unicode else FACES_ASCII

    def face(self, advance: bool = True) -> str:
        """The current frame. ``advance`` steps the animation."""
        frames = self.faces()[self.mood]
        if advance and self.animate:
            self._frame += 1
        index = (self._frame // 3) % len(frames) if self.animate else 0
        return frames[index]

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
        options = REMARKS.get(event)
        return random.choice(options) if options else ""

    def greeting(self) -> str:
        if not self.enabled:
            return ""
        return f"{self.face(advance=False)}  {self.name} — {self.remark('greet')}"
