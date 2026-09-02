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

from rich.text import Text

from dataclasses import dataclass, field
from enum import Enum


class Mood(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    READING = "reading"
    SEARCHING = "searching"
    WORKING = "working"
    TESTING = "testing"
    RUNNING = "running"
    ASKING = "asking"
    HAPPY = "happy"
    SAD = "sad"
    CELEBRATING = "celebrating"
    """The plan finished. Distinct from HAPPY, which is a reaction to one
    good step -- this is the end of the work, and it gets the big face."""
    SLEEPY = "sleepy"
    """Idle for a long while. The companion dozing off is a quieter way of
    saying nothing is happening than a spinner that never stops."""


# One cat, drawn rather than spelled.
#
# This replaces a one-line kaomoji -- the ₍ᐢ∙ᴥ∙ᐢ₎ shape -- which was
# width-stable, glyph-safe and bidi-safe, and still read as punctuation
# rather than as an animal. Every property it had was worth having and none
# of them was the point: a face that has to be explained is not a mascot.
#
# The mascot is three rows now, and the thing that makes it read as a cat is
# the pair of diagonal strokes in the first one. No single Unicode glyph
# draws an ear as well as "/\" does at this size, which is why the drawn
# version and the plain-terminal version are the same drawing rather than a
# good one and an apologetic fallback.
#
# Three rows will not fit on a status line, and that is the other half of
# the design: the mascot is no longer a status indicator. It appears where
# the session has room for a character -- the header, the farewell -- and
# the status line carries a single mark instead. See MARKS below.
#
# The character never changes. The ears, the cheeks and the body are
# identical in every frame; only the eyes and the mouth move. That is what
# makes a frame change read as an expression rather than as a different
# animal.

EARS = r" /\_/\ "
"""The top row, in every frame of every mood. Seven cells."""

WIDTH = 7
"""Cells every row occupies. One number for the whole set."""

HEIGHT = 3

FRAMES: dict[Mood, list[tuple[str, str]]] = {
    #                   eyes         mouth
    Mood.IDLE:        [("( o.o )", " > ^ < "), ("( o.o )", " > ^ < "),
                       ("( o.o )", " > ^ < "), ("( -.- )", " > ^ < ")],
    Mood.THINKING:    [("( o.- )", " > ^ < "), ("( -.o )", " > ^ < ")],
    Mood.READING:     [("( O.O )", " > ^ < "), ("( O.O )", " > ^ < "),
                       ("( -.- )", " > ^ < ")],
    Mood.SEARCHING:   [("( O.o )", " > ^ < "), ("( o.O )", " > ^ < ")],
    Mood.WORKING:     [("( >.< )", " > ^ < "), ("( >.< )", " > ^ < "),
                       ("( o.o )", " > ^ < ")],
    Mood.TESTING:     [("( O.- )", " > ^ < "), ("( -.O )", " > ^ < ")],
    Mood.RUNNING:     [("( o.o )", " >-^-< "), ("( o.o )", " > ^ < ")],
    Mood.ASKING:      [("( O.o )", " > ~ < "), ("( O.o )", " > ^ < ")],
    Mood.HAPPY:       [("( ^.^ )", " > w < ")],
    Mood.SAD:         [("( x.x )", " > _ < ")],
    Mood.CELEBRATING: [("( ^.^ )", " >*^*< "), ("( O.O )", " > w < ")],
    Mood.SLEEPY:      [("( -.- )", " > ^ < "), ("( -.- )", " > z < ")],
}

# The status line's share of the mascot: one mark, whose colour carries the
# rest. Four of them, because four is what a glance tells apart -- resting,
# working, went well, went wrong -- and because a status line competing with
# the answer above it has already lost.
MARKS: dict[Mood, str] = {
    Mood.IDLE: "◦", Mood.SLEEPY: "◦",
    Mood.THINKING: "◉", Mood.READING: "◉",
    Mood.SEARCHING: "◉", Mood.WORKING: "◉",
    Mood.TESTING: "◉", Mood.RUNNING: "◉",
    Mood.ASKING: "◉",
    Mood.HAPPY: "✓", Mood.CELEBRATING: "✓",
    Mood.SAD: "✗",
}
PULSE = ("◌", "◍", "◉", "◍")
PULSE_ASCII = (".", "o", "O", "o")
"""The one thing that moves while a tool runs.

The strip carries a single cell, so the animation has to live inside it:
one shape gaining and losing weight, which reads as breathing rather than
as a widget. It replaced the spinner and then, for one pass, nothing at all
-- the mark was drawn straight from the mood table and the whole strip went
still, so a call that took thirty seconds looked identical to a wedged one
apart from the elapsed count.

Every frame is East Asian Width Neutral and one cell wide. The strip is
width-exact and redrawn a dozen times a second; an Ambiguous glyph here
draws two cells in a CJK locale and tears the line on every other frame.
"""

MARKS_ASCII: dict[Mood, str] = {
    Mood.IDLE: ".", Mood.SLEEPY: ".",
    Mood.THINKING: "o", Mood.READING: "o", Mood.SEARCHING: "o",
    Mood.WORKING: "o", Mood.TESTING: "o", Mood.RUNNING: "o",
    Mood.ASKING: "o",
    Mood.HAPPY: "+", Mood.CELEBRATING: "+",
    Mood.SAD: "x",
}

# The mascot's colour, by the job the colour does rather than by name. Roles,
# not literals: these used to be grey62, bright_cyan and bright_magenta,
# which made the mascot the one thing on screen that /theme could not reach.
MOOD_ROLES: dict[Mood, str] = {
    Mood.IDLE: "muted",
    Mood.SLEEPY: "faint",
    Mood.THINKING: "bar_accent",
    Mood.READING: "bar_accent",
    Mood.SEARCHING: "bar_accent",
    Mood.WORKING: "bar_accent",
    Mood.TESTING: "bar_accent",
    Mood.RUNNING: "bar_accent",
    Mood.ASKING: "warn",
    Mood.HAPPY: "good",
    Mood.CELEBRATING: "good",
    Mood.SAD: "bad",
}

# Which activity name maps to which mood. Anything unrecognised stays THINKING,
# which is the honest default: something is happening and we did not label it.
_BUSY = frozenset({Mood.THINKING, Mood.READING, Mood.SEARCHING,
                   Mood.WORKING, Mood.TESTING, Mood.RUNNING})
"""The moods that mean a tool or the model is working right now."""

ACTIVITY_MOODS: dict[str, Mood] = {
    "thinking": Mood.THINKING,
    "planning": Mood.THINKING,
    "critiquing plan": Mood.THINKING,
    "reconciling": Mood.THINKING,
    "compacting context": Mood.THINKING,
    "reading": Mood.READING,
    "listing": Mood.READING,
    "finding": Mood.READING,
    "searching": Mood.SEARCHING,
    "testing": Mood.TESTING,
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
    "proud": ["nice one", "well done", "good work"],
    "bye": ["see you", "later", "until next time"],
}

REMARKS_KAWAII: dict[str, list[str]] = {
    "greet": ["ready when you are~", "what are we making today?", "listening~", "let's do our best! ♡", "nya~ ready~"],
    "done": ["all done~", "there we go", "finished~", "yay! completed! ✨", "mission accomplished~ ♡"],
    "denied": ["okay, leaving it", "no worries~", "that's fine~ ♡", "understood~"],
    "error": ["ah, that didn't work", "hit a wall, sorry", "uh oh~ 😭", "let's try again~"],
    "long": ["still going~", "this one's slow", "almost...", "hanging in there~", "patience~ ✨"],
    "interrupted": ["stopped~", "okay, dropping it", "understood~ ♡"],
    "proud": ["yay! ♡", "so proud of us~", "amazing~ ✨", "we did it! ♡"],
    "bye": ["bye bye~ ♡", "come back soon~", "see you~ ✨", "take care of yourself, okay?~"],
}

REMARKS_MOMMY: dict[str, list[str]] = {
    "greet": ["there's my goodboy~ what are we building?", "ready when you are, goodboy", "mommy's listening~", "let's do our best, goodboy ♡"],
    "done": ["goodboy, it's done~", "there we go, goodboy", "finished~ mommy's proud of you", "all done, goodboy ✨"],
    "denied": ["okay, leaving it then", "as you say, goodboy~", "fine, we'll leave it ♡"],
    "error": ["that didn't work, goodboy", "hit a wall -- let's look together", "no worries, goodboy, we'll fix it", "hmm, that one slipped -- try again~"],
    "long": ["still going, goodboy", "this one's slow, hang on", "almost there, goodboy", "patience, goodboy~ ✨"],
    "interrupted": ["stopped, goodboy", "okay, dropping it then", "as you say~ ♡"],
    "proud": ["goodboy, well done~ ♡", "mommy's proud of you, goodboy", "there's my goodboy ✨", "you did so well, goodboy~"],
    "bye": ["goodbye, goodboy~ mommy's here when you need her", "see you soon, goodboy ♡", "off you go, goodboy~ take care", "until next time, goodboy~"],
}


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
    _last_remark: dict = field(default_factory=dict, repr=False)
    """The line used last for each event, so the next one differs."""

    # -- appearance --------------------------------------------------------

    def _frame_index(self) -> int:
        frames = FRAMES[self.mood]
        if not self.animate:
            return 0
        return (self._frame // max(1, self.pace)) % len(frames)

    def rows(self, advance: bool = True) -> list[str]:
        """The mascot, as three rows of equal width.

        ``advance`` steps the animation. ``pace`` divides the frame counter,
        so a lower value animates faster; it follows the effort level,
        because choosing ultra should visibly cost something and a companion
        working visibly harder says that more cheaply than a number.
        """
        if advance and self.animate:
            self._frame += 1
        eyes, mouth = FRAMES[self.mood][self._frame_index()]
        return [EARS, eyes, mouth]

    def mark(self, advance: bool = False) -> str:
        """The one-cell stand-in, for lines with no room for a character.

        The status strip used to carry the whole mascot, which is how a
        companion ends up competing with the answer above it. It carries
        this instead: one cell whose colour says the mood.

        While something is running the cell breathes, on the same frame
        counter the drawing uses -- one clock, so nothing here can drift
        against the header or run on after the turn. At rest, waiting, or
        finished it holds still: motion means work in progress, and a mark
        that pulses after the answer has landed says the opposite.
        """
        table = MARKS if self.unicode else MARKS_ASCII
        if self.mood not in _BUSY or not self.animate:
            return table[self.mood]
        if advance:
            self._frame += 1
        frames = PULSE if self.unicode else PULSE_ASCII
        return frames[(self._frame // max(1, self.pace)) % len(frames)]

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
        """The mascot's colour right now, from the theme in force."""
        from . import theme

        return theme.active().role(MOOD_ROLES[self.mood])

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
        """One short line for an event, or "" when the pet is off.

        Never the same line twice running for the same event. There are only
        three or four of each and a session uses two or three of them, so a
        plain random choice repeats often enough to be noticed -- and a
        companion that greets you with the identical sentence every time you
        open it reads as a string constant rather than as a character.
        """
        if not self.enabled:
            return ""
        table = REMARKS
        if self.style_name == "kawaii":
            table = REMARKS_KAWAII
        elif self.style_name == "mommy":
            table = REMARKS_MOMMY
        options = table.get(event)
        if not options:
            return ""
        fresh = [line for line in options if line != self._last_remark.get(event)]
        chosen = random.choice(fresh or options)
        self._last_remark[event] = chosen
        return chosen

    def block(self, right: list | None = None, advance: bool = False):
        """The mascot with text set beside it, as a rich Text.

        One helper, because the header, the greeting and the farewell all
        want the same thing: the character on the left and a short stack of
        lines to its right, aligned to a single baseline. ``right`` is one
        entry per row; shorter lists leave the remaining rows blank.
        """
        rows = self.rows(advance=advance)
        right = list(right or [])
        right += [None] * (len(rows) - len(right))
        out = Text()
        for index, (art, beside) in enumerate(zip(rows, right)):
            if index:
                out.append("\n")
            out.append("  ")
            out.append(art, style=self.style())
            if beside is not None:
                out.append("   ")
                out.append_text(beside if isinstance(beside, Text)
                                else Text(str(beside)))
        return out

    def greeting(self) -> str:
        if not self.enabled:
            return ""
        return f"{self.name} — {self.remark('greet')}"
