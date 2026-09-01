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


# One cat, in one size.
#
# Two rules hold the whole set together, and both were broken before.
#
# EVERY FRAME IS EXACTLY ``BOX`` CELLS. Frames used to be padded to the
# widest frame *of the current mood*, which fixes jitter inside a mood and
# not between them: idle was 7 cells and running 8, so every time the agent
# picked up a tool the entire status line shifted sideways by one column.
# The ASCII set was worse, moving between 5 and 7. The box is one width for
# the whole session, so nothing after the mascot can ever move.
#
# EVERY GLYPH IS SAFE TO PUT IN A LINE OF TEXT -- unambiguous width, and
# left-to-right. The old eyes were U+2022 BULLET,
# whose East Asian Width is "Ambiguous" -- one cell in a Western locale and
# two in a CJK one. So were the breve, the ≧≦ squint, the ╥ tears and the ×.
# WORKING was built from combining accents, which terminals place however
# they like. Those are all fine in prose and wrong in an animation, where a
# glyph that measures 1 and draws 2 tears the line on the frame it appears.
# Everything here is width "N" or "Na" and carries no combining marks.
#
# The muzzle is U+1D25 LATIN LETTER AIN, the one from ʕ•ᴥ•ʔ, and not the
# Arabic presentation form that looks the same: that glyph's bidi class is
# AL, so a terminal implementing the bidirectional algorithm is entitled to
# reorder the neutrals either side of it -- the eyes -- and the face comes
# apart. A mascot may not depend on the reading direction of the sentence it
# lands in.
#
# The character is the same in every frame: the ears, the body and the mouth
# never move. Only the eyes change, plus one optional cell to the right for
# a paw, a question mark or a sleepy z. That is what makes a frame change
# read as an expression rather than as a different animal.

BOX = 8
"""Cells every frame occupies: seven for the face, one for the accessory."""

FACES: dict[Mood, list[str]] = {
    Mood.IDLE:        ["₍ᐢ∙ᴥ∙ᐢ₎ ", "₍ᐢ∙ᴥ∙ᐢ₎ ", "₍ᐢ∙ᴥ∙ᐢ₎ ", "₍ᐢ-ᴥ-ᐢ₎ "],
    Mood.THINKING:    ["₍ᐢ∙ᴥ-ᐢ₎ ", "₍ᐢ-ᴥ∙ᐢ₎ ", "₍ᐢ∙ᴥ-ᐢ₎ ", "₍ᐢ∙ᴥ∙ᐢ₎ "],
    Mood.READING:     ["₍ᐢ◉ᴥ◉ᐢ₎ ", "₍ᐢ◉ᴥ◉ᐢ₎ ", "₍ᐢ◉ᴥ◉ᐢ₎ ", "₍ᐢ-ᴥ-ᐢ₎ "],
    # The eyes track left and right: a cat looking for something.
    Mood.SEARCHING:   ["₍ᐢ◉ᴥ∙ᐢ₎ ", "₍ᐢ∙ᴥ◉ᐢ₎ ", "₍ᐢ◉ᴥ∙ᐢ₎ "],
    Mood.WORKING:     ["₍ᐢ>ᴥ<ᐢ₎ ", "₍ᐢ>ᴥ<ᐢ₎ ", "₍ᐢ∙ᴥ∙ᐢ₎ "],
    # Scrutiny: one eye narrowed, then the other. Reading is two wide eyes
    # held still, and the two states have to be told apart at a glance.
    Mood.TESTING:     ["₍ᐢ◉ᴥ-ᐢ₎ ", "₍ᐢ◉ᴥ-ᐢ₎ ", "₍ᐢ-ᴥ◉ᐢ₎ ", "₍ᐢ-ᴥ◉ᐢ₎ "],
    Mood.RUNNING:     ["₍ᐢ∙ᴥ∙ᐢ₎ฅ", "₍ᐢ∙ᴥ∙ᐢ₎ﾉ", "₍ᐢ∙ᴥ∙ᐢ₎ฅ", "₍ᐢ∙ᴥ∙ᐢ₎ﾉ"],
    Mood.ASKING:      ["₍ᐢ◉ᴥ∙ᐢ₎?", "₍ᐢ◉ᴥ∙ᐢ₎ ", "₍ᐢ◉ᴥ∙ᐢ₎?", "₍ᐢ◉ᴥ∙ᐢ₎ "],
    Mood.HAPPY:       ["₍ᐢ‿ᴥ‿ᐢ₎ ", "₍ᐢ‿ᴥ‿ᐢ₎✧"],
    Mood.SAD:         ["₍ᐢxᴥxᐢ₎ ", "₍ᐢxᴥxᐢ₎ ", "₍ᐢ-ᴥ-ᐢ₎ "],
    Mood.CELEBRATING: ["₍ᐢ‿ᴥ‿ᐢ₎✧", "₍ᐢ◉ᴥ◉ᐢ₎ ", "₍ᐢ‿ᴥ‿ᐢ₎✦", "₍ᐢ◉ᴥ◉ᐢ₎ "],
    Mood.SLEEPY:      ["₍ᐢ-ᴥ-ᐢ₎ ", "₍ᐢ-ᴥ-ᐢ₎z", "₍ᐢ-ᴥ-ᐢ₎z", "₍ᐢ-ᴥ-ᐢ₎ "],
}

# The same cat where the font cannot be trusted with anything but ASCII.
# Same grammar, same box, same expressions -- ears, eyes, nose, accessory --
# so it is recognisably the character rather than a different mascot for
# people with a plainer terminal.
FACES_ASCII: dict[Mood, list[str]] = {
    Mood.IDLE:        ["=^o.o^= ", "=^o.o^= ", "=^o.o^= ", "=^-.-^= "],
    Mood.THINKING:    ["=^o.-^= ", "=^-.o^= ", "=^o.-^= ", "=^o.o^= "],
    Mood.READING:     ["=^O.O^= ", "=^O.O^= ", "=^O.O^= ", "=^-.-^= "],
    Mood.SEARCHING:   ["=^O.o^= ", "=^o.O^= ", "=^O.o^= "],
    Mood.WORKING:     ["=^>.<^= ", "=^>.<^= ", "=^o.o^= "],
    Mood.TESTING:     ["=^O.-^= ", "=^O.-^= ", "=^-.O^= ", "=^-.O^= "],
    Mood.RUNNING:     ["=^o.o^=/", "=^o.o^=-", "=^o.o^=\\", "=^o.o^=-"],
    Mood.ASKING:      ["=^O.o^=?", "=^O.o^= ", "=^O.o^=?", "=^O.o^= "],
    Mood.HAPPY:       ["=^u.u^= ", "=^u.u^=*"],
    Mood.SAD:         ["=^x.x^= ", "=^x.x^= ", "=^-.-^= "],
    Mood.CELEBRATING: ["=^u.u^=*", "=^O.O^= ", "=^u.u^=+", "=^O.O^= "],
    Mood.SLEEPY:      ["=^-.-^= ", "=^-.-^=z", "=^-.-^=z", "=^-.-^= "],
}

# The mascot's colour, by the job the colour does rather than by name. Four
# roles, because four is what a glance can tell apart: resting, busy, went
# well, went wrong. The face already says *which* kind of busy -- the eyes
# differ per mood and the strip spells the activity out beside it -- so a
# fifth and sixth hue would be the same fact a third time.
#
# Roles, not literals, is the point. These used to be grey62, bright_cyan,
# bright_magenta and so on, which made the mascot the one thing on screen
# that /theme could not reach: catboy's violet never touched the cat.
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
    _last_remark: dict = field(default_factory=dict, repr=False)
    """The line used last for each event, so the next one differs."""

    # -- appearance --------------------------------------------------------

    def faces(self) -> dict[Mood, list[str]]:
        """The frame set for this terminal.

        One cat, in two tiers: the drawn one, and the ASCII one for a font
        that cannot be trusted with the rest. There used to be a third --
        the kawaii voice got a different animal from the default voice --
        so what the mascot *was* depended on a personality setting. Voice
        changes what it says; it does not change the species.
        """
        return FACES if self.unicode else FACES_ASCII

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
        """The mascot's colour right now, from the theme in force."""
        from . import theme

        return theme.active().role(MOOD_ROLES[self.mood])

    def width(self) -> int:
        """Cells the mascot occupies. One number, for the whole session.

        It used to be the widest frame *of the current mood*, which stops
        the jitter inside a mood and not between them -- idle was seven
        cells and running eight, so the whole status line stepped sideways
        every time the agent picked up a tool. The frames are all one width
        by construction now; this stays as the guarantee, so a frame added
        later cannot quietly reintroduce the shift.
        """
        return BOX

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

    def greeting(self) -> str:
        if not self.enabled:
            return ""
        return f"{self.face(advance=False)}  {self.name} — {self.remark('greet')}"
