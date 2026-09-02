"""The companion's voice: its name, and the short lines it says.

This module used to draw as well. It held a face -- a kaomoji, then line
art -- that sat in the header and in the status strip, while a second
character with a laptop lived unrendered in ``companion.py`` and a third
copy of that one was wrapped for previews in ``motion.py``. Three drawings
of one animal, of which the one you actually saw was the smallest.

The drawing is now ``sprite.py``, once, and this is what is left: who the
companion is and how it talks. The split is worth having on its own terms.
A voice is text and a picture is pixels; they change for different reasons,
they are configured separately (``/pet voice`` against ``/animate``), and
nothing here needs to know how wide a terminal is.

It never gates behaviour and never speaks for the model.
"""

from __future__ import annotations

import random

from dataclasses import dataclass, field


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
    """The companion's name and voice.

    No mood and no frames. What the companion is *doing* is not a property
    of this object: it is a fact about the agent, read from the task state
    and the running tool when the sprite is drawn. Keeping a second copy
    here is how a character ends up cheerfully animating through a turn
    that failed a minute ago.
    """

    name: str = "wyn"
    enabled: bool = True
    animate: bool = True
    unicode: bool = True
    style_name: str = "default"
    """Which set of remarks to draw from: default, kawaii or mommy."""
    _last_remark: dict = field(default_factory=dict, repr=False)
    """The line used last for each event, so the next one differs."""

    def remark(self, event: str) -> str:
        """One short line for an event, or "" when the companion is off.

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
        fresh = [line for line in options
                 if line != self._last_remark.get(event)]
        chosen = random.choice(fresh or options)
        self._last_remark[event] = chosen
        return chosen

    def greeting(self) -> str:
        if not self.enabled:
            return ""
        return f"{self.name} — {self.remark('greet')}"
