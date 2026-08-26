"""Saying the answer out loud.

You type, she talks. Speech is an *output* channel only -- there is no
microphone anywhere in wynxo, and adding one would change what this program
is. Typing stays the input.

Nothing here is a Python dependency. Every engine is a program that is
already on the machine (``say`` on macOS, PowerShell's speech synthesiser on
Windows, ``termux-tts-speak`` on Android) or one the user installed on
purpose (``piper``, ``espeak-ng``). That keeps wynxo's install a pure-Python
one: no wheels, no compiler, no 60MB model downloaded behind your back.

The other half of the problem is *what* to say. A coding agent's answer is
full of paths, fences and diffs; read literally it is unlistenable. So
speakable() throws away everything that is not prose before a word reaches
the synthesiser -- see the tests for the shapes that matters for.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

from .platforms import is_macos, is_termux, is_windows

MAX_SPOKEN = 600
"""Characters. Past this she is reading an essay at you, not answering."""


@dataclass(frozen=True)
class Engine:
    name: str
    """What to show the user in /voice."""

    binary: str
    """The executable to look for, or "" when it is built into the OS."""

    quality: str
    """Plain-language, because "which TTS backend" is not a question anyone
    should have to have an opinion about."""


# Best first: the picker takes the first one that is actually present.
ENGINES: list[Engine] = [
    Engine("piper", "piper", "neural, most natural -- needs a voice model"),
    Engine("say", "say", "built into macOS, good quality"),
    Engine("powershell", "", "built into Windows"),
    Engine("termux", "termux-tts-speak", "Android's own voice"),
    Engine("espeak-ng", "espeak-ng", "robotic but always works"),
    Engine("espeak", "espeak", "robotic but always works"),
    Engine("flite", "flite", "robotic, very small"),
    Engine("spd-say", "spd-say", "whatever speech-dispatcher is set to"),
]


def _powershell() -> str | None:
    for candidate in ("pwsh", "powershell"):
        if shutil.which(candidate):
            return candidate
    return None


def available() -> list[Engine]:
    """Every engine that could speak on this machine, best first."""
    found: list[Engine] = []
    for engine in ENGINES:
        if engine.name == "powershell":
            if is_windows() and _powershell():
                found.append(engine)
            continue
        if engine.name == "say" and not is_macos():
            continue
        if engine.name == "termux" and not is_termux():
            continue
        if engine.binary and shutil.which(engine.binary):
            found.append(engine)
    return found


def pick(preferred: str = "auto") -> Engine | None:
    """Resolve a configured engine name to something that exists."""
    options = available()
    if not options:
        return None
    if preferred and preferred != "auto":
        for engine in options:
            if engine.name == preferred:
                return engine
        return None
    return options[0]


def install_hint() -> str:
    """What to install, for the platform actually in front of them."""
    if is_termux():
        return "pkg install termux-api   (and the Termux:API app from F-Droid)"
    if is_windows():
        return "PowerShell is required and is part of Windows."
    if is_macos():
        return "`say` is part of macOS and should already be there."
    return ("sudo apt install espeak-ng     (Debian/Ubuntu)\n"
            "sudo dnf install espeak-ng     (Fedora)\n"
            "For a far better voice, install piper: "
            "https://github.com/rhasspy/piper")


# -- what is worth saying --------------------------------------------------

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# The link text cannot contain "[" either. Without that exclusion the
# class eats to the end of the answer at every "[", fails to find the
# closing bracket, and gives the characters back one at a time -- and
# an answer full of unclosed brackets (a log dump, array indexing, raw
# escape sequences) is not unusual. CPython 3.11 optimises the worst of
# it away; on 3.10 an 80k answer took three and a half seconds.
#
# It also reads the way CommonMark does: in "[a [b](c)" the inner link
# is the link.
_LINK = re.compile(r"\[([^\][]+)\]\([^)]*\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(.+?)\1", re.DOTALL)
_PATHY = re.compile(r"(?<!\S)\S*[/\\]\S*")
"""Anything with a slash in it. Read aloud, a path is a stream of
punctuation names -- and the answer almost always says what the file *is*
right next to it anyway.

Only a run that starts a word can match, which is what the greedy
form found anyway -- but saying so keeps it linear. Left open, \\S*
ate to the end at every position and gave the characters back one at
a time: a 40k answer took ten seconds to prepare for speech.
"""


def speakable(text: str, limit: int = MAX_SPOKEN) -> str:
    """Reduce an answer to the part a person would actually want read out.

    Code, diffs, tool markup and paths come out. What is left is the
    sentences, which is the half that carries the meaning.
    """
    if not text:
        return ""

    out = _THINK.sub(" ", text)
    out = _TOOL_CALL.sub(" ", out)
    out = _FENCE.sub(" ", out)
    out = _LINK.sub(r"\1", out)
    out = _HEADING.sub("", out)
    out = _BULLET.sub("", out)
    out = _EMPHASIS.sub(r"\2", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _PATHY.sub(" ", out)

    # Table rows and horizontal rules read as noise.
    lines = [line for line in out.splitlines()
             if not line.strip().startswith("|")
             and not re.fullmatch(r"\s*[-=_]{3,}\s*", line)]
    out = " ".join(lines)
    out = re.sub(r"\s+", " ", out).strip()

    if len(out) <= limit:
        return out
    # Cut at a sentence end so she does not stop mid-word.
    clipped = out[:limit]
    for stop in (". ", "! ", "? "):
        cut = clipped.rfind(stop)
        if cut > limit // 2:
            return clipped[: cut + 1].strip()
    return clipped.rsplit(" ", 1)[0].strip()


# -- saying it -------------------------------------------------------------

def command(engine: Engine, text: str, voice: str = "", rate: int = 0,
            model: str = "") -> list[str] | None:
    """The argv to run. Text goes as an argument, never through a shell."""
    if engine.name == "say":
        args = ["say"]
        # Samantha ships with macOS and is the closest thing to a default
        # female voice; -v is ignored gracefully if she is not installed.
        args += ["-v", voice or "Samantha"]
        if rate:
            args += ["-r", str(rate)]
        return args + ["--", text]

    if engine.name == "powershell":
        shell = _powershell()
        if not shell:
            return None
        # Build the script with the text injected as a single-quoted
        # PowerShell literal: doubling ' is the whole escaping rule there,
        # and it cannot break out of the string.
        literal = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.SelectVoiceByHints('Female');"
        )
        if rate:
            script += f"$s.Rate = {max(-10, min(10, rate))};"
        script += f"$s.Speak('{literal}')"
        return [shell, "-NoProfile", "-NonInteractive", "-Command", script]

    if engine.name == "termux":
        args = ["termux-tts-speak"]
        if rate:
            args += ["-r", str(rate)]
        return args + ["--", text]

    if engine.name in ("espeak-ng", "espeak"):
        # +f3 is espeak's third female variant: the least robotic of them.
        args = [engine.binary, "-v", voice or "en+f3"]
        if rate:
            args += ["-s", str(rate)]
        return args + ["--", text]

    if engine.name == "flite":
        return ["flite", "-voice", voice or "slt", "-t", text]

    if engine.name == "spd-say":
        args = ["spd-say", "-w"]
        if voice:
            args += ["-y", voice]
        return args + ["--", text]

    if engine.name == "piper":
        if not model:
            return None
        return ["piper", "--model", model, "--output-raw"]

    return None


class Speaker:
    """Speaks one thing at a time, and never blocks the agent loop.

    A new utterance replaces whatever is still being said: the answer you
    just got is more interesting than the tail of the previous one, and
    letting them queue up means she falls further behind the longer you use
    it.
    """

    def __init__(self, engine: Engine | None = None, voice: str = "",
                 rate: int = 0, model: str = "") -> None:
        self.engine = engine
        self.voice = voice
        self.rate = rate
        self.model = model
        self.enabled = engine is not None
        self._process: subprocess.Popen | None = None
        self.last_error = ""

    def stop(self) -> None:
        """Cut her off. Ctrl-C should silence her, not just the model."""
        process, self._process = self._process, None
        if process and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            # Reaped rather than abandoned. A speech engine goes on SIGTERM
            # immediately, so this returns at once; without it the child sits
            # as a zombie until something else happens to make a subprocess.
            try:
                process.wait(timeout=0.5)
            except Exception:
                pass

    def say(self, text: str) -> bool:
        """Start speaking. Returns False if nothing was said."""
        if not self.enabled or self.engine is None:
            return False
        spoken = speakable(text)
        if not spoken:
            return False

        argv = command(self.engine, spoken, self.voice, self.rate, self.model)
        if argv is None:
            return False

        self.stop()
        try:
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE if self.engine.name == "piper" else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if self.engine.name == "piper" and self._process.stdin:
                # piper reads its text on stdin and writes raw audio out;
                # without a player that is silence, so this path stays
                # unsupported rather than pretending to work.
                self._process.stdin.write(spoken.encode("utf-8"))
                self._process.stdin.close()
        except OSError as exc:
            self.last_error = str(exc)
            self.enabled = False
            return False
        return True

    async def say_async(self, text: str) -> bool:
        """Speak without holding up whatever called it."""
        return await asyncio.get_running_loop().run_in_executor(None, self.say, text)

    def is_speaking(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def describe(self) -> str:
        if not self.enabled or self.engine is None:
            return "off"
        bits = [self.engine.name]
        if self.voice:
            bits.append(self.voice)
        return " ".join(bits)
