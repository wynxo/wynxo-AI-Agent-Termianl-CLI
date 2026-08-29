"""Saying the answer out loud.

You type, she talks. Speech is an *output* channel only -- there is no
microphone anywhere in wynxo, and adding one would change what this program
is. Typing stays the input.

Nothing here is a Python dependency. Every engine is a program that is
already on the machine (``say`` on macOS, PowerShell's speech synthesiser on
Windows, ``termux-tts-speak`` on Android) or one the user installed on
purpose (``piper``, ``espeak-ng``, ``edge-tts``). That keeps wynxo's install
a pure-Python one: no wheels, no compiler, no 60MB model downloaded behind
your back.

The one voice worth going out of your way for is ``edge-tts``: a free
``pip install edge-tts`` and she speaks with Microsoft's neural voices (the
same ones Edge reads aloud with), which sound like a person rather than a
phone lady. Windows' built-in SAPI voices are the robotic ones -- wynxo
prefers any installed natural voice (Aria, Jenny, ...) over Zira on its
own.

The other half of the problem is *what* to say. A coding agent's answer is
full of paths, fences and diffs; read literally it is unlistenable. So
speakable() throws away everything that is not prose before a word reaches
the synthesiser -- see the tests for the shapes that matters for.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    # Microsoft's neural voices (the ones Edge reads aloud with): free, no
    # API key, and genuinely human-sounding -- the difference between a
    # woman and a phone lady. Needs `pip install edge-tts` once; synthesis
    # goes over the network and is played back by ffplay/mpv/afplay (or the
    # default player).
    Engine("edge-tts", "edge-tts",
           "Microsoft's neural voices -- most human, needs pip install edge-tts"),
    Engine("piper", "piper", "neural, most natural -- needs a voice model"),
    Engine("say", "say", "built into macOS, good quality"),
    Engine("powershell", "", "built into Windows"),
    Engine("termux", "termux-tts-speak", "Android's own voice"),
    Engine("espeak-ng", "espeak-ng", "robotic but always works"),
    Engine("espeak", "espeak", "robotic but always works"),
    Engine("flite", "flite", "robotic, very small"),
    Engine("spd-say", "spd-say", "whatever speech-dispatcher is set to"),
]


# Female natural voices, best-known first. A Windows install that has any of
# them (they are free from Settings > Time & language > Speech in Windows
# 11) sounds like a person; without them the only female SAPI voice is the
# robotic Zira.
_NATURAL_VOICES = ("Aria", "Jenny", "Michelle", "Ana", "Emma", "Cora",
                   "Libby", "Neerja", "Sonia", "Zira")


def _player() -> list[str] | None:
    """How to play a synthesized audio file, quietest first.

    ffplay's -nodisp means no window pops up; mpv --no-video likewise. On
    Windows with neither, the default player is the fallback -- it opens a
    window, but it plays.
    """
    for binary, args in (("ffplay", ["-nodisp", "-autoexit"]),
                         ("mpv", ["--no-video"]),
                         ("afplay", [])):
        if shutil.which(binary):
            return [binary, *args]
    if is_windows():
        return ["cmd", "/c", "start", ""]
    return None


def _powershell() -> str | None:
    for candidate in ("pwsh", "powershell"):
        if shutil.which(candidate):
            return candidate
    return None


def _edge_tts_here() -> list[str] | None:
    """The argv that runs edge-tts in this interpreter, or None.

    Suites by importability rather than PATH: ``python -m edge_tts`` works
    the moment ``pip install edge-tts`` put the package in the same
    interpreter wynxo runs in, which is the setup where a PATH-only check
    silently misses it and wynxo falls back to the robotic SAPI voice.
    """
    try:
        if importlib.util.find_spec("edge_tts") is None:
            return None
    except (ImportError, ValueError):
        return None
    return [sys.executable, "-m", "edge_tts"]


def available() -> list[Engine]:
    """Every engine that could speak on this machine, best first."""
    found: list[Engine] = []
    for engine in ENGINES:
        if engine.name == "edge-tts":
            # Available when the module is importable in this interpreter
            # (run via ``python -m edge_tts``) *or* a bare binary is on
            # PATH -- either way it beats the robotic SAPI voice.
            if _edge_tts_here() or (engine.binary
                                    and shutil.which(engine.binary)):
                found.append(engine)
            continue
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
        return ("PowerShell is required and is part of Windows.\n"
                "For a voice that sounds like a person rather than a phone "
                "lady: pip install edge-tts, then /speak engine edge-tts. "
                "Or install a free natural voice (Aria, Jenny, ...) from "
                "Settings > Time & language > Speech and wynxo will prefer "
                "it over Zira automatically.")
    if is_macos():
        return "`say` is part of macOS and should already be there."
    return ("sudo apt install espeak-ng     (Debian/Ubuntu)\n"
            "sudo dnf install espeak-ng     (Fedora)\n"
            "For a far better voice, install piper: "
            "https://github.com/rhasspy/piper")


# Warm female Microsoft neural voices, best first. edge-tts speaks these
# over the network; the list stays short because browsing all six hundred
# machine voices is not choosing. 'voice' lets them pick one, and a custom
# name can always be typed in.
MOMMY_VOICES: list[tuple[str, str]] = [
    ("en-US-JennyNeural", "warm, soft -- the mommy default"),
    ("en-GB-SoniaNeural", "British, very natural"),
    ("en-CA-ClaraNeural", "warm and friendly"),
    ("en-US-AriaNeural", "clear and expressive"),
    ("en-US-MichelleNeural", "warm and calm"),
    ("en-GB-LibbyNeural", "warm British"),
    ("en-AU-NatashaNeural", "warm Australian"),
]


def _pip(argv: list[str]) -> tuple[bool, str]:
    """Run pip through the interpreter wynxo runs with."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip"] + argv,
            capture_output=True, text=True, timeout=600)
    except Exception as exc:            # noqa: BLE001 - one message, any cause
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def install_edge_tts() -> tuple[bool, str]:
    """Bring Microsoft's neural voices into this interpreter.

    Returns (installed_ok, detail). Installs into the same interpreter
    wynxo runs in, which is exactly where ``python -m edge_tts`` looks,
    so a success makes the voices available without a PATH change.
    """
    if _edge_tts_here() is not None:
        return True, "already installed"
    ok, out = _pip(["install", "-q", "edge-tts"])
    if _edge_tts_here() is not None:
        return True, "installed"
    return False, (out or "pip exited without installing.").strip()[-400:]


async def list_edge_voices() -> list[tuple[str, str]]:
    """The female neural voices edge-tts can presently use, newest first.

    Fetched live from Microsoft's list so the picker is never out of date;
    empty when offline or edge-tts is missing.
    """
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
    except Exception:
        return []
    female = [v for v in voices
              if str(v.get("Gender", "")).lower() == "female"]
    ordered = sorted(female,
                     key=lambda v: (str(v.get("Locale", "")),
                                    str(v.get("ShortName", ""))))
    out: list[tuple[str, str]] = []
    for v in ordered:
        name = v.get("ShortName")
        if not name:
            continue
        hint = f"{v.get('Locale', '')}  {v.get('FriendlyName', '')}".strip()
        out.append((str(name), hint))
    return out


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
        )
        if voice:
            # An explicit voice wins. SelectVoice throws on a name it does
            # not know, so the fallback is still a female voice.
            v = voice.replace("'", "''")
            script += (
                f"try {{ $s.SelectVoice('{v}') }} "
                f"catch {{ $s.SelectVoiceByHints('Female') }};"
            )
        else:
            # Otherwise pick the most natural female voice installed, not
            # just "a female voice" -- on a stock Windows install that is
            # the robotic Zira, and a natural voice (Aria, Jenny, ...) is
            # free from Settings > Time & language > Speech. They are
            # discoverable by name; prefer the known ones in order.
            names = "|".join(_NATURAL_VOICES)
            script += (
                "$g = $s.GetInstalledVoices() | ForEach-Object "
                "{ $_.VoiceInfo } | Where-Object "
                "{ $_.Gender -eq 'Female' };"
                f"$pick = $g | Where-Object {{ $_.Name -match '{names}' }} "
                "| Select-Object -First 1;"
                "if ($pick) { $s.SelectVoice($pick.Name) } "
                "elseif ($g) { $s.SelectVoice($g[0].Name) } "
                "else { $s.SelectVoiceByHints('Female') };"
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

    if engine.name == "edge-tts":
        # Two steps: synthesise to a file, then play it. The synthesis half
        # is here; the Speaker owns the play half so it can clean up the
        # file afterwards. Run through ``python -m edge_tts`` in *this*
        # interpreter when it is importable, so the neural voice works
        # without the venv's Scripts dir being on PATH -- the usual way a
        # robotic Zira sneaks in. Fall back to a bare ``edge-tts`` binary
        # that happens to be on PATH.
        launcher = _edge_tts_here() or ["edge-tts"]
        return [*launcher, "--voice", voice or "en-US-AriaNeural",
                "--text", text, "--write-media"]

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
        self._media_path: str = ""
        """A synthesized audio file still being played (edge-tts). Removed
        when she is stopped or replaced, so temp files do not pile up."""
        self.last_error = ""

    def stop(self) -> None:
        """Cut her off. Ctrl-C should silence her, not just the model."""
        import os as _os

        process, self._process = self._process, None
        if self._media_path:
            try:
                _os.unlink(self._media_path)
            except OSError:
                pass
            self._media_path = ""
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

        if self.engine.name == "edge-tts":
            return self._say_edge_tts(spoken)

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

    def _say_edge_tts(self, text: str) -> bool:
        """Synthesise with edge-tts, then play the file it wrote.

        edge-tts (Microsoft's neural voices) writes an audio file rather
        than speaking directly, so this is two steps: synth to a temp file,
        then hand it to a quiet player. The file lives for the length of
        the utterance and is removed when she is stopped or replaced.
        """
        player = _player()
        if player is None:
            self.last_error = "no audio player found (ffplay, mpv, afplay)"
            self.enabled = False
            return False

        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        argv = command(self.engine, text, self.voice, self.rate, self.model)
        if argv is None:
            os.unlink(path)
            return False
        argv = argv + [path]
        try:
            synth = subprocess.run(argv, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            os.unlink(path)
            self.last_error = str(exc)
            self.enabled = False
            return False
        if synth.returncode != 0:
            os.unlink(path)
            detail = synth.stderr.decode("utf-8", "replace").strip()[:200]
            self.last_error = f"edge-tts failed ({synth.returncode}): {detail}"
            self.enabled = False
            return False

        self.stop()
        try:
            self._process = subprocess.Popen(
                player + [path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._media_path = path
        except OSError as exc:
            os.unlink(path)
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
