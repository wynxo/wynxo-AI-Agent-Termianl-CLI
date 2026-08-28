"""Kawaii effects and animations for the terminal UI.

Sparkles, cute text effects, enhanced animations, and visual polish to make
the terminal a delightful place to work. This module provides a collection of
decorative effects that can be sprinkled throughout the UI to enhance the
aesthetic without cluttering the functional content.
"""

from __future__ import annotations

import random
from typing import Sequence

# Sparkle and shine effects
SPARKLES = ["✨", "✦", "★", "⭐", "🌟", "💫", "✧", "◆"]
SPARKLES_SUBTLE = ["·", "•", "◦", "°", "¸", "·", "˙"]
HEARTS = ["♥", "💕", "♡", "💖", "💗"]
STARS = ["✨", "⭐", "🌟", "✦", "★", "✧"]
PAWS = ["🐾", "🐕", "🐈", "ㄑ", "ㄣ"]
ARROWS_KAWAII = ["→", "→", "⟶", "➜", "→"]
BOXES_KAWAII = ["◜", "◝", "◟", "◞"]

# Extra cute symbols
FLOWERS = ["✿", "❀", "✾", "❁", "❋"]
MOONS = ["☾", "☽", "🌙", "🌛", "🌜"]
CATS = ["ᐛ", "ᐜ", "ᐩ", "ᐨ"]
SPARKLE_VARIATIONS = ["✧", "✦", "★", "⭐", "✨", "💫", "🌟"]

# Cute loading animations
LOADING_SPARKLE = ["✧･ﾟ: *✧･ﾟ:*", "*:･ﾟ✧*:･ﾟ✧", "✧*:･ﾟ✧*:･ﾟ", "✧･ﾟ: *✧*:･ﾟ✧"]
LOADING_PULSE = ["◑", "◒", "◐", "◓"]
LOADING_BREATHING = ["●", "⊙", "◉", "⊙"]
LOADING_DOTS = ["", "·", "··", "···", "····"]
LOADING_WAVES = ["▌ ", " ▌", "  ▌", " ▌"]
LOADING_CAT = ["= ^.^ =", "( ^.^ )", "= ^.^ =", "( ^.^ )"]
LOADING_STARS = ["✦", "✧", "★", "⭐", "✨", "🌟", "⭐", "★"]
LOADING_HEARTS = ["♡", "♥", "💕", "💖", "💗", "💖", "💕", "♥"]

# Text decoration
def sparkle_text(text: str, marker: str = "✨") -> str:
    """Wrap text in sparkles."""
    return f"{marker} {text} {marker}"


def heart_text(text: str) -> str:
    """Wrap text with hearts."""
    heart = random.choice(HEARTS)
    return f"{heart} {text} {heart}"


def paw_text(text: str) -> str:
    """Add paw prints around text."""
    paw = random.choice(PAWS)
    return f"{paw} {text} {paw}"


def cute_box(text: str, corner: str = "◜") -> str:
    """Simple cute box around text."""
    return f"│ {text} │"


def add_sparkles(text: str, density: float = 0.3) -> str:
    """Randomly add sparkles throughout text, respecting word boundaries.
    
    density: fraction of words to decorate (0.0-1.0)
    """
    words = text.split()
    for i, word in enumerate(words):
        if random.random() < density and not word.startswith("@"):
            words[i] = f"✨{word}"
    return " ".join(words)


def kawaii_loading(frame: int, style: str = "sparkle") -> str:
    """Get a kawaii loading animation frame.
    
    frame: current frame number (will be cycled)
    style: 'sparkle', 'pulse', 'breathing', 'dots', 'waves', 'cat', 'stars', or 'hearts'
    """
    animations = {
        "sparkle": LOADING_SPARKLE,
        "pulse": LOADING_PULSE,
        "breathing": LOADING_BREATHING,
        "dots": LOADING_DOTS,
        "waves": LOADING_WAVES,
        "cat": LOADING_CAT,
        "stars": LOADING_STARS,
        "hearts": LOADING_HEARTS,
    }
    frames = animations.get(style, LOADING_SPARKLE)
    return frames[frame % len(frames)]


def rainbow_text(text: str, colors: Sequence[str] | None = None) -> str:
    """Return text with rainbow color codes (for rich markup).
    
    colors: sequence of rich color names/codes, defaults to rainbow
    """
    if colors is None:
        colors = ["#ff0000", "#ff7f00", "#ffff00", "#00ff00", "#0000ff", "#4b0082", "#9400d3"]
    
    result = ""
    for i, char in enumerate(text):
        color = colors[i % len(colors)]
        result += f"[{color}]{char}[/]"
    return result


# Cute status messages that replace boring technical ones
STATUS_MESSAGES = {
    "planning": [
        "thinking about it~",
        "strategizing... ✨",
        "plotting course...",
        "making a plan~ ♡",
        "ideas brewing~",
        "scheming~ ✧",
        "brain go brrr~",
    ],
    "reading": [
        "examining files~",
        "reading carefully... ◉ω◉",
        "taking a look~",
        "analyzing~ 📖",
        "investigating~",
        "scanning files~",
        "peeking at code~",
    ],
    "writing": [
        "creating magic~ ✨",
        "writing code~",
        "making changes... 🖊️",
        "crafting~ ♡",
        "working hard~",
        "code flowing~",
        "typing typing~",
    ],
    "waiting": [
        "waiting for Ollama~",
        "Ollama is thinking... 💭",
        "processing~",
        "still waiting~",
        "patience~ ✨",
        "thinking hard...",
        "Ollama go brrrr~",
    ],
    "error": [
        "oops! something went wrong~",
        "hit a snag... 😭",
        "error encountered~",
        "uh oh~ ✧",
        "things went sideways...",
        "big oof~",
        "disaster! 💔",
    ],
    "success": [
        "all done! ✨",
        "success~ ♡",
        "mission accomplished!",
        "perfection~ 💕",
        "yay! finished~",
        "uwu success!",
        "flawless victory~",
    ],
}


def cute_status(stage: str, frame: int = 0) -> str:
    """Get a cute status message for a given stage.
    
    stage: 'planning', 'reading', 'writing', 'waiting', 'error', or 'success'
    frame: frame number for cycling (optional)
    """
    messages = STATUS_MESSAGES.get(stage, ["working~"])
    return messages[frame % len(messages)]


# ASCII cat faces (additional to pet.py faces)
EXTRA_CAT_FACES = {
    "happy": "^(ᐛ)^",
    "heart_eyes": "♡(◕ω◕)♡",
    "shy": "/(^・ω・^)\\",
    "excited": "ᐛ",
    "sleepy": "(´ι _` )",
    "thinking": "(´・ω・`)",
    "grumpy": "(ᐛ)≡⊃",
    "uwu": "✧ w ✧",
    "owo": "✧ ω ✧",
    "blush": "ଘ(੭ˊ꒳ˋ)੭♡",
    "angry": "(╯°□°)╯︵ ┻━┻",
}


def border_with_stars(text: str, width: int = 40) -> str:
    """Create a bordered text with stars."""
    star = "★"
    top = f"{star} " * (width // 3) + "\n"
    bottom = f"\n{star} " * (width // 3)
    return f"{top}{text}{bottom}"


def kawaii_separator(width: int = 60, char: str = "･") -> str:
    """Create a cute separator line."""
    return f"｡･:{char}:･ﾟ★,｡･:{char}:･ﾟ★".replace("{char}", char)[:width]


def catboy_reaction(text: str = "") -> str:
    """Generate a cute catboy reaction with faces and hearts."""
    faces = ["٩(◕‿◕｡)۶", "ヾ(๑❛ ▿◠๑ )ﻭ✧", "✧(๑☆‿☆๑)✧", "(๑•́ ₃•̀๑)"]
    face = random.choice(faces)
    hearts = " ".join(random.choice(HEARTS) for _ in range(2))
    return f"{face} {hearts}" + (f" {text}" if text else "")


def uwu_face() -> str:
    """The classic UwU face."""
    return "✧ w ✧"


def owo_face() -> str:
    """The classic OwO face."""
    return "✧ ω ✧"


def nya_text(text: str) -> str:
    """Add 'nya~' to text."""
    return f"{text}~ nya ♡"


# Animated transitions
def fade_in_frames(text: str, frames: int = 3) -> list[str]:
    """Generate fade-in animation frames."""
    result = []
    opacity_chars = ["░", "▒", "▓", "█"]
    
    for i in range(frames):
        fade_level = i * len(opacity_chars) // frames
        char = opacity_chars[min(fade_level, len(opacity_chars) - 1)]
        result.append(f"{char} {text}")
    
    result.append(text)
    return result


def sparkle_reveal_frames(text: str, frames: int = 3) -> list[str]:
    """Generate sparkle reveal animation frames."""
    result = []
    for i in range(frames):
        revealed = text[: len(text) * (i + 1) // frames]
        if i < frames - 1:
            revealed += "✨"
        result.append(revealed)
    return result


def bouncy_text(text: str, frames: int = 4) -> list[str]:
    """Generate bouncy animation frames for text."""
    result = []
    spaces = ["", " ", "  "]
    for i in range(frames):
        space = spaces[i % len(spaces)]
        result.append(f"{space}{text}")
    return result


def twinkle_frames(text: str, frames: int = 6) -> list[str]:
    """Generate twinkling animation frames."""
    result = []
    for i in range(frames):
        if i % 2 == 0:
            result.append(f"✨ {text} ✨")
        else:
            result.append(f"✧ {text} ✧")
    return result


def heart_pulse_frames(text: str, frames: int = 4) -> list[str]:
    """Generate heart pulsing animation frames."""
    result = []
    hearts_amounts = ["♡", "♥", "♡ ♥ ♡"]
    for i in range(frames):
        heart = hearts_amounts[i % len(hearts_amounts)]
        result.append(f"{heart} {text} {heart}")
    return result


def cat_walk_frames() -> list[str]:
    """Generate frames of a cat walking across screen."""
    return [
        "  ヾ(๑❛ ▿◠๑)ﻭ✧",
        "   ヾ(๑❛ ▿◠๑)ﻭ✧",
        "    ヾ(๑❛ ▿◠๑)ﻭ✧",
        "     ヾ(๑❛ ▿◠๑)ﻭ✧",
    ]


# Party effects for celebrations
def confetti_line(width: int = 60) -> str:
    """Generate a random confetti line."""
    confetti_chars = list("✧✦★⭐✨💫🌟") + list("･｡:")
    return "".join(random.choice(confetti_chars) for _ in range(width))


def celebrate_success(text: str = "Success!") -> str:
    """Generate celebration message with confetti."""
    confetti = confetti_line(40)
    return f"\n{confetti}\n✨ {text} ✨\n{confetti}"

