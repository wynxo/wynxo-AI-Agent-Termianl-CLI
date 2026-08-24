"""An inline arrow-key chooser.

prompt_toolkit ships dialogs, but they take over the whole screen, which is
jarring for a question asked in the middle of a setup flow. This renders a
few rows below the cursor, moves a marker with the arrow keys, and erases
itself on the way out.

It always keeps a typed fallback: number keys select directly, and where
there is no terminal to draw on -- a pipe, CI, a dumb terminal -- the caller
gets told so and asks its question the old way. An interface that only works
with arrow keys is one that cannot be scripted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

CURSOR = "❯"
CURSOR_ASCII = ">"

STYLE = Style.from_dict({
    "row": "",
    "row.selected": "bold",
    "cursor": "ansicyan bold",
    "hint": "ansibrightblack",
    "badge": "ansigreen",
    "badge.warn": "ansiyellow",
    "badge.muted": "ansibrightblack",
    "title": "ansicyan bold",
    "footer": "ansibrightblack",
})


@dataclass
class Choice:
    """One row. ``value`` is what the caller gets back."""

    value: Any
    label: str
    badge: str = ""
    badge_style: str = "badge"
    hint: str = ""
    extra: dict = field(default_factory=dict)


def supported() -> bool:
    """Whether a terminal is attached that can draw this."""
    import sys

    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


async def choose(
    choices: list[Choice],
    *,
    title: str = "",
    default: int = 0,
    footer: str = "",
    width: int = 80,
    unicode: bool = True,
) -> Any | None:
    """Show the list and return the chosen value, or None if cancelled."""
    if not choices:
        return None

    index = max(0, min(default, len(choices) - 1))
    label_width = min(max(len(c.label) for c in choices), max(12, width - 34))
    badge_width = max((len(c.badge) for c in choices), default=0)
    cursor = CURSOR if unicode else CURSOR_ASCII

    def render():
        lines: list[tuple[str, str]] = []
        if title:
            lines.append(("class:title", f"  {title}\n"))
        for i, choice in enumerate(choices):
            selected = i == index
            lines.append(("class:cursor", f"  {cursor} " if selected else "    "))
            label = choice.label
            if len(label) > label_width:
                label = label[: label_width - 1] + ("…" if unicode else "~")
            style = "class:row.selected" if selected else "class:row"
            lines.append((style, label.ljust(label_width)))
            if badge_width:
                lines.append((f"class:{choice.badge_style}",
                              "  " + choice.badge.ljust(badge_width)))
            if choice.hint:
                lines.append(("class:hint", f"  {choice.hint}"))
            lines.append(("", "\n"))
        if footer:
            lines.append(("class:footer", f"  {footer}"))
        return to_formatted_text(lines)

    bindings = KeyBindings()
    result: dict[str, Any] = {"value": None}

    @bindings.add("up")
    @bindings.add("k")
    @bindings.add("c-p")
    def _(event):
        nonlocal index
        index = (index - 1) % len(choices)

    @bindings.add("down")
    @bindings.add("j")
    @bindings.add("c-n")
    def _(event):
        nonlocal index
        index = (index + 1) % len(choices)

    @bindings.add("home")
    @bindings.add("pageup")
    def _(event):
        nonlocal index
        index = 0

    @bindings.add("end")
    @bindings.add("pagedown")
    def _(event):
        nonlocal index
        index = len(choices) - 1

    @bindings.add("enter")
    @bindings.add("right")
    def _(event):
        result["value"] = choices[index].value
        event.app.exit()

    @bindings.add("escape", eager=True)
    @bindings.add("c-c")
    @bindings.add("c-d")
    def _(event):
        event.app.exit()

    # Number keys jump straight to a row, so the list stays usable without
    # arrows and matches the numbers people are used to typing.
    for position in range(min(9, len(choices))):
        @bindings.add(str(position + 1))
        def _(event, position=position):
            nonlocal index
            index = position
            result["value"] = choices[position].value
            event.app.exit()

    application: Application = Application(
        layout=Layout(HSplit([
            Window(FormattedTextControl(render), dont_extend_height=True),
        ])),
        key_bindings=bindings,
        style=STYLE,
        color_depth=ColorDepth.DEFAULT,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
    )
    await application.run_async()
    return result["value"]
