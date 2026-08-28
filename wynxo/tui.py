"""Terminal chat UI helpers."""

from __future__ import annotations

import asyncio
import io
import string
import time
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output import ColorDepth
from rich.console import Console

MIN_WIDTH = 20
MAX_SCROLLBACK = 4_000


class Transcript:
    def __init__(self, width: int = 80):
        self._buffer = io.StringIO()
        self.lines: list[str] = []
        self.width = max(MIN_WIDTH, width)
        self.console = self._make_console()
        self.on_change: Callable[[], None] | None = None

    def _make_console(self) -> Console:
        return Console(file=self._buffer, force_terminal=True,
                       color_system="truecolor", highlight=False,
                       soft_wrap=False, width=self.width, height=10_000)

    def resize(self, width: int) -> None:
        width = max(MIN_WIDTH, width)
        if width == self.width:
            return
        self.width = width
        self.console.width = width

    def drain(self) -> None:
        text = self._buffer.getvalue()
        if not text:
            return
        self._buffer.seek(0)
        self._buffer.truncate(0)
        pieces = text.split("\n")
        if pieces and pieces[-1] == "":
            pieces.pop()
        self.lines.extend(pieces)
        if len(self.lines) > MAX_SCROLLBACK:
            del self.lines[:len(self.lines) - MAX_SCROLLBACK]
        if self.on_change:
            self.on_change()

    def visible(self, height: int, offset: int = 0) -> list[str]:
        if height <= 0:
            return []
        end = max(0, min(len(self.lines) - max(0, offset), len(self.lines)))
        return self.lines[max(0, end - height):end]

    def max_offset(self, height: int) -> int:
        return max(0, len(self.lines) - height)

    def clear(self) -> None:
        self.lines.clear()
        if self.on_change:
            self.on_change()


def _output():
    try:
        from prompt_toolkit.output.defaults import create_output
        return create_output()
    except Exception:
        from prompt_toolkit.output import DummyOutput
        return DummyOutput()


_ANSWER_KEYS = string.ascii_letters + string.digits


class ChatUI:
    HEADER_ROWS = 2
    COMPOSER_ROWS = 3
    STATUS_ROWS = 1
    MAX_STATUS_ROWS = 14

    def __init__(self, status: Callable[[], str] | None = None,
                 completer=None, on_interrupt: Callable[[], None] | None = None,
                 on_thinking: Callable[[], None] | None = None,
                 on_tools: Callable[[], None] | None = None,
                 unicode: bool = True, accent: str = "ansimagenta",
                 width: int | None = None,
                 header: Callable[[], str] | None = None):
        self.transcript = Transcript(width or _terminal_width())
        self.transcript.on_change = self._changed
        self.submissions: asyncio.Queue[str] = asyncio.Queue()
        self.scroll = 0
        self._status = status or (lambda: "")
        self._header = header or (lambda: "")
        self._on_interrupt = on_interrupt
        self._on_thinking = on_thinking
        self._on_tools = on_tools
        self._unicode = unicode
        self._accent = accent
        self._closed = False
        self.question = ""
        self.answers: dict[str, str] = {}
        self.answer: asyncio.Future[str] | None = None
        self.picker: dict | None = None
        self.picked: asyncio.Future[str | None] | None = None
        self.on_resize: Callable[[int], None] | None = None
        self._last_width = 0
        self._status_lines = 1
        self.typed: asyncio.Future[str] | None = None
        self.default = ""
        self.buffer = Buffer(multiline=False, completer=completer,
                             complete_while_typing=True,
                             accept_handler=self._accept)
        self.app = self._build()

    def size(self) -> tuple[int, int]:
        if self.app.is_running:
            try:
                size = self.app.output.get_size()
                return self._measured(max(MIN_WIDTH, size.columns), max(4, size.rows))
            except Exception:
                pass
        return self._measured(max(MIN_WIDTH, _terminal_width()),
                              max(4, _terminal_height()))

    def _measured(self, width: int, rows: int) -> tuple[int, int]:
        if width != self._last_width:
            self._last_width = width
            if self.on_resize:
                self.on_resize(width)
        return width, rows

    def status_rows(self) -> int:
        _, rows = self.size()
        room = max(1, rows - self.HEADER_ROWS - self.COMPOSER_ROWS - 3)
        return max(self.STATUS_ROWS, min(self._status_lines, self.MAX_STATUS_ROWS, room))

    def transcript_rows(self) -> int:
        _, rows = self.size()
        return max(1, rows - self.HEADER_ROWS - self.COMPOSER_ROWS - self.status_rows() - 2)

    def _transcript_fragments(self):
        self.transcript.drain()
        width, _ = self.size()
        self.transcript.resize(width)
        rows = self.transcript_rows()
        self.scroll = min(self.scroll, self.transcript.max_offset(rows))
        lines = self.transcript.visible(rows, self.scroll)
        if picker := self._picker_lines(width):
            room = max(0, rows - len(picker))
            lines = self.transcript.visible(room, self.scroll) + picker
        if len(lines) < rows:
            lines = [""] * (rows - len(lines)) + lines
        return ANSI("\n".join(lines[-rows:]))

    def _header_fragments(self):
        return ANSI(self._header())

    def _rule_fragments(self):
        width, _ = self.size()
        bar = "─" if self._unicode else "-"
        return [("class:edge", bar * max(0, width))]

    def _status_fragments(self):
        text = self._status()
        if self.scroll > 0:
            marker = "  ^ scrolled back -- End to follow again"
            text = f"{text}{marker}" if text else marker.strip()
        self._status_lines = text.count("\n") + 1 if text else 1
        return ANSI(text)

    def _edge(self, top: bool):
        def render():
            width, _ = self.size()
            if self._unicode:
                left, right, bar = (("╭", "╮", "─") if top else ("╰", "╯", "─"))
            else:
                left, right, bar = "+", "+", "-"
            return [("class:edge", left + bar * max(0, width - 2) + right)]
        return render

    def _build(self) -> Application:
        transcript = Window(content=FormattedTextControl(self._transcript_fragments,
                                                         focusable=False),
                            wrap_lines=False)
        status = Window(content=FormattedTextControl(self._status_fragments,
                                                     focusable=False),
                        height=lambda: self.status_rows())
        composer = Window(content=BufferControl(buffer=self.buffer), height=1,
                          get_line_prefix=lambda *_: [("class:prompt", self._composer_prefix())])
        body = HSplit([
            Window(content=FormattedTextControl(self._header_fragments), height=1),
            Window(content=FormattedTextControl(self._rule_fragments), height=1),
            transcript,
            status,
            Window(content=FormattedTextControl(self._edge(True)), height=1),
            composer,
            Window(content=FormattedTextControl(self._edge(False)), height=1),
        ])
        layout = Layout(FloatContainer(
            content=body,
            floats=[Float(xcursor=True, ycursor=True,
                           content=CompletionsMenu(max_height=8, scroll_offset=1))],
        ), focused_element=composer)
        return Application(layout=layout, key_bindings=self._keys(), full_screen=True,
                           mouse_support=False, color_depth=ColorDepth.TRUE_COLOR,
                           erase_when_done=True, output=_output())

    def _composer_prefix(self) -> str:
        if self.asking or self.typing:
            return f"│ {self.question} "
        return "│ > "

    def _accept(self, buff: Buffer) -> bool:
        text = buff.text
        if self.typing:
            self._resolve_typed(text.strip())
            return False
        if self.asking:
            chosen = text.strip().lower()
            for key in self.answers:
                if chosen == key or (chosen and chosen[0] == key):
                    self._resolve(key)
                    return False
            if not chosen and self.default:
                self._resolve(self.default)
            return False
        if text.strip():
            self.submissions.put_nowait(text)
        return False

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()
        asking = Condition(lambda: self.asking)

        def answer_or_type(event):
            if self.buffer.text:
                self.buffer.insert_text(event.data)
                return
            key = str(event.data).lower()
            if key in self.answers:
                self._resolve(key)
            else:
                self.buffer.insert_text(event.data)

        for character in _ANSWER_KEYS:
            keys.add(character, filter=asking, eager=True)(answer_or_type)

        picking = Condition(lambda: self.picking)

        @keys.add("up", filter=picking, eager=True)
        def _(event):
            count = len(self.picker["options"])
            self.picker["index"] = (self.picker["index"] - 1) % count
            self.invalidate()

        @keys.add("down", filter=picking, eager=True)
        def _(event):
            count = len(self.picker["options"])
            self.picker["index"] = (self.picker["index"] + 1) % count
            self.invalidate()

        @keys.add("enter", filter=picking, eager=True)
        def _(event):
            if self.picked is not None and not self.picked.done():
                option = self.picker["options"][self.picker["index"]]
                self.picked.set_result(option[2] if len(option) > 2 else option[0])

        @keys.add("escape", filter=picking, eager=True)
        def _(event):
            if self.picked is not None and not self.picked.done():
                self.picked.set_result(None)

        @keys.add("c-c")
        def _(event):
            if self.picking:
                self.picked.set_result(None)
                return
            if self.typing:
                self._resolve_typed("")
                return
            if self.asking:
                self._resolve("q" if "q" in self.answers else "")
                return
            if self._on_interrupt:
                self._on_interrupt()

        @keys.add("c-o")
        def _(event):
            if self._on_thinking:
                self._on_thinking()

        @keys.add("c-t")
        def _(event):
            if self._on_tools:
                self._on_tools()

        @keys.add("c-d")
        def _(event):
            if not self.buffer.text:
                self.submissions.put_nowait("/quit")

        @keys.add("pageup")
        def _(event):
            self.scroll = min(self.transcript.max_offset(self.transcript_rows()),
                              self.scroll + max(1, self.transcript_rows() - 1))
            self.invalidate()

        @keys.add("pagedown")
        def _(event):
            self.scroll = max(0, self.scroll - max(1, self.transcript_rows() - 1))
            self.invalidate()

        @keys.add("end")
        def _(event):
            self.scroll = 0
            self.invalidate()

        return keys

    async def choose(self, title: str, options: list[tuple], current: str = "") -> str | None:
        self.picker = {"title": title, "options": options,
                       "index": max(0, next((i for i, option in enumerate(options)
                                              if option[0] == current), 0))}
        self.picked = asyncio.get_running_loop().create_future()
        self.invalidate()
        try:
            return await self.picked
        finally:
            self.picker = None
            self.picked = None
            self.invalidate()

    @property
    def picking(self) -> bool:
        return self.picked is not None and not self.picked.done()

    def _picker_lines(self, width: int) -> list[str]:
        picker = self.picker
        if not picker:
            return []
        dim, reset = "\x1b[38;5;247m", "\x1b[0m"
        mark = "❯" if self._unicode else ">"
        phase = int(time.monotonic() * 12)
        sweep = ((255,120,200),(255,96,190),(246,74,186),(228,64,190),
                 (204,62,200),(176,70,214),(150,84,226),(132,104,236),
                 (150,84,226),(176,70,214),(204,62,200),(228,64,190),
                 (246,74,186),(255,96,190),(255,120,200))
        rgb = lambda c: f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m"
        lines = [f"{rgb(sweep[phase % len(sweep)])}  {picker['title']}{reset}"]
        for i, option in enumerate(picker["options"]):
            name, hint = option[0], option[1]
            if i == picker["index"]:
                lit = "".join(f"{rgb(sweep[(phase+n)%len(sweep)])}{ch}" for n,ch in enumerate(f"{mark} {name}"))
                body = f"{lit}{reset}" + (f"  {dim}{hint}{reset}" if hint else "")
            else:
                body = f"  {dim}{name}{reset}" + (f"  {dim}{hint}{reset}" if hint else "")
            lines.append("  " + body)
        lines.append(f"{dim}  arrows move  ·  enter chooses  ·  esc cancels{reset}")
        return lines

    async def ask(self, question: str, answers: dict[str, str], default: str = "") -> str:
        self.question = question
        self.answers = answers
        self.default = default
        self.answer = asyncio.get_running_loop().create_future()
        self.invalidate()
        try:
            return await self.answer
        finally:
            self.question = ""
            self.answers = {}
            self.default = ""
            self.answer = None
            self.invalidate()

    async def prompt(self, question: str, default: str = "") -> str:
        self.question = question
        self.typed = asyncio.get_running_loop().create_future()
        self.buffer.text = default
        self.buffer.cursor_position = len(default)
        self.invalidate()
        try:
            return await self.typed
        finally:
            self.question = ""
            self.typed = None
            self.buffer.text = ""
            self.invalidate()

    def _resolve(self, key: str) -> None:
        if self.answer is not None and not self.answer.done():
            self.answer.set_result(key)

    def _resolve_typed(self, text: str) -> None:
        if self.typed is not None and not self.typed.done():
            self.typed.set_result(text)

    @property
    def asking(self) -> bool:
        return self.answer is not None and not self.answer.done()

    @property
    def typing(self) -> bool:
        return self.typed is not None and not self.typed.done()

    def _changed(self) -> None:
        pass

    def flush(self) -> None:
        self.transcript.drain()
        self.invalidate()

    def invalidate(self) -> None:
        if self._closed:
            return
        try:
            self.app.invalidate()
        except Exception:
            pass

    async def next_message(self) -> str:
        return await self.submissions.get()

    async def repaint_loop(self, interval: float = 0.1) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                self.invalidate()
        except asyncio.CancelledError:
            pass

    def exit(self) -> None:
        self._closed = True
        try:
            self.app.exit()
        except Exception:
            pass


def render_to_ansi(renderable, width: int, max_rows: int = 1) -> str:
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=True, color_system="truecolor",
                      highlight=False, soft_wrap=False,
                      width=max(MIN_WIDTH, width), height=max(4, max_rows + 2))
    try:
        console.print(renderable, end="")
    except Exception:
        return ""
    lines = sink.getvalue().split("\n")
    if len(lines) <= max_rows:
        return "\n".join(lines)
    return "\n".join(lines[-max_rows:])


def _terminal_width(default: int = 80) -> int:
    import shutil
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except (OSError, ValueError):
        return default


def _terminal_height(default: int = 24) -> int:
    import shutil
    try:
        return shutil.get_terminal_size((80, default)).lines
    except (OSError, ValueError):
        return default


MIN_ROWS = ChatUI.HEADER_ROWS + ChatUI.COMPOSER_ROWS + ChatUI.STATUS_ROWS + 2


def usable() -> bool:
    import sys
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError, OSError):
        return False
    term = __import__('os').environ.get("TERM", "").lower()
    if term in ("dumb", "unknown"):
        return False
    if _terminal_height() < MIN_ROWS:
        return False
    if not term:
        return sys.platform == "win32"
    return True
