"""Entry point and REPL."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import select
import signal
import stat
import sys
import threading
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from . import __version__
from . import fullscreen
from .agent import Agent, Callbacks, Interrupted
from .config import Config, Endpoint, data_dir, is_configured, load, normalise_url
from .doctor import run_doctor
from .effort import ORDER, resolve
from .permissions import Decision
from .provider import OllamaClient, ProviderError, check_context
from .queue import Pending
from .session import Session
from .keys import KeyWatcher, describe_bindings
from .journal import Journal, recent as recent_logs
from .memory import Memory
from .pet import Mood, Pet
from .select import (
    HINT, HINT_ASCII, Choice, choose, silence_cpr_warning,
    supported as arrows_supported)
from .scope import Mode, Scope, resolve as resolve_scope
from .status import Status, WARN
from .tools import build_registry
from rich.text import Text

from .ui import (ACCENT, BAR_ACCENT, MUTED, ActivityBar, CodeStreamer,
                 ThoughtStreamer, UI, effort_meter)

# What the activity bar says while each tool runs.
_ACTIVITY = {
    "read_file": "reading", "write_file": "writing file", "edit_file": "editing",
    "list_dir": "listing", "glob": "finding", "grep": "searching",
    "shell": "running", "todo_write": "planning",
}
_LANGUAGE = {"read_file": "python", "shell": "console"}

# Keys that work *while the agent is running*, not just at the prompt.
LIVE_KEYS = {"ctrl+o": "thinking", "ctrl+t": "detail"}
from .platforms import (
    is_dumb_terminal, ollama_server_help as server_help,
    suspicious_workspace)
from .wizard import probe, run_wizard

# Short forms for the prefixes that are genuinely ambiguous. An exact command
# is matched before any of these, so /mode still means /mode.
ALIASES = {
    "/exit": "/quit", "/q": "/quit", "/?": "/help", "/h": "/help",
    "/m": "/model", "/mo": "/model", "/mod": "/model",
    "/e": "/effort", "/eff": "/effort",
    "/t": "/theme", "/th": "/theme",
    "/mem": "/memory", "/sc": "/scope", "/st": "/stats", "/se": "/sessions",
    "/c": "/clear", "/co": "/compact",
}


class _NoPicker:
    """Sentinel: this terminal cannot draw an arrow picker."""

    def __repr__(self) -> str:
        return "NO_PICKER"


NO_PICKER = _NoPicker()


def _first(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _clean_commit_message(text: str) -> str:
    """Take the message out of whatever the model wrapped it in.

    Small models fence things, label them, and add "Here is the commit
    message:" no matter how firmly the prompt says not to.
    """
    import re as _re

    out = _re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", text,
                  flags=_re.DOTALL | _re.IGNORECASE).strip()
    fenced = _re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", out, _re.DOTALL)
    if fenced:
        out = fenced.group(1)
    out = _re.sub(r"^\s*(here'?s?\s+(is\s+)?)?(the\s+)?commit\s+message\s*:?\s*\n+",
                  "", out, flags=_re.IGNORECASE)
    lines = [ln.rstrip() for ln in out.strip().splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def resolve_command(name: str) -> str | None:
    """Expand an abbreviation to a command, when it is unambiguous.

    Typing /mo for /model is the sort of thing people do without thinking, and
    refusing it is friction for no safety benefit -- but /m matching both
    /model and /memory must not silently pick one, so an ambiguous prefix is
    resolved only by an explicit alias.
    """
    if name in ALIASES:
        return ALIASES[name]
    if name in COMMANDS:
        return name
    # A trailing "s" is the other thing people type without thinking: the
    # command lists things, so /themes and /models are the natural plurals.
    # Prefix matching alone cannot catch them -- "/theme" does not start with
    # "/themes" -- so they used to come back as unknown commands.
    if name.endswith("s") and name[:-1] in COMMANDS:
        return name[:-1]
    matches = [c for c in COMMANDS if c.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def _theme_summary(name: str) -> str:
    return {
        "purple": "deep violet (default)",
        "sakura": "pink and violet, turned up",
        "midnight": "cool blue",
        "ember": "warm orange",
        "plain": "your terminal's own 16 colours",
    }.get(name, "")


def _voice_summary(voice: str) -> str:
    return {
        "plain": "direct and professional (default)",
        "warm": "friendly, still honest about failures",
        "mentor": "explains the reasoning behind decisions",
        "blunt": "the fewest words that say what happened",
        "kawaii": "cheerful and affectionate, same engineering underneath",
    }.get(voice, "")


def _escape(text: str) -> str:
    """prompt_toolkit's HTML helper parses its input, so a model tag with an
    angle bracket in it would raise rather than render."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


COMMANDS = {
    "/help": "show this",
    "/effort": "change effort level (low|medium|high|xhigh|max|ultra)",
    "/model": "switch model, or list what the server has",
    "/endpoint": "list | use <name> | add <url> | test -- where Ollama serves",
    "/ctx": "show or set the context window (num_ctx)",
    "/tools": "list the tools the agent can call",
    "/pet": "the companion: on | off | name <x> | voice <x>",
    "/theme": "colour palette: purple | sakura | midnight | ember | plain",
    "/fullscreen": "draw on the alternate screen, like vim: on | off",
    "/secrets": "credential protection: on | off | allow <path>",
    "/speak": "read answers out loud: on | off | test | engine <name>",
    "/talker": "small model that does the talking: <model> | off",
    "/log": "where this session is being recorded",
    "/mode": "plan | manual | auto | yolo -- how much it asks first",
    "/scope": "folder | repo | machine, or a path to work in",
    "/cd": "work in another directory",
    "/repo": "clone a GitHub repo and work in it",
    "/undo": "revert the last file change",
    "/memory": "show, add to, or forget long-term memory",
    "/thinking": "show or hide the model's reasoning",
    "/plan": "show the current plan",
    "/new": "start a new chat: fresh history, screen and log",
    "/resume": "pick up an earlier conversation where it stopped",
    "/commit": "write a commit message from the staged diff, then commit",
    "/clear": "start a fresh conversation",
    "/compact": "summarise the conversation to reclaim context",
    "/stats": "tokens, speed, context use",
    "/doctor": "check the server and model for problems",
    "/yolo": "stop asking permission for this session",
    "/sessions": "list recent sessions",
    "/init": "write a WYNXO.md describing this project",
    "/map": "the project layout the model is given, or rebuild it",
    "/quit": "exit",
}


class CommandCompleter(Completer):
    """Suggests slash commands, and files after an "@".

    Nothing else is completed. A menu opening over ordinary prose -- which
    is what you are typing almost all of the time -- would be in the way
    rather than helpful.
    """

    def __init__(self, workspace_getter=None):
        self._workspace = workspace_getter

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # @path, anywhere in the line: it is a reference inside a sentence.
        word = text.rsplit(" ", 1)[-1]
        if word.startswith("@") and self._workspace is not None:
            from .mentions import candidates

            prefix = word[1:]
            for path in candidates(self._workspace(), prefix):
                yield Completion(
                    "@" + path, start_position=-len(word),
                    display=path,
                    display_meta="directory" if path.endswith("/") else "file")
            return

        if not text.startswith("/") or " " in text:
            return

        seen: set[str] = set()
        for name, description in COMMANDS.items():
            if name.startswith(text):
                seen.add(name)
                yield Completion(
                    name, start_position=-len(text),
                    display=name, display_meta=description)

        # Aliases that are not prefixes of what they expand to -- /q for
        # /quit, /? for /help. Prefix matching alone would never find them.
        for alias, target in sorted(ALIASES.items()):
            if alias.startswith(text) and target not in seen:
                seen.add(target)
                yield Completion(
                    target, start_position=-len(text),
                    display=f"{target}  ({alias})",
                    display_meta=COMMANDS.get(target, ""))


class TerminalCallbacks(Callbacks):
    """Wires the agent's events to the terminal."""

    def __init__(self, ui: UI, prompt_session: PromptSession | None = None):
        self.ui = ui
        # None in non-interactive mode, where nothing can be asked anyway.
        self.prompt_session = prompt_session
        self._streaming = False
        self._thinking_chars = 0
        self._thinker: ThoughtStreamer | None = None
        self.bar: ActivityBar | None = None
        self.journal: Journal | None = None
        self.watcher: KeyWatcher | None = None
        """Set while a turn runs. It holds the terminal in cbreak mode, so it
        must be stopped before prompt_toolkit is asked to read a line --
        otherwise both read stdin and keystrokes go to whichever wins."""
        self.streamer: CodeStreamer | None = None
        self.verbose_tools = False
        """Ctrl-T: show full tool output instead of a one-line summary."""
        self.tokens = 0

    # -- live toggles, called from the key watcher thread -------------------

    def toggle_thinking(self) -> None:
        self.ui.show_thinking = not self.ui.show_thinking
        if not self.ui.show_thinking:
            # Collapse what is already on screen's worth of buffer, so hiding
            # takes effect now rather than after the current block finishes.
            self._end_thinking()
        self._note(f"thinking {'shown' if self.ui.show_thinking else 'hidden'}")

    def toggle_verbose(self) -> None:
        self.verbose_tools = not self.verbose_tools
        self._note(f"tool output {'full' if self.verbose_tools else 'summarised'}")

    def _note(self, message: str) -> None:
        """Surface a toggle without disturbing whatever is streaming."""
        if self.bar is not None:
            self.bar.update(detail=message)
        else:
            self.ui.info(message)

    def _end_stream(self) -> None:
        """Close whichever transient block is open, so the next starts clean."""
        self._end_thinking()
        if self._streaming:
            if self.streamer is not None:
                self.streamer.finish()
                self.streamer = None
            self._streaming = False

    async def on_thinking(self, text: str) -> None:
        self._thinking_chars += len(text)
        self.tokens += 1
        if self.bar is not None:
            self.bar.update(activity="thinking", tokens=self.tokens)

        if not self.ui.show_thinking:
            return
        if self._streaming:
            self._end_stream()
        if self._thinker is None:
            self.ui.console.print()
            self.ui.console.print(Text("  thinking", style=f"bold {MUTED}"))
            self._thinker = ThoughtStreamer(self.ui)
        self._thinker.feed(text)

    def _end_thinking(self) -> None:
        if self._thinker is not None:
            self._thinker.finish()
            self._thinker = None
            self._thinking_chars = 0

    async def on_content(self, text: str) -> None:
        # Ollama streams roughly one token per chunk, so counting chunks gives
        # a live figure that tracks generation instead of a character estimate
        # that lurches. The exact count arrives with the final chunk.
        self.tokens += 1
        if not self._streaming:
            self._end_stream()
            self.streamer = CodeStreamer(self.ui)
            self._streaming = True
        if self.bar is not None:
            self.bar.update(activity="writing", detail="", tokens=self.tokens)
        self.streamer.feed(text)

    async def on_stage(self, name: str, detail: str = "") -> None:
        if self.journal is not None:
            self.journal.stage(name, detail)
        self._end_stream()
        if self.bar is not None:
            self.bar.update(activity=name, detail=detail)
        suffix = f" [{MUTED}]{detail}[/]" if detail else ""
        self.ui.console.print(f"  [{ACCENT}]{self.ui.g.arrow}[/] [{MUTED}]{name}[/]{suffix}")

    async def on_tool_start(self, name: str, summary: str) -> None:
        if self.journal is not None:
            self.journal.tool(name, {"summary": summary})
        self._end_stream()
        if self.bar is not None:
            self.bar.update(activity=_ACTIVITY.get(name, name), detail=summary)
            if name == "todo_write":
                return       # the pinned plan is the announcement
        self.ui.tool_start(name, summary)

    async def on_tool_result(self, name: str, ok: bool, display: str, output: str) -> None:
        if self.journal is not None:
            self.journal.tool_result(name, ok, output)
        # The pinned plan already shows every step and its state, so a result
        # line per todo_write is the same information a second time -- and it
        # scrolls, which is exactly what pinning the panel was meant to stop.
        if name == "todo_write" and ok and self.bar is not None:
            return
        if self.verbose_tools and output.strip():
            self.ui.tool_result(name, ok, "", "")
            self.ui.code(output[:4000], _LANGUAGE.get(name, "text"))
        else:
            self.ui.tool_result(name, ok, display, output)

    async def on_tool_output(self, name: str, line: str) -> None:
        """A line from a command while it is still running.

        Only shell gets this. A build or a test run is the case where
        waiting in silence is worst: the output that explains what went
        wrong arrives long before the exit code does, and if the command
        times out it is the only output there will ever be.
        """
        if name != "shell":
            return
        self._end_stream()
        self.ui.tool_output(line)
        if self.bar is not None:
            # Doubles as the keep-alive: the pinned bar now says what the
            # command is doing right now, so a slow build looks busy rather
            # than hung.
            self.bar.update(detail=line.strip()[:60])

    async def on_todos(self, rendered: str) -> None:
        """Pin the plan in the live region rather than printing it again.

        It used to print a fresh panel on every update, so a five-step plan
        left five panels in the scrollback and the current one was whichever
        had scrolled past last. Now there is one, it is redrawn in place, and
        it ticks itself off and leaves when the work is done.
        """
        if self.bar is None:
            self.ui.todos(rendered)      # non-interactive: print it once
            return
        self.bar.set_plan(rendered)
        if self.bar.plan_is_complete():
            await self.bar.finish_plan()

    async def on_warning(self, message: str) -> None:
        self._end_stream()
        self.ui.warn(message)

    async def ask_permission(self, name: str, summary: str, preview: str) -> Decision:
        self._end_stream()
        if self.prompt_session is None:
            return Decision.ALLOW
        try:
            return await self._ask(name, summary, preview)
        finally:
            # Hand the terminal back so the rest of the turn keeps its status
            # line and its live keys.
            self._resume_live()

    def _suspend_live(self) -> None:
        """Release the terminal before prompt_toolkit reads a line."""
        if self.watcher is not None:
            self.watcher.stop()
        if self.bar is not None:
            self.bar.stop()

    def _resume_live(self) -> None:
        if self.bar is not None:
            self.bar.start()
        if self.watcher is not None:
            self.watcher.start()

    async def _ask(self, name: str, summary: str, preview: str) -> Decision:
        self._suspend_live()
        self.ui.console.print()
        verb = {
            "write_file": "write to",
            "edit_file": "edit",
            "shell": "run",
        }.get(name, name)
        self.ui.console.print(f"  [bold {ACCENT}]{verb}[/] [bold]{summary}[/]")
        if preview:
            self.ui.diff(preview) if preview.lstrip().startswith(("---", "+", "-")) else self.ui.code(preview)

        while True:
            try:
                answer = (await self.prompt_session.prompt_async(
                    HTML('<ansicyan>  [y] yes  [a] always  [n] no  [q] stop: </ansicyan>')
                )).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return Decision.ABORT
            if answer in ("", "y", "yes"):
                return Decision.ALLOW
            if answer in ("a", "always"):
                return Decision.ALLOW_ALWAYS
            if answer in ("n", "no"):
                return Decision.DENY
            if answer in ("q", "quit", "stop"):
                return Decision.ABORT
            self.ui.warn("y, a, n or q.")


class Repl:
    def __init__(self, config: Config, workspace: Path, ui: UI,
                 scope: Scope = Scope.FOLDER, mode: Mode = Mode.MANUAL):
        self.config = config
        self.workspace = workspace
        self.ui = ui
        self.client = OllamaClient(config)
        self.policy = resolve(config.effort)
        self.boundary = resolve_scope(workspace, scope)
        self.mode = mode
        self.memory = Memory(workspace)
        self.journal = Journal.open(
            self.agent_session_id(), enabled=config.log)
        self.pending = Pending()
        self.pet = Pet(
            name=config.pet_name,
            enabled=config.pet,
            animate=config.animations,
            unicode=ui.g.unicode,
        )
        self.pet.style_name = "kawaii" if config.voice == "kawaii" else "default"
        self.pet.set_pace(self.policy.name)
        self._last_elapsed = 0.0

        # The talker speaks; the coder works. Constructed here so /talker can
        # turn it on and off mid-session without rebuilding the agent.
        from .duo import Talker
        from .prompts import VOICES
        from .speech import Speaker, pick as pick_engine

        self.talker: Talker | None = None
        if config.talker:
            self.talker = Talker(self.client, config.talker,
                                 voice_block=VOICES.get(config.voice, ""))

        # Speech is opt-in and degrades to silence: a missing synthesiser is
        # not a reason to refuse to start.
        engine = pick_engine(config.speech_engine) if config.speak else None
        self.speaker = Speaker(engine, voice=config.speech_voice,
                               rate=config.speech_rate, model=config.speech_model)

        history_file = data_dir() / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event):
            """Alt-Enter inserts a newline; Enter submits."""
            event.current_buffer.insert_text("\n")

        @bindings.add("c-o")
        def _(event):
            """Ctrl-O toggles the thinking display, at the prompt or mid-turn."""
            self.callbacks.toggle_thinking()

        @bindings.add("c-t")
        def _(event):
            """Ctrl-T toggles full tool output."""
            self.callbacks.toggle_verbose()

        @bindings.add("c-e")
        def _(event):
            """Ctrl-E steps the effort level up; Ctrl-B steps it down."""
            self._shift_effort(1)
            event.app.invalidate()

        @bindings.add("c-b")
        def _(event):
            self._shift_effort(-1)
            event.app.invalidate()

        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(history_file)),
            completer=CommandCompleter(lambda: self.workspace),
            complete_while_typing=True,
            key_bindings=bindings,
            multiline=False,
            # The default reserves eight rows for a completion dropdown,
            # which shows up as a slab of empty screen under every prompt.
            # Readline-style completion prints inline and needs none.
            # Enough rows for the menu to open downward without the prompt
            # jumping, but not so many that an empty slab sits under the
            # input the rest of the time.
            reserve_space_for_menu=6,
            complete_style=CompleteStyle.COLUMN,
        )
        silence_cpr_warning(self.prompt_session.app)
        self.callbacks = TerminalCallbacks(ui, self.prompt_session)
        self.callbacks.journal = self.journal
        self.agent = Agent(self.client, config, self.policy, workspace, self.callbacks,
                           boundary=self.boundary, memory=self.memory)
        self.agent.permissions.mode = mode
        self.agent.refresh_system_prompt()
        self._task: asyncio.Task | None = None
        # Set by amain() once the alternate screen is bracketed around the
        # session. Left as an inert Screen so /fullscreen works the same in
        # tests and in an embedded Repl that nobody wrapped.
        self.screen = fullscreen.Screen(enabled=False)

    def agent_session_id(self) -> str:
        import uuid

        return uuid.uuid4().hex[:8]

    async def _connect(self) -> bool:
        """Reach the server, report what is loaded, adapt to the model."""
        if self.config.clear_on_start:
            self.ui.clear()
        status = Status()
        problems: list[tuple[str, str, str]] = []

        def note(state: str, message: str, detail: str = "") -> None:
            """Collect rather than print. A wall of green OK lines on every
            start is noise; the things that are wrong are the news."""
            problems.append((state, message, detail))

        try:
            version = await self.client.ping()
        except ProviderError as exc:
            status.fail("ollama", self.client.base_url)
            status.close()
            self.ui.error(str(exc))
            self.ui.console.print(server_help())
            return False

        info = None
        try:
            info = await self.client.show(self.config.model)
        except ProviderError:
            pass
        if info is not None and info.capabilities_known and not info.supports_tools:
            note(WARN, self.config.model, "no native tool calling")

        if warning := await check_context(self.client, self.config):
            note(WARN, f"context {self.config.num_ctx}", warning.split(".")[0])

        self._refresh_map(note)
        await self.agent.detect_capabilities()
        # EffortPolicy is immutable: a capability downgrade inside the agent
        # produces a new object rather than mutating self.policy in place, so
        # without this the status bar and /effort table would keep showing
        # "thinking on" after the agent had silently turned it off.
        self.policy = self.agent.policy
        if not self.agent.native_tools:
            note(WARN, "tools", "hermes (prompted), not native")

        if reason := suspicious_workspace(self.workspace):
            note(WARN, f"scope {self.boundary.scope.value}", reason)

        if problems:
            print()
            for state, message, detail in problems:
                status.line(state, message, detail)
        status.close()

        self.ui.wake(self.pet, self.pet.name)
        self.ui.banner(
            self.config.model,
            f"{self.client.base_url} (ollama {version})",
            self.policy.name,
            str(self.workspace),
        )
        if self.pet.enabled:
            self.ui.console.print(
                Text("  ") + Text(self.pet.face(advance=False),
                                  style=f"bold {self.pet.style()}")
                + Text(f"  {self.pet.name} {self.ui.g.dot} "
                       f"{self.pet.remark('greet')}", style=MUTED))
            self.ui.console.print()
        return True

    async def start(self) -> int:
        if not await self._connect():
            return 1
        return await self._loop()

    async def _loop(self) -> int:
        while True:
            try:
                self._open_box()
                text = await self.prompt_session.prompt_async(
                    self._prompt_message, bottom_toolbar=self._bottom_toolbar)
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

            text = text.strip()
            if not text:
                continue

            if text.startswith("/"):
                if await self._guarded(self.command(text)) is False:
                    break
                continue

            await self._guarded(self.turn(text))
            if await self._guarded(self._drain_queue()) is False:
                break

        await self.client.aclose()
        self.ui.console.print(f"  [{MUTED}]bye[/]")
        return 0

    async def start_with(self, prompt: str) -> int:
        """Run one prompt, then drop into the REPL. `wynxo "fix the tests"`."""
        if not await self._connect():
            return 1
        await self.turn(prompt)
        return await self._loop()

    async def _guarded(self, coro):
        """Run one turn or command; survive anything it raises.

        A crash here used to end the process and take the conversation with
        it -- and the causes are all things a local model does on a bad day:
        a template that cannot parse its own tool-call output, a tool that
        raises something unforeseen, a bug of mine. None of that is worth
        losing the session over, and the traceback is written to the log
        where it can be read afterwards.

        Interrupted and CancelledError pass through: those are Ctrl-C, which
        the caller already handles, and swallowing them would make Ctrl-C
        look broken again.
        """
        import traceback

        try:
            return await coro
        except (Interrupted, asyncio.CancelledError):
            raise
        except ProviderError as exc:
            self.callbacks._end_stream()
            self.ui.error(str(exc))
            self.journal.error(str(exc))
            return None
        except Exception as exc:                      # noqa: BLE001
            self.callbacks._end_stream()
            self.ui.error(f"{type(exc).__name__}: {exc}")
            self.ui.info("the conversation is intact -- this was not fatal")
            self.journal.error("".join(traceback.format_exception(exc)))
            if self.journal.path is not None:
                self.ui.info(f"details in {self.journal.path}")
            return None

    async def turn(self, text: str) -> None:
        """Run one request, with a live status bar and mid-flight keybinds."""
        self.journal.user(text)
        text = self._expand_mentions(text)
        self.callbacks.tokens = 0
        self.callbacks._thinking_chars = 0

        bar = ActivityBar(self.ui, self.policy.name, describe_bindings(LIVE_KEYS),
                          model=self.config.model, pet=self.pet)
        review_mark = self.agent.checkpoints.mark()
        bar.animate = self.config.animations
        bar.queued = self.pending.preview(ellipsis=self.ui.g.ellipsis)
        used = self.agent.session.token_estimate()
        limit = self.policy.context_budget or self.config.num_ctx
        bar.context_pct = 100 * used / max(1, limit)
        self.callbacks.bar = bar
        self.ui.bar = bar

        def typed(char: str) -> None:
            """A keystroke that no binding claimed, while a turn is running."""
            self.pending.key(char)
            bar.queued = self.pending.preview(ellipsis=self.ui.g.ellipsis)
            bar.refresh()

        watcher = KeyWatcher(
            {
                "ctrl+o": self.callbacks.toggle_thinking,
                "ctrl+t": self.callbacks.toggle_verbose,
                "ctrl+u": lambda: (self.pending.clear(),
                                   setattr(bar, "queued", ""), bar.refresh()),
                # The watcher holds the terminal in cbreak mode for the whole
                # turn, so it sees Ctrl-C as a keypress. Handling it here
                # works even where the tty driver does not raise SIGINT --
                # a pipe, a pty without a controlling terminal, or a
                # platform that swallows it.
                "ctrl+c": self.interrupt,
            },
            on_key=typed,
        )

        self.callbacks.watcher = watcher
        self._arm_interrupt()
        # The talker answers first: a 1B model is quick enough that the
        # acknowledgement lands before the coder has produced a token.
        if self.talker is not None:
            await self._talk(await self.talker.opening(text))
        self._task = asyncio.ensure_future(self.agent.run(text))
        bar.start()
        watcher.start()
        try:
            result = await self._task
        except (asyncio.CancelledError, Interrupted):
            self.pet.react(Mood.IDLE)
            self.ui.console.print()
            self.ui.warn("Interrupted. The conversation is intact; ask me something else.")
            return
        finally:
            # Order matters: the terminal must be restored before anything
            # tries to read from it again, and any in-progress line has to be
            # flushed to the real scrollback before the bar stops -- it is a
            # transient Live, which erases its render area on stop, taking an
            # unflushed line with it.
            watcher.stop()
            self.callbacks._end_stream()
            bar.stop()
            self.callbacks.bar = None
            self.ui.bar = None
            self.callbacks.watcher = None
            self._task = None

        if result.errors:
            self.pet.react(Mood.SAD)
            for message in result.errors:
                self.journal.error(message)
            self.ui.error("\n".join(result.errors))
            return
        self.journal.assistant(result.content, tokens=self.callbacks.tokens,
                               seconds=result.elapsed)
        self.pet.react(Mood.HAPPY if not result.interrupted else Mood.IDLE)

        if result.content and not self.config.stream:
            self.ui.assistant_markdown(result.content)
        elif result.content:
            self.ui.console.print()

        # No stats line here: the pinned bar under the input already shows
        # tokens, rate and context, and printing them again above the next
        # prompt was the same numbers twice, scrolling away from where you
        # are actually looking.
        self._last_elapsed = result.elapsed
        if result.compacted:
            self.ui.info("context was compacted during this turn")

        await self._review_changes(review_mark)
        await self._narrate(text, result)

    async def _review_changes(self, mark: int) -> None:
        """In review mode, put the whole turn's changes up as one diff.

        Manual mode interrupts a ten-file refactor ten times; auto never
        shows you the shape of what happened. This waits until the work is
        finished, then asks once.
        """
        if self.agent.permissions.mode is not Mode.REVIEW:
            return
        changes = self.agent.checkpoints.changes_since(mark)
        if not changes:
            return

        from .tools.files import make_diff

        diffs: list[tuple[str, str]] = []
        for snapshot in changes:
            name = self.ui.shorten_path(str(snapshot.path))
            try:
                now = snapshot.path.read_text(encoding="utf-8",
                                              errors="surrogateescape") \
                    if snapshot.path.exists() else ""
            except OSError as exc:
                diffs.append((name, f"(could not re-read: {exc})"))
                continue
            before = snapshot.content or ""
            if before == now:
                continue
            diffs.append((name, make_diff(before, now, name)))
        if not diffs:
            return

        self.ui.console.print()
        self.ui.console.print(Text(
            f"  {len(diffs)} file{'s' if len(diffs) != 1 else ''} changed",
            style=f"bold {ACCENT}"))
        for name, diff in diffs:
            self.ui.console.print()
            self.ui.console.print(Text(f"  {name}", style=f"bold {ACCENT}"))
            self.ui.diff(diff)

        self.ui.console.print()
        answer = (await self.prompt_session.prompt_async(
            HTML('<ansicyan>  [k] keep  [r] revert all  [s] step through: </ansicyan>')
        )).strip().lower()

        if answer in ("", "k", "keep", "y", "yes"):
            self.ui.success(f"kept {len(diffs)} file(s)")
            return
        if answer in ("r", "revert", "n", "no"):
            reverted, problems = self.agent.checkpoints.revert_since(mark)
            for problem in problems:
                self.ui.warn(problem)
            self.ui.success(f"reverted {reverted} change(s)")
            return
        await self._step_through(mark, changes)

    async def _step_through(self, mark: int, changes) -> None:
        """One file at a time, keeping the rest.

        Reverting a single file cannot go through the checkpoint stack --
        that is ordered, and taking one out of the middle would put the
        others back too. The snapshot holds the original, so it is written
        straight back.
        """
        kept = reverted = 0
        for snapshot in changes:
            name = self.ui.shorten_path(str(snapshot.path))
            answer = (await self.prompt_session.prompt_async(
                HTML(f'<ansicyan>  {_escape(name)}  [k] keep  [r] revert: </ansicyan>')
            )).strip().lower()
            if answer in ("r", "revert", "n", "no"):
                try:
                    if snapshot.existed:
                        snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                        snapshot.path.write_text(snapshot.content or "",
                                                 encoding="utf-8", newline="",
                                                 errors="surrogateescape")
                    elif snapshot.path.exists():
                        snapshot.path.unlink()
                    reverted += 1
                except OSError as exc:
                    self.ui.warn(f"could not revert {name}: {exc}")
            else:
                kept += 1
        self.ui.success(f"kept {kept}, reverted {reverted}")

    def _expand_mentions(self, text: str) -> str:
        """Inline any @path the user referenced, and say what could not be.

        A mention that quietly did nothing is worse than one that reports
        why, so problems are printed rather than swallowed.
        """
        from .mentions import expand, find

        if not find(text):
            return text
        expanded, problems = expand(text, self.workspace, self.boundary)
        for problem in problems:
            self.ui.warn(problem)
        if expanded != text:
            count = expanded.count("### ")
            self.ui.info(f"read {count} referenced "
                         f"{'file' if count == 1 else 'files'}")
        return expanded

    async def _narrate(self, request: str, result) -> None:
        """Have the talker say what the coder did, and speak it.

        With no talker configured the coder's own answer is what gets read
        out, so speech works on its own -- the two features are independent.
        """
        if self.talker is not None:
            line = await self.talker.report(
                request, "\n".join(result.errors) if result.errors else result.content,
                failed=bool(result.errors))
            if line:
                await self._talk(line)
                return
            if self.talker.last_error:
                self.ui.warn(f"talker: {self.talker.last_error}")
        if result.content and not result.errors:
            self.speaker.say(result.content)

    async def _talk(self, line: str) -> None:
        """Show one line from the talker, and say it out loud."""
        if not line:
            return
        self.ui.console.print()
        self.ui.console.print(
            Text("  " + self.pet.face(advance=False) + "  ",
                 style=f"bold {self.pet.style()}") + Text(line, style=ACCENT))
        self.speaker.say(line)

    def _prompt_message(self) -> HTML:
        """The left edge of the input box.

        Re-evaluated on every redraw, so a mid-prompt Ctrl-E shows up at once.
        """
        if is_dumb_terminal():
            return HTML('<b>&gt;</b> ')
        edge = self.ui.g.vbar
        return HTML(
            '<ansicyan>%s</ansicyan> <b><ansicyan>&gt;</ansicyan></b> ' % edge)

    def _open_box(self) -> None:
        """Top edge of the input box, printed just before the prompt.

        Skipped on a dumb terminal: prompt_toolkit falls back to a plain
        readline there and draws neither the prompt nor the toolbar inside
        the frame, so an opening edge with no closing one is worse than none.
        """
        if is_dumb_terminal():
            return
        g = self.ui.g
        width = max(24, self.ui.width)
        self.ui.console.print(
            Text(g.tl + g.hbar * (width - 2) + g.tr, style=ACCENT))

    def _bottom_toolbar(self):
        """The closing edge of the input box, with the status set into it.

        One line rather than two: a multi-line toolbar does not render on
        terminals that cannot answer a cursor-position request, and the border
        is the natural place for the status anyway.
        """
        from rich.cells import cell_len

        g = self.ui.g
        width = max(30, self.ui.width)
        left = self._status_line()
        hint = "^O thinking   ^C stop"

        # "<bl><hbar> " + status + " " + fill + " " + hint + " <hbar><br>"
        def total(status: str, tail: str, fill: int) -> int:
            head = 3 + cell_len(status) + 1 + fill
            return head + (1 + cell_len(tail) + 3 if tail else 1)

        if total(left, hint, 1) > width:
            hint = ""
        if total(left, hint, 1) > width:
            left = left[: max(0, width - 10)]
        fill = max(1, width - total(left, hint, 0))

        cyan, dim, reset = "\x1b[36m", "\x1b[38;5;247m", "\x1b[0m"
        line = f"{cyan}{g.bl}{g.hbar}{reset} {dim}{left}{reset} {cyan}" + g.hbar * fill
        if hint:
            line += f"{reset} {dim}{hint}{reset} {cyan}{g.hbar}"
        line += f"{g.br}{reset}"
        return ANSI(line)

    def _border_plain(self) -> str:
        """The bottom border with the escapes stripped, for tests."""
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", self._bottom_toolbar().value)

    def _status_line(self) -> str:
        """The idle half of the pinned bar.

        prompt_toolkit renders this immediately below the input, in the same
        place the activity bar occupies while a turn runs -- so the strip is
        there the whole time and only its contents change.
        """
        usage = self.agent.session.usage
        used = self.agent.session.token_estimate()
        limit = self.policy.context_budget or self.config.num_ctx
        pieces = []
        if self.pet.enabled:
            pieces.append(f"{self.pet.face()} {self.pet.name}")
        pieces += [self.config.model,
                   f"{effort_meter(self.policy.name, self.ui.g.unicode)} "
                   f"{self.policy.name}",
                   f"ctx {100 * used / max(1, limit):.0f}%"]
        if usage.completion_tokens:
            pieces.append(f"{usage.completion_tokens} tok")
            if speed := usage.tokens_per_second():
                pieces.append(f"{speed:.0f} tok/s")
            if self._last_elapsed:
                pieces.append(f"{self._last_elapsed:.1f}s")
        if self.agent.permissions.mode is not Mode.MANUAL:
            pieces.append(self.agent.permissions.mode.value)

        return f"  {self.ui.g.dot}  ".join(pieces)

    async def _drain_queue(self) -> bool:
        """Run whatever was typed during the turn, oldest first.

        Shown before each one runs: a message typed a minute ago and then
        silently executed is startling.
        """
        while (queued := self.pending.take()) is not None:
            self.ui.console.print()
            self.ui.console.print(
                Text("  \u203a ", style=f"bold {ACCENT}") + Text(queued))
            if queued.startswith("/"):
                if await self.command(queued) is False:
                    return False
                continue
            await self.turn(queued)
        return True

    def _shift_effort(self, delta: int) -> None:
        """Step the effort level without typing a command."""
        policy = self.policy.bump(delta)
        if policy.name == self.policy.name:
            return
        self.config.effort = policy.name
        self.agent.set_effort(policy)
        self.policy = self.agent.policy
        self.pet.set_pace(self.policy.name)
        self.ui.info(f"effort: {self.policy.name} -- {self.policy.headline}")

    def _arm_interrupt(self) -> None:
        """Re-install the SIGINT handler. Must run before every turn.

        prompt_toolkit's Application installs its own SIGINT handler while it
        reads a line and calls loop.remove_signal_handler(SIGINT) in its
        finally -- and there is only one handler per signal, so that removes
        ours rather than restoring it. After the first prompt there is no
        handler left at all, which is why Ctrl-C did nothing for the whole
        rest of the session.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                signal.signal(
                    signal.SIGINT,
                    lambda *_: loop.call_soon_threadsafe(self.interrupt))
            return
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGINT, self.interrupt)

    def interrupt(self) -> None:
        # Silence her first: a voice still talking about the thing you just
        # cancelled is the most annoying possible response to Ctrl-C.
        self.speaker.stop()
        if self._task and not self._task.done():
            self._task.cancel()

    # -- slash commands ----------------------------------------------------

    async def command(self, text: str) -> bool:
        parts = text.split()
        name, args = parts[0].lower(), parts[1:]

        if name not in COMMANDS and name not in ALIASES:
            resolved = resolve_command(name)
            if resolved is None:
                near = [c for c in COMMANDS if c.startswith(name)]
                self.ui.warn(
                    f"unknown command {name}."
                    + (f" Did you mean {' or '.join(near)}?" if near
                       else " /help for the list."))
                return True
            name = resolved

        if name in ("/quit", "/exit", "/q"):
            return False

        if name == "/help":
            self.ui.table(
                ["command", "what it does"],
                [(c, d) for c, d in COMMANDS.items()],
                title="commands",
            )
            self.ui.table(
                ["key", "does"],
                [("Ctrl-O", "show or hide the model's thinking (works mid-answer)"),
                 ("Ctrl-T", "full tool output vs. one-line summary (mid-answer)"),
                 ("Ctrl-E", "step effort up"),
                 ("Ctrl-B", "step effort down"),
                 ("Ctrl-C", "interrupt the current turn, keep the conversation"),
                 ("Alt-Enter", "newline instead of submitting"),
                 ("Up / Down", "history")],
                title="keys",
            )
            return True

        if name == "/effort":
            return await self.cmd_effort(args)
        if name == "/model":
            return await self.cmd_model(args)
        if name == "/endpoint":
            return await self.cmd_endpoint(args)
        if name == "/ctx":
            return await self.cmd_ctx(args)

        if name == "/tools":
            self.ui.table(
                ["tool", "writes?", "what it does"],
                [(t.signature(), "yes" if t.mutating else "", t.description[:70])
                 for t in self.agent.tools],
                title=f"{len(self.agent.tools)} tools"
                + ("" if self.agent.native_tools else "  (Hermes prompted mode)"),
            )
            return True

        if name == "/thinking":
            self.ui.show_thinking = not self.ui.show_thinking
            self.config.show_thinking = self.ui.show_thinking
            self.ui.info(f"thinking display {'on' if self.ui.show_thinking else 'off'}")
            return True

        if name == "/plan":
            todo = self.agent.tools.get("todo_write")
            rendered = todo.render() if todo and hasattr(todo, "render") else ""
            self.ui.todos(rendered) if rendered else self.ui.info("no plan yet")
            return True

        if name == "/clear":
            self.agent.session = Session(workspace=self.workspace)
            self.agent.refresh_system_prompt()
            self.ui.info("conversation cleared")
            return True

        if name == "/new":
            return self.cmd_new()

        if name == "/compact":
            before = self.agent.session.token_estimate()
            with self.ui.status("compacting..."):
                await self.agent._compact()
            after = self.agent.session.token_estimate()
            self.ui.success(f"{before} -> {after} tokens")
            return True

        if name == "/stats":
            return self.cmd_stats()

        if name == "/doctor":
            from .doctor import Doctor
            await Doctor(self.client, self.config, self.ui).run()
            return True

        if name == "/mode":
            return await self.cmd_mode(args)
        if name == "/scope":
            return await self.cmd_scope(args)
        if name == "/cd":
            return self.cmd_cd(args)
        if name == "/repo":
            return await self.cmd_repo(args)
        if name == "/undo":
            return self.cmd_undo(args)
        if name == "/memory":
            return self.cmd_memory(args)

        if name == "/pet":
            return await self.cmd_pet(args)

        if name == "/theme":
            return await self.cmd_theme(args)

        if name == "/fullscreen":
            return await self.cmd_fullscreen(args)

        if name == "/secrets":
            return await self.cmd_secrets(args)

        if name == "/speak":
            return await self.cmd_speak(args)

        if name == "/talker":
            return self.cmd_talker(args)

        if name == "/log":
            return self.cmd_log(args)

        if name == "/yolo":
            self.agent.permissions.yolo = not self.agent.permissions.yolo
            if self.agent.permissions.yolo:
                self.ui.warn("Permission prompts off. The agent can write files and run commands freely.")
            else:
                self.ui.info("Permission prompts back on.")
            return True

        if name == "/resume":
            return await self.cmd_resume(args)

        if name == "/commit":
            return await self.cmd_commit(args)

        if name == "/map":
            return self.cmd_map(args)

        if name == "/sessions":
            rows = Session.recent()
            if not rows:
                self.ui.info("no saved sessions")
                return True
            self.ui.table(
                ["id", "messages", "started with"],
                [(r["session_id"], r["messages"], r["preview"]) for r in rows],
                title="recent sessions",
            )
            return True

        if name == "/init":
            await self.turn(
                "Look at this project -- its layout, build files, tests, and "
                "conventions -- then write a WYNXO.md at the root that tells a "
                "new agent what it needs to know: what the project is, how to "
                "build and test it, and the conventions to follow. Be concrete "
                "and brief. No filler."
            )
            return True

        self.ui.warn(f"unknown command {name}. /help for the list.")
        return True

    async def cmd_commit(self, args: list[str]) -> bool:
        """Write a commit message from what is actually staged, then commit.

        Reads the staged diff, has the model describe it, shows the result,
        and only commits once you say so. Nothing is staged for you: what to
        include is a decision the message should describe, not one this
        should make on your behalf.
        """
        from .prompts import COMMIT_PROMPT
        from .repo import run_git

        ok, _ = run_git(["rev-parse", "--git-dir"], cwd=self.workspace, timeout=10)
        if not ok:
            self.ui.warn("not a git repository")
            return True

        ok, staged = run_git(["diff", "--staged"], cwd=self.workspace, timeout=60)
        if not ok:
            self.ui.warn(f"could not read the staged diff: {staged}")
            return True
        if not staged.strip():
            self.ui.info("nothing staged")
            _, unstaged = run_git(["status", "--short"], cwd=self.workspace,
                                  timeout=30)
            if unstaged.strip():
                self.ui.info("stage what you want first:  git add -p")
            return True

        _, stat = run_git(["diff", "--staged", "--stat"], cwd=self.workspace,
                          timeout=30)
        for line in stat.strip().splitlines()[-1:]:
            self.ui.info(line.strip())

        # Cap what is sent: a huge diff would blow the context window, and
        # the top of it describes the change well enough to name it.
        body = staged if len(staged) <= 24_000 else (
            staged[:24_000] + "\n\n... [diff truncated]")

        message = " ".join(args).strip()
        if not message:
            with self.ui.status("reading the diff..."):
                turn = await self.agent._call_model(
                    messages=[{"role": "user",
                               "content": f"{COMMIT_PROMPT}\n\n```diff\n{body}\n```"}],
                    use_tools=False, stream_content=False)
            message = _clean_commit_message(turn.content)
        if not message:
            self.ui.warn("the model did not produce a message; write one yourself")
            return True

        self.ui.console.print()
        self.ui.console.print(Text("  " + message.splitlines()[0], style=f"bold {ACCENT}"))
        for line in message.splitlines()[1:]:
            self.ui.console.print(Text("  " + line, style=MUTED))
        self.ui.console.print()

        answer = (await self.prompt_session.prompt_async(
            HTML('<ansicyan>  [y] commit  [e] edit  [n] no: </ansicyan>')
        )).strip().lower()

        if answer in ("e", "edit"):
            edited = (await self.prompt_session.prompt_async(
                HTML('<ansicyan>  message: </ansicyan>'), default=message.splitlines()[0]
            )).strip()
            if not edited:
                self.ui.info("cancelled")
                return True
            message = edited
        elif answer not in ("", "y", "yes"):
            self.ui.info("not committed")
            return True

        ok, output = run_git(["commit", "-m", message], cwd=self.workspace,
                             timeout=60)
        if not ok:
            self.ui.warn(f"commit failed: {_first(output)}")
            return True
        self.ui.success(_first(output) or "committed")
        self.journal.note("committed", subject=message.splitlines()[0])
        return True

    async def cmd_resume(self, args: list[str]) -> bool:
        """Carry on an earlier conversation, with its history.

        Sessions were already being written to disk after every turn; there
        was simply no way back into one, which made them a debugging
        artefact rather than something you could use.
        """
        import time as _time

        rows = Session.recent()
        if not rows:
            self.ui.info("no saved conversations yet")
            return True

        if args:
            wanted = args[0]
            match = next((r for r in rows if r["session_id"].startswith(wanted)), None)
            if match is None:
                self.ui.warn(f"no saved conversation starting {wanted!r}")
                return True
            return self._load_session(match["session_id"])

        def age(stamp: float) -> str:
            if not stamp:
                return "?"
            seconds = max(0, _time.time() - stamp)
            for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
                if seconds >= size:
                    return f"{int(seconds // size)}{unit} ago"
            return "just now"

        options = [
            Choice(value=r["session_id"],
                   label=age(r.get("updated_at", 0)),
                   badge=f"{r['messages']} msgs",
                   badge_style="badge.muted",
                   hint=(r["preview"] or "(no messages)"))
            for r in rows
        ]
        if arrows_supported():
            chosen = await choose(
                options, title="resume which conversation?", default=0,
                footer=HINT if self.ui.g.unicode else HINT_ASCII,
                width=self.ui.width, unicode=self.ui.g.unicode)
            if not chosen:
                return True
            return self._load_session(chosen)

        self.ui.table(
            ["id", "when", "messages", "started with"],
            [(r["session_id"][:8], age(r.get("updated_at", 0)),
              str(r["messages"]), r["preview"]) for r in rows],
            title="recent conversations",
        )
        self.ui.info("/resume <id> to pick one up")
        return True

    def _load_session(self, session_id: str) -> bool:
        restored = Session.load(session_id, self.workspace)
        if restored is None:
            self.ui.warn(f"could not read conversation {session_id[:8]}")
            return True

        # The system prompt is rebuilt rather than restored: effort, scope,
        # mode and memory may all have moved on since, and the saved one
        # would quietly reinstate the old ones.
        self.agent.session = restored
        self.agent.checkpoints.clear()
        self.agent.refresh_system_prompt()
        self._last_elapsed = 0.0

        self.ui.success(f"resumed {session_id[:8]} -- "
                        f"{len(restored.messages)} messages")
        self.ui.info("undo history is not restored; it belonged to that run")
        return True

    def cmd_new(self) -> bool:
        """A new chat, the way opening a new tab is new.

        /clear empties the message list in place. This goes further: a new
        session id and a new log file, so the old conversation stays intact
        and reviewable rather than being half-overwritten, undo history reset
        because those snapshots belong to the chat that is over, and a clean
        screen so what is in front of you matches what the model can see.

        Memory survives on purpose -- it is the thing that is supposed to
        outlive a conversation.
        """
        self.agent.session = Session(workspace=self.workspace)
        self.agent.checkpoints.clear()
        self.agent.refresh_system_prompt()
        self.callbacks.tokens = 0
        self._last_elapsed = 0.0
        self.pending.clear()

        self.journal = Journal.open(self.agent_session_id(),
                                    enabled=self.config.log)
        self.callbacks.journal = self.journal

        self.ui.clear()
        self.ui.banner(self.config.model, self.client.base_url,
                       self.policy.name, str(self.workspace))
        self.ui.console.print()
        self.ui.info("new chat -- memory kept, history and undo reset")
        return True

    async def cmd_effort(self, args: list[str]) -> bool:
        if not args:
            options = [(n, resolve(n).headline) for n in ORDER]
            chosen = await self._pick("effort", options, self.policy.name)
            if chosen is None:
                return True
            if chosen is NO_PICKER:
                self.ui.table(
                    ["level", "behaviour"],
                    [(n + ("  <-" if n == self.policy.name else ""),
                      resolve(n).describe()) for n in ORDER],
                    title="effort levels",
                )
                return True
            args = [chosen]
        try:
            policy = resolve(args[0])
        except KeyError as exc:
            self.ui.warn(str(exc))
            return True
        previous = self.policy.name
        self.config.effort = policy.name
        self.agent.set_effort(policy)
        # set_effort() may downgrade thinking for the current model; read the
        # policy back rather than trusting the one just resolved, so the
        # status bar reports what the agent will actually do.
        self.policy = self.agent.policy
        self.pet.set_pace(self.policy.name)
        await self._effort_surge(previous, self.policy.name)
        self.ui.success(f"effort: {self.policy.name} -- {self.policy.describe()}")
        return True

    async def _effort_surge(self, previous: str, current: str) -> None:
        """Make stepping up to the top two levels feel like it costs
        something, because it does.

        Only on the way up, and only into max or ultra: an animation that
        played on every change would be noise, and one that played on the way
        down would be celebrating the wrong direction.
        """
        from .ui import surge

        heavy = {"max": (BAR_ACCENT, "MAX EFFORT"),
                 "ultra": (ACCENT, "ULTRA")}
        if current not in heavy or current == previous:
            return
        if ORDER.index(current) <= ORDER.index(previous):
            return
        if not self.config.animations:
            return
        style, label = heavy[current]
        await surge(self.ui, label, style)

    async def cmd_model(self, args: list[str]) -> bool:
        """Switch model. With no argument this is the same capability-aware
        picker the first-run wizard uses, rather than a second, dumber list."""
        from .provider import inspect_all
        from .wizard import _model_choice, _print_model_rows

        try:
            with self.ui.status("asking the server what it has..."):
                models = await self.client.list_models()
        except ProviderError as exc:
            self.ui.error(str(exc))
            return True
        if not models:
            self.ui.warn("that server has no models installed")
            return True

        if args:
            return await self._switch_model(args[0], [m.name for m in models])

        with self.ui.status(f"checking what {len(models)} model(s) can do..."):
            models = await inspect_all(self.client, models)
        models.sort(key=lambda m: (not m.supports_tools, m.name))
        current = next((i for i, m in enumerate(models)
                        if m.name == self.config.model), 0)

        if arrows_supported():
            chosen = await choose(
                [_model_choice(m) for m in models],
                default=current,
                footer=HINT if self.ui.g.unicode else HINT_ASCII,
                width=self.ui.width,
                unicode=self.ui.g.unicode,
            )
            if chosen and chosen != self.config.model:
                return await self._switch_model(chosen, [m.name for m in models])
            if chosen:
                self.ui.info(f"already using {chosen}")
            return True

        self.ui.console.print()
        _print_model_rows(self.ui, models)
        self.ui.info("switch with /model <name>")
        return True

    async def _switch_model(self, target: str, names: list[str]) -> bool:
        if target not in names:
            matches = [n for n in names if n.split(":")[0] == target
                       or n.startswith(target)]
            if len(matches) == 1:
                target = matches[0]
            else:
                self.ui.warn(f"{target} is not installed on that server.")
                if matches:
                    self.ui.info(f"did you mean: {', '.join(matches)}")
                else:
                    self.ui.info(f"pull it first:  ollama pull {target}")
                return True

        self.config.model = target
        self.agent.config.model = target
        await self.agent.detect_capabilities()
        self.policy = self.agent.policy
        self.ui.success(f"model: {target}")
        self.journal.note("model switched", model=target)
        if warning := await check_context(self.client, self.config):
            self.ui.warn(warning)
        return True

    async def cmd_endpoint(self, args: list[str]) -> bool:
        action = args[0].lower() if args else "list"

        if action == "list" and not args:
            # Bare /endpoint: pick which server to talk to. With more than
            # one configured that is almost always what you meant.
            options = [(e.name, e.url) for e in self.config.endpoints]
            picked = (await self._pick("ollama server", options,
                                       self.config.active_endpoint)
                      if len(options) > 1 else NO_PICKER)
            if picked is None:
                return True
            if picked is not NO_PICKER:
                if picked == self.config.active_endpoint:
                    self.ui.info(f"already using {picked}")
                    return True
                return await self.cmd_endpoint(["use", picked])

        if action == "list":
            self.ui.table(
                ["name", "url"],
                [(e.name + ("  <-" if e.name == self.config.active_endpoint else ""), e.url)
                 for e in self.config.endpoints],
                title="ollama servers",
            )
            dot = self.ui.g.dot
            self.ui.info(f"/endpoint add <url> {dot} /endpoint use <name> {dot} /endpoint test")
            return True

        if action == "test":
            for endpoint in self.config.endpoints:
                with self.ui.status(f"checking {endpoint.url}..."):
                    version = await probe(endpoint.url, timeout=5.0)
                if version:
                    self.ui.success(f"{endpoint.name}: {endpoint.url} (ollama {version})")
                else:
                    self.ui.warn(f"{endpoint.name}: {endpoint.url} unreachable")
            return True

        if action == "add" and len(args) > 1:
            url = normalise_url(args[1])
            name = args[2] if len(args) > 2 else url.split("://")[1].split(":")[0]
            with self.ui.status(f"checking {url}..."):
                version = await probe(url, timeout=6.0)
            if not version:
                self.ui.error(f"Nothing answering at {url}.\n\n{server_help()}")
                return True
            self.config.endpoints = [e for e in self.config.endpoints if e.name != name]
            self.config.endpoints.append(Endpoint(name=name, url=url))
            self.config.save()
            self.ui.success(f"added {name} -> {url} (ollama {version})")
            self.ui.info(f"switch to it with /endpoint use {name}")
            return True

        if action == "use" and len(args) > 1:
            name = args[1]
            if not any(e.name == name for e in self.config.endpoints):
                self.ui.warn(f"no endpoint named {name}. /endpoint list to see them.")
                return True
            self.config.active_endpoint = name
            self.config.save()
            await self.client.aclose()
            self.client = OllamaClient(self.config)
            self.agent.client = self.client
            try:
                version = await self.client.ping()
            except ProviderError as exc:
                self.ui.error(str(exc))
                return True
            self.ui.success(f"now using {name}: {self.client.base_url} (ollama {version})")
            return True

        self.ui.warn("usage: /endpoint [list | test | add <url> [name] | use <name>]")
        return True

    async def cmd_mode(self, args: list[str]) -> bool:
        if not args:
            options = [(m.value, m.describe()) for m in Mode]
            chosen = await self._pick("permission mode", options,
                                      self.agent.permissions.mode.value)
            if chosen is None:
                return True
            if chosen is NO_PICKER:
                self.ui.table(
                    ["mode", "behaviour"],
                    [(m.value + ("  <-" if m is self.agent.permissions.mode else ""),
                      m.describe()) for m in Mode],
                    title="permission modes",
                )
                return True
            if chosen == self.agent.permissions.mode.value:
                self.ui.info(f"already in {chosen} mode")
                return True
            args = [chosen]
        try:
            mode = Mode.parse(args[0])
        except KeyError as exc:
            self.ui.warn(str(exc))
            return True
        self.agent.permissions.mode = mode
        self.agent.refresh_system_prompt()
        self.ui.success(f"mode: {mode.value} -- {mode.describe()}")
        if mode is Mode.YOLO:
            self.ui.warn("Nothing will ask for approval from here on.")
        return True

    def _refresh_map(self, note=None) -> None:
        """Rebuild the project map if the files have moved on.

        Never fatal: a project that cannot be walked, or a read-only one that
        cannot cache the result, still gets a session -- just without the
        head start.
        """
        from . import projectmap

        try:
            self.agent.project_map = projectmap.load(self.workspace)
        except Exception as exc:
            self.agent.project_map = ""
            if note is not None:
                note(WARN, "project map", str(exc))
            return
        self.agent.refresh_system_prompt()

    def cmd_map(self, args: list[str]) -> bool:
        """Show the map, or force a rebuild."""
        from . import projectmap

        if args and args[0].lower() in ("rebuild", "refresh", "again"):
            path = projectmap.cache_path(self.workspace)
            with contextlib.suppress(OSError):
                path.unlink()
            with self.ui.status("mapping the project..."):
                self._refresh_map()
            self.ui.success(projectmap.summarise(self.agent.project_map)
                            or "nothing to map here")
            return True

        if not self.agent.project_map:
            self.ui.info("no map for this project")
            self.ui.info("/map rebuild to try again")
            return True
        self.ui.console.print()
        for line in self.agent.project_map.splitlines():
            style = f"bold {ACCENT}" if line.startswith("#") else MUTED
            self.ui.console.print(Text("  " + line, style=style))
        self.ui.console.print()
        self.ui.info(f"{projectmap.cache_path(self.workspace)}  "
                     f"{self.ui.g.dot}  /map rebuild")
        return True

    def _boundary_summary(self, boundary) -> str:
        """describe(), but with the path shortened for the terminal.

        Boundary.describe() is also what goes into the system prompt, where
        the model needs the real, full path -- so the shortening happens
        here instead, only for what the user reads.
        """
        if boundary.unrestricted:
            return "the whole machine"
        root = self.ui.shorten_path(str(boundary.root))
        return f"the repository at {root}" if boundary.scope is Scope.REPO else root

    async def cmd_scope(self, args: list[str]) -> bool:
        if not args:
            current = self.agent.boundary
            means = {"folder": "only the directory wynxo started in",
                     "repo": "the whole git repository",
                     "machine": "no path restriction at all"}
            options = [(s.value, means[s.value]) for s in Scope]
            chosen = await self._pick(
                "scope", options, current.scope.value if current else "")
            if chosen is None:
                return True
            if chosen is NO_PICKER:
                self.ui.table(
                    ["scope", "means"],
                    [(s.value + ("  <-" if current and s is current.scope else ""),
                      means[s.value]) for s in Scope],
                    title="scope",
                )
                if current:
                    self.ui.info(f"currently: {self._boundary_summary(current)}")
                self.ui.info("/scope <path> or /cd <path> moves to another directory")
                return True
            if current and chosen == current.scope.value:
                self.ui.info(f"already scoped to {chosen}")
                return True
            args = [chosen]
        try:
            scope = Scope.parse(args[0])
        except KeyError:
            # Not one of the three words, so read it as a directory. Pointing
            # wynxo at another project mid-session is the obvious thing to
            # want from a command called /scope, and refusing it over a
            # vocabulary mismatch helps nobody.
            return self.cmd_cd(args)

        if scope is Scope.MACHINE:
            self.ui.warn(
                "Machine scope removes the path restriction entirely: the agent "
                "can read and write anywhere your user account can.")
            try:
                answer = (await self.prompt_session.prompt_async(
                    HTML('<ansicyan>  really widen to the whole machine? [y/N]: </ansicyan>')
                )).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer not in ("y", "yes"):
                self.ui.info("left unchanged")
                return True

        self._apply_scope(scope)
        self.ui.success(f"scope: {self._boundary_summary(self.agent.boundary)}")
        return True

    async def cmd_repo(self, args: list[str]) -> bool:
        """Clone a GitHub repository and move the workspace into it."""
        from . import repo as repo_module

        if not args:
            current = repo_module.status(self.workspace)
            if current:
                self.ui.info(f"{self.ui.shorten_path(str(self.workspace))}  "
                             f"on {current}")
            else:
                self.ui.info("not a git checkout")
            self.ui.info(f"/repo owner/name  {self.ui.g.dot}  /repo <url>")
            return True

        if not repo_module.git_available():
            self.ui.error("git is not installed, so wynxo cannot clone anything.")
            return True

        target = repo_module.parse(" ".join(args))
        if target is None:
            self.ui.warn("that does not look like a repository. Try owner/name "
                         "or a GitHub URL.")
            return True

        with self.ui.status(f"fetching {target.slug}..."):
            ok, path, message = repo_module.clone_or_update(target)
        if not ok:
            self.ui.error(message)
            return True

        self.ui.success(message)
        self.cmd_cd([str(path)])
        # A checkout is a repository, so widen to it rather than pinning the
        # agent to whichever subdirectory happens to be the clone root.
        self._apply_scope(Scope.REPO)
        if branch := repo_module.status(path):
            self.ui.info(f"on {branch}")
        self.ui.info("wynxo does not push. Ask it to run git, and approve it.")
        return True

    def cmd_cd(self, args: list[str]) -> bool:
        """Move the agent to another directory, keeping the conversation."""
        if not args:
            self.ui.info(f"{self.workspace}")
            return True

        raw = " ".join(args).strip().strip('"').strip("'")
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = (self.workspace / target)
        try:
            target = target.resolve()
        except OSError as exc:
            self.ui.warn(f"cannot resolve {raw}: {exc}")
            return True

        if not target.exists():
            self.ui.warn(f"{target} does not exist")
            return True
        if not target.is_dir():
            self.ui.warn(f"{target} is a file, not a directory")
            return True

        self.workspace = target
        self.memory = Memory(target)
        self.agent.workspace = target
        self.agent.memory = self.memory
        self._apply_scope(self.boundary.scope)
        self._refresh_map()
        self.ui.success(f"working in {self.ui.shorten_path(str(target))}")
        if reason := suspicious_workspace(target):
            self.ui.warn(reason)
        self.journal.note("workspace changed", path=str(target))
        return True

    def _apply_scope(self, scope: Scope) -> None:
        boundary = resolve_scope(self.workspace, scope)
        self.boundary = boundary
        if boundary.scope is not scope and scope is Scope.REPO:
            self.ui.warn("Not inside a git repository; staying with folder scope.")

        # The registry is rebuilt with new tool instances, so carry over the
        # state that lives on a tool -- otherwise changing scope silently
        # wipes the plan the agent is working through.
        previous_todo = self.agent.tools.get("todo_write")
        self.agent.boundary = boundary
        self.agent.tools = build_registry(
            self.workspace, allow_shell=self.config.allow_shell,
            boundary=boundary, memory=self.agent.memory)
        current_todo = self.agent.tools.get("todo_write")
        if previous_todo is not None and current_todo is not None:
            current_todo.items = previous_todo.items
        self.agent.refresh_system_prompt()

    def cmd_undo(self, args: list[str]) -> bool:
        if args and args[0] in ("list", "history"):
            history = self.agent.checkpoints.history()
            if not history:
                self.ui.info("nothing to undo")
                return True
            self.ui.table(
                ["file", "changed by"],
                [(s.label or s.path.name, s.tool) for s in history],
                title="undo history (most recent first)",
            )
            return True

        count = 1
        if args and args[0].isdigit():
            count = max(1, min(20, int(args[0])))
        for _ in range(count):
            done, message = self.agent.checkpoints.undo()
            if not done:
                self.ui.info(message)
                break
            self.ui.success(message)
        return True

    async def cmd_speak(self, args: list[str]) -> bool:
        """Turn the voice on and off, and say which synthesiser is doing it."""
        from .speech import (Speaker, available, install_hint,
                             pick as pick_engine)

        action = args[0].lower() if args else "show"

        if action in ("show", "status"):
            self.ui.info(f"speech: {self.speaker.describe()}")
            options = available()
            if options:
                self.ui.info("available: " + ", ".join(
                    f"{e.name} ({e.quality})" for e in options))
            else:
                self.ui.warn("No speech synthesiser found on this machine.")
                for line in install_hint().splitlines():
                    self.ui.info(line)
            return True

        if action in ("on", "off"):
            want = action == "on"
            if want and self.speaker.engine is None:
                engine = pick_engine(self.config.speech_engine)
                if engine is None:
                    self.ui.warn("No speech synthesiser found on this machine.")
                    for line in install_hint().splitlines():
                        self.ui.info(line)
                    return True
                self.speaker = Speaker(engine, voice=self.config.speech_voice,
                                       rate=self.config.speech_rate,
                                       model=self.config.speech_model)
            self.speaker.enabled = want
            if not want:
                self.speaker.stop()
            self.config.speak = want
            self.ui.success(f"speech: {self.speaker.describe()}")
            return True

        if action == "test":
            if not self.speaker.say("Hello. If you can hear this, the voice works."):
                self.ui.warn("Nothing was said. " +
                             (self.speaker.last_error or "Speech is off."))
                return True
            self.ui.success(f"spoke through {self.speaker.describe()}")
            return True

        if action == "engine" and len(args) == 1:
            options = [(e.name, e.quality) for e in available()]
            if not options:
                self.ui.warn("No speech synthesiser found on this machine.")
                for line in install_hint().splitlines():
                    self.ui.info(line)
                return True
            picked = await self._pick("speech engine", options,
                                      self.config.speech_engine)
            if picked is None:
                return True
            if picked is NO_PICKER:
                self.ui.info("available: " + ", ".join(n for n, _ in options))
                return True
            args = [action, picked]

        if action in ("engine", "voice") and len(args) > 1:
            if action == "engine":
                engine = pick_engine(args[1])
                if engine is None:
                    self.ui.warn(f"{args[1]} is not available here.")
                    return True
                self.config.speech_engine = args[1]
                self.speaker = Speaker(engine, voice=self.config.speech_voice,
                                       rate=self.config.speech_rate,
                                       model=self.config.speech_model)
            else:
                self.config.speech_voice = args[1]
                self.speaker.voice = args[1]
            self.speaker.enabled = self.config.speak
            self.ui.success(f"speech: {self.speaker.describe()}")
            return True

        self.ui.info("/speak on | off | test | engine <name> | voice <name>")
        return True

    def cmd_talker(self, args: list[str]) -> bool:
        """Set the small model that does the talking, or turn it off."""
        from .duo import Talker
        from .prompts import VOICES

        if not args:
            if self.talker is None:
                self.ui.info("no talker -- one model does both jobs")
                self.ui.info("/talker <model> to have a small one do the talking")
            else:
                self.ui.info(f"talker: {self.talker.model}   "
                             f"coder: {self.config.model}")
            return True

        if args[0].lower() in ("off", "none"):
            self.talker = None
            self.config.talker = ""
            self.ui.success("talker off -- one model does both jobs")
            return True

        self.config.talker = args[0]
        self.talker = Talker(self.client, args[0],
                             voice_block=VOICES.get(self.config.voice, ""))
        self.ui.success(f"talker: {args[0]}   coder: {self.config.model}")
        return True

    async def cmd_theme(self, args: list[str]) -> bool:
        from . import theme as theme_module

        if not args:
            options = [(n, _theme_summary(n)) for n in theme_module.names()]
            chosen = await self._pick("theme", options, self.config.theme)
            if chosen is None:
                return True
            if chosen is NO_PICKER:
                self.ui.table(
                    ["theme", "look"],
                    [(n + ("  <-" if n == self.config.theme else ""), summary)
                     for n, summary in options],
                    title="themes",
                )
                self.ui.info("/theme <name> to change it")
                return True
            if chosen == self.config.theme:
                self.ui.info(f"already using {chosen}")
                return True
            args = [chosen]

        choice = args[0].lower()
        if choice not in theme_module.names():
            self.ui.warn(f"unknown theme; choose one of "
                         f"{', '.join(theme_module.names())}")
            return True
        self.config.theme = choice
        self.config.save()
        # Rebind now so the rest of this session picks it up, and say plainly
        # that anything already drawn keeps the old colours.
        from .ui import apply_palette

        self.ui.palette = theme_module.resolve(choice)
        apply_palette(self.ui.palette)
        self.ui.code_theme = self.ui.palette.code_theme
        self.pet.style_name = self.pet.style_name   # keep the face set
        self.ui.success(f"theme: {choice}")
        self._preview_theme()
        return True

    async def _pick(self, title: str, options: list[tuple[str, str]],
                    current: str) -> str | None:
        """Arrow-key chooser for a simple setting.

        Returns the chosen value, None if the user pressed escape, or
        NO_PICKER when this terminal cannot draw one. Cancelling and being
        unable to offer a choice are different things: escape means "never
        mind", and printing the table anyway ignores that.
        """
        if not arrows_supported():
            return NO_PICKER
        chosen = await choose(
            [Choice(value=name,
                    label=name,
                    badge="current" if name == current else "",
                    badge_style="badge",
                    hint=summary)
             for name, summary in options],
            title=title,
            default=next((i for i, (n, _) in enumerate(options) if n == current), 0),
            footer=HINT if self.ui.g.unicode else HINT_ASCII,
            width=self.ui.width,
            unicode=self.ui.g.unicode,
        )
        return chosen

    async def cmd_fullscreen(self, args: list[str]) -> bool:
        """Switch screens now, and remember the choice.

        Takes effect immediately rather than at next start: a setting that
        says it changed while the screen plainly did not is the kind of
        thing that makes people stop trusting the whole settings menu.
        """
        want = args[0].lower() if args else ""
        if want not in ("on", "off"):
            chosen = await self._pick(
                "fullscreen",
                [("on", "take over the terminal; your shell comes back "
                        "untouched on exit, but this screen has no scrollback"),
                 ("off", "stay in the normal scrolling terminal")],
                "on" if self.config.fullscreen else "off",
            )
            if chosen is NO_PICKER:
                state = "on" if self.config.fullscreen else "off"
                self.ui.info(f"fullscreen is {state}  {self.ui.g.dot}  "
                             "/fullscreen on | off")
                return True
            if chosen is None:
                return True
            want = chosen

        enable = want == "on"
        if enable and not fullscreen.supported():
            self.ui.warn("this terminal cannot switch screens, so fullscreen "
                         "would do nothing here")
            return True

        self.config.fullscreen = enable
        self.config.save()

        screen = getattr(self, "screen", None)
        if screen is not None:
            screen.enabled = enable or screen.active
            if enable:
                screen.enter()
            else:
                screen.leave()
                screen.enabled = False

        if enable:
            # The new screen is blank, so the conversation so far is not on
            # it. Say where it went rather than letting it look lost.
            self.ui.info(fullscreen.note(True, self.ui.g.unicode))
            self.ui.info("the scrollback from before is waiting on the other "
                         "screen, and comes back when wynxo exits")
        else:
            self.ui.success("back to the scrolling terminal")
        return True

    async def cmd_secrets(self, args: list[str]) -> bool:
        shield = self.agent.shield

        if args and args[0].lower() == "allow":
            if len(args) < 2:
                self.ui.info("/secrets allow <path>  -- let the agent read "
                             "one file it would otherwise refuse")
                return True
            target = " ".join(args[1:])
            shield.allow(target)
            self.ui.warn(f"{target} can now be read this session. It will be "
                         "sent to the model like any other file.")
            return True

        want = args[0].lower() if args else ""
        if want not in ("on", "off"):
            chosen = await self._pick(
                "secrets",
                [("on", "refuse .env files and private keys; mask credentials "
                        "found inside ordinary files, and in the session log"),
                 ("off", "no filtering -- every file goes to the model as-is")],
                "on" if self.config.protect_secrets else "off",
            )
            if chosen is NO_PICKER:
                state = "on" if self.config.protect_secrets else "off"
                self.ui.info(f"secret protection is {state}  {self.ui.g.dot}  "
                             "/secrets on | off | allow <path>")
                return True
            if chosen is None:
                return True
            want = chosen

        enabled = want == "on"
        self.config.protect_secrets = enabled
        self.config.save()
        shield.enabled = enabled
        if enabled:
            self.ui.success("credentials will be kept out of the model and "
                            "the log")
        else:
            # Warned rather than confirmed. The endpoint may well be another
            # machine, so this is a decision about what leaves this one.
            self.ui.warn("secret protection off -- .env files, keys and "
                         "tokens will be sent to the model verbatim and "
                         "written to the session log")
        return True

    def _preview_theme(self) -> None:
        """Show the new colours immediately, on this line.

        A theme change that only affects text drawn later looks like it did
        nothing -- the whole screen above is still the old palette, so
        without a sample there is nothing to see.
        """
        from rich.text import Text as _T

        row = _T("  ")
        # Straight off the palette, not the module-level names: cli.py
        # already has a WARN, and it is the status tag, not a colour.
        palette = self.ui.palette
        for label, style in (("accent", palette.accent), ("text", palette.text),
                             ("muted", palette.muted), ("ok", palette.good),
                             ("warn", palette.warn), ("error", palette.bad)):
            row.append(f" {label} ", style=f"bold {style}")
            row.append(" ")
        self.ui.console.print(row)
        self.ui.console.print()

    def cmd_log(self, args: list[str]) -> bool:
        action = args[0].lower() if args else "show"

        if action in ("off", "on"):
            self.journal.enabled = action == "on"
            self.config.log = self.journal.enabled
            self.config.save()
            self.ui.success(f"logging {action}")
            return True

        if action in ("list", "sessions"):
            logs = recent_logs()
            if not logs:
                self.ui.info("no logs yet")
                return True
            self.ui.table(
                ["log", "size"],
                [(p.name, f"{p.stat().st_size // 1024}KB") for p in logs],
                title="recent sessions",
            )
            return True

        if action == "tail":
            for record in self.journal.tail(20):
                kind = record.get("kind", "?")
                body = (record.get("text") or record.get("message")
                        or record.get("name") or "")
                self.ui.console.print(
                    f"  [{MUTED}]{kind:12}[/] {str(body)[:90]}")
            return True

        if not self.journal.enabled or self.journal.path is None:
            self.ui.info(f"logging is off  {self.ui.g.dot}  /log on to enable it")
            return True
        self.ui.info(f"{self.journal.path}")
        dot = self.ui.g.dot
        self.ui.info(f"{self.journal.size() // 1024}KB  {dot}  /log tail  {dot}  "
                     f"/log list  {dot}  /log off")
        return True

    async def cmd_pet(self, args: list[str]) -> bool:
        from .prompts import VOICES

        if not args:
            self.ui.console.print()
            for mood in Mood:
                self.pet.react(mood)
                self.pet._frame = 0
                frames = "  ".join(self.pet.face() for _ in range(0, 12, 3))
                self.ui.console.print(
                    f"    [{self.pet.style()}]{frames}[/]  [{MUTED}]{mood.value}[/]")
            self.pet.rest()
            self.ui.console.print()
            self.ui.info(f"name: {self.pet.name}   voice: {self.config.voice}   "
                         f"{'on' if self.pet.enabled else 'off'}"
                         f"{'' if self.pet.animate else ', still'}")
            self.ui.info(f"/pet off {self.ui.g.dot} /pet name <x> {self.ui.g.dot} /pet voice "
                         + " | ".join(VOICES))
            return True

        action = args[0].lower()

        if action in ("on", "off"):
            self.pet.enabled = action == "on"
            self.config.pet = self.pet.enabled
            self.config.save()
            self.ui.success(f"pet {action}")
            return True

        if action in ("still", "animate"):
            self.pet.animate = action == "animate"
            self.config.animations = self.pet.animate
            self.config.save()
            self.ui.success(f"animation {'on' if self.pet.animate else 'off'}")
            return True

        if action == "name" and len(args) > 1:
            self.pet.name = " ".join(args[1:])[:24]
            self.config.pet_name = self.pet.name
            self.config.save()
            self.ui.success(f"{self.pet.face(advance=False)}  hello, {self.pet.name}")
            return True

        if action == "voice":
            if len(args) < 2:
                options = [(v, _voice_summary(v)) for v in VOICES]
                picked = await self._pick("voice", options, self.config.voice)
                if picked is None:
                    return True
                if picked is NO_PICKER:
                    self.ui.table(
                        ["voice", "sounds like"],
                        [(v + ("  <-" if v == self.config.voice else ""),
                          _voice_summary(v)) for v in VOICES],
                        title="voice",
                    )
                    return True
                if picked == self.config.voice:
                    self.ui.info(f"already using {picked}")
                    return True
                args = [action, picked]
            choice = args[1].lower()
            if choice not in VOICES:
                self.ui.warn(f"unknown voice; choose one of {', '.join(VOICES)}")
                return True
            self.config.voice = choice
            self.config.save()
            self.pet.style_name = "kawaii" if choice == "kawaii" else "default"
            self.agent.refresh_system_prompt()
            self.ui.success(f"voice: {choice} -- {_voice_summary(choice)}")
            return True

        self.ui.warn("usage: /pet [on|off|still|animate|name <x>|voice <x>]")
        return True

    def cmd_memory(self, args: list[str]) -> bool:
        memory = self.agent.memory

        if not args or args[0] in ("show", "list"):
            project, user = memory.counts()
            if not project and not user:
                self.ui.info("nothing remembered yet")
                self.ui.info("the agent writes here itself; or: /memory add <note>")
                return True
            if user:
                self.ui.table(["about you"], [(e.lstrip("-* "),)
                                              for e in memory.user.entries()])
            if project:
                self.ui.table(["about this project"], [(e.lstrip("-* "),)
                                                       for e in memory.project.entries()])
            self.ui.info(f"{memory.project.path}")
            return True

        action, rest = args[0], " ".join(args[1:]).strip()

        if action in ("add", "remember") and rest:
            scope = "user" if rest.startswith("user:") else "project"
            note = rest.split(":", 1)[1].strip() if rest.startswith("user:") else rest
            added, message = memory.remember(note, scope)
            self.ui.success(f"remembered: {message}") if added else self.ui.info(message)
            self.agent.refresh_system_prompt()
            return True

        if action in ("forget", "remove") and rest:
            count, message = memory.forget(rest)
            if not count:
                count, message = memory.forget(rest, "user")
            self.ui.info(message)
            self.agent.refresh_system_prompt()
            return True

        if action == "edit":
            path = memory.project.path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(memory.project.header, encoding="utf-8")
            self.ui.info(f"open it in your editor: {path}")
            self.ui.info("changes are picked up with /memory reload")
            return True

        if action == "reload":
            self.agent.refresh_system_prompt()
            project, user = memory.counts()
            self.ui.success(f"reloaded: {project} project, {user} user")
            return True

        self.ui.warn("usage: /memory [show | add <note> | forget <text> | edit | reload]")
        return True

    async def cmd_ctx(self, args: list[str]) -> bool:
        if not args:
            used = self.agent.session.token_estimate()
            self.ui.info(
                f"num_ctx={self.config.num_ctx}, roughly {used} tokens in use "
                f"({100 * used / max(1, self.config.num_ctx):.0f}%)"
            )
            return True
        try:
            value = int(args[0])
        except ValueError:
            self.ui.warn("usage: /ctx <number>")
            return True
        self.config.num_ctx = value
        self.agent.config.num_ctx = value
        self.ui.success(f"num_ctx = {value}")
        if warning := await check_context(self.client, self.config):
            self.ui.warn(warning)
        return True

    def cmd_stats(self) -> bool:
        usage = self.agent.session.usage
        used = self.agent.session.token_estimate()
        limit = self.policy.context_budget or self.config.num_ctx
        self.ui.table(
            ["", ""],
            [
                ("model", self.config.model),
                ("server", self.client.base_url),
                ("effort", f"{self.policy.name} -- {self.policy.describe()}"),
                ("requests", str(usage.requests)),
                ("tool calls", str(usage.tool_calls)),
                ("prompt tokens", str(usage.prompt_tokens)),
                ("output tokens", str(usage.completion_tokens)),
                ("speed", f"{usage.tokens_per_second():.1f} tok/s"),
                ("context", f"~{used} / {limit} ({100 * used / max(1, limit):.0f}%)"),
                ("compactions", str(self.agent.session.compactions)),
                ("tool mode", "native" if self.agent.native_tools else "hermes (prompted)"),
                ("session", self.agent.session.session_id),
            ],
            title="session",
        )
        return True


def read_piped_stdin(grace: float = 0.25) -> str:
    """Read piped stdin, or return "" -- but never block waiting for a writer.

    A plain ``sys.stdin.read()`` here hangs forever whenever stdin is an open
    pipe that nobody writes to, which is the normal state of affairs under CI,
    process supervisors and editor terminals. The agent then sits there with no
    output and no explanation, which looks exactly like a crash.
    """
    if sys.stdin.isatty():
        return ""

    try:
        if stat.S_ISREG(os.fstat(sys.stdin.fileno()).st_mode):
            # A redirect from a real file always has an EOF coming.
            return sys.stdin.read().strip()
    except (OSError, ValueError):
        return ""

    if hasattr(select, "select"):
        try:
            ready, _, _ = select.select([sys.stdin], [], [], grace)
        except (OSError, ValueError):
            return ""
        if not ready:
            return ""
        try:
            return sys.stdin.read().strip()
        except (OSError, UnicodeDecodeError):
            return ""

    # Windows cannot select on a pipe, so read in a thread we can abandon.
    # A slow producer may miss the window; redirect from a file to be certain.
    result: list[str] = []

    def _read() -> None:
        try:
            result.append(sys.stdin.read())
        except (OSError, UnicodeDecodeError, ValueError):
            pass

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(grace)
    return result[0].strip() if result else ""


async def run_once(config: Config, workspace: Path, ui: UI, prompt: str,
                   scope: Scope = Scope.FOLDER, mode: Mode = Mode.YOLO) -> int:
    """Non-interactive mode: answer one prompt and exit."""
    client = OllamaClient(config)
    try:
        await client.ping()
    except ProviderError as exc:
        ui.error(str(exc))
        await client.aclose()
        return 1

    callbacks = TerminalCallbacks(ui)
    agent = Agent(client, config, resolve(config.effort), workspace, callbacks,
                  boundary=resolve_scope(workspace, scope), memory=Memory(workspace))
    # No human to answer prompts, so nothing can be asked; the caller opted
    # into that by using -p. An explicit --mode still wins, so `-p --mode plan`
    # gives a read-only run.
    agent.permissions.mode = mode
    agent.refresh_system_prompt()
    await agent.detect_capabilities()

    try:
        result = await agent.run(prompt)
    except ProviderError as exc:
        ui.error(str(exc))
        await client.aclose()
        return 1
    callbacks._end_stream()
    await client.aclose()

    if result.errors:
        ui.error("\n".join(result.errors))
        return 1
    if result.content and not config.stream:
        ui.assistant_markdown(result.content)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wynxo",
        description="A terminal coding agent for local models via Ollama.",
    )
    parser.add_argument("prompt", nargs="*", help="run one prompt and exit")
    parser.add_argument("-p", "--print", action="store_true",
                        help="non-interactive: answer the prompt and exit")
    parser.add_argument("-e", "--effort", choices=list(ORDER), help="effort level for this run")
    parser.add_argument("-m", "--model", help="model to use")
    parser.add_argument("--endpoint", help="Ollama URL, e.g. http://homelab:11434")
    parser.add_argument("--ctx", type=int, help="context window size (num_ctx)")
    parser.add_argument("-C", "--cwd", help="project directory (default: here)")
    parser.add_argument("--repo", metavar="OWNER/NAME",
                        help="clone a GitHub repository and work in it")
    parser.add_argument("--talker", metavar="MODEL",
                        help="small model that talks while the coder works")
    parser.add_argument("--coder", metavar="MODEL",
                        help="model that does the work when --talker is set")
    parser.add_argument("--speak", action="store_true",
                        help="read answers out loud")
    parser.add_argument("--no-speak", action="store_true",
                        help="stay quiet even if speech is on in the config")
    parser.add_argument("--setup", action="store_true", help="re-run first-time setup")
    parser.add_argument("--doctor", action="store_true",
                        help="check the server and model, and report what will not work")
    parser.add_argument("--fullscreen", action="store_true",
                        help="draw on the alternate screen, like vim; your "
                             "terminal is restored exactly on exit")
    parser.add_argument("--no-fullscreen", action="store_true",
                        help="stay in the normal scrolling terminal")
    parser.add_argument("--no-stream", action="store_true", help="wait for the full response")
    parser.add_argument("--no-thinking", action="store_true", help="hide model reasoning")
    parser.add_argument("--yolo", action="store_true",
                        help="never ask permission (same as --mode yolo)")
    parser.add_argument("--mode", choices=[m.value for m in Mode],
                        help="how much it asks first: plan, manual, auto, yolo")
    parser.add_argument("--scope", choices=[s.value for s in Scope],
                        help="what it may touch: folder, repo, machine")
    parser.add_argument("--version", action="version", version=f"wynxo {__version__}")
    return parser


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    if not workspace.is_dir():
        print(f"No such directory: {workspace}", file=sys.stderr)
        return 1

    ui = UI(show_thinking=not args.no_thinking)

    if args.repo:
        from . import repo as repo_module

        if not repo_module.git_available():
            print("git is not installed, so --repo cannot clone anything.",
                  file=sys.stderr)
            return 1
        target = repo_module.parse(args.repo)
        if target is None:
            print(f"{args.repo!r} does not look like a repository. "
                  "Try owner/name or a GitHub URL.", file=sys.stderr)
            return 1
        with ui.status(f"fetching {target.slug}..."):
            ok, path, message = repo_module.clone_or_update(target)
        if not ok:
            ui.error(message)
            return 1
        ui.success(message)
        workspace = path

    # An endpoint supplied on the command line or in the environment is enough
    # to run: do not drag the user through setup for something they answered.
    endpoint_supplied = bool(
        args.endpoint or os.environ.get("WYNXO_ENDPOINT") or os.environ.get("OLLAMA_HOST")
    )
    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if args.setup or (not is_configured() and not endpoint_supplied):
        if not interactive:
            print(
                "wynxo is not configured yet, and there is no terminal to ask on.\n"
                "Either run `wynxo --setup` interactively, or point it at a server:\n"
                "  wynxo --endpoint homelab:11434 -p \"your prompt\"\n"
                "  WYNXO_ENDPOINT=homelab:11434 wynxo -p \"your prompt\"",
                file=sys.stderr,
            )
            return 1
        try:
            config = await run_wizard(ui)
        except (KeyboardInterrupt, EOFError):
            ui.console.print()
            ui.info("setup cancelled")
            return 1
    else:
        config = load(workspace)

    if args.endpoint:
        url = normalise_url(args.endpoint)
        config.endpoints = [Endpoint(name="cli", url=url)]
        config.active_endpoint = "cli"
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    if args.ctx:
        config.num_ctx = args.ctx
    if args.no_stream:
        config.stream = False
    if args.fullscreen:
        config.fullscreen = True
    if args.no_fullscreen:
        config.fullscreen = False
    if args.no_thinking:
        config.show_thinking = False
    if args.talker:
        config.talker = args.talker
    if args.coder:
        config.coder = args.coder
    if args.speak:
        config.speak = True
    if args.no_speak:
        config.speak = False
    ui.show_thinking = config.show_thinking

    if args.doctor:
        return await run_doctor(config, ui)

    prompt = " ".join(args.prompt).strip()

    # Piped input becomes context for the prompt: `git diff | wynxo -p "review"`.
    piped = read_piped_stdin()
    if piped:
        prompt = (
            f"{prompt}\n\n<piped-input>\n{piped}\n</piped-input>"
            if prompt
            else f"Here is some input:\n\n<piped-input>\n{piped}\n</piped-input>"
        )
        args.print = True  # nothing to be interactive with

    if prompt and args.print:
        once_mode = Mode.parse(args.mode) if args.mode else Mode.YOLO
        once_scope = Scope.parse(args.scope) if args.scope else Scope.FOLDER
        return await run_once(config, workspace, ui, prompt, once_scope, once_mode)

    mode = Mode.parse(args.mode) if args.mode else Mode.MANUAL
    if args.yolo:
        mode = Mode.YOLO
    scope = Scope.parse(args.scope) if args.scope else Scope.FOLDER

    repl = Repl(config, workspace, ui, scope=scope, mode=mode)
    if mode is Mode.YOLO:
        ui.warn("Nothing will ask for approval: the agent writes and runs freely.")
    if scope is Scope.MACHINE:
        ui.warn("Machine scope: no path restriction. The agent can reach anything "
                "your user account can.")

    # Ctrl-C cancels the running turn rather than killing the process.
    loop = asyncio.get_running_loop()
    if sys.platform == "win32":
        # No add_signal_handler on Windows. The handler runs in the main thread,
        # so hop back onto the loop before touching the task.
        signal.signal(
            signal.SIGINT,
            lambda *_: loop.call_soon_threadsafe(repl.interrupt),
        )
    else:
        try:
            loop.add_signal_handler(signal.SIGINT, repl.interrupt)
        except NotImplementedError:
            pass

    # raw=True is load-bearing. patch_stdout routes everything through
    # prompt_toolkit's Vt100_Output.write, which replaces every ESC byte with
    # "?" as an escape-injection guard -- turning every colour code from rich
    # and from the status lines into literal "?[1;32m" garbage on screen.
    # write_raw passes them through, which is what a terminal UI needs.
    # The alternate screen is entered before anything draws and left after
    # everything, so the session is bracketed exactly. Screen() is a no-op
    # when fullscreen is off or the terminal cannot do it, so there is no
    # branch here.
    with fullscreen.Screen(config.fullscreen) as screen:
        repl.screen = screen
        with patch_stdout(raw=True):
            if prompt:
                return await repl.start_with(prompt)
            return await repl.start()


def _write_crash_report(exc: BaseException) -> "Path | None":
    """Put the traceback somewhere it can be read later."""
    import traceback

    try:
        directory = data_dir() / "crashes"
        directory.mkdir(parents=True, exist_ok=True)
        import time as _time

        path = directory / f"{_time.strftime('%Y%m%d-%H%M%S')}.txt"
        path.write_text(
            f"wynxo {__version__}\n"
            f"python {sys.version}\n"
            f"platform {sys.platform}\n\n"
            + "".join(traceback.format_exception(exc)),
            encoding="utf-8")
        return path
    except Exception:
        return None


def main() -> None:
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except BaseException as exc:                       # noqa: BLE001
        # Last resort. Everything inside a session is already guarded, so
        # reaching here means start-up broke or something got past all of
        # it -- and a raw Python traceback is a bug report the person
        # reading it cannot act on. Say what happened in one line, keep the
        # traceback on disk for when it is actually wanted.
        report = _write_crash_report(exc)
        print(f"\nwynxo hit an unexpected error and had to stop.",
              file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        if report is not None:
            print(f"\n  The full details are in {report}", file=sys.stderr)
            print("  That file is worth attaching to a bug report.",
                  file=sys.stderr)
        print("  Nothing you had open was written to; run wynxo again to "
              "carry on.\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
