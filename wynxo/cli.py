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
import time
from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.formatted_text.html import html_escape as escape
from prompt_toolkit.history import FileHistory
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from . import __version__
from . import config as config_module
from . import logo
from . import stt_devices
from .agent import Agent, Callbacks, Interrupted
from .events import ToolEvent
from .task_state import TaskState
from .discovery import Discovery
from .config import Config, Endpoint, data_dir, is_configured, load, normalise_url
from . import livediff
from .doctor import run_doctor
from .effort import ORDER, resolve
from .permissions import Decision
from .provider import ProviderError, check_context, make_client
from .queue import Pending
from .session import Session
from .keys import KeyWatcher
from .journal import Journal, recent as recent_logs
from .memory import Memory
from .pet import Mood, Pet
from .select import (
    HINT, HINT_ASCII, Choice, choose, silence_cpr_warning,
    supported as arrows_supported)
from .scope import Mode, Scope, resolve as resolve_scope
from .status import Status, WARN
from .tools import build_registry
from .tools.appcatalog import ApplicationCatalog
from rich.text import Text

from .ui import (ACCENT, BAD, BAR_ACCENT, FAINT, GOOD, MUTED, ActivityBar,
                 CodeStreamer, ThoughtStreamer, UI, _ansi_of, effort_meter)

# What the activity bar says while each tool runs.
_ACTIVITY = {
    "read_file": "reading", "write_file": "writing file", "edit_file": "editing",
    "list_dir": "listing", "glob": "finding", "grep": "searching",
    "shell": "running", "todo_write": "planning", "launch_application": "launching",
    "run_tests": "testing",
}
_LANGUAGE = {"read_file": "python", "shell": "console"}

# Keys that work *while the agent is running*, not just at the prompt.
LIVE_KEYS = {"ctrl+o": "thinking", "ctrl+t": "detail",
             "ctrl+d": "diff", "ctrl+c": "stop"}
from .platforms import (
    is_dumb_terminal, ollama_server_help as server_help,
    suspicious_workspace)
from .wizard import probe, run_wizard


def live_note(note):
    """(text to show, note to keep) for a transient prompt note.

    A function rather than a method because both strips that can show one --
    the classic prompt's bottom border and the chat layout's footer -- need
    identical expiry, and the second one grew its own copy that never ran.
    """
    if note is None:
        return "", None
    message, until = note
    if time.monotonic() < until:
        return message, note
    return "", None


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
        "kawaii": "soft candy pink with sparkles",
        "midnight": "cool blue",
        "ember": "warm orange",
        "catboy": "soft pastel blue and pink",
        "plain": "your terminal's own 16 colours",
        "minimal": "plain grey, no animation (reduced motion)",
    }.get(name, "")


def _voice_summary(voice: str) -> str:
    return {
        "plain": "direct and human, no support-bot filler",
        "warm": "friendly, still honest about failures",
        "mentor": "explains the reasoning behind decisions",
        "blunt": "the fewest words that say what happened",
        "kawaii": "cheerful and affectionate, same engineering underneath",
        "mommy": "warm, playful, doting -- your goodboy, her mommy (default) -- same engineering underneath",
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
    "/apps": "list the applications discovered on this machine (add refresh to rescan)",
    "/pet": "the companion: on | off | name <x> | voice <x> | show <mood>",
    "/mommy": "mommy-style talking: on | off (no argument toggles)",
    "/animate": "preview the animations: list, or <name> for a scene",
    "/todo": "show the current plan",
    "/queue": "what you typed while it was working: show | run | clear",
    "/dictate": "record one spoken message onto the prompt (Ctrl-R)",
    "/theme": "colour palette: purple | sakura | midnight | ember | plain | minimal",
    "/secrets": "credential protection: on | off | allow <path>",
    "/logo": "the start-up logo: pick one, or off",
    "/speak": "read answers out loud: on | off | test | engine <name>",
    "/talker": "small model that does the talking: <model> | off",
    "/log": "where this session is being recorded",
    "/mode": "plan | manual | auto | yolo -- how much it asks first",
    "/scope": "folder | repo | machine, or a path to work in",
    "/cd": "work in another directory",
    "/repo": "clone a GitHub repo and work in it",
    "/undo": "revert the last file change",
    "/copy": "copy the conversation to the clipboard (/copy last = last answer)",
    "/memory": "show, add to, or forget long-term memory",
    "/thinking": "show or hide the model's reasoning",
    "/plan": "show the current plan",
    "/new": "start a new chat: fresh history, screen and log",
    "/resume": "pick up an earlier conversation where it stopped",
    "/gh": "work on a GitHub repo in the cloud: status | open | ls | cat | edit | branch | pr | close",
    "/commit": "write a commit message from the staged diff, then commit",
    "/review": "ask the model to review the working-tree changes",
    "/diff": "show uncommitted changes (or /diff staged)",
    "/test": "run the project's detected test command",
    "/clear": "start a fresh conversation",
    "/compact": "summarise the conversation to reclaim context",
    "/stats": "tokens, speed, context use",
    "/session": "show current session, model, tools, and context",
    "/doctor": "check the server and model for problems",
    "/yolo": "stop asking permission for this session",
    "/sessions": "list recent sessions",
    "/init": "write a WYNXO.md describing this project",
    "/map": "the project layout the model is given, or rebuild it",
    "/pull": "download a model from Ollama, then switch to it",
    "/quit": "exit",
}


_SUBCOMMAND_VALUES: dict[str, tuple[str, ...]] = {
    "/effort": ("low", "medium", "high", "xhigh", "max", "ultra"),
    "/mode": ("plan", "manual", "auto", "yolo"),
    "/scope": ("folder", "repo", "machine"),
    "/theme": ("purple", "sakura", "kawaii", "midnight", "ember",
                "catboy", "plain", "minimal"),
    "/log": ("tail", "list", "off"),
    "/pet": ("on", "off", "still", "name", "voice"),
    "/mommy": ("on", "off"),
    "/copy": ("last",),
    "/gh": ("status", "login", "open", "ls", "cat", "edit",
             "branch", "pr", "close"),
}
"""Values offered after a slash command's space, e.g. ``/effort h``."""


class CommandCompleter(Completer):
    """Suggests slash commands, their subcommand values, and files after
    an "@".

    Nothing else is completed. A menu opening over ordinary prose -- which
    is what you are typing almost all of the time -- would be in the way
    rather than helpful.
    """

    def __init__(self, workspace_getter=None, model_names_getter=None):
        self._workspace = workspace_getter
        self._model_names = []
        self._model_names_getter = model_names_getter

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

        if not text.startswith("/"):
            return
        if text.startswith("/model "):
            prefix = text.split(None, 1)[1]
            names = self._model_names_getter() if self._model_names_getter else self._model_names
            for model in names:
                if model.startswith(prefix):
                    yield Completion(model, start_position=-len(prefix), display=model,
                                     display_meta="installed model")
            return
        command, _, arg = text.partition(" ")
        if command in _SUBCOMMAND_VALUES and arg:
            prefix = arg
            for value in _SUBCOMMAND_VALUES[command]:
                if value.startswith(prefix):
                    yield Completion(value, start_position=-len(prefix),
                                     display=value)
            return
        if " " in text:
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
        self._thinker: ThoughtStreamer | None = None
        self.bar: ActivityBar | None = None
        self.journal: Journal | None = None
        self.watcher: KeyWatcher | None = None
        """Set while a turn runs. It holds the terminal in cbreak mode, so it
        must be stopped before prompt_toolkit is asked to read a line --
        otherwise both read stdin and keystrokes go to whichever wins."""
        self.thinking_asked_for: "Callable[[], bool] | None" = None
        """Whether the effort level actually asks the model to think. Set by
        Repl.

        Showing reasoning and *having* reasoning are two settings, and only
        one of them is on this keypress. Below "high" the policy sends no
        ``think`` at all, so turning the display on produced a session that
        said "thinking shown" and then never showed a word -- indefinitely,
        with nothing on screen to say why. The toggle has to be able to say
        there is nothing to reveal."""
        self.rearm_interrupt: "Callable[[], None] | None" = None
        """Put the session's SIGINT handler back. Set by Repl.

        Every prompt_toolkit read removes it: the Application installs its
        own for the length of the read and calls
        ``loop.remove_signal_handler(SIGINT)`` in its finally, and there is
        only one handler per signal -- so it removes ours rather than
        restoring it. A permission prompt in the middle of a turn therefore
        left the rest of that turn with no handler at all, and Ctrl-C did
        nothing while the agent went on working. The key watcher is no help
        there: cbreak leaves ISIG set, so the driver turns ^C into a signal
        and the byte never reaches a reader."""
        self._card: livediff.DiffCard | None = None
        """The edit being streamed right now, or the last one finished. Fed
        by on_code with the provider's real fragments. Assigned through the
        property below, which keeps the live region in step -- the card is
        drawn there, and a second place to remember is a place to forget."""
        self.detail_diffs = False
        """Ctrl-D. Whether a finished edit prints its whole diff into the
        transcript instead of one summary line."""
        self.workspace = None
        """Set by Repl, so a card can read the file it is about to replace
        and diff against it."""
        self.boundary = None
        """The same wall the tools use. The card reads independently of the
        tool, so without it a path the tool would refuse to write was still
        read and drawn."""
        self.active_tool = ""
        """The tool running right now, or "". The companion's scene is chosen
        from this and the task state together: "executing" is equally true of
        reading a file and of writing one, and those must not look alike."""
        self.streamer: CodeStreamer | None = None
        self.verbose_tools = False
        self._tool_started = 0.0
        """When the running tool began, for deciding whether anybody is
        waiting on its output."""
        self._held_output: list[str] = []
        """Lines from a command too young to be worth showing yet."""
        """Ctrl-T: show full tool output instead of a one-line summary."""
        self.tokens = 0
        self._coder: CodeStreamer | None = None
        """The file currently being written, while a tool call streams."""
        self._thinking_buffer: list[str] = []
        """Every thought of the whole session, shown or not.

        Capturing never stops. Hiding is a display state, not a decision to
        throw the reasoning away, so the record survives both the toggle and
        the turn boundary -- ``/thinking all`` and a mid-session Ctrl-O can
        both go back over everything the model has thought."""
        self._thinking_unsent: list[str] = []
        """Chunks that arrived while the panel was collapsed and have not
        been fed to a thinker yet. This is the un-shown suffix: opening the
        panel drains it, so a reopen never replays what already printed, and
        no full-buffer re-join is needed to find the boundary -- collapsed
        thinking stays O(1) per chunk instead of O(n) per chunk."""
        self._thinking_turns: list[str] = []
        """Completed turns' thinking, oldest first, so a replay can show them
        with a divider rather than as one undifferentiated wall."""
        self._thinking_total = 0
        """len of the concatenated buffer, kept incrementally so a live panel
        never re-joins the whole scratchpad per token."""
        self._thinking_words = 0
        """Running space count for the collapse note, so the note is O(1)
        per chunk rather than a recount of the whole buffer."""
        self._status_message = ""
        self._status_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()

    @property
    def card(self) -> "livediff.DiffCard | None":
        return self._card

    @card.setter
    def card(self, value: "livediff.DiffCard | None") -> None:
        self._card = value
        if self.bar is not None:
            self.bar.card = value

    # -- live toggles, called from the key watcher thread -------------------

    def thinking_is_off_at_this_effort(self) -> bool:
        """Whether the model is being asked to think at all right now."""
        if self.thinking_asked_for is None:
            return False
        try:
            return not self.thinking_asked_for()
        except Exception:
            return False

    NOTHING_TO_THINK = ("no reasoning at this effort "
                        "\u00b7 /effort high to turn it on")
    """Shown when the display is switched on at a level that sends no
    ``think``. Without it the display is on, nothing ever appears, and that
    reads as a broken feature rather than as a level that does not think."""

    def toggle_thinking(self) -> None:
        self.ui.show_thinking = not self.ui.show_thinking
        nothing_to_show = (self.ui.show_thinking
                           and self.thinking_is_off_at_this_effort())
        # Announced before the panel opens, not after: the note is a line of
        # its own, and printing it once the backlog is streaming drops it
        # into the middle of a sentence.
        #
        # And mid-turn it is not printed at all. This runs from the key
        # watcher while an answer is streaming a word at a time, so anything
        # written to the terminal here lands between two words of a
        # sentence. The live region is where a transient note belongs: the
        # status bar already reflects the state, so the note rides in its
        # detail slot and disappears with it.
        if nothing_to_show:
            self._status_message = self.NOTHING_TO_THINK
        else:
            self._status_message = "Thinking..." if self.ui.show_thinking else ""
        if self.bar is not None:
            self.bar.update(detail=self._status_message)
        else:
            self.ui.info("thinking shown" if self.ui.show_thinking
                         else "thinking hidden")
            if nothing_to_show:
                self.ui.info("this effort level does not ask the model to "
                             "think -- /effort high or above to give it "
                             "reasoning to show")
        if self.ui.show_thinking:
            self._open_thinking()
        else:
            # Collapse what is already on screen's worth of buffer, so hiding
            # takes effect now rather than after the current block finishes.
            self._end_thinking()

    THINKING_TURNS_KEPT = 12
    """How many finished turns of reasoning stay replayable. Enough to look
    back over a session; bounded so a long one cannot grow without limit."""

    def _open_thinking(self, whole_session: bool = False) -> None:
        """Show the reasoning, then let the rest stream in.

        Showing means showing all of it. Opening part-way through prints
        everything that has arrived and the live stream picks up from there,
        so the panel reads the same whether it was opened at the start of the
        turn or in the middle of it.

        ``whole_session`` also replays the turns already finished, under a
        divider each. That is the answer to "show me everything it has
        thought": hiding never discarded any of it, so there is always a full
        record to go back to.

        The common case -- a mid-turn Ctrl-O -- still costs only the
        collapsed backlog, because the chunks already fed to a thinker were
        printed live and are not in ``_thinking_unsent``.
        """
        if self._streaming:
            self._end_stream()
        history = self._thinking_turns if whole_session else []
        # Only the leading space is trimmed. Stripping the tail would glue
        # the backlog to whichever word streams in next.
        backlog = "".join(self._thinking_unsent).lstrip()
        if not backlog and not history:
            return

        if history:
            self.ui.console.print()
            self.ui.console.print(
                Text(f"  everything thought this session "
                     f"({len(history)} earlier turn"
                     f"{'s' if len(history) != 1 else ''})",
                     style=f"bold {MUTED}"))
            for index, thought in enumerate(history, start=1):
                if not thought.strip():
                    continue
                self.ui.console.print(
                    Text(f"  turn {index}", style=FAINT))
                past = ThoughtStreamer(self.ui)
                past.feed(thought.lstrip())
                past.finish()

        if not backlog:
            return
        self.ui.console.print()
        self.ui.console.print(Text("  thinking", style=f"bold {MUTED}"))
        if self._thinker is None:
            self._thinker = ThoughtStreamer(self.ui)
        self._thinker.feed(backlog)
        self._thinking_unsent.clear()

    def _thinking_note(self) -> str:
        """The collapsed form: how much thinking there is, and how to read it.

        Collapsed does not mean invisible, and it does not mean lost. The
        count keeps rising while hidden because the reasoning is still being
        captured -- hiding only stops it being drawn -- so the note doubles
        as the promise that Ctrl-O will still have something to show.
        """
        words = self._thinking_words
        if words < 3:
            return ""
        if self.ui.show_thinking:
            return ""
        return f"{words} words thought  ^O to read"

    def toggle_diff_detail(self) -> None:
        """Ctrl-D. Show the whole diff, or just the summary line.

        Works during a stream and after it: while an edit is live the card
        in the overlay grows to full height, and a finished edit prints its
        diff into the transcript on the spot rather than making the user
        cause another one to see it.
        """
        self.detail_diffs = not self.detail_diffs
        self._note("Diffs: full" if self.detail_diffs else "")
        card = self.card
        if self.detail_diffs and card is not None and not card.live:
            for line in card.render(self.ui.g, self.ui.width - 4,
                                    expanded=True):
                self.ui.console.print(Text(f"  {line}", style=MUTED))

    OUTPUT_AFTER_SECONDS = 1.5
    """How long a command must run before its output goes to the screen.

    Under this it reads as instant and its transcript is noise; over it,
    somebody is watching a progress bar that does not exist yet."""
    HELD_OUTPUT_LINES = 400
    """How much of a young command's output to keep for the flush. A bound,
    so a chatty command cannot grow this without limit before the threshold."""

    def _streaming_output(self) -> bool:
        """Whether a running command's output belongs on the screen yet."""
        if self.verbose_tools:
            return True          # asked for explicitly; show everything
        if not self._tool_started:
            return True          # no start time: never hold on a guess
        return time.monotonic() - self._tool_started >= self.OUTPUT_AFTER_SECONDS

    def toggle_verbose(self) -> None:
        self.verbose_tools = not self.verbose_tools
        self._status_message = "Tool output: full" if self.verbose_tools else ""
        if self.bar is not None:
            self.bar.update(detail=self._status_message)

    def _note(self, message: str) -> None:
        """Update the transient status without adding transcript noise."""
        self._status_message = message
        if self.bar is not None:
            self.bar.update(detail=message)

    def _end_stream(self) -> None:
        """Close whichever transient block is open, so the next starts clean."""
        self._end_thinking()
        if self._streaming:
            if self.streamer is not None:
                self.streamer.finish()
                self.streamer = None
            self._streaming = False

    async def on_thinking(self, text: str) -> None:
        if not text:
            return
        async with self._status_lock:
            self.tokens += 1
            # Kept whether or not it is being shown. Collapsed is a display
            # state, not a decision to throw the reasoning away -- without this
            # buffer, opening the panel part-way through could only ever show
            # what came after the keypress, and the thought you wanted to read
            # was the one that had already gone by.
            self._thinking_buffer.append(text)
            self._thinking_total += len(text)
            self._thinking_words += text.count(" ")
            if self.bar is not None:
                self.bar.update(activity="thinking", tokens=self.tokens,
                                detail=self._thinking_note())

            if not self.ui.show_thinking:
                # Held for when the panel opens. Nothing is fed to a thinker
                # while collapsed, so the reopen can show exactly what was
                # missed without re-printing what already went out.
                self._thinking_unsent.append(text)
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

    async def on_content(self, text: str) -> None:
        if not text:
            return
        async with self._status_lock:
            # Ollama streams roughly one token per chunk, so counting chunks gives
            # a live figure that tracks generation instead of a character estimate
            # that lurches. The exact count arrives with the final chunk.
            self.tokens += 1
            if not self._streaming:
                self._end_stream()
                # Indented to the same column the user's own line, the tool
                # lines and the greeting all sit at. The answer -- the thing
                # the screen exists for -- was the one kind of content
                # starting hard against column zero, so it read as overflow
                # rather than as the reply to the line above it.
                self.streamer = CodeStreamer(self.ui, indent="  ")
                self._streaming = True
            if self.bar is not None:
                self.bar.update(activity="writing", detail="", tokens=self.tokens)
            self.streamer.feed(text)

    async def on_stage(self, name: str, detail: str = "") -> None:
        async with self._status_lock:
            await self._on_stage_locked(name, detail)

    async def _on_stage_locked(self, name: str, detail: str = "") -> None:
        if self.journal is not None:
            self.journal.stage(name, detail)
        self._end_stream()
        # The pet is a visualisation of the agent state: the stage change
        # is the state, and the face/scene follows it. One source of truth.
        pet = getattr(self, "pet", None)
        if pet is not None:
            pet.set_activity(name)
        if self.bar is not None:
            # The live strip owns this. A stage is a state, not an event: it
            # is true for a while and then stops being true, which is what a
            # status region shows and what a transcript line cannot. Printing
            # it as well was tolerable while stages were rare, but there is
            # one before every model request now -- so a five-step coding
            # turn wrote "thinking" into the conversation five times, between
            # the tool results the user actually wanted to read.
            self.bar.update(activity=name, detail=detail)
            self._last_stage = name
            return
        # No live region -- a pipe, -p, a dumb terminal. A printed line is
        # the only way to show it there, so it is printed, but never twice
        # in a row for the same state.
        if name == getattr(self, "_last_stage", ""):
            return
        self._last_stage = name
        suffix = f" [{MUTED}]{detail}[/]" if detail else ""
        self.ui.console.print(f"  [{ACCENT}]{self.ui.g.arrow}[/] [{MUTED}]{name}[/]{suffix}")

    async def on_tool_start(self, name: str, summary: str, event: ToolEvent | None = None) -> None:
        async with self._status_lock:
            await self._on_tool_start_locked(name, summary, event)

    async def _on_tool_start_locked(self, name: str, summary: str, event: ToolEvent | None = None) -> None:
        self.active_tool = name
        self._last_stage = ""      # a tool ends the stage it followed
        self._tool_started = time.monotonic()
        self._held_output.clear()
        self._end_code()
        if livediff.is_edit(name):
            # The card may already exist: the arguments stream arrives before
            # the call has been parsed, so on_code opens one as soon as the
            # first fragment lands and this fills in what only becomes known
            # now -- which tool it was, which file, and the file's contents
            # before the tool replaces them.
            path = ""
            if event is not None:
                path = str(getattr(event, "target", "") or "")
            if not path:
                path = summary.strip().split()[-1] if summary.strip() else ""
            before = (livediff.read_before(self.workspace, path,
                                           self.boundary)
                      if self.workspace is not None else "")
            streamed = self.card.streamed if self.card is not None \
                and self.card.live else ""
            self.card = livediff.DiffCard(tool=name, path=path, before=before,
                                          streamed=streamed)
        elif self.card is not None and self.card.live:
            # A different tool ran while a card was open: whatever was
            # streaming was not an edit after all.
            self.card = None
        if self.journal is not None:
            self.journal.tool(name, {"summary": summary})
        self._end_stream()
        pet = getattr(self, "pet", None)
        if pet is not None:
            pet.set_activity(_ACTIVITY.get(name, name))
        if self.bar is not None:
            self.bar.update(activity=_ACTIVITY.get(name, name), detail=summary)
            if event is not None:
                self.bar.record_tool_event(event)
            if name == "todo_write":
                return       # the pinned plan is the announcement
        self.ui.tool_start(name, summary)

    async def on_tool_result(self, name: str, ok: bool, display: str, output: str, event: ToolEvent | None = None) -> None:
        async with self._status_lock:
            await self._on_tool_result_locked(name, ok, display, output, event)

    async def _on_tool_result_locked(self, name: str, ok: bool, display: str, output: str, event: ToolEvent | None = None) -> None:
        self.active_tool = ""
        if self.card is not None and self.card.tool == name and self.card.live:
            settled = ""
            if ok and not self.card.streamed and self.workspace is not None:
                # Nothing streamed: an atomic provider. The file itself is
                # the honest source for what changed.
                settled = livediff.read_before(self.workspace,
                                               self.card.path, self.boundary)
            self.card.finish(ok=ok, error="" if ok else (output or "")[:200],
                             settled=settled)
            self._commit_card()
            return
        # Normally on_tool_start has already closed whatever was streaming.
        # Not always: an unknown tool, one blocked by the mode, or one the
        # user declined never starts, and the result went out while the
        # sentence before it was still held in the streamer -- so "Let me
        # check that for you." appeared *after* the error it preceded, run
        # into the next turn's first words. In plan mode, where every write
        # is refused, that was every single tool call.
        self._end_code()
        self._end_stream()
        if self.journal is not None:
            self.journal.tool_result(name, ok, output)
        # The pinned plan already shows every step and its state, so a result
        # line per todo_write is the same information a second time -- and it
        # scrolls, which is exactly what pinning the panel was meant to stop.
        if name == "todo_write" and ok and self.bar is not None:
            return
        # Whatever was held back because the command looked too quick to be
        # worth watching. It worked out is the only reason holding it was
        # safe; a failure is exactly when those lines are the whole point,
        # and the result line carries only the first of them.
        if not ok and self._held_output:
            for held in self._held_output:
                self.ui.tool_output(held)
        self._held_output.clear()
        if self.verbose_tools and output.strip():
            # The marker line first, then the whole output under it. It used
            # to be asked for with empty arguments, and tool_result returns
            # early on empty text -- so verbose mode printed the output and
            # no tick or cross at all, and whether the tool had succeeded was
            # something you had to work out from the text.
            self.ui.tool_result(name, ok, display, output)
            # And only when the block says more than the marker already did.
            # A one-line result printed as a line and then again as a
            # syntax-highlighted block is the same sentence twice, which is
            # not what asking for detail meant.
            if len(output.strip().splitlines()) > 1:
                self.ui.code(output[:4000], _LANGUAGE.get(name, "text"))
        else:
            self.ui.tool_result(name, ok, display, output)

    async def on_code(self, text: str) -> None:
        if not text:
            return
        async with self._status_lock:
            await self._on_code_locked(text)

    async def _on_code_locked(self, text: str) -> None:
        """Code arriving inside a tool call, shown as it is written.

        Its own streamer rather than the prose one: this is known to be a
        file's contents, so none of the fence-detection guesswork applies and
        every character can go straight out.
        """
        if self._streaming:
            self._end_stream()
        if self.card is None or not self.card.live:
            # The fragments arrive before the call has been parsed, so there
            # is no tool name or path yet. Opened blank and filled in by
            # on_tool_start; opening it there instead meant the card was
            # created after the stream it existed to catch.
            self.card = livediff.DiffCard(tool="write_file")
        if self.card.live:
            # An edit in flight: the fragments belong to its card, which
            # draws in the live region. Streaming them into the conversation
            # as well would put the whole file on screen twice.
            self.card.feed(text)
            if self.bar is not None:
                self.bar.refresh()
            return
        if self._coder is None:
            self.ui.console.print()
            self.ui.console.print(Text("  writing", style=f"bold {MUTED}"))
            self._coder = CodeStreamer(self.ui, indent="    ",
                                       style=MUTED, code=False, literal=True)
        self._coder.feed(text)

    def _commit_card(self) -> None:
        """Put the finished edit into the transcript.

        One line by default. The live body was drawn in the live region,
        which is a layer rather than a record -- the conversation on screen
        is append-only, so anything written into it while streaming could
        never be taken back.
        """
        card = self.card
        if card is None:
            return
        # Out of the live region first: the summary about to be printed is
        # the committed record of this edit, and a card still drawn above it
        # would be the same edit twice, one of them saying "streaming".
        if self.bar is not None:
            self.bar.card = None
        self.ui.console.print(Text(f"  {card.summary(self.ui.g)}",
                                   style=GOOD if card.state == "done" else BAD))
        if self.detail_diffs:
            for line in card.render(self.ui.g, self.ui.width - 4,
                                    expanded=True):
                self.ui.console.print(Text(f"  {line}", style=MUTED))

    def close_card(self) -> None:
        """Close an edit left streaming, at the end of a turn.

        A cancelled turn never delivers a tool result, so the card stayed
        live and the live region went on saying "streaming..." into the next
        turn -- describing an edit that had already stopped. Deliberately not
        folded into _end_stream(): that runs *during* a turn as well, and
        would close the card moments after on_code opened it.
        """
        if self.card is not None and self.card.live:
            self.card.finish(ok=False, error="interrupted")
            self._commit_card()
        self.active_tool = ""

    def _end_code(self) -> None:
        if self._coder is not None:
            self._coder.finish()
            self._coder = None

    async def on_tool_output(self, name: str, line: str) -> None:
        if not line:
            return
        async with self._status_lock:
            await self._on_tool_output_locked(name, line)

    async def _on_tool_output_locked(self, name: str, line: str) -> None:
        """A line from a command while it is still running.

        Only shell gets this. A build or a test run is the case where
        waiting in silence is worst: the output that explains what went
        wrong arrives long before the exit code does, and if the command
        times out it is the only output there will ever be.
        """
        if name != "shell":
            return
        # Held until somebody is actually waiting. The reason for showing a
        # running command's output is that silence is worst while you wait --
        # but a command that finishes in a moment was never waited on, and
        # dumping its whole transcript into an append-only conversation left
        # eleven rows of pytest preamble behind for a result of "1 passed",
        # twice per coding turn. Below the threshold nothing lands here; the
        # pinned bar below still shows the current line, so it is visible
        # that something is happening.
        #
        # Held rather than dropped, so a slow command still shows its output
        # from the beginning rather than from whenever the clock ran out.
        if not self._streaming_output():
            self._held_output.append(line)
            del self._held_output[:-self.HELD_OUTPUT_LINES]
            if self.bar is not None:
                self.bar.update(detail=line.strip()[:60])
            return
        self._end_stream()
        if self._held_output:
            for held in self._held_output:
                self.ui.tool_output(held)
            self._held_output.clear()
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
            self.ui.todos(rendered)
            return

        self.bar.set_plan(rendered)
        if self.bar.plan_is_complete():
            # Every step ticked is the one moment in a turn worth a face of
            # its own: HAPPY is the reaction to a single good step, this is
            # the end of the work. The pet hangs off the bar rather than off
            # the callbacks, which never owned one.
            pet = self.bar.pet
            if pet is not None and pet.enabled:
                pet.react(Mood.CELEBRATING)
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
            # The classic prompt temporarily releases its reader.
            self._resume_live()

    def _suspend_live(self) -> None:
        """Release the terminal before prompt_toolkit reads a line."""
        if self.watcher is not None:
            self.watcher.stop()
        if self.bar is not None:
            self.bar.stop()

    def _resume_live(self) -> None:
        # Before anything else: the read that just finished took the SIGINT
        # handler with it, and the rest of this turn is exactly when Ctrl-C
        # is wanted.
        if self.rearm_interrupt is not None:
            self.rearm_interrupt()
        if self.bar is not None:
            self.bar.start()
        # The watcher is the only reader of stdin during a turn, and it was
        # stopped by _suspend_live so prompt_toolkit could read the answer.
        # Restart it or the rest of the turn has no type-ahead and no
        # mid-turn keys.
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

        question = "[y] yes  [a] always  [n] no  [q] stop:"
        while True:
            try:
                answer = (await self.prompt_session.prompt_async(
                    HTML(f'<style fg="{ACCENT}">  {question} </style>')
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


class _LazyPromptSession:
    """A PromptSession stand-in built on first use.

    Everything that touches ``prompt_session`` -- the classic loop, the
    free-text input, the permission question -- sees a normal session
    through attribute delegation; the expensive constructor (which opens
    the Windows console and can raise NoConsoleScreenBufferError in Git
    Bash and remote shells) runs only when the classic path actually
    prompts. Chat mode never does, so it never pays the cost.
    """

    def __init__(self, factory: Callable[[], PromptSession]):
        self._factory = factory
        self._session: PromptSession | None = None

    def _get(self) -> PromptSession:
        if self._session is None:
            self._session = self._factory()
        return self._session

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


class Repl:
    def __init__(self, config: Config, workspace: Path, ui: UI,
                 scope: Scope = Scope.FOLDER, mode: Mode = Mode.MANUAL):
        self.config = config
        self.workspace = workspace
        self.ui = ui
        self.client = make_client(config)
        self.policy = resolve(config.effort)
        self.boundary = resolve_scope(workspace, scope)
        self.mode = mode
        self.memory = Memory(workspace)
        self._prompt_note: tuple[str, float] | None = None
        """A transient line for the bottom toolbar: (message, expiry).

        Key-bindings that change state (Ctrl-E effort) must not print into
        the live prompt -- prompt_toolkit would render the input below the
        stray line, wedging it inside the box. They park the change here
        instead, and the toolbar shows it until it expires."""
        self.discovery = Discovery(workspace)
        self.journal = Journal.open(
            self.agent_session_id(), enabled=config.log)
        self.pending = Pending()
        self.pet = Pet(
            name=config.pet_name,
            enabled=config.pet,
            animate=config.animations,
            unicode=ui.g.unicode,
        )
        self.pet.style_name = ("kawaii" if config.voice == "kawaii"
                                else "mommy" if config.voice == "mommy"
                                else "default")
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

        @bindings.add("c-d", filter=Condition(
            lambda: bool(get_app().current_buffer.text)))
        def _(event):
            """Ctrl-D expands or collapses the edit detail -- but only with
            something typed.

            On an empty composer it is left alone, so prompt_toolkit's own
            end-of-file binding runs and Ctrl-D leaves the session, the way
            it does in bash, python and psql. This used to be bound
            unconditionally, which took that away: the one universal way out
            of a terminal program silently toggled a diff instead, and the
            ``except EOFError`` in the prompt loop was unreachable from the
            keyboard.

            It was unconditional for a reason that has since gone. These
            bindings were also handed to the old full-screen layout, whose
            application ran for the whole *turn* -- so there an unfiltered
            Ctrl-D really would have quit mid-edit. The prompt only runs
            when nothing else does, and mid-turn the key watcher binds
            Ctrl-D to the same toggle, so both cases are covered without
            spending the convention.
            """
            self.callbacks.toggle_diff_detail()

        @bindings.add("c-e")
        def _(event):
            """Ctrl-E steps the effort level up; Ctrl-B steps it down."""
            self._shift_effort(1)
            event.app.invalidate()

        @bindings.add("c-b")
        def _(event):
            self._shift_effort(-1)
            event.app.invalidate()

        @bindings.add("c-r")
        def _(event):
            """Ctrl-R records one spoken message for the prompt."""
            self.start_dictation()
            event.app.invalidate()

        self._prompt_bindings = bindings
        # The prompt is built lazily, on its first real use. Its constructor
        # builds a whole prompt_toolkit application, and on Windows that
        # opens the console -- which raises NoConsoleScreenBufferError when
        # TERM is set but no console buffer exists (Git Bash, mintty, a
        # remote shell). A session that only ever runs `-p` never needs one,
        # and must not crash at start-up building one it will not use.
        self.prompt_session: PromptSession | None = _LazyPromptSession(
            self._make_prompt_session)
        self.callbacks = TerminalCallbacks(ui, self.prompt_session)
        self.callbacks.workspace = workspace
        self.callbacks.rearm_interrupt = self._arm_interrupt
        self.callbacks.thinking_asked_for = lambda: bool(self.policy.thinking)
        self.callbacks.boundary = self.boundary
        self.callbacks.journal = self.journal
        self.callbacks.pet = self.pet
        self.project_info = self.discovery.scan()
        self.agent = Agent(self.client, config, self.policy, workspace, self.callbacks,
                           boundary=self.boundary, memory=self.memory)
        launcher = self.agent.tools.get("launch_application")
        self._app_catalog = launcher.catalog if launcher else ApplicationCatalog()
        """The one application scan this session shares: /apps and the
        launch tool must agree on what this machine has installed."""
        self._model_names: list[str] = []
        self.agent.permissions.mode = mode
        self.agent.refresh_system_prompt()
        self._task: asyncio.Task | None = None
        self._dictation_task: asyncio.Task | None = None
        self._interrupt_armed = 0.0
        """When a first Ctrl-C on an idle prompt stops meaning "quit"."""
        self._dictation_draft = ""
        """A transcription waiting on the next prompt line, for review."""
        self.gh = None
        """The lazy GitHubClient; created on the first /gh use."""
        self.gh_ws = None
        """The open cloud workspace: owner, repo, branch, default, tree."""

    def _make_prompt_session(self) -> PromptSession:
        """Build the prompt on demand.

        Constructing a PromptSession eagerly builds its whole application
        (and opens the Windows console), which crashed start-up under Git
        Bash and remote shells. Built here, on the first real prompt.
        """
        history_file = data_dir() / "history"
        session = PromptSession(
            history=FileHistory(str(history_file)),
            completer=CommandCompleter(
                lambda: self.workspace,
                model_names_getter=lambda: self._model_names),
            complete_while_typing=True,
            key_bindings=self._prompt_bindings,
            multiline=False,
            # Nothing is reserved for a completion dropdown, because a
            # dropdown needs rows *below* the input and the composer is
            # seated on the bottom rows of the screen. Reserving them put an
            # empty slab between the caret and the closing border -- the box
            # was three rows of frame around six rows of nothing, and it was
            # there on every prompt whether or not a completion was open.
            #
            # Readline-style completion needs none of it: the candidates are
            # printed above the composer like any other output, and scroll
            # away with the transcript.
            reserve_space_for_menu=0,
            complete_style=CompleteStyle.READLINE_LIKE,
            # The composer takes its rows back when it closes. Without this
            # every accepted turn left its whole frame behind, so the
            # scrollback filled with stranded top borders and half-boxes --
            # the transcript wants one line saying what was asked, which
            # _echo_prompt prints instead.
            erase_when_done=True,
        )
        silence_cpr_warning(session.app)
        return session

    def _pet_mood(self) -> str:
        """The pet's current mood, for the top-right scene."""
        return self.pet.mood.value if self.pet.enabled else ""

    def agent_session_id(self) -> str:
        import uuid

        return uuid.uuid4().hex[:8]

    async def _connect(self) -> bool:
        """Reach the server, report what is loaded, adapt to the model."""
        self._arm_resize()
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
            # The configured server is dead. Before giving up, discover
            # every Ollama that *is* answering -- loopback, this machine's
            # own LAN address, and the network -- and let the user pick.
            picked = await self._offer_endpoint_discovery()
            if picked is None:
                self.ui.console.print(server_help())
                return False
            self._adopt_endpoint(picked)
            try:
                version = await self.client.ping()
            except ProviderError as exc2:
                self.ui.error(str(exc2))
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

        # A settings file that could not be read is the reason the endpoint
        # list, the model and the theme are suddenly back to their defaults.
        # Falling back is right; doing it in silence is not.
        for problem in config_module.LOAD_PROBLEMS:
            note(WARN, "settings", problem)

        if problems:
            print("", file=status.stream)
            for state, message, detail in problems:
                status.line(state, message, detail)
        status.close()

        if self.config.logo and self.config.logo != "off":
            await logo.play(self.ui, self.config.logo, self.config.animations)
        else:
            self.ui.wake(self.pet, self.pet.name)
        self.ui.banner(
            self.config.model,
            f"{self.client.base_url} (ollama {version})",
            self.policy.name,
            str(self.workspace),
        )
        if self.pet.enabled and (greeting := self.pet.remark("greet")):
            line = Text("  ")
            line.append(self.pet.face(advance=False),
                        style=f"bold {self.pet.style()}")
            line.append("  ")
            line.append(f"{self.pet.name} {self.ui.g.dot} {greeting}", style=MUTED)
            self.ui.console.print(line)
            self.ui.console.print()
        return True

    async def _offer_endpoint_discovery(self) -> str | None:
        """The configured server answered nothing: look on this machine and
        the LAN, show every hit, and let the user pick one (or type an
        address). Returns the chosen URL, or None to give up."""
        from .discovery import discover

        with self.ui.status("looking for Ollama on this machine and the LAN..."):
            found = await discover()
        if not found:
            self.ui.warn("nothing answering on this machine or the network.")
            return None
        self.ui.console.print(f"  [{ACCENT}]Found Ollama on:[/]")
        for i, hit in enumerate(found, 1):
            self.ui.console.print(
                f"    [bold]{i}[/]  {hit.url}  "
                f"[{MUTED}]v{hit.version} {self.ui.g.dot} {hit.where}[/]")
        self.ui.console.print()
        answers = {str(i): hit.url for i, hit in enumerate(found, 1)}
        answers["m"] = "type a different address"
        picked = await self._question(
            f"where does Ollama serve? [1-{len(found)} or m]:",
            answers, default="1")
        if not picked:
            return None
        if picked == "m":
            raw = await self._type_in("address:", "")
            if not raw:
                return None
            return normalise_url(raw if "://" in raw else f"http://{raw}")
        return found[int(picked) - 1].url

    def _adopt_endpoint(self, url: str) -> None:
        """Point this session at a discovered server, and remember it."""
        url = normalise_url(url)
        for endpoint in self.config.endpoints:
            if endpoint.url == url:
                self.config.active_endpoint = endpoint.name
                break
        else:
            name = f"discovered-{len(self.config.endpoints) + 1}"
            self.config.endpoints.append(Endpoint(name=name, url=url))
            self.config.active_endpoint = name
        self.config.save()
        self.client = make_client(self.config)

    async def start(self) -> int:
        if not await self._connect():
            return 1
        return await self._loop()

    async def _loop(self) -> int:
        try:
            return await self._prompt_loop()
        finally:
            # She stops when wynxo does. A speech process is a child that
            # outlives its parent, so quitting mid-sentence used to leave the
            # voice talking to an empty terminal. In a finally at the
            # outermost level rather than at the bottom of the prompt loop,
            # so no early return can skip it.
            self.speaker.stop()
            # And so does anything started with shell(background=true). It is
            # in its own session so the whole command can be killed at once,
            # which also means the terminal never hangs it up -- a watcher or
            # a dev server would go on writing into the project long after
            # wynxo was gone.
            from .tools.shell import shutdown_background

            with contextlib.suppress(Exception):
                shutdown_background()
            await self.client.aclose()
            # The pet signs off -- a warm end, but never louder than a plain
            # "bye" when it is disabled.
            farewell = self.pet.remark("bye")
            if farewell:
                self.ui.console.print(f"  {self.pet.name} — {farewell}")
            else:
                self.ui.console.print(f"  [{MUTED}]bye[/]")


    async def _prompt_loop(self) -> int:
        while True:
            # The window may have been resized while this prompt was
            # waiting, and prompt_toolkit owned SIGWINCH for all of it. A
            # draw is about to start, so measure now.
            self.ui.refresh_size()
            try:
                # The draft is dictation text waiting to be submitted. An
                # empty draft must stay an empty string -- `or None` would
                # pass None as prompt_toolkit's default, and Document(None)
                # is a TypeError on every startup.
                draft = self._dictation_draft or ""
                self._dictation_draft = ""
                text = await self.prompt_session.prompt_async(
                    self._prompt_message, bottom_toolbar=self._bottom_toolbar,
                    default=draft)
            except KeyboardInterrupt:
                # prompt_toolkit throws away the line and raises. One press
                # is "forget what I typed" and always was; a second within
                # the window is how every other terminal program is left,
                # and until now there was none -- only /quit or Ctrl-D.
                if self._quit_is_armed():
                    self.ui.console.print()
                    break
                self._interrupt_armed = time.monotonic() + self.QUIT_WINDOW
                self._prompt_note = ("press Ctrl-C again to quit",
                                     self._interrupt_armed)
                continue
            except EOFError:
                break

            # A line got typed, so the earlier Ctrl-C was not the start of a
            # quit. Disarming here rather than on a timer keeps "twice in a
            # row" literal.
            self._interrupt_armed = 0.0
            text = text.strip()
            if not text:
                continue
            # The composer erases itself on accept (erase_when_done), so the
            # transcript gets one compact line saying what was asked instead
            # of a stranded box. See _echo_prompt.
            self._echo_prompt(text)

            # Commands and the queue drain run with the same Ctrl-C handler
            # a turn gets. Without it, prompt_toolkit's own teardown had
            # already put SIGINT back to Python's default, so Ctrl-C during
            # a slow command (/model probing a server, /gh reaching the API)
            # raised KeyboardInterrupt straight out of the loop and ended
            # the session, conversation and all.
            self._arm_interrupt()
            if text.startswith("/"):
                if await self._guarded(self.command(text)) is False:
                    break
                continue

            # Only a turn that finished lets the queue run. An interrupt or
            # an error returns anything but True, and then whatever was
            # typed during the turn stays queued rather than starting the
            # moment you pressed stop -- the status strip says how much is
            # waiting, and Enter or /queue decides what happens to it.
            finished = await self._guarded(self.turn(text)) is True
            if not finished:
                self._report_held_queue()
                continue
            if await self._guarded(self._drain_queue()) is False:
                break

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
        look broken again. A bare KeyboardInterrupt is Ctrl-C too, but with
        nothing holding it -- it is reported here rather than re-raised,
        because the alternative was the process exiting.
        """
        import traceback

        try:
            return await coro
        except (Interrupted, asyncio.CancelledError):
            raise
        except KeyboardInterrupt:
            # A SIGINT that landed somewhere with no task to cancel -- most
            # often during a command. It means "stop this", never "throw the
            # session away", and as a BaseException it walked straight past
            # the handler below on its way out of the process.
            self.callbacks._end_stream()
            self.ui.console.print()
            self.ui.warn("Interrupted. The conversation is intact; "
                         "ask me something else.")
            return None
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

    async def turn(self, text: str) -> bool:
        """Run one request, with a live status bar and mid-flight keybinds.

        Returns False if it was interrupted, so a queue drain can stop
        rather than launching the next thing the moment you press Ctrl-C.
        """
        async with self.callbacks._turn_lock:
            cb = self.callbacks
            # The previous turn's reasoning is retired into the history
            # rather than dropped: showing thinking is meant to show all of
            # it, and a turn boundary is a divider in that record, not the
            # end of it. Only the live buffers reset.
            if cb._thinking_buffer:
                cb._thinking_turns.append("".join(cb._thinking_buffer))
                del cb._thinking_turns[:-cb.THINKING_TURNS_KEPT]
            cb._thinking_buffer.clear()
            cb._thinking_unsent.clear()
            cb._thinking_total = 0
            cb._thinking_words = 0
            return await self._turn_locked(text)

    async def _turn_locked(self, text: str) -> bool:
        self.journal.user(text)
        text = self._expand_mentions(text)
        self.callbacks.tokens = 0
        # Cleared per turn: opening the panel should show this answer's
        # reasoning, not everything the model has thought this session.

        # One hint, not four. The whole set -- ^O thinking, ^T detail, ^D
        # diff, ^C stop -- is forty-four cells of key names competing with
        # the answer to "what is it doing" on every frame, and it is already
        # spelled out on the prompt's own border where there is room for it.
        # Mid-turn there is one binding somebody actually reaches for.
        bar = ActivityBar(self.ui, self.policy.name, "^C stop",
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
                # Advertised in LIVE_KEYS and drawn into the activity bar as
                # "^D diff", but never bound: mid-turn it fell through to
                # type-ahead, which drops it for not being printable. The
                # prompt has had the same binding all along.
                "ctrl+d": self.callbacks.toggle_diff_detail,
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
        interrupted = False
        self._arm_interrupt()
        # The talker answers first: a 1B model is quick enough that the
        # acknowledgement lands before the coder has produced a token.
        if self.talker is not None:
            await self._talk(await self.talker.opening(text))
        self._task = asyncio.ensure_future(self.agent.run(text))
        bar.start()
        # Only one thing may read stdin. prompt_toolkit is not reading during
        # a turn -- the prompt has returned -- so the watcher can hold the
        # terminal in cbreak mode and read it directly for the length of the
        # turn, which is what makes mid-turn keys and type-ahead possible.
        watcher.start()
        try:
            result = await self._task
        except (asyncio.CancelledError, Interrupted):
            # Reported *after* the teardown below, never here. Printing from
            # this block put "Interrupted. The conversation is intact" on
            # screen while the live region was still up -- so for that frame
            # the card above it said "streaming" and the strip said the tool
            # was running, underneath a line saying the work had stopped. The
            # rule is one way round: stop showing the work, then say it
            # stopped.
            interrupted = True
        finally:
            # Order matters: the terminal must be restored before anything
            # tries to read from it again, and any in-progress line has to be
            # flushed to the real scrollback before the bar stops -- it is a
            # transient Live, which erases its render area on stop, taking an
            # unflushed line with it.
            watcher.stop()
            self.callbacks.close_card()
            self.callbacks._end_stream()
            bar.stop()
            self.callbacks.bar = None
            self.ui.bar = None
            self.callbacks.watcher = None
            self._task = None
            # Stages are transient. Never leave a terminal-level status or
            # pet activity behind after the live bar is disposed.
            self.callbacks._status_message = ""
            self.pet.react(Mood.IDLE)

        if interrupted:
            self.ui.console.print()
            self.ui.warn("Interrupted. The conversation is intact; "
                         "ask me something else.")
            return False

        if result.errors:
            self.pet.react(Mood.SAD)
            for message in result.errors:
                self.journal.error(message)
            self.ui.error("\n".join(result.errors))
            return False
        self.journal.assistant(result.content, tokens=self.callbacks.tokens,
                               seconds=result.elapsed)
        self.pet.react(Mood.HAPPY if not result.interrupted else Mood.IDLE)

        if result.content and not self.config.stream:
            self.ui.assistant_markdown(result.content)
        elif result.content:
            self.ui.console.print()

        # The completion report is built from recorded task state -- changed
        # files, checks that ran, failures that remain -- never from the
        # model's prose. A "fixed" claim cannot outrun the evidence. Pure
        # conversation has no evidence and gets no report.
        if not result.interrupted:
            report = self.agent.task_state.completion_report()
            if report:
                # At the same margin as everything else in the conversation.
                # It was the last block still starting hard against column
                # zero, which reads as output that escaped the transcript
                # rather than as the turn's own summing-up.
                self.ui.console.print()
                self.ui.console.print(
                    Text("\n".join(f"  {line}" for line in
                                    report.splitlines()), style="dim"))

        # No stats line here: the pinned bar under the input already shows
        # tokens, rate and context, and printing them again above the next
        # prompt was the same numbers twice, scrolling away from where you
        # are actually looking.
        self._last_elapsed = result.elapsed
        if result.compacted:
            self.ui.info("context was compacted during this turn")

        await self._review_changes(review_mark)
        await self._narrate(text, result)
        return not result.interrupted

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
        answer = await self._question(
            "[k] keep  [r] revert all  [s] step through:",
            {"k": "keep", "r": "revert", "s": "step",
             # The two words a hand types without reading the options.
             "y": "yes", "n": "no"},
            default="k")

        if answer in ("", "k", "y"):
            self.ui.success(f"kept {len(diffs)} file(s)")
            return
        if answer in ("r", "n"):
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
            answer = await self._question(
                f"{name}  [k] keep  [r] revert:",
                {"k": "keep", "r": "revert", "n": "no"}, default="k")
            if answer in ("r", "n"):
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
        expanded, problems = expand(text, self.workspace, self.boundary,
                                    self.agent.shield)
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
            # Off the event loop. speakable() runs a stack of regexes over
            # the whole answer -- seconds on a long one -- and say() also
            # starts the synthesiser process; doing either here froze the
            # chat UI for exactly as long.
            await self.speaker.say_async(result.content)

    async def _talk(self, line: str) -> None:
        """Show one line from the talker, and say it out loud."""
        if not line:
            return
        self.ui.console.print()
        self.ui.console.print(
            Text("  " + self.pet.face(advance=False) + "  ",
                 style=f"bold {self.pet.style()}") + Text(line, style=ACCENT))
        await self.speaker.say_async(line)

    def _prompt_message(self) -> HTML:
        """The composer's top edge and its caret, as one prompt_toolkit
        message.

        The top border used to be printed separately with console.print while
        prompt_toolkit drew the closing toolbar, and the two could not stay
        together. prompt_toolkit seats its toolbar on the last row by
        *emitting newlines* until it gets there -- so whatever gap the manual
        cursor arithmetic left behind became a void of blank rows between the
        top border and the rest of the box, with the border stranded halfway
        up the screen. Making the border part of the message puts the whole
        box inside prompt_toolkit's own render, so it is seated as one piece
        and there is no gap to leave.

        Re-evaluated on every redraw, so a mid-prompt Ctrl-E shows up at once.
        """
        if is_dumb_terminal():
            return HTML('<b>&gt;</b> ')
        g = self.ui.g
        width = max(24, self.ui.width)
        top = escape(g.tl + g.hbar * (width - 2) + g.tr)
        return HTML(
            '<style fg="%s">%s</style>\n'
            '<style fg="%s">%s</style> <b><style fg="%s">&gt;</style></b> '
            % (ACCENT, top, ACCENT, escape(g.vbar), ACCENT))

    def _echo_prompt(self, text: str) -> None:
        """Put what was asked into the transcript, compactly.

        The composer erases itself when it closes, which is what keeps a
        stranded frame from piling up in the scrollback on every turn. The
        cost is that the question would vanish with it, so it is reprinted
        here as a single line -- the shape a transcript wants anyway.
        """
        if is_dumb_terminal():
            return
        body = Text()
        body.append("  > ", style=f"bold {ACCENT}")
        first, *rest = text.splitlines() or [""]
        body.append(first, style="bold")
        for line in rest:
            body.append("\n    " + line, style="bold")
        self.ui.console.print(body)

    def _bottom_toolbar(self):
        """The closing edge of the input box, with the status set into it.

        One line rather than two: a multi-line toolbar does not render on
        terminals that cannot answer a cursor-position request, and the border
        is the natural place for the status anyway.

        Frame, status and hint are all palette colours, so the whole box
        changes with /theme -- the top edge was already accent; making the
        bottom and the caret the same hue turned a two-tone frame (violet
        top, cyan bottom) into one object.
        """
        from rich.cells import cell_len

        g = self.ui.g
        width = max(30, self.ui.width)
        left = self._status_line()

        # "<bl><hbar> " + status + " " + fill + " " + hint + " <hbar><br>"
        def total(status: str, tail: str, fill: int) -> int:
            head = 3 + cell_len(status) + 1 + fill
            return head + (1 + cell_len(tail) + 3 if tail else 1)

        # A transient note (Ctrl-E effort changes) replaces the hints for a
        # moment; the hints return once it expires.
        note, self._prompt_note = live_note(self._prompt_note)

        # The hints that fit, most useful first. ^C stop must survive the
        # longest, so it is trimmed last rather than with the rest.
        base = ["^O think", "^T detail", "^D diff", "^E effort", "^R talk"]
        if note:
            base = [note]
        hint = ""
        for keep in range(len(base), -1, -1):
            candidate = "  ".join(base[:keep] + ["^C stop"])
            if total(left, candidate, 1) <= width:
                hint = candidate
                break
        if total(left, hint, 1) > width:
            left = left[: max(0, width - 10)]
        fill = max(1, width - total(left, hint, 0))

        frame = _ansi_of(ACCENT)
        status_style = _ansi_of(MUTED)
        hint_style = _ansi_of(FAINT)
        reset = "\x1b[0m"
        line = f"{frame}{g.bl}{g.hbar}{reset} {status_style}{left}{reset} " \
            f"{frame}" + g.hbar * fill
        if hint:
            line += f"{reset} {hint_style}{hint}{reset} {frame}{g.hbar}"
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
        # What the agent is doing, first and in front of the numbers.
        bar = getattr(getattr(self, "callbacks", None), "bar", None)
        if bar is not None and (activity := getattr(bar, "activity", "")):
            phase = f"{self.ui.g.gear} {activity}"
            if detail := getattr(bar, "detail", ""):
                phase += f" {detail}"
            pieces.append(phase)
        # Before the numbers: something you typed and have not seen run
        # outranks a token count. Without this a queue held by an interrupt
        # was invisible at the prompt and fired on the next thing typed.
        if waiting := len(self.pending):
            pieces.append(f"\u203a {waiting} queued")
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

        A Ctrl-C stops the drain and leaves the rest queued. Pressing stop
        and having the next thing start immediately is the opposite of what
        the key means -- you interrupt because you want to look at
        something, and the queue firing on regardless takes that away. What
        is left is reported, and stays visible in the status strip until it
        runs or is dropped.
        """
        while (queued := self.pending.take()) is not None:
            self.ui.console.print()
            self.ui.console.print(
                Text("  \u203a ", style=f"bold {ACCENT}") + Text(queued))
            if queued.startswith("/"):
                if await self.command(queued) is False:
                    return False
                continue
            if await self.turn(queued) is False:
                self._report_held_queue()
                return True
        return True

    def _report_held_queue(self) -> None:
        """Say what an interrupt left waiting, and how to deal with it."""
        waiting = len(self.pending)
        if not waiting:
            return
        what = "message" if waiting == 1 else "messages"
        self.ui.info(f"{waiting} queued {what} still waiting "
                     f"{self.ui.g.dot} /queue run, or /queue clear")

    def _shift_effort(self, delta: int) -> None:
        """Step the effort level without typing a command.

        The change is shown in the toolbar, not printed: this runs from a
        key binding while the prompt is on screen, and a printed line would
        wedge itself between the box edge and the input.
        """
        policy = self.policy.bump(delta)
        if policy.name == self.policy.name:
            return
        self.config.effort = policy.name
        self.agent.set_effort(policy)
        self.policy = self.agent.policy
        self.pet.set_pace(self.policy.name)
        self._prompt_note = (f"effort: {self.policy.name} -- "
                             f"{self.policy.headline}", time.monotonic() + 3)

    def _arm_resize(self) -> None:
        """Hear a resize immediately, while a turn is running.

        This is a nudge, not the mechanism. prompt_toolkit's Application
        takes SIGWINCH for the length of every read it does -- it saves our
        handler and restores it afterwards, which is correct of it, but for
        the whole time somebody is sitting at the prompt the signal is its
        and ``ui.width`` heard nothing. Resizing the window and *then*
        typing is the ordinary case, and it left every wrap in the session
        computed against the width the terminal had at launch: on a window
        made narrower, every streamed line ran off the edge and wrapped
        twice.

        So the honest re-measure happens at the two places a draw begins --
        once per prompt in ``_prompt_loop``, and once per repaint in
        ``ActivityBar._render`` -- and between them they cover the whole
        session. This handler only makes it instant rather than up to a
        repaint late. Windows has no SIGWINCH and does not need one for the
        same reason.
        """
        if sys.platform == "win32":
            return
        with contextlib.suppress(NotImplementedError, RuntimeError):
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGWINCH, self.ui.refresh_size)

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

    QUIT_WINDOW = 2.0
    """How long a first Ctrl-C on an idle prompt stays armed. Long enough to
    be a deliberate second press, short enough that a Ctrl-C now and another
    a minute later is not a quit."""

    def _quit_is_armed(self) -> bool:
        """Whether a previous Ctrl-C is still close enough to mean quit."""
        armed = getattr(self, "_interrupt_armed", 0.0)
        return bool(armed) and time.monotonic() < armed

    # -- speech to text ------------------------------------------------------

    def start_dictation(self) -> None:
        """Ctrl-R or /dictate: microphone -> transcription -> next prompt.

        The transcript is put on the next prompt line for review, never
        submitted -- what the microphone heard is a draft, and sending it is
        the user's decision. Safe to press while a dictation is already
        running: the second press cancels the first.
        """
        if self._dictation_task is not None and not self._dictation_task.done():
            self._dictation_task.cancel()
            return
        if not self.config.stt_enabled:
            self.ui.error("speech input is disabled: set stt_enabled=true in "
                          "config (or use /config)")
            return
        self._dictation_task = asyncio.ensure_future(self._dictate())

    async def _dictate(self) -> None:
        session, hint = stt_devices.create_session(
            stt_devices.SpeechConfig(
                language=self.config.stt_language,
                device=self.config.stt_device or None,
                silence_timeout=self.config.stt_silence_timeout,
                max_duration=self.config.stt_max_duration,
                transcription_timeout=self.config.stt_transcription_timeout,
            ),
            on_state=self._on_speech_state,
            backend=self.config.stt_backend,
        )
        if session is None:
            self.ui.error(hint)
            return
        try:
            text = await session.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.ui.error(f"speech input failed: {exc}")
            return
        if session.state is stt_devices.SpeechState.CANCELLED:
            return
        if text:
            # A draft on the next prompt line, for review. It is never sent
            # on its own: the prompt shows it and Enter submits, edits first
            # if the microphone got something wrong.
            self._dictation_draft = text.strip()
            self.ui.info(f"heard: {self._dictation_draft}")
            self.ui.info("review it at the prompt, then Enter to send")
        elif session.state is stt_devices.SpeechState.ERROR:
            self.ui.error("speech input failed; nothing was heard")

    def _on_speech_state(self, snapshot) -> None:
        """Mirror the speech state machine into the transcript."""
        mic = "🎙" if self.ui.g.unicode else "o"
        state = snapshot.state.value
        labels = {
            "listening": f"{mic} Listening... (Ctrl-R again to stop)",
            "transcribing": f"{self.ui.g.gear} Transcribing...",
        }
        if message := labels.get(state):
            self.ui.info(message)

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
                 ("Ctrl-D", "expand or collapse the edit diff (during or after)"),
                 ("Ctrl-R", "speak a message: record, transcribe, review in the composer"),
                 ("Ctrl-E", "step effort up"),
                 ("Ctrl-B", "step effort down"),
                 ("Ctrl-C", "interrupt the current turn, keep the conversation"),
                 ("Alt-Enter", "newline instead of submitting"),
                 ("Up / Down", "history"),
                 ("Mouse wheel", "scroll back -- your terminal's own scrollback"),
                 ("Drag", "select text; copy the way you always do"),
                 ("/copy", "the whole conversation, or /copy last, to the clipboard")],
                title="keys",
            )
            return True

        if name == "/effort":
            return await self.cmd_effort(args)
        if name == "/model":
            return await self.cmd_model(args)
        if name == "/pull":
            return await self.cmd_pull(args)
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

        if name == "/apps":
            return self.cmd_apps(args)

        if name == "/todo":
            return await self.cmd_todo(args)

        if name == "/queue":
            return await self.cmd_queue(args)

        if name == "/dictate":
            self.start_dictation()
            return True

        if name == "/thinking":
            return await self.cmd_thinking(args)

        if name == "/plan":
            todo = self.agent.tools.get("todo_write")
            rendered = todo.render() if todo and hasattr(todo, "render") else ""
            self.ui.todos(rendered) if rendered else self.ui.info("no plan yet")
            return True

        if name == "/clear":
            self.agent.session = Session(workspace=self.workspace)
            self._leave_conversation()
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

        if name in ("/status", "/session"):
            return self.cmd_session()

        if name == "/stats":
            return self.cmd_stats()

        if name == "/doctor":
            from .doctor import Doctor
            await Doctor(self.client, self.config, self.ui,
                         workspace=self.workspace).run()
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
        if name == "/gh":
            return await self.cmd_gh(args)
        if name == "/copy":
            return self.cmd_copy(args)
        if name == "/memory":
            return self.cmd_memory(args)

        if name == "/pet":
            return await self.cmd_pet(args)

        if name == "/mommy":
            return self.cmd_mommy(args)

        if name == "/animate":
            return await self.cmd_animate(args)

        if name == "/theme":
            return await self.cmd_theme(args)

        if name == "/secrets":
            return await self.cmd_secrets(args)

        if name == "/logo":
            return await self.cmd_logo(args)

        if name == "/speak":
            return await self.cmd_speak(args)

        if name == "/talker":
            return await self.cmd_talker(args)

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

        if name == "/review":
            return await self.cmd_review(args)

        if name == "/diff":
            return self.cmd_diff(args)

        if name == "/test":
            return await self.cmd_test(args)

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

    def cmd_session(self) -> bool:
        session = self.agent.session
        limit = min(self.policy.max_iterations, self.config.max_tool_iterations)
        self.ui.info(
            f"state {getattr(getattr(self.agent, 'task_state', None), 'state', TaskState.IDLE).value}  "
            f"project {self.project_info.summary() if hasattr(self, 'project_info') else 'unknown project'}\n"
            f"session {session.session_id}  model {self.config.model}  "
            f"workspace {self.ui.shorten_path(str(self.workspace))}\n"
            f"tools {len(self.agent.tools)}  iteration limit {limit}  "
            f"context {session.token_estimate()}/{self.config.num_ctx} tokens  "
            f"requests {session.usage.requests}  tool calls {session.usage.tool_calls}"
        )
        return True

    def cmd_diff(self, args: list[str]) -> bool:
        """Show a read-only Git diff without involving the model."""
        from .repo import run_git

        if args and args != ["staged"]:
            self.ui.warn("usage: /diff [staged]")
            return True
        ok, _ = run_git(["rev-parse", "--git-dir"], cwd=self.workspace, timeout=10)
        if not ok:
            self.ui.warn("not a git repository")
            return True
        command = ["diff", "--staged"] if args else ["diff"]
        ok, diff = run_git(command, cwd=self.workspace, timeout=60)
        if not ok:
            self.ui.warn(f"could not read the diff: {_first(diff)}")
            return True
        if not diff.strip():
            self.ui.info("no " + ("staged " if args else "unstaged ") + "changes")
            return True
        files = []
        additions = deletions = 0
        for line in diff.splitlines():
            if line.startswith("diff --git "):
                files.append(line.rsplit(" b/", 1)[-1])
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        self.ui.info(f"{len(files)} file(s) changed  +{additions} -{deletions}")
        if files:
            self.ui.table(["file"], [(f,) for f in files[:100]], title="changed files")
        self.ui.diff(diff[:100_000])
        if len(diff) > 100_000:
            self.ui.info("diff truncated visually; use git diff for the complete patch")
        return True

    async def cmd_review(self, args: list[str]) -> bool:
        """Ask the model to review the current changes. Read-only.

        Nothing is changed or committed; this is the review pass that happens
        before a commit decides it is done. Diff only, so a clean tree is an
        honest "nothing to review".
        """
        from .prompts import REVIEW_PROMPT
        from .repo import run_git

        target = args[0].lower() if args else "all"
        if target not in ("staged", "unstaged", "all"):
            self.ui.warn("usage: /review [staged | unstaged]")
            return True
        ok, _ = run_git(["rev-parse", "--git-dir"], cwd=self.workspace, timeout=10)
        if not ok:
            self.ui.warn("not a git repository")
            return True

        if target == "staged":
            git_commands = [["diff", "--staged"]]
        elif target == "unstaged":
            git_commands = [["diff"]]
        else:
            git_commands = [["diff"], ["diff", "--staged"]]
        diffs = []
        for git_command in git_commands:
            ok, diff = run_git(git_command, cwd=self.workspace, timeout=60)
            if not ok:
                self.ui.warn(f"could not read the diff: {_first(diff)}")
                return True
            if diff.strip():
                diffs.append(diff)
        if not diffs:
            self.ui.info("no changes to review")
            return True
        diff = "\n".join(diffs)

        files = additions = deletions = 0
        for line in diff.splitlines():
            if line.startswith("diff --git "):
                files += 1
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        body = diff if len(diff) <= 24_000 else\
            diff[:24_000] + "\n\n... [diff truncated]"
        self.ui.info(f"reviewing {files} file(s), +{additions} -{deletions}...")
        with self.ui.status("reviewing the changes..."):
            turn = await self.agent._call_model(
                messages=[{"role": "user",
                           "content": f"{REVIEW_PROMPT}\n\n```diff\n{body}\n```"}],
                use_tools=False, stream_content=False)
        review = turn.content.strip()
        if not review:
            self.ui.warn("the model did not produce a review")
            return True
        self.ui.console.print()
        self.ui.console.print(Text("review", style=f"bold {ACCENT}"))
        self.ui.console.print(Text(review))
        return True

    async def cmd_test(self, args: list[str]) -> bool:
        """Run only the test command the project itself makes unambiguous."""
        if args:
            self.ui.warn("usage: /test")
            return True
        from . import testing

        runner = testing.detect(self.workspace)
        if runner is None:
            self.ui.warn("could not identify this project's test command")
            return True
        shell = self.agent.tools.get("shell")
        if shell is None:
            self.ui.warn("the shell tool is disabled; enable it to run tests")
            return True
        await self.callbacks.on_stage("testing", runner.command)
        result = await shell.invoke(
            {"command": runner.command, "timeout": testing.DEFAULT_TIMEOUT},
            timeout=testing.DEFAULT_TIMEOUT + 30,
        )
        if result.ok:
            self.ui.success(f"tests passed: {runner.command}")
        else:
            self.ui.error(f"tests failed: {runner.command}")
        if result.output.strip():
            self.ui.code(result.output, language="text")
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

        answer = await self._question(
            "[y] commit  [e] edit  [n] no:",
            {"y": "commit", "e": "edit", "n": "no"}, default="y")

        if answer == "e":
            edited = await self._type_in("message:",
                                         default=message.splitlines()[0])
            if not edited:
                self.ui.info("cancelled")
                return True
            message = edited
        elif answer != "y":
            self.ui.info("not committed")
            return True

        ok, output = run_git(["commit", "-m", message], cwd=self.workspace,
                             timeout=60)
        if not ok:
            self.ui.warn(f"commit failed: {_first(output)}")
            return True
        self.ui.success(_first(output) or "committed")
        if proud := self.pet.remark("proud"):
            self.ui.console.print(f"  {self.pet.name} — {proud}")
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
        chosen = await self._pick("resume which conversation?", options,
                                  options[0].value if options else "")
        if chosen is not NO_PICKER:
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
        self._leave_conversation()
        self.agent.refresh_system_prompt()
        self._last_elapsed = 0.0

        self.ui.success(f"resumed {session_id[:8]} -- "
                        f"{len(restored.messages)} messages")
        self.ui.info("undo history is not restored; it belonged to that run")
        return True

    def _leave_conversation(self) -> None:
        """Drop everything that belonged to the conversation being left.

        Three commands replace the conversation -- /clear, /new and /resume
        -- and each kept its own idea of what that meant. Only /clear reset
        the task state, and none of them cleared the checklist, so work from
        an abandoned chat followed you into the next one: a recovery block
        citing failures from a different task, a completion report claiming
        files changed in another conversation, the repeat detector treating
        a first action as a repetition, a compaction summary told that
        someone else's steps were "still outstanding", and the plan panel
        sitting in the corner insisting on a checklist nobody was working
        on.

        The conversation's *history* is the caller's business -- restored,
        replaced or emptied depending on which command it is. This is only
        the state that has no meaning outside the chat it came from.
        """
        self.agent.task_state.reset()
        todo = self.agent.tools.get("todo_write")
        if todo is not None and hasattr(todo, "items"):
            todo.items = []

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
        self._leave_conversation()
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
        self.config.effort = self.policy.name
        self.config.save()
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
        from .ui import celebrate, surge

        if current == previous or not self.config.animations:
            return
        step_up = ORDER.index(current) - ORDER.index(previous)
        if step_up <= 0:
            return          # celebrating the way down would be the wrong mood

        level = ORDER.index(current)
        label = {"max": "MAX EFFORT", "ultra": "ULTRA"}.get(
            current, current.upper())

        if not self.ui.live_ok:
            # No live bar to animate into (a pipe, non-interactive): the band
            # is drawn once instead.
            celebrate(self.ui, label, level, step_up)
            return

        # The top two still get the moving version, which is worth the extra
        # half second precisely because it does not happen often.
        if current in ("max", "ultra"):
            await surge(self.ui, label, ACCENT if current == "ultra" else BAR_ACCENT)
        else:
            celebrate(self.ui, label, level, step_up)

    async def cmd_pull(self, args: list[str]) -> bool:
        """Download a model from Ollama, then make it the active one.

        `ollama pull` is a separate step the wizard and /model used to send
        people off to run by hand; the provider already streams progress, so
        the command only needs to surface it and hand over to the switcher.
        """
        if not args:
            self.ui.info("usage: /pull <model>, e.g. /pull qwen3-coder:30b")
            return True
        target = args[0]
        if target not in (m.name for m in await self.client.list_models()):
            self.ui.info(f"pulling {target}...")
            last = ""
            try:
                async for line in self.client.pull(target):
                    line = line.strip()
                    if line and line != last and not line.endswith("success"):
                        last = line
                        self.ui.info(line)
            except ProviderError as exc:
                self.ui.error(str(exc))
                return True
        self._model_names = [m.name for m in await self.client.list_models()]
        return await self._switch_model(target, self._model_names)

    async def cmd_model(self, args: list[str]) -> bool:
        """Switch model. With no argument this is the same capability-aware
        picker the first-run wizard uses, rather than a second, dumber list."""
        from .provider import inspect_all
        from .wizard import _model_choice, _print_model_rows

        try:
            with self.ui.status("asking the server what it has..."):
                models = await self.client.list_models()
            self._model_names = [model.name for model in models]
        except ProviderError as exc:
            self.ui.error(str(exc))
            return True
        self._model_names = [model.name for model in models]
        if not models:
            self.ui.warn("that server has no models installed")
            return True

        if args:
            return await self._switch_model(args[0], [m.name for m in models])

        with self.ui.status(f"checking what {len(models)} model(s) can do..."):
            models = await inspect_all(self.client, models)
        models.sort(key=lambda m: (not m.supports_tools, m.name))

        chosen = await self._pick("model", [_model_choice(m) for m in models],
                                  self.config.model)
        if chosen is not NO_PICKER:
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
                    self.ui.info(f"pull it first:  /pull {target}")
                return True

        self.config.model = target
        self.agent.config.model = target
        self.config.save()
        await self.agent.detect_capabilities()
        self.policy = self.agent.policy
        self.ui.success(f"model: {target}")
        self.journal.note("model switched", model=target)
        if warning := await check_context(self.client, self.config):
            self.ui.warn(warning)
        return True

    _ENDPOINT_ADD = "\x00add"
    _ENDPOINT_TEST = "\x00test"
    """Sentinels for the two actions in the server picker. Not the plain
    words: a server can be named "add", and choosing it must select it."""

    async def cmd_endpoint(self, args: list[str]) -> bool:
        action = args[0].lower() if args else "list"

        if action == "list" and not args:
            # Bare /endpoint: pick which server to talk to. The actions live
            # in the same list as the servers, so this opens a picker even
            # with a single server configured -- which is the usual case, and
            # was the one that used to get a table and a line of syntax to
            # copy. Adding a second server is exactly what someone with one
            # server is here to do.
            #
            # Sentinel values rather than the words: a server may legitimately
            # be named "add" or "test", and picking it must not run a command.
            options = [
                Choice(value=endpoint.name, label=endpoint.name,
                       badge=("current" if endpoint.name ==
                              self.config.active_endpoint else ""),
                       hint=endpoint.url)
                for endpoint in self.config.endpoints
            ]
            options.append(Choice(value=self._ENDPOINT_ADD, label="add a server",
                                  hint="another machine running Ollama"))
            options.append(Choice(value=self._ENDPOINT_TEST, label="test all",
                                  hint="check which of them answer"))

            picked = await self._pick("ollama server", options,
                                      self.config.active_endpoint)
            if picked is None:
                return True
            if picked is not NO_PICKER:
                if picked == self._ENDPOINT_TEST:
                    return await self.cmd_endpoint(["test"])
                if picked == self._ENDPOINT_ADD:
                    typed = await self._type_in(
                        "url of the Ollama server (host, or host:port):")
                    if not typed:
                        return True
                    return await self.cmd_endpoint(["add", typed])
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
            self.client = make_client(self.config)
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
            self.discovery.workspace = self.workspace.resolve()
            self.project_info = self.discovery.scan(force=True)
            self.agent.project_map = projectmap.load(self.workspace)
        except Exception as exc:
            self.agent.project_map = ""
            if note is not None:
                note(WARN, "project map", str(exc))
            return
        self.agent.refresh_system_prompt()

    def cmd_apps(self, args: list[str]) -> bool:
        """The applications actually discovered on this machine.

        Generated from the OS -- Start Menu shortcuts, App Paths, PATH, and
        the platform equivalents -- never from a code-level list, so what is
        shown is exactly what launch_application can launch.
        """
        if args and args[0].lower() in ("refresh", "rescan", "again"):
            with self.ui.status("scanning for installed applications..."):
                entries = self._app_catalog.refresh()
        else:
            entries = self._app_catalog.entries()
        if not entries:
            self.ui.info("no applications discovered on this machine")
            return True
        self.ui.table(
            ["application", "found in", "target"],
            [(e.name, e.where, self.ui.shorten_path(str(e.path))) for e in entries],
            title=f"{len(entries)} applications",
        )
        self.ui.info("launched by name via launch_application; /apps refresh to rescan")
        return True

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
            answer = await self._question(
                "really widen to the whole machine? [y/N]:",
                {"y": "yes", "n": "no"}, default="n")
            if answer != "y":
                self.ui.info("left unchanged")
                return True

        self._apply_scope(scope)
        self.ui.success(f"scope: {self._boundary_summary(self.agent.boundary)}")
        return True

    # A leading keyword is a subcommand, never a repository. Parsed in one
    # place so invalid subcommands get structured help instead of the clone
    # path swallowing them as "that does not look like a repository".
    _REPO_KEYWORDS = ("help", "status", "list", "ls", "clone", "add")

    def _repo_help(self) -> None:
        self.ui.table(
            ["command", "does"],
            [
                ("repo", "the current checkout"),
                ("repo status", "which branch the workspace is on"),
                ("repo list", "the current checkout (wynxo keeps no registry)"),
                ("repo owner/name", "clone a GitHub repository"),
                ("repo <url>", "clone from a full URL"),
                ("repo clone owner/name", "same, spelled out"),
            ],
            title="repo",
        )

    def _repo_status(self) -> None:
        from . import repo as repo_module

        current = repo_module.status(self.workspace)
        if current:
            self.ui.info(f"{self.ui.shorten_path(str(self.workspace))}  "
                         f"on {current}")
        else:
            self.ui.info("this folder is not a git checkout")

    async def cmd_repo(self, args: list[str]) -> bool:
        """Clone a GitHub repository and move the workspace into it."""
        from . import repo as repo_module

        if not args:
            self._repo_status()
            self._repo_help()
            return True

        first = args[0].lower()
        if first in self._REPO_KEYWORDS:
            if first in ("list", "ls"):
                self._repo_status()
                self.ui.info("wynxo keeps no registry of clones; to work in "
                             "one, /repo <owner/name> or /cd into it.")
                return True
            if first == "status":
                self._repo_status()
                return True
            if first == "help":
                self._repo_help()
                return True
            if first in ("clone", "add"):
                args = args[1:]          # repo clone owner/name -> owner/name
                if not args:
                    self._repo_help()
                    return True

        if not repo_module.git_available():
            self.ui.error("git is not installed, so wynxo cannot clone anything.")
            return True

        target = repo_module.parse(" ".join(args))
        if target is None:
            self._repo_help()
            self.ui.warn("that does not look like a repository name or URL.")
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
            boundary=boundary, memory=self.agent.memory,
            app_catalog=self._app_catalog)
        current_todo = self.agent.tools.get("todo_write")
        if previous_todo is not None and current_todo is not None:
            current_todo.items = previous_todo.items
        self.agent.refresh_system_prompt()

    def cmd_copy(self, args: list[str]) -> bool:
        """Copy the conversation, or the last answer, to the system clipboard.

        Selecting with the mouse works normally -- wynxo never captures it
        -- but a long conversation is easier to take in one piece than to
        drag over. Built from the session messages, so what you get is plain
        text, not ANSI escapes.
        """
        from .platforms import copy_to_clipboard

        messages = self.agent.session.messages
        only_last = bool(args) and args[0] == "last"
        if only_last:
            blocks = [m.get("content", "") for m in messages
                      if m.get("role") == "assistant" and m.get("content")]
            text = blocks[-1] if blocks else ""
            label = "last answer"
        else:
            lines: list[str] = []
            for message in messages:
                role = message.get("role")
                content = str(message.get("content") or "").strip()
                if role == "user" and content and not content.startswith("/"):
                    lines.append(f"> {content}")
                elif role == "assistant" and content:
                    lines.append(content)
            text = "\n\n".join(lines)
            label = "conversation"
        if not text.strip():
            self.ui.info("nothing to copy yet")
            return True
        if copy_to_clipboard(text):
            self.ui.success(f"copied the {label} ({len(text)} chars) to the clipboard")
        else:
            self.ui.error("could not copy: no clipboard tool on this machine")
        return True

    async def cmd_gh(self, args: list[str]) -> bool:
        """Work on a GitHub repository in the cloud, through the gh CLI.

        Nothing is cloned: the tree, files and branches live on GitHub, and
        every operation goes through the API using the account ``gh auth
        login`` stored. This is the person-sized twin of the github_read /
        github_write tools the agent gets. Edits commit to the workspace
        branch; /gh branch creates one to work on, /gh pr ships it.
        """
        from .gh import GitHubClient, GitHubError

        if self.gh is None:
            self.gh = GitHubClient()
        action = args[0].lower() if args else "status"
        known = {"status", "login", "open", "ls", "cat", "edit",
                 "branch", "pr", "close"}
        if action not in known:
            self.ui.warn(f"unknown /gh action {action}. "
                         "status | login | open | ls | cat | edit | "
                         "branch | pr | close")
            return True
        ws = self.gh_ws

        try:
            if action == "login":
                self.ui.info("run `gh auth login` in a terminal to connect "
                             "your GitHub account, then /gh status here.")
                return True
            if action == "status":
                user = self.gh.auth_user()
                if ws:
                    files = sum(1 for e in ws["tree"]
                                if e.get("type") == "blob")
                    self.ui.info(f"GitHub: {user} · {ws['owner']}/{ws['repo']} "
                                 f"@{ws['branch']} · {files} files")
                else:
                    self.ui.info(f"GitHub: logged in as {user}. No repository "
                                 "open — /gh open owner/repo.")
                return True
            if action == "open":
                if len(args) < 2 or "/" not in args[1]:
                    self.ui.error("usage: /gh open owner/repo [branch]")
                    return True
                owner, repo = args[1].strip().split("/", 1)
                default = self.gh.repo_default_branch(owner, repo)
                branch = args[2] if len(args) > 2 else default
                tree = self.gh.tree(owner, repo, branch)
                self.gh_ws = {"owner": owner, "repo": repo,
                              "branch": branch, "default": default,
                              "tree": tree.entries}
                self.ui.success(
                    f"opened {owner}/{repo} @ {branch} in the cloud "
                    f"({len(tree.files)} files). /gh ls to browse, /gh cat "
                    f"<path> to read, /gh edit <path> to change.")
                if tree.truncated:
                    # The same thing the tool tells the model, told to the
                    # person: a listing this large is not the repository.
                    self.ui.warn(
                        "GitHub truncated this file listing, so it is only "
                        "part of the repository -- /gh ls will not show "
                        "everything.")
                return True
            if ws is None:
                self.ui.error("no repository open — /gh open owner/repo first")
                return True
            owner, repo, branch = ws["owner"], ws["repo"], ws["branch"]
            if action == "ls":
                prefix = args[1] if len(args) > 1 else ""
                lines = self._gh_ls(ws, prefix)
                self.ui.console.print("\n".join(lines) if lines else "nothing here.")
                return True
            if action == "cat":
                if len(args) < 2:
                    self.ui.error("usage: /gh cat <path>")
                    return True
                lines = self.gh.read(owner, repo, args[1], branch).text.splitlines()
                self.ui.console.print("\n".join(lines[:500]))
                if len(lines) > 500:
                    self.ui.info(f"... ({len(lines) - 500} more lines)")
                return True
            if action == "edit":
                if len(args) < 2:
                    self.ui.error("usage: /gh edit <path> [commit message]")
                    return True
                return await self._gh_edit(ws, args[1], " ".join(args[2:]) or None)
            if action == "branch":
                if len(args) < 2:
                    self.ui.error("usage: /gh branch <name>")
                    return True
                head = self.gh.ref_sha(owner, repo, branch)
                self.gh.create_branch(owner, repo, args[1], head)
                self.gh_ws["branch"] = args[1]
                self.ui.success(f"now working on {args[1]} in {owner}/{repo}.")
                return True
            if action == "pr":
                if branch == ws["default"]:
                    self.ui.warn("on the default branch — /gh branch <name> "
                                 "first, then /gh pr.")
                    return True
                title = " ".join(args[1:]) or None
                body = self._gh_pr_body(ws)
                url = self.gh.open_pr(
                    owner, repo, ws["default"], branch,
                    title or f"wynxo: changes on {branch}", body)
                self.ui.success(url)
                return True
            if action == "close":
                self.gh_ws = None
                self.ui.info("closed the cloud workspace.")
                return True
        except GitHubError as exc:
            self.ui.error(str(exc))
            return True
        return True

    def _gh_ls(self, ws: dict, prefix: str) -> list[str]:
        """The tree entries directly under a prefix, directories first."""
        prefix = prefix.strip("/")
        entries = ws["tree"]
        if prefix:
            entries = [e for e in entries
                       if e["path"] == prefix
                       or e["path"].startswith(prefix + "/")]
        lines: list[str] = []
        for entry in sorted(entries,
                            key=lambda e: (e.get("type") != "tree", e["path"])):
            path = entry["path"]
            if path == prefix:
                continue
            rest = path[len(prefix) + 1:] if prefix else path
            if "/" in rest:
                continue
            if entry.get("type") == "tree":
                lines.append(f"{path}/")
            else:
                size = entry.get("size")
                lines.append(f"{path}  ({size} B)" if size else path)
        return lines

    async def _gh_edit(self, ws: dict, path: str, message: str | None) -> bool:
        """Fetch a cloud file into a temp file, open the editor, review the
        diff, and commit back -- nothing goes up blind."""
        import os
        import shlex
        import subprocess
        import tempfile

        from .gh import GitHubError

        owner, repo, branch = ws["owner"], ws["repo"], ws["branch"]
        try:
            blob = self.gh.read(owner, repo, path, branch)
            content, sha = blob.text, blob.sha
        except GitHubError as exc:
            self.ui.error(str(exc))
            return True
        suffix = ".md" if path.endswith((".md", ".markdown")) else ".txt"
        fd, tmp = tempfile.mkstemp(prefix="wynxo-gh-", suffix=suffix)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR"))
            if editor:
                subprocess.run([*shlex.split(editor), tmp])
            elif sys.platform == "win32":
                # notepad returns immediately; Start-Process -Wait blocks
                # until the window is closed.
                subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Start-Process -Wait notepad "
                                "-ArgumentList $args[0]", tmp])
            else:
                subprocess.run(["nano", tmp])
            with open(tmp, encoding="utf-8") as handle:
                updated = handle.read()
            if updated == content:
                self.ui.info("no changes saved.")
                return True
            diff = self._gh_diff(content, updated, path)
            self.ui.console.print()
            self.ui.diff(diff)
            answer = await self._question(
                f"commit {path} to {owner}/{repo} @ {branch}? [y/n]",
                {"y": "yes", "n": "no"}, default="y")
            if answer != "y":
                self.ui.info("nothing committed.")
                return True
            commit = self.gh.write(owner, repo, path, updated,
                                   message or f"wynxo: edit {path}",
                                   branch, sha=sha)
            self.ui.success(f"committed {path} on {branch} ({commit[:10]}).")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return True

    @staticmethod
    def _gh_diff(before: str, after: str, path: str) -> str:
        """A unified diff of what the editor changed, for the review prompt."""
        import difflib

        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        lines = list(difflib.unified_diff(
            before_lines, after_lines, fromfile=f"a/{path}",
            tofile=f"b/{path}", n=2))
        return "".join(lines) if lines else "(no text changes)"

    def _gh_pr_body(self, ws: dict) -> str:
        """A PR body listing the commits made on the workspace branch."""
        from .gh import GitHubError

        try:
            messages = self.gh.commits(ws["owner"], ws["repo"], ws["branch"])
        except GitHubError:
            messages = []
        lines = [f"Changes on `{ws['branch']}` → `{ws['default']}`:"]
        lines += [f"- {m.splitlines()[0]}" for m in messages if m.splitlines()]
        return "\n".join(lines) or f"Changes on `{ws['branch']}`."

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
        from .speech import (MOMMY_VOICES, Speaker, available, install_edge_tts,
                             install_hint, list_edge_voices,
                             pick as pick_engine)

        action = args[0].lower() if args else ""

        if not action:
            # Bare /speak used to print the status and stop, which told you
            # what the setting was and gave you no way to change it. Every
            # other setting opens a picker; this one offers the actions.
            engines = available()
            if not engines:
                self.ui.warn("No speech synthesiser found on this machine.")
                for line in install_hint().splitlines():
                    self.ui.info(line)
                return True
            edge_tts_missing = not any(
                e.name == "edge-tts" for e in engines)
            menu = [("on", "read answers out loud"),
                    ("off", "stay quiet"),
                    ("test", "say a sentence now, to check it works"),
                    ("engine", f"which synthesiser speaks "
                               f"({len(engines)} available here)"),
                    ("voice", "pick a voice / heard it speak first")]
            if edge_tts_missing:
                menu.append(("install",
                             "get Microsoft's natural voices -- no more robot"))
            chosen = await self._pick(
                "speech", menu,
                "on" if self.config.speak else "off",
            )
            if chosen is None:
                return True
            if chosen is NO_PICKER:
                action = "show"
            else:
                return await self.cmd_speak([chosen])

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
            self.config.save()
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

        if action == "voice" and len(args) == 1:
            # With edge-tts around, a voice is a person-shaped choice: offer
            # the warm picks, then every female neural voice Microsoft has.
            # Without it, it is just an engine-specific name to type.
            if any(e.name == "edge-tts" for e in available()):
                with self.ui.status("fetching Microsoft's voices..."):
                    all_voices = await list_edge_voices()
                curated = dict(MOMMY_VOICES)
                options = list(MOMMY_VOICES)
                options += [(n, h) for n, h in all_voices if n not in curated]
                current = self.config.speech_voice or "en-US-JennyNeural"
                picked = await self._pick("voice", options, current)
                if picked is None:
                    return True
                if picked is NO_PICKER:
                    typed = await self._type_in(
                        "voice name (blank to cancel):", current)
                    if not typed:
                        return True
                    args = [action, typed]
                else:
                    args = [action, picked]
            else:
                typed = await self._type_in("voice name (blank to cancel):",
                                            self.config.speech_voice or "")
                if not typed:
                    return True
                args = [action, typed]

        if action == "install":
            if any(e.name == "edge-tts" for e in available()):
                self.ui.info("edge-tts is already here.")
                return await self.cmd_speak(["engine", "edge-tts"])
            self.ui.info("installing Microsoft's neural voices...")
            ok, detail = install_edge_tts()
            if not ok:
                self.ui.warn(f"could not install edge-tts: {detail}")
                return True
            self.ui.success("installed -- the natural voices are ready.")
            self.config.speak = True
            self.config.speech_engine = "edge-tts"
            self.speaker = Speaker(
                pick_engine("edge-tts"), voice=self.config.speech_voice,
                rate=self.config.speech_rate, model=self.config.speech_model)
            self.speaker.enabled = True
            self.config.save()
            self.ui.info(f"speech: {self.speaker.describe()}  "
                         f"(/speak voice to pick which)")
            return True

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
            self.config.save()
            self.speaker.enabled = self.config.speak
            self.ui.success(f"speech: {self.speaker.describe()}")
            return True

        self.ui.info("/speak on | off | test | engine <name> | voice <name>")
        return True

    async def cmd_talker(self, args: list[str]) -> bool:
        """Set the small model that does the talking, or turn it off."""
        from .duo import Talker
        from .prompts import VOICES

        if not args:
            # Bare /talker used to name the current one and then tell you the
            # syntax for changing it. Ask the server what it has and offer
            # that instead -- a talker is a model, and you cannot pick a
            # model you cannot remember the tag of.
            try:
                with self.ui.status("asking the server what it has..."):
                    models = await self.client.list_models()
            except ProviderError as exc:
                self.ui.error(str(exc))
                return True

            options = [("off", "one model does both jobs")]
            for model in models:
                if model.name == self.config.model:
                    continue        # the coder cannot also be the talker
                hint = " ".join(part for part in (model.human_size(),
                                                   model.parameter_size,
                                                   model.quantization) if part)
                options.append((model.name, hint or "installed"))
            if len(options) == 1:
                self.ui.info("no other model on that server to talk with")
                return True

            chosen = await self._pick("talker", options,
                                      self.config.talker or "off")
            if chosen is None:
                return True
            if chosen is NO_PICKER:
                if self.talker is None:
                    self.ui.info("no talker -- one model does both jobs")
                    self.ui.info("/talker <model> to have a small one do the talking")
                else:
                    self.ui.info(f"talker: {self.talker.model}   "
                                 f"coder: {self.config.model}")
                return True
            args = [chosen]

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
        if choice == "minimal":
            # Reduced motion: no animated faces, no waveforms, no effects.
            # The interface stays fully usable -- just still.
            if self.config.animations:
                self.config.animations = False
                self.config.save()
            self.pet.animate = False
        else:
            # Leaving minimal restores whatever motion was configured.
            self.pet.animate = self.config.animations
        # Rebind now so the rest of this session picks it up, and say plainly
        # that anything already drawn keeps the old colours.
        from .ui import apply_palette

        self.ui.palette = theme_module.resolve(choice)
        apply_palette(self.ui.palette)
        self.ui.code_theme = self.ui.palette.code_theme
        self.ui.success(f"theme: {choice}")
        self._preview_theme()
        return True

    async def cmd_animate(self, args: list[str]) -> bool:
        """Preview the animation scenes, or toggle motion on and off.

        Deterministic by design: previews are frame strips printed once, not
        live loops, so the command never owns the screen or needs a timer.
        """
        from .motion import SCENES, preview

        if args and args[0].lower() in ("on", "off"):
            on = args[0].lower() == "on"
            self.config.animations = on
            self.config.save()
            self.pet.animate = on
            self.ui.success(f"animations {'on' if on else 'off'}"
                            + ("  (reduced motion)" if not on else ""))
            return True

        if args:
            name = args[0].lower()
            if name not in SCENES:
                self.ui.warn(f"unknown scene; pick one of {', '.join(SCENES)}")
                return True
            self.ui.console.print(
                Text(f"  {name}", style="bold")
                + Text(f"  {SCENES[name].label}", style=MUTED))
            self.ui.console.print(
                Text(preview(name, unicode=self.ui.g.unicode),
                     style=f"bold {self.pet.style()}"))
            self.ui.console.print()
            return True

        self.ui.table(
            ["scene", "what it shows", "speed", "plays"],
            [(n, s.label, f"{s.fps:.0f} fps",
              "loop" if s.loops else "once") for n, s in SCENES.items()],
            title="animations",
        )
        self.ui.info(
            f"preview one: /animate <scene>   motion: "
            f"{'on' if self.config.animations else 'off'}"
            + ("  (/animate off for reduced motion)" if self.config.animations else ""))
        return True

    async def cmd_todo(self, args: list[str]) -> bool:
        """Show the current plan."""
        todo = self.agent.tools.get("todo_write")
        rendered = todo.render() if todo and hasattr(todo, "render") else ""
        self.ui.todos(rendered) if rendered else self.ui.info("no plan yet")
        return True

    async def cmd_queue(self, args: list[str]) -> bool:
        """Show what is waiting, run it, or drop it.

        Type-ahead is collected while a turn runs and normally drains the
        moment it ends, so this is for the case where it did not: a Ctrl-C
        holds the rest of the queue rather than launching it, and then there
        has to be a way to see what is held and decide about it.
        """
        want = args[0].lower() if args else ""
        if want in ("clear", "drop", "empty"):
            dropped = self.pending.clear()
            self.ui.success(dropped or "nothing was queued")
            return True
        if want in ("run", "go", "resume"):
            if not self.pending.summary():
                self.ui.info("nothing queued")
                return True
            return await self._drain_queue()
        if args:
            self.ui.warn("/queue shows what is waiting; run or clear decides")
            return True
        waiting = self.pending.summary()
        if not waiting:
            self.ui.info("nothing queued")
            return True
        self.ui.console.print()
        for index, message in enumerate(waiting, start=1):
            self.ui.console.print(
                Text(f"  {index}. ", style=MUTED) + Text(message))
        self.ui.console.print()
        self.ui.info("/queue run to send them, /queue clear to drop them")
        return True

    async def _question(self, question: str, answers: dict[str, str],
                        default: str = "") -> str:
        """Ask a short question.

        Every question in the session comes through here, so the escaping
        and the cancelled-vs-default rule can only be got wrong once.

        Returns the key that was answered, or "" for cancelled.
        """
        try:
            typed = (await self.prompt_session.prompt_async(
                HTML(f'<style fg="{ACCENT}">  {_escape(question)} </style>')
            )).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ""
        finally:
            # prompt_toolkit's teardown removes the loop's SIGINT handler.
            # Without this, Ctrl-C is dead for the rest of the command that
            # asked the question.
            self._arm_interrupt()
        if not typed:
            return default
        for key, meaning in answers.items():
            if typed in (key, meaning) or typed[0] == key:
                return key
        return ""

    async def _type_in(self, question: str, default: str = "") -> str:
        """Read a line of free text."""
        try:
            return (await self.prompt_session.prompt_async(
                HTML(f'<style fg="{ACCENT}">  {_escape(question)} </style>'),
                default=default)).strip()
        except (EOFError, KeyboardInterrupt):
            return ""
        finally:
            self._arm_interrupt()

    async def _pick(self, title: str, options: list[tuple[str, str]],
                    current: str) -> str | None:
        """Arrow-key chooser for a simple setting.

        Returns the chosen value, None if the user pressed escape, or
        NO_PICKER when this terminal cannot draw one. Cancelling and being
        unable to offer a choice are different things: escape means "never
        mind", and printing the table anyway ignores that.
        """
        choices = [
            option if isinstance(option, Choice) else
            Choice(value=option[0],
                   label=option[0],
                   badge="current" if option[0] == current else "",
                   badge_style="badge",
                   hint=option[1])
            for option in options
        ]
        if not arrows_supported():
            return NO_PICKER
        try:
            return await choose(
                choices,
                title=title,
                default=next((i for i, c in enumerate(choices)
                              if c.value == current), 0),
                footer=HINT if self.ui.g.unicode else HINT_ASCII,
                width=self.ui.width,
                unicode=self.ui.g.unicode,
            )
        finally:
            # The chooser is a prompt_toolkit Application too, and its
            # teardown takes the SIGINT handler with it.
            self._arm_interrupt()

    async def cmd_thinking(self, args: list[str]) -> bool:
        """Show or hide the model's reasoning.

        A picker rather than a bare toggle. Every other setting opens one,
        and a toggle is the one shape that cannot tell you what it is about
        to do: you press it to find out, and if it was already what you
        wanted you have to press it twice more to get back. With two named
        options the current one is marked and choosing it again is a no-op.
        """
        want = args[0].lower() if args else ""
        if want in ("show", "yes"):
            want = "on"
        elif want in ("hide", "no"):
            want = "off"
        elif want in ("all", "history", "replay"):
            # Nothing was thrown away while it was hidden, so there is always
            # a full record to go back over.
            self.callbacks._open_thinking(whole_session=True)
            if not self.callbacks._thinking_turns and not self.callbacks._thinking_buffer:
                self.ui.info("nothing thought yet this session")
            return True
        if want not in ("on", "off"):
            chosen = await self._pick(
                "thinking",
                [("on", "show the model's reasoning as it works"),
                 ("off", "keep only the answer; ^O reveals the reasoning "
                         "for one turn"),
                 ("all", "replay everything thought this session")],
                "on" if self.ui.show_thinking else "off",
            )
            if chosen is NO_PICKER:
                state = "on" if self.ui.show_thinking else "off"
                self.ui.info(f"thinking display is {state}  {self.ui.g.dot}  "
                             "/thinking on | off")
                return True
            if chosen is None:
                return True
            if chosen == "all":
                self.callbacks._open_thinking(whole_session=True)
                return True
            want = chosen

        self.ui.show_thinking = want == "on"
        self.config.show_thinking = self.ui.show_thinking
        self.config.save()
        self.ui.info(f"thinking display {'on' if self.ui.show_thinking else 'off'}")
        if self.ui.show_thinking and not self.policy.thinking:
            self.ui.warn(f"{self.policy.name} effort does not ask the model to "
                         "think, so there will be nothing to show. "
                         "/effort high or above turns reasoning on.")
        return True

    async def cmd_logo(self, args: list[str]) -> bool:
        """Choose the start-up logo, and show it straight away.

        Shown on choosing rather than described: the whole point of a logo
        is what it looks like, and "logo: wordmark" tells you nothing.
        """
        names = logo.available()
        want = args[0].lower() if args else ""
        if want not in names and want != "off":
            chosen = await self._pick(
                "logo",
                [(n, "the start-up picture" if n != "off" else "")
                 for n in names] + [("off", "no logo at all")],
                self.config.logo,
            )
            if chosen is NO_PICKER:
                self.ui.info(f"logo: {self.config.logo}  {self.ui.g.dot}  "
                             f"/logo {' | '.join(names)} | off")
                return True
            if chosen is None:
                return True
            want = chosen

        self.config.logo = want
        self.config.save()
        if want == "off":
            self.ui.success("no logo at start-up")
            return True
        await logo.play(self.ui, want, self.config.animations)
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
            # The moods are worth seeing, so they stay. What used to follow
            # them was a line of usage text -- it told you what you could
            # type and left you to type it. The picker acts instead.
            chosen = await self._pick(
                "companion",
                [("on", "she reacts as the work goes"),
                 ("off", "no companion"),
                 ("animate", "let the faces move"),
                 ("still", "one face, held"),
                 ("name", f"what to call her (now: {self.pet.name})"),
                 ("voice", f"how she writes (now: {self.config.voice})")],
                ("on" if self.pet.enabled else "off"),
            )
            if chosen is None:
                return True
            if chosen is NO_PICKER:
                self.ui.info(f"name: {self.pet.name}   voice: {self.config.voice}   "
                             f"{'on' if self.pet.enabled else 'off'}"
                             f"{'' if self.pet.animate else ', still'}")
                self.ui.info(f"/pet off {self.ui.g.dot} /pet name <x> "
                             f"{self.ui.g.dot} /pet voice " + " | ".join(VOICES))
                return True
            return await self.cmd_pet([chosen])

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

        if action == "show":
            from .motion import preview, scene_for
            moods = [m.value for m in Mood]
            if len(args) > 1 and args[1].lower() in moods:
                moods = [args[1].lower()]
            self.ui.console.print()
            for mood in moods:
                scene = scene_for(mood)
                self.ui.console.print(
                    Text(f"  {mood}", style="bold")
                    + Text(f"  {scene.label}", style=MUTED))
                self.ui.console.print(
                    Text(preview(mood, unicode=self.ui.g.unicode),
                         style=f"bold {self.pet.style()}"))
            self.ui.console.print()
            self.ui.info("moods: " + " | ".join(m.value for m in Mood))
            return True

        if action == "name" and len(args) == 1:
            typed = await self._type_in("what should she be called?",
                                        self.pet.name)
            if not typed:
                return True
            args = [action, typed]

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
            self._set_voice(choice)
            return True

        self.ui.warn("usage: /pet [on|off|still|animate|show <mood>|name <x>|voice <x>]")
        return True

    def _set_voice(self, choice: str) -> None:
        """Switch how the agent talks, everywhere the voice lives.

        One call instead of the four places a voice change must land in:
        the saved config, the pet's style, the talker's persona, and the
        agent's system prompt. Missing one of them means the new voice is
        spoken by half the program.
        """
        from .prompts import VOICES

        self.config.voice = choice
        self.config.save()
        self.pet.style_name = ("kawaii" if choice == "kawaii"
                               else "mommy" if choice == "mommy"
                               else "default")
        if self.talker is not None:
            self.talker.voice_block = VOICES.get(choice, "")
        self.agent.refresh_system_prompt()
        self.ui.success(f"voice: {choice} -- {_voice_summary(choice)}")

    def cmd_mommy(self, args: list[str]) -> bool:
        """Mommy-style talking, on or off; no argument toggles.

        A single-word convenience over the full voice picker: on is the
        doting mommy persona, off is the plain professional voice. The
        engineering underneath is identical either way.
        """
        current = self.config.voice
        if not args or args[0].lower() in ("toggle", "t"):
            want = "plain" if current == "mommy" else "mommy"
        else:
            action = args[0].lower()
            if action in ("on", "mommy"):
                want = "mommy"
            elif action in ("off", "plain", "default"):
                want = "plain"
            else:
                self.ui.warn(
                    f"usage: /mommy [on|off]  (now: {_voice_summary(current)})")
                return True
        if want == current:
            self.ui.info(
                f"mommy style already {'on' if want == 'mommy' else 'off'}")
            return True
        self._set_voice(want)
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
            added, message = memory.remember(note, scope, explicit=True)
            self.ui.success(f"remembered: {message}") if added else self.ui.info(message)
            self.agent.refresh_system_prompt()
            return True

        if action in ("forget", "remove") and rest:
            count, message = memory.forget(rest, explicit=True)
            if not count:
                count, message = memory.forget(rest, "user", explicit=True)
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

    CTX_SIZES = (4096, 8192, 16384, 32768, 65536, 131072)
    """The window sizes worth offering. Powers of two because that is what
    every model is trained and every runner tuned for."""

    async def cmd_ctx(self, args: list[str]) -> bool:
        if not args:
            return await self._pick_context()
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

    async def _pick_context(self) -> bool:
        """Choose the context window from a list rather than be told the number.

        Bare /ctx used to report the setting and stop -- which is the one
        thing you already knew, since you asked. The sizes are the ones
        worth having; the model's own maximum is marked when the server
        told us what it is, because going past it costs memory and buys
        nothing.
        """
        used = self.agent.session.token_estimate()
        native = 0
        info = getattr(self.agent, "model_info", None)
        if info is not None:
            native = getattr(info, "context_length", 0) or 0

        sizes = sorted(set(self.CTX_SIZES) | {self.config.num_ctx}
                       | ({native} if native else set()))
        options = []
        for size in sizes:
            note = []
            if native and size == native:
                note.append("the model's own window")
            elif native and size > native:
                note.append(f"past the model's {native}")
            if size < 8192:
                note.append("too small for real work")
            if used and size < used:
                note.append(f"smaller than the {used} tokens already in use")
            options.append((str(size), ", ".join(note) or f"{size // 1024}k tokens"))
        options.append(("custom", "type an exact number"))

        chosen = await self._pick("context window", options,
                                  str(self.config.num_ctx))
        if chosen is None:
            return True
        if chosen is NO_PICKER:
            self.ui.info(
                f"num_ctx={self.config.num_ctx}, roughly {used} tokens in use "
                f"({100 * used / max(1, self.config.num_ctx):.0f}%)"
            )
            return True
        if chosen == "custom":
            typed = await self._type_in("context window, in tokens:",
                                        str(self.config.num_ctx))
            if not typed:
                return True
            chosen = typed
        return await self.cmd_ctx([chosen])

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
                ("reclaimed", f"~{self.agent.session.superseded_chars // 4} tokens"
                              " of superseded reads and test runs"),
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

    # Gated on the platform, not on hasattr: Windows *has* select.select, it
    # just cannot use it on a pipe. Testing for the attribute sent Windows
    # down this branch, where the call raised and the input was dropped -- so
    # `git diff | wynxo -p "review"` silently reviewed nothing.
    if os.name != "nt":
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

    # Windows: ask the pipe itself whether anything is waiting.
    #
    # The obvious alternative -- read in a thread and abandon it after the
    # grace period -- leaves a thread blocked in a read on stdin for the rest
    # of the process's life. That is not merely untidy: the handle stays
    # busy, and anything that later waits on this process's pipes can wait
    # for it forever.
    return _windows_pipe_text(grace)


def _windows_pipe_text(grace: float) -> str:
    """Whatever is already in the pipe, without blocking on it.

    PeekNamedPipe reports how many bytes are available and returns
    immediately either way, which is the thing select cannot do for a pipe
    on Windows. Nothing waiting means nothing piped.
    """
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
    except (ImportError, AttributeError):
        return ""

    try:
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
    except (OSError, ValueError):
        return ""

    peek = ctypes.windll.kernel32.PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                     ctypes.POINTER(wintypes.DWORD),
                     ctypes.POINTER(wintypes.DWORD),
                     ctypes.POINTER(wintypes.DWORD)]
    available = wintypes.DWORD(0)

    # Polled rather than asked once: a producer that is a fraction of a
    # second behind would otherwise look like an empty pipe.
    deadline = time.monotonic() + max(0.0, grace)
    while True:
        try:
            ok = peek(handle, None, 0, None, ctypes.byref(available), None)
        except OSError:
            return ""
        if not ok:
            return ""          # closed, or not a pipe at all
        if available.value:
            break
        if time.monotonic() >= deadline:
            return ""
        time.sleep(0.02)

    try:
        return sys.stdin.read().strip()
    except (OSError, UnicodeDecodeError, ValueError):
        return ""


async def run_once(config: Config, workspace: Path, ui: UI, prompt: str,
                   scope: Scope = Scope.FOLDER, mode: Mode = Mode.YOLO,
                   json_output: bool = False) -> int:
    """Non-interactive mode: answer one prompt and exit.

    With ``json_output`` the whole run is silent and a single JSON object
    -- answer, errors, usage -- is printed on stdout, for scripts.
    """
    import io
    import json as _json

    from rich.console import Console

    if json_output:
        # Tool progress and stage lines are chat, not data. Swap the
        # console for a sink so stdout carries nothing but the JSON.
        ui.console = Console(file=io.StringIO())

    client = make_client(config)
    try:
        await client.ping()
    except ProviderError as exc:
        if json_output:
            print(_json.dumps({"ok": False, "content": "", "errors": [str(exc)]}))
        else:
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
        if json_output:
            print(_json.dumps({"ok": False, "content": "", "errors": [str(exc)]}))
        else:
            ui.error(str(exc))
        await client.aclose()
        return 1
    callbacks._end_stream()
    usage = agent.session.usage
    await client.aclose()

    if json_output:
        print(_json.dumps({
            "ok": not result.errors,
            "content": result.content,
            "errors": result.errors,
            "model": config.model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "requests": usage.requests,
                "tool_calls": usage.tool_calls,
            },
        }))
        return 0 if not result.errors else 1

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
    parser.add_argument("--json", action="store_true",
                        help="with -p: print the answer as one JSON object "
                             "on stdout, nothing else")
    parser.add_argument("-e", "--effort", choices=list(ORDER), help="effort level for this run")
    parser.add_argument("-m", "--model", help="model to use")
    parser.add_argument("--endpoint", help="Ollama URL, e.g. http://homelab:11434")
    parser.add_argument("--ctx", type=int, help="context window size (num_ctx)")
    parser.add_argument("-C", "--cwd", help="project directory (default: here)")
    parser.add_argument("--repo", metavar="OWNER/NAME",
                        help="clone a GitHub repository and work in it")
    parser.add_argument("--talker", metavar="MODEL",
                        help="small model that talks while the coder works")
    parser.add_argument("--speak", action="store_true",
                        help="read answers out loud")
    parser.add_argument("--no-speak", action="store_true",
                        help="stay quiet even if speech is on in the config")
    parser.add_argument("--setup", action="store_true", help="re-run first-time setup")
    parser.add_argument("--doctor", action="store_true",
                        help="check the server and model, and report what will not work")
    parser.add_argument("--ascii", metavar="IMAGE",
                        help="turn a picture into ASCII art and print it")
    parser.add_argument("--ascii-width", type=int, default=100,
                        metavar="N", help="columns wide (default 100)")
    parser.add_argument("--ascii-style", default="detail",
                        choices=("detail", "simple", "blocks"),
                        help="character ramp to draw with")
    parser.add_argument("--ascii-invert", action="store_true",
                        help="for a light terminal, where the ramp reads "
                             "the other way round")
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


def apply_flags(config, args) -> None:
    """Let this run's flags override the saved settings.

    Its own function so it can be checked directly. The interesting rules
    are the ones where two flags interact, and those are exactly the ones
    that are impossible to test through main().
    """
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    if args.ctx:
        config.num_ctx = args.ctx
    if args.no_stream:
        config.stream = False
    if args.no_thinking:
        config.show_thinking = False
    if args.talker:
        config.talker = args.talker
    if args.speak:
        config.speak = True
    if args.no_speak:
        config.speak = False


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    if not workspace.is_dir():
        print(f"No such directory: {workspace}", file=sys.stderr)
        return 1

    ui = UI(show_thinking=not args.no_thinking)

    # Before the configuration gate on purpose: turning a local picture into
    # text needs no model and no server, so being unconfigured is irrelevant
    # to it and asking the user to run setup first would be nonsense.
    if args.ascii:
        return _print_ascii(args, ui)

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
    apply_flags(config, args)
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
        return await run_once(config, workspace, ui, prompt, once_scope,
                              once_mode, json_output=args.json)

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
    #
    # Output goes to the terminal's own scrollback and the mouse is never
    # captured, so scrolling, selecting and copying are the terminal's own
    # -- no alternate screen, no mouse reporting, nothing for the user to
    # fight.
    with patch_stdout(raw=True):
        if prompt:
            return await repl.start_with(prompt)
        return await repl.start()


def _print_ascii(args, ui: UI) -> int:
    """Print a picture as text.

    Deliberately not part of a turn: it is a local conversion of a local
    file, so it neither needs a model nor sends the image anywhere.
    """
    from . import asciiart

    source = Path(args.ascii).expanduser()
    if not source.is_file():
        ui.error(f"{source} is not a file.")
        return 1
    width = max(20, min(400, args.ascii_width))
    try:
        art = asciiart.from_image(source, width=width, style=args.ascii_style,
                                  invert=args.ascii_invert)
    except asciiart.ImageSupportMissing as exc:
        ui.error(str(exc))
        return 1
    except (OSError, ValueError) as exc:
        ui.error(f"Could not read {source.name}: {exc}")
        return 1
    # Straight to stdout, unstyled: this is meant to be redirected into a
    # file and pasted into a banner, and rich would wrap and colour it.
    print(art)
    return 0


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
        print("\nwynxo hit an unexpected error and had to stop.",
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
