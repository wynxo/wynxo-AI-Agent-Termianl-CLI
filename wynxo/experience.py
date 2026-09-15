"""Chat-first product behaviour layered over the stable CLI core.

The REPL and agent are intentionally large, mature modules. This layer keeps
small product-policy decisions -- conversational identity, command completion,
and the compact launch surface -- out of those hot paths. It is installed by
:mod:`wynxo.bootstrap` after the existing product and clean UI layers.
"""

from __future__ import annotations

import re

from prompt_toolkit.completion import Completion
from rich.table import Table
from rich.text import Text

from . import __version__
from .platforms import is_dumb_terminal

_INSTALLED = False
_PREV_HOME = None
_ORIGINAL_SMALL_TALK = None
_ORIGINAL_MODEL_CALL = None
_ORIGINAL_TASK_SIGNAL = None

# Affection and tiny human moments should never be mistaken for a coding task.
# Keep this deliberately narrow: "I love Python, fix this" is work; "ilyyy"
# and "i lov yuuu" are conversation.
_AFFECTION = re.compile(
    r"^\s*(?:"
    r"(?:i\s+)?(?:love|luv|lov|adore|miss)\s+(?:you+|u+|y+u+|ya+)"
    r"|il+y+|ily+|ilysm+|luv\s+(?:you+|u+|y+u+|ya+)"
    r"|(?:you(?:'re|\s+are)|ur)\s+(?:cute|sweet|nice|lovely|the\s+best)"
    r"|<3+"
    r")\s*[!.?~❤♡]*\s*$",
    re.IGNORECASE,
)

# build_chat_prompt formats these three fields. Positive instructions are
# intentional here: smaller local models follow a concrete conversational
# behaviour more reliably than a long list of forbidden support-bot phrases.
CHAT_PROMPT = """You are Wynxo, the user's local AI companion in their terminal.
You are a person to talk to, ask things, brainstorm with, and work with. Coding
is one capability you have; it is not the topic you drag every conversation
back to.

Respond to what the user actually said and stay on that subject.
- If they are affectionate, playful, celebrating, venting, or just chatting,
  respond to that moment naturally and stop. Do not pivot to features, coding,
  debugging, tasks, or "what next" unless they asked.
- If they ask a normal question, answer the question. Do not turn it into a
  project workflow.
- If they ask for writing, brainstorming, explaining, comparing, translating,
  planning, or recommendations, do that directly like a general assistant.
- If they ask for coding or project work, help with it directly.
- Match their energy and length. Tiny messages get tiny replies.
- Keep scratch work private. Never print a "thinking" section or narrate hidden
  reasoning; give the user the answer.
- Do not end replies with a generic offer to help or a question just to keep
  the conversation alive. Ask only when an answer is actually needed.
- Never invent facts, memories, actions, or results.

{voice_tag}
{serious_line}
{memory}"""

_OLD_CODE_IDENTITY = (
    "You are wynxo, a coding agent that runs in the user's terminal, on their "
    "machine, against their real files."
)
_NEW_CODE_IDENTITY = (
    "You are Wynxo, a local-first AI assistant running in the user's terminal, "
    "on their machine, with project tools available when a task needs them. "
    "Coding is a capability, not your whole personality: ordinary questions "
    "and casual messages stay ordinary, while project requests are handled as "
    "a precise coding agent."
)

_PRIMARY_COMMANDS = (
    "/chat", "/code", "/github", "/help", "/model", "/context", "/clear", "/quit",
)
_PRIMARY_RANK = {name: index for index, name in enumerate(_PRIMARY_COMMANDS)}

# People remember what they want to do more easily than the exact command name.
# These are completion keywords, not dispatcher aliases: typing `/chatter` can
# *suggest* /chat without silently changing what an entered command means.
_COMMAND_TERMS = {
    "/chat": ("chatter", "conversation", "companion", "chatting"),
    "/code": ("agent", "coding", "project", "developer", "dev"),
    "/github": ("git", "repo", "repository", "remote", "pull", "pr"),
    "/help": ("commands", "command", "docs", "documentation"),
    "/model": ("llm", "ollama", "models"),
    "/context": ("ctx", "tokens", "window"),
    "/clear": ("reset", "clean"),
    "/quit": ("bye", "leave"),
}

_COMMAND_META = {
    "/chat": "talk without project tools",
    "/code": "work on the local project",
    "/github": "work with a GitHub repository",
    "/help": "show commands and shortcuts",
    "/model": "choose the local model",
    "/context": "inspect context usage",
    "/clear": "start a clean conversation",
    "/quit": "leave Wynxo",
}

_CHAT_MARKER = "local AI companion in their terminal"

# The core signal deliberately errs toward "work" and historically treated a
# generic verb by itself ("explain", "write", "create", "find", "show me") as
# proof that the user wanted the project agent. That made "explain black holes"
# a coding turn. Product routing calls something obvious project work only when
# it carries project evidence. Obvious general-assistant requests can now take
# the direct chat path; genuinely ambiguous turns still get the cheap intent
# classifier rather than being guessed at.
_PROJECT_ACTION = (
    r"(?:fix|add|write|create|make|build|run|test|refactor|implement|remove|"
    r"delete|rename|update|change|edit|debug|explain|review|check|find|search|"
    r"install|deploy|commit|push|merge|read|open|show\s+me)"
)
_PROJECT_NOUN = (
    r"(?:code|bug|parser|function|class|method|script|tool|shell|tests?|"
    r"repo(?:sitory)?|project|files?|folders?|director(?:y|ies)|workspace|"
    r"module|package|dependencies?|cli|api|endpoint|branch|commit|diff|build|"
    r"config(?:uration)?|readme)"
)
_PRODUCT_TASK_SIGNAL = re.compile(
    rf"(?:"
    rf"\.(?:py|js|ts|tsx|jsx|go|rs|rb|java|c|h|cpp|cs|sh|md|json|ya?ml|toml|txt|html|css)\b"
    rf"|[/\\][\w.-]"
    rf"|```"
    rf"|\b(?:commit|push|merge|deploy|refactor|debug)\b"
    rf"|\b{_PROJECT_ACTION}\b.{{0,100}}\b{_PROJECT_NOUN}\b"
    rf"|\b{_PROJECT_NOUN}\b.{{0,100}}\b{_PROJECT_ACTION}\b"
    rf"|^\s*(?:sudo\s+)?(?:git|pytest|npm|pnpm|yarn|pip|pipx|cargo|python3?|node|go|cmake|ninja)\b"
    rf"|^\s*(?:sudo\s+)?make(?:\s+(?:-[\w-]+|all|build|test|install|clean|release|debug))?\s*$"
    rf")",
    re.IGNORECASE,
)

# Requests that are plainly general-assistant work rather than project work.
# They are only used after _PRODUCT_TASK_SIGNAL has ruled out paths, code
# fences, project nouns paired with actions, and developer commands. This is a
# latency optimisation: a local model should not have to answer a classifier
# request before it can answer "write me a poem".
_GENERAL_REQUEST = re.compile(
    r"^\s*(?:"
    r"explain|teach(?:\s+me)?|help\s+me\s+understand|summari[sz]e|translate|"
    r"brainstorm|recommend|suggest|compare|write|draft|create|make|find|"
    r"give\s+me"
    r")\b",
    re.IGNORECASE,
)
_GENERAL_QUESTION = re.compile(
    r"^\s*(?:what|who|where|when|why|how)\s+"
    r"(?:is|are|was|were|does|do|did|can|could|would|should)\b",
    re.IGNORECASE,
)
_PROJECT_WORD = re.compile(rf"\b{_PROJECT_NOUN}\b", re.IGNORECASE)
# "create an app" is project creation even though "app" is deliberately not
# in _PROJECT_NOUN (where it would make ordinary questions about apps look
# like repository work). Keep construction targets as a separate guard.
_BUILD_TARGET = re.compile(
    r"\b(?:website|web\s*app|app(?:lication)?|program|service|server|database|"
    r"frontend|backend|component)\b",
    re.IGNORECASE,
)


def extra_conversation(request: str) -> bool:
    """True when Code mode can answer directly with the chat prompt.

    This is deliberately a certainty fast path, not a second intent router.
    Project evidence and build-like targets still go through the normal
    routing/tool path; anything unclear still gets the model classifier.
    """
    text = (request or "").strip()
    if not text:
        return False
    if len(text) <= 120 and _AFFECTION.fullmatch(text):
        return True
    if _PRODUCT_TASK_SIGNAL.search(text):
        return False
    if _GENERAL_REQUEST.match(text):
        return not _BUILD_TARGET.search(text)
    if _GENERAL_QUESTION.match(text):
        # "what is photosynthesis?" is obvious chat; "where is config?" or
        # "what does this function do?" needs project context or a classifier.
        return not _PROJECT_WORD.search(text)
    return False


def _semantic_commands(cli_mod, text: str) -> list[str]:
    """Commands matching the user's *intent word*, not only their spelling."""
    raw = (text or "").strip().lower()
    if raw in cli_mod.COMMANDS or raw in cli_mod.ALIASES:
        # An exact command is authoritative. `/repo` is a real command and
        # must never be displaced by the semantic keyword "repo" for /github.
        return []
    stem = raw.lstrip("/")
    if len(stem) < 2 or " " in stem:
        return []

    matches: list[tuple[int, int, str]] = []
    for command, terms in _COMMAND_TERMS.items():
        if command not in cli_mod.COMMANDS:
            continue
        matching = [term for term in terms if term.startswith(stem)]
        if not matching:
            continue
        exact = 0 if stem in terms else 1
        shortest = min(len(term) for term in matching)
        matches.append((exact, shortest, command))
    matches.sort(key=lambda item: (item[0], item[1], _PRIMARY_RANK.get(item[2], 999)))
    return [command for _, _, command in matches]


def _ranked_commands(cli_mod, text: str, limit: int = 8) -> list[str]:
    """Command suggestions with useful front-door commands first.

    A bare slash is discovery, so show the handful that explain the product.
    Once the user types characters, preserve the core resolver's alias, prefix,
    and fuzzy-spelling behaviour, then add intent-based discovery such as
    `/chatter` -> /chat and `/llm` -> /model.
    """
    if text == "/":
        return [name for name in _PRIMARY_COMMANDS if name in cli_mod.COMMANDS][:limit]

    suggestions = cli_mod.suggest_commands(text, limit=max(limit, 12))
    for command in _semantic_commands(cli_mod, text):
        if command not in suggestions:
            suggestions.append(command)
    suggestions.sort(key=lambda name: (_PRIMARY_RANK.get(name, 999), name))
    return suggestions[:limit]


def _product_command_hints(cli_mod, buffer: str) -> list[str]:
    """The footer and popup use one suggestion policy, so they never disagree."""
    text = (buffer or "").strip().lower()
    if not text.startswith("/") or " " in text or len(text) < 2:
        return []
    matches = _ranked_commands(cli_mod, text, limit=5)
    return [] if matches == [text] else matches


def command_completer_class(cli_mod):
    """Build a completer that keeps the core path/model completion intact."""

    class ProductCommandCompleter(cli_mod.CommandCompleter):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor

            # Command names: include aliases, intent words, and typo recovery
            # in the popup rather than only after Enter reports an unknown
            # command.
            if text.startswith("/") and " " not in text:
                for name in _ranked_commands(cli_mod, text):
                    yield Completion(
                        name,
                        start_position=-len(text),
                        display=name,
                        display_meta=_COMMAND_META.get(name, cli_mod.COMMANDS.get(name, "")),
                    )
                return

            # Enumerated arguments should appear as soon as the space is
            # typed. The old completer required one character first, which
            # made `/effort `, `/theme `, `/voice `, etc. look unsupported.
            if text.startswith("/") and " " in text:
                command, _, raw_arg = text.partition(" ")
                canonical = cli_mod.resolve_command(command) or command
                values = cli_mod._SUBCOMMAND_VALUES.get(canonical)
                # Only complete the first value token. Once the user has
                # typed trailing whitespace, replacing `len(strip())` chars
                # would replace that whitespace rather than the value.
                clean_arg = raw_arg.strip()
                one_token = not raw_arg or raw_arg == clean_arg
                if values is not None and one_token and " " not in clean_arg:
                    prefix = clean_arg.lower()
                    for value in values:
                        if value.lower().startswith(prefix):
                            yield Completion(
                                value,
                                start_position=-len(clean_arg),
                                display=value,
                                display_meta=f"{canonical[1:]} option",
                            )
                    return

            # @file references, installed model names, and future completion
            # types remain owned by the battle-tested core completer.
            yield from super().get_completions(document, complete_event)

    ProductCommandCompleter.__name__ = "CommandCompleter"
    ProductCommandCompleter.__qualname__ = "CommandCompleter"
    return ProductCommandCompleter


def _prompt_hint(repl) -> str:
    """Short, mode-aware composer copy with no coding-only assumption."""
    from . import cli as cli_mod

    note, repl._prompt_note = cli_mod.live_note(repl._prompt_note)
    if note:
        return note
    typed = _product_command_hints(cli_mod, cli_mod._composer_text(repl))
    if typed:
        return "  ".join(typed)
    if getattr(repl.agent, "working_mode", "code") == "chat":
        return "Chat mode  ·  /code for project work"
    return "Ask anything  ·  / for commands"


def _home(self, model: str, workspace: str, *, mode: str = "agent",
          companion: str = "ready", version: str = "",
          show_companion: bool = False, show_art: bool = False,
          show_static_controls: bool = False) -> None:
    """A compact front door: Wynxo is an assistant first, not a dashboard."""
    self.refresh_size()
    if is_dumb_terminal() or not self.console.is_terminal:
        return _PREV_HOME(
            self, model, workspace, mode=mode, companion=companion,
            version=version, show_companion=show_companion, show_art=show_art,
            show_static_controls=show_static_controls,
        )

    palette = self.palette
    self.console.print()

    brand = Text()
    brand.append("WYNXO", style=f"bold {palette.accent}")
    brand.append(f"  {version or __version__}", style=palette.faint)
    brand.append("  LOCAL AI", style=palette.muted)
    self.console.print(brand)

    if self.width >= 72:
        context = Table.grid(padding=(0, 2), expand=False)
        context.add_column(no_wrap=True)
        context.add_column(no_wrap=False)
        context.add_row(
            Text.assemble(("MODEL  ", palette.faint), (str(model), palette.muted)),
            Text.assemble(
                ("WORKSPACE  ", palette.faint),
                (self.shorten_path(workspace), palette.muted),
            ),
        )
        self.console.print(context)
    else:
        self.console.print(Text.assemble(
            ("MODEL      ", palette.faint), (str(model), palette.muted)))
        self.console.print(Text.assemble(
            ("WORKSPACE  ", palette.faint),
            (self.shorten_path(workspace), palette.muted)))
    self.console.print()

    intro = Text()
    intro.append("Talk normally. ", style=f"bold {palette.text}")
    intro.append(
        "Wynxo can chat, answer questions, or work on the project when you ask.",
        style=palette.muted,
    )
    self.console.print(intro)
    self.console.print()

    quick = Table.grid(padding=(0, 2))
    quick.add_column(width=10, no_wrap=True)
    quick.add_column()
    for command, description in (
        ("/chat", "tool-free conversation"),
        ("/code", "local project agent"),
        ("/github", "work with a GitHub repository"),
        ("/help", "commands and shortcuts"),
    ):
        quick.add_row(
            Text(command, style=f"bold {palette.accent}"),
            Text(description, style=palette.muted),
        )
    self.console.print(quick)
    self.console.print()


class _ConversationCallbacks:
    """Keep a chat turn conversational even when developer tracing is enabled.

    Global thinking display remains useful in Code mode. A normal conversation
    is different: showing the model's scratchpad between "i lov yuuu" and the
    reply turns a human moment into a debugger trace. The model may still use
    its reasoning internally; this proxy only keeps it out of the transcript.
    """

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    async def on_thinking(self, text: str) -> None:
        return None

    async def on_stage(self, name: str, detail: str = "") -> None:
        # Keep a lightweight sign of life for a slow local model without
        # exposing an implementation term as the headline of casual chat.
        if name == "thinking":
            name = "responding"
        await self._base.on_stage(name, detail)


def _is_chat_messages(messages) -> bool:
    """Whether a model call is using Wynxo's conversational system prompt."""
    if not isinstance(messages, list):
        return False
    marker = _CHAT_MARKER.lower()
    for message in messages[:3]:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        if marker in str(message.get("content") or "").lower():
            return True
    return False


def _install_chat_rendering() -> None:
    """Hide scratch reasoning only for calls that carry the chat prompt."""
    global _ORIGINAL_MODEL_CALL
    from . import agent

    if _ORIGINAL_MODEL_CALL is None:
        _ORIGINAL_MODEL_CALL = agent.Agent._call_model
    original = _ORIGINAL_MODEL_CALL

    async def _call_model(self, *args, **kwargs):
        if not _is_chat_messages(kwargs.get("messages")):
            return await original(self, *args, **kwargs)

        callbacks = self.cb
        self.cb = _ConversationCallbacks(callbacks)
        try:
            return await original(self, *args, **kwargs)
        finally:
            self.cb = callbacks

    agent.Agent._call_model = _call_model


def _install_prompts() -> None:
    from . import prompts

    prompts.CHAT_PROMPT = CHAT_PROMPT
    if _OLD_CODE_IDENTITY in prompts.BASE:
        prompts.BASE = prompts.BASE.replace(
            _OLD_CODE_IDENTITY, _NEW_CODE_IDENTITY, 1,
        )


def _install_router() -> None:
    global _ORIGINAL_SMALL_TALK, _ORIGINAL_TASK_SIGNAL
    from . import agent

    if _ORIGINAL_SMALL_TALK is None:
        _ORIGINAL_SMALL_TALK = agent.is_small_talk
    if _ORIGINAL_TASK_SIGNAL is None:
        _ORIGINAL_TASK_SIGNAL = agent._TASK_SIGNAL

    # The original function resolves _TASK_SIGNAL from its module at call
    # time, so replacing the signal improves both small-talk and intent-routing
    # decisions without copying the core routing function into this layer.
    agent._TASK_SIGNAL = _PRODUCT_TASK_SIGNAL
    original = _ORIGINAL_SMALL_TALK

    def is_small_talk(request: str) -> bool:
        return extra_conversation(request) or original(request)

    agent.is_small_talk = is_small_talk


def install() -> None:
    """Install the chat-first experience after the normal CLI modules load."""
    global _INSTALLED, _PREV_HOME
    if _INSTALLED:
        return

    from . import cli as cli_mod
    from . import product_ui
    from . import ui as ui_mod

    _install_prompts()
    _install_router()
    _install_chat_rendering()

    # Repl instances are created after bootstrap finishes, so replacing the
    # class here changes completion without mutating an already-running prompt.
    cli_mod.CommandCompleter = command_completer_class(cli_mod)

    # clean_ui installs the final home method before this layer. Keep it only
    # as a non-interactive fallback; narrow interactive terminals still get
    # the same assistant-first identity, just with stacked context rows.
    _PREV_HOME = ui_mod.UI.home
    ui_mod.UI.home = _home

    # clean_ui's prompt renderer calls product_ui._prompt_hint dynamically, so
    # changing this one helper updates both product and clean shells.
    product_ui._prompt_hint = _prompt_hint

    _INSTALLED = True
