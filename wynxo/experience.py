"""Chat-first product behaviour layered over the stable CLI core.

The REPL and agent are intentionally large, mature modules.  This layer keeps
small product-policy decisions -- conversational identity, command completion,
and the compact launch surface -- out of those hot paths.  It is installed by
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

# build_chat_prompt formats these three fields.  Positive instructions are
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
- If they ask for coding or project work, help with it directly.
- Match their energy and length. Tiny messages get tiny replies.
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


def extra_conversation(request: str) -> bool:
    """Return True for small human messages the core router should not plan."""
    text = (request or "").strip()
    return bool(text and len(text) <= 120 and _AFFECTION.fullmatch(text))


def _ranked_commands(cli_mod, text: str, limit: int = 8) -> list[str]:
    """Command suggestions with useful front-door commands first.

    A bare slash is discovery, so show the handful that explain the product.
    Once the user types characters, preserve the core resolver's alias, prefix,
    and fuzzy-spelling behaviour.
    """
    if text == "/":
        return [name for name in _PRIMARY_COMMANDS if name in cli_mod.COMMANDS][:limit]
    return cli_mod.suggest_commands(text, limit=limit)


def command_completer_class(cli_mod):
    """Build a completer that keeps the core path/model completion intact."""

    class ProductCommandCompleter(cli_mod.CommandCompleter):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor

            # Command names: include aliases and typo recovery in the popup,
            # rather than only after Enter reports an unknown command.
            if text.startswith("/") and " " not in text:
                for name in _ranked_commands(cli_mod, text):
                    yield Completion(
                        name,
                        start_position=-len(text),
                        display=name,
                        display_meta=cli_mod.COMMANDS.get(name, ""),
                    )
                return

            # Enumerated arguments should appear as soon as the space is
            # typed.  The old completer required one character first, which
            # made `/effort `, `/theme `, `/voice `, etc. look unsupported.
            if text.startswith("/") and " " in text:
                command, _, raw_arg = text.partition(" ")
                canonical = cli_mod.resolve_command(command) or command
                values = cli_mod._SUBCOMMAND_VALUES.get(canonical)
                if values is not None and " " not in raw_arg.strip():
                    prefix = raw_arg.strip().lower()
                    for value in values:
                        if value.lower().startswith(prefix):
                            yield Completion(
                                value,
                                start_position=-len(raw_arg.strip()),
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
    typed = cli_mod.command_hints(cli_mod._composer_text(repl))
    if typed:
        return "  ".join(typed)
    if getattr(repl.agent, "working_mode", "code") == "chat":
        return "Chat mode  ·  /code for project work"
    return "Chat or ask for project work  ·  / for commands"


def _home(self, model: str, workspace: str, *, mode: str = "agent",
          companion: str = "ready", version: str = "",
          show_companion: bool = False, show_art: bool = False,
          show_static_controls: bool = False) -> None:
    """A compact front door: Wynxo is an assistant first, not a dashboard."""
    self.refresh_size()
    if is_dumb_terminal() or self.narrow or not self.console.is_terminal:
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


def _install_prompts() -> None:
    from . import prompts

    prompts.CHAT_PROMPT = CHAT_PROMPT
    if _OLD_CODE_IDENTITY in prompts.BASE:
        prompts.BASE = prompts.BASE.replace(
            _OLD_CODE_IDENTITY, _NEW_CODE_IDENTITY, 1,
        )


def _install_router() -> None:
    global _ORIGINAL_SMALL_TALK
    from . import agent

    if _ORIGINAL_SMALL_TALK is None:
        _ORIGINAL_SMALL_TALK = agent.is_small_talk

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

    # Repl instances are created after bootstrap finishes, so replacing the
    # class here changes completion without mutating an already-running prompt.
    cli_mod.CommandCompleter = command_completer_class(cli_mod)

    # clean_ui installs the final home method before this layer. Keep it as the
    # fallback for narrow/dumb terminals, then use the compact assistant-first
    # home on normal terminals.
    _PREV_HOME = ui_mod.UI.home
    ui_mod.UI.home = _home

    # clean_ui's prompt renderer calls product_ui._prompt_hint dynamically, so
    # changing this one helper updates both product and clean shells.
    product_ui._prompt_hint = _prompt_hint

    _INSTALLED = True
