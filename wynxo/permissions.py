"""Deciding whether a tool call is allowed to happen.

The rule is simple: reads are free, writes ask. What makes it usable rather
than infuriating is remembering the answer -- per tool, or per exact command,
for the rest of the session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .scope import Mode


class Decision(Enum):
    ALLOW = "allow"
    ALLOW_ALWAYS = "allow_always"
    DENY = "deny"
    ABORT = "abort"


# Read-only commands that are pointless to prompt for. Matched on the first
# word (and second, for subcommands) so `git status` is free but `git push`
# is not.
SAFE_COMMANDS = {
    "ls", "dir", "pwd", "cat", "head", "tail", "wc", "file", "stat", "which",
    "where", "echo", "date", "whoami", "env", "printenv", "tree", "du", "df",
    "grep", "rg", "find", "fd", "diff", "sort", "uniq", "cut", "awk", "sed",
    "python --version", "node --version", "go version", "cargo --version",
}
SAFE_SUBCOMMANDS = {
    "git": {"status", "log", "diff", "show", "branch", "remote", "config",
            "rev-parse", "ls-files", "blame", "describe", "stash list"},
    "npm": {"ls", "list", "view", "outdated"},
    "pip": {"list", "show", "freeze"},
    "cargo": {"tree", "metadata"},
    "docker": {"ps", "images", "version"},
    "kubectl": {"get", "describe", "logs"},
}

# Things that reach outside the machine or rewrite history. Always prompt,
# even when the user has said "always allow shell", because the blast radius
# is not local.
ALWAYS_CONFIRM = re.compile(
    r"\b(git\s+push|git\s+reset\s+--hard|git\s+clean|git\s+rebase|"
    r"npm\s+publish|cargo\s+publish|twine\s+upload|"
    r"curl|wget|ssh|scp|rsync|nc|"
    r"sudo|doas|chmod\s+777|chown|"
    r"docker\s+(run|rm|rmi)|kubectl\s+(apply|delete)|terraform\s+apply)\b",
    re.IGNORECASE,
)


@dataclass
class PermissionStore:
    """Remembers what the user has already said yes to, for this session."""

    always_allowed_tools: set[str] = field(default_factory=set)
    always_allowed_commands: set[str] = field(default_factory=set)
    denied_this_session: list[str] = field(default_factory=list)
    mode: Mode = Mode.MANUAL

    @property
    def yolo(self) -> bool:
        return self.mode is Mode.YOLO

    @yolo.setter
    def yolo(self, value: bool) -> None:
        self.mode = Mode.YOLO if value else Mode.MANUAL

    def preapprove(self, names: list[str]) -> None:
        self.always_allowed_tools.update(names)

    def blocked(self, tool_name: str, mutating: bool, internal: bool = False) -> str | None:
        """Whether the current mode forbids this outright.

        Plan mode is the only one that refuses rather than asks: the point of
        it is that nothing changes, so a prompt would defeat it. Internal
        writes are exempt -- a read-only session should still be able to note
        what it worked out.
        """
        if self.mode is Mode.PLAN and mutating and not internal:
            return (
                f"{tool_name} would change something, and wynxo is in plan mode "
                "(read-only). Investigate and describe what you would do instead. "
                "The user can switch with /mode auto or /mode manual."
            )
        return None

    def needs_prompt(self, tool_name: str, mutating: bool, args: dict,
                     internal: bool = False) -> bool:
        if self.mode is Mode.YOLO:
            return False
        if not mutating or internal:
            return False

        if self.mode is Mode.AUTO:
            # Edits in scope go through; anything that runs a command or
            # reaches off the machine still asks.
            if tool_name != "shell":
                return False

        if tool_name == "shell":
            command = str(args.get("command", "")).strip()
            if ALWAYS_CONFIRM.search(command):
                return True
            if command in self.always_allowed_commands:
                return False
            if is_read_only_command(command):
                return False
            if "shell" in self.always_allowed_tools:
                return False
            return True

        return tool_name not in self.always_allowed_tools

    def remember(self, tool_name: str, args: dict) -> None:
        if tool_name == "shell":
            command = str(args.get("command", "")).strip()
            # Remember the exact command, not "all shell commands" -- approving
            # `npm test` forever should not also approve `rm -rf build`.
            if command and not ALWAYS_CONFIRM.search(command):
                self.always_allowed_commands.add(command)
        else:
            self.always_allowed_tools.add(tool_name)

    def record_denial(self, tool_name: str, reason: str) -> None:
        self.denied_this_session.append(f"{tool_name}: {reason}")


def is_read_only_command(command: str) -> bool:
    """Whether a command only observes. Conservative: unsure means no."""
    text = command.strip()
    if not text:
        return False
    # Anything chained or redirected is analysed as a whole and refused: the
    # safe half tells you nothing about the other half.
    if any(token in text for token in ("&&", "||", ";", "|", ">", "<", "`", "$(")):
        return False

    parts = text.split()
    head = parts[0].lower()
    if head in SAFE_COMMANDS:
        return True
    if head in SAFE_SUBCOMMANDS and len(parts) > 1:
        return parts[1].lower() in SAFE_SUBCOMMANDS[head]
    if len(parts) == 2 and parts[1] in ("--version", "-v", "--help", "-h"):
        return True
    return False


def summarise_call(tool_name: str, args: dict, workspace=None) -> str:
    """A one-line description of what is about to happen.

    Paths are shown relative to the project. Models routinely pass absolute
    paths, and a line reading `read_file C:\\Users\\you\\proj\\src\\a.py`
    buries the only part that matters.
    """
    if tool_name == "shell":
        return str(args.get("command", ""))
    if pattern := args.get("pattern"):
        where = args.get("glob") or args.get("path")
        shown = shorten_path(where, workspace) if where else ""
        return f"{pattern}" + (f"  in {shown}" if shown and shown != "." else "")
    if path := args.get("path"):
        return shorten_path(path, workspace)
    bits = [f"{k}={v!r}" for k, v in list(args.items())[:3]
            if not isinstance(v, (dict, list))]
    return ", ".join(bits)


def shorten_path(raw, workspace=None) -> str:
    """Relative to the workspace where possible, else just the tail."""
    from pathlib import Path

    text = str(raw)
    if workspace is None:
        return text
    try:
        return str(Path(text).resolve().relative_to(Path(workspace).resolve()))
    except (ValueError, OSError):
        pass
    parts = Path(text).parts
    return str(Path(*parts[-2:])) if len(parts) > 2 else text
