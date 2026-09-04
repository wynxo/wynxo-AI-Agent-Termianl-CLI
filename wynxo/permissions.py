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

    @staticmethod
    def _as_asked(tool_name: str, args: dict) -> tuple[str, dict]:
        """What the user is really being asked about.

        A launch that carries a command is a command, whatever the tool is
        called. It runs outside every guard the shell tool has -- no output
        ceiling, no workspace, no read-only test -- in a window wynxo does
        not own, so it is asked about on the same terms and never waved
        through as "just opening an application".

        Both the asking and the remembering go through here. They did not
        once: needs_prompt() did the rewrite and remember() did not, so
        "always allow" on `konsole running python3 main.py` remembered the
        *tool* -- it asked again for that same command every time, and in
        exchange granted silent launching of every other application on the
        machine, which is not what anyone approved.
        """
        if tool_name == "launch_application" and str(
                args.get("command", "")).strip():
            return "shell", {"command": args["command"]}
        return tool_name, args

    def needs_prompt(self, tool_name: str, mutating: bool, args: dict,
                     internal: bool = False) -> bool:
        if self.mode is Mode.YOLO:
            return False
        if not mutating or internal:
            return False

        tool_name, args = self._as_asked(tool_name, args)

        if self.mode in (Mode.AUTO, Mode.REVIEW):
            # Edits in scope go through; anything that runs a command or
            # reaches off the machine still asks. Review mode defers the
            # question rather than skipping it -- the whole diff is put up
            # once the turn finishes.
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
        tool_name, args = self._as_asked(tool_name, args)
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


def _sed_writes_in_place(args: list[str]) -> bool:
    # sed's -i is only ever a short flag, optionally bundled with others
    # (-ni, -i.bak) or standing alone -- never a long --option, so any
    # single-dash token with an 'i' among its letters means in-place.
    return any(a.startswith("-") and not a.startswith("--") and "i" in a[1:]
              for a in args)


def _awk_writes_in_place(args: list[str]) -> bool:
    # gawk's in-place extension is loaded as `-i inplace` (two tokens).
    return "inplace" in args


def _find_mutates(args: list[str]) -> bool:
    # -delete and -exec/-execdir/-ok/-okdir run arbitrary actions per match
    # with no shell metacharacter in sight; -fprint(f) writes a file too.
    return bool({"-delete", "-exec", "-execdir", "-ok", "-okdir",
                "-fprint", "-fprintf"} & set(args))


def _writes_to_o_flag(args: list[str]) -> bool:
    # `sort -o file file` and `tree -o file` write to an arbitrary path --
    # including, for sort, back over one of the files it just read.
    return any(a in ("-o", "--output") or a.startswith("--output=") for a in args)


# Commands in SAFE_COMMANDS whose normal, read-only form has a flag that
# turns it into a write instead -- checked before the SAFE_COMMANDS lookup
# so that flag is never missed just because the bare command name is safe.
_HIDDEN_WRITE_FLAGS = {
    "sed": _sed_writes_in_place,
    "awk": _awk_writes_in_place,
    "gawk": _awk_writes_in_place,
    "find": _find_mutates,
    "sort": _writes_to_o_flag,
    "tree": _writes_to_o_flag,
}


def _git_config_only_reads(args: list[str]) -> bool:
    """`git config` reads or writes depending on how many arguments it has.

    `git config user.email` prints the value; `git config user.email x` sets
    it, and with --global it sets it for every repository on the machine.
    Both were being waved through as "git config, that's a read".

    So only the forms that say out loud that they are reading count.
    """
    if not args:
        return True                      # bare `git config` prints usage
    reading = {"--get", "--get-all", "--get-regexp", "--list", "-l"}
    if any(a in reading or a.startswith("--get") for a in args):
        return True
    # One argument that is a name, not a flag: `git config user.email`.
    named = [a for a in args if not a.startswith("-")]
    return len(named) == 1 and len(args) == 1


def is_read_only_command(command: str) -> bool:
    """Whether a command only observes. Conservative: unsure means no."""
    text = command.strip()
    if not text:
        return False
    # Anything chained or redirected is analysed as a whole and refused: the
    # safe half tells you nothing about the other half.
    #
    # The newline is not a nicety. Every shell treats it as a command
    # separator, so "ls\nrm -rf build" was read as the safe command "ls" and
    # run without asking. A bare & separates commands too -- "ls & rm -rf
    # build" backgrounds the first and runs the second. Both were missing.
    if any(token in text for token in
           ("&&", "||", ";", "|", ">", "<", "`", "$(", "&", "\n", "\r")):
        return False

    parts = text.split()
    head = parts[0].lower()
    if head in _HIDDEN_WRITE_FLAGS and _HIDDEN_WRITE_FLAGS[head](parts[1:]):
        return False
    if head in SAFE_COMMANDS:
        return True
    if head in SAFE_SUBCOMMANDS and len(parts) > 1:
        if parts[1].lower() not in SAFE_SUBCOMMANDS[head]:
            return False
        if head == "git" and parts[1].lower() == "config":
            return _git_config_only_reads(parts[2:])
        return True
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
