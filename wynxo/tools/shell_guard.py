"""Host-independent parsing for commands that must never be agent-executed.

The command string belongs to the shell Wynxo is about to invoke, not to the
Python host running the guard. Using ``shlex.split(..., posix=os.name != 'nt')``
made the safety decision change on Windows: POSIX shell snippets such as
``bash -c 'rm -rf /'`` kept their quote characters and the nested destructive
command was never inspected. Safety must be at least as strict on Windows as
it is on Linux, so inspect both common quoting interpretations and refuse when
either unambiguously exposes a destructive command.
"""

from __future__ import annotations

import re
import shlex

_EVERYTHING = {"/", "/*", "/.", "~", "~/", "~/*", "*", "/usr", "/etc",
               "/home", "/var", "/bin", "/lib", "/boot", "/sys", "/proc"}
_FORMATTERS = ("mkfs", "mke2fs", "mkdosfs", "newfs", "diskpart")
_TURNS_IT_OFF = {"shutdown", "reboot", "halt", "poweroff"}
_FORK_BOMB = ":(){:|:&};:"
_RAW_DISK = re.compile(r">\s*/dev/(sd|nvme|hd|disk|vd)", re.IGNORECASE)
_WINDOWS_ROOT = re.compile(r"^[a-z]:[\\/]?$", re.IGNORECASE)
_SEPARATORS = re.compile(r"&&|\|\||[;|&\n\r]")
_WRAPPERS = {"sudo", "doas", "env", "nice", "ionice", "nohup", "time",
             "command", "exec", "stdbuf", "xargs"}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "ash", "busybox"}


def _split_segments(line: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            elif char == "\\" and index + 1 < len(line) and quote == '"':
                index += 1
                current.append(line[index])
        elif char in "'\"":
            quote = char
            current.append(char)
        elif match := _SEPARATORS.match(line, index):
            segments.append("".join(current))
            current = []
            index = match.end()
            continue
        else:
            current.append(char)
        index += 1
    segments.append("".join(current))
    return [segment for segment in segments if segment.strip()]


def _tokenizations(segment: str) -> list[list[str]]:
    """Useful shell interpretations, independent of Python's host OS."""
    variants: list[list[str]] = []
    for posix in (True, False):
        try:
            tokens = shlex.split(segment, posix=posix)
        except ValueError:
            continue
        if tokens and tokens not in variants:
            variants.append(tokens)
    if not variants:
        crude = segment.split()
        if crude:
            variants.append(crude)
    return variants


def _basename(token: str) -> str:
    return token.lower().replace("\\", "/").rsplit("/", 1)[-1].strip("'\"")


def _unwrap(tokens: list[str]) -> list[str]:
    tokens = list(tokens)
    while tokens and _basename(tokens[0]) in _WRAPPERS:
        tokens = tokens[1:]
        while tokens and (tokens[0].startswith("-") or "=" in tokens[0]):
            tokens = tokens[1:]
    return tokens


def _script_of(tokens: list[str]) -> str | None:
    if not tokens or _basename(tokens[0]) not in _SHELLS:
        return None
    rest = tokens[1:]
    while rest:
        flag, rest = rest[0], rest[1:]
        clean = flag.strip("'\"")
        if not clean.startswith("-"):
            return None
        letters = clean[1:]
        if (not clean.startswith("--") and "c" in letters) or clean == "--command":
            return rest[0].strip("'\"") if rest else None
    return None


def _commands_in(line: str, depth: int = 0) -> list[list[str]]:
    out: list[list[str]] = []
    for segment in _split_segments(line):
        for tokens in _tokenizations(segment):
            out.append(tokens)
            script = _script_of(_unwrap(tokens))
            if script and depth < 4:
                out.extend(_commands_in(script, depth + 1))
    return out


def hard_refusal(line: str) -> str:
    """Why a command is too destructive for the agent to run, or ``""``."""
    if _FORK_BOMB in "".join(line.split()):
        return "a fork bomb"
    if _RAW_DISK.search(line):
        return "a write straight to a raw disk device"

    filesystem_roots = {entry.rstrip("/") for entry in _EVERYTHING}
    for raw_tokens in _commands_in(line):
        tokens = _unwrap(raw_tokens)
        if not tokens:
            continue
        head = _basename(tokens[0])
        arguments = [token.strip("'\"") for token in tokens[1:]
                     if not token.startswith("-")]

        if head == "rm" and any(
                arg.rstrip("/") in filesystem_roots or arg in _EVERYTHING
                for arg in arguments):
            return "a recursive delete of the whole filesystem"
        if head.startswith(_FORMATTERS):
            return "formatting a filesystem"
        if head == "format" and any(_WINDOWS_ROOT.match(arg) for arg in arguments):
            return "formatting a drive"
        if head in ("del", "rd", "rmdir") and any(
                _WINDOWS_ROOT.match(arg) for arg in arguments):
            return "deleting a whole drive"
        if head in _TURNS_IT_OFF:
            return "shutting the machine down"
        if head == "dd" and any(
                token.strip("'\"").startswith("of=/dev/") for token in tokens[1:]):
            return "writing straight to a device"
    return ""
