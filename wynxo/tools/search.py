"""Finding things: by filename, and by content."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from ..schema import Field, Schema
from .base import Tool, ToolResult
from .files import IGNORED, _looks_binary, _read_text

MAX_MATCHES = 200
MAX_FILES_SCANNED = 20_000


class GlobInput(Schema):
    pattern = Field(str, "Glob pattern, e.g. '**/*.py' or 'src/**/test_*.ts'. "
                         "A pattern with no '/' matches the file name "
                         "anywhere in the tree.")
    path = Field(str, "Directory to search from.", default=".")


def _matcher(pattern: str):
    """A glob pattern as a predicate on a relative path.

    Written out rather than handed to ``fnmatch``, which has no notion of a
    path separator: its ``*`` matches ``/`` like any other character. So
    ``src/*.py`` matched ``src/deep/nested/thing.py``, and ``*.py`` matched
    every Python file in the tree rather than the ones beside you. And the
    pattern people actually type, ``**/*.py``, matched *nothing* in the root
    directory, because ``**`` collapses to nothing and the literal ``/``
    then has to be there. All three at once meant the answer was roughly
    "files whose name ends in .py, somewhere", whatever was asked.

    The segment rules, which are what every other glob implements:

        **/     zero or more directories
        **      everything below here
        *       anything within one name
        ?       one character within one name

    A pattern with no separator in it is matched against the file name
    anywhere in the tree, which is what ``fd '*.py'`` and ``git ls-files
    '*.py'`` do and what someone typing it means.
    """
    bare = "/" not in pattern
    body, index, length = [], 0, len(pattern)
    while index < length:
        char = pattern[index]
        if pattern.startswith("**/", index):
            body.append("(?:[^/]+/)*")
            index += 3
        elif pattern.startswith("**", index):
            body.append(".*")
            index += 2
        elif char == "*":
            body.append("[^/]*")
            index += 1
        elif char == "?":
            body.append("[^/]")
            index += 1
        elif char == "[":
            close = pattern.find("]", index + 1)
            if close == -1:
                body.append(re.escape(char))
                index += 1
            else:
                inner = pattern[index + 1:close].replace("\\", "\\\\")
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                body.append(f"[{inner}]")
                index = close + 1
        else:
            body.append(re.escape(char))
            index += 1
    compiled = re.compile("".join(body) + r"\Z")

    def matches(relative: str) -> bool:
        if compiled.match(relative):
            return True
        return bare and bool(compiled.match(relative.rsplit("/", 1)[-1]))

    return matches


def _display_path(tool: Tool, path: Path) -> str:
    """Stable workspace path for model/tool output on every OS.

    Tool paths are protocol text, not native shell paths. Returning backslashes
    on Windows made the same glob or grep produce different strings depending
    on the runner and also broke patterns written with the documented `/`
    separator. Keep native ``Path`` objects for filesystem access and normalize
    only at the display/protocol boundary.
    """
    return tool.relative(path).replace("\\", "/")


class Glob(Tool):
    name = "glob"
    description = "Find files by name pattern. Use this to locate files before reading them."
    Input = GlobInput

    async def run(self, args: GlobInput) -> ToolResult:
        root = self.resolve_path(args.path)
        if not root.is_dir():
            return ToolResult.failure(f"{self.relative(root)} is not a directory.")

        matches = await asyncio.to_thread(self._collect, root, args.pattern)
        if not matches:
            return ToolResult.success(
                f"No files match {args.pattern!r}. Try a broader pattern, "
                "or list_dir to see the layout."
            )
        shown = matches[:MAX_MATCHES]
        body = "\n".join(shown)
        if len(matches) > len(shown):
            body += f"\n... and {len(matches) - len(shown)} more"
        return ToolResult.success(body, display=f"glob {args.pattern} -> {len(matches)} files")

    def _collect(self, root: Path, pattern: str) -> list[str]:
        matches = _matcher(pattern.replace("\\", "/").lstrip("./"))
        out: list[tuple[str, float]] = []
        for path in root.rglob("*"):
            if len(out) > MAX_MATCHES * 4:
                break
            try:
                is_file = path.is_file()
            except OSError:
                continue
            if not is_file:
                continue
            if any(part in IGNORED or part.startswith(".") for part in path.parts):
                continue
            rel = _display_path(self, path)
            if matches(rel):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                out.append((rel, mtime))
        # Most-recently-modified first: when a model is hunting for the file it
        # just changed, that is nearly always the one it wants. Keep the native
        # path only for stat above; rebuilding a Windows path from the normalized
        # display string used to make sorting itself platform-dependent.
        out.sort(key=lambda item: -item[1])
        return [rel for rel, _mtime in out]


class GrepInput(Schema):
    pattern = Field(str, "Text or regular expression to search for.")
    literal = Field(bool, "Treat pattern literally instead of as regex.", default=False)
    path = Field(str, "Directory or file to search in.", default=".")
    glob = Field(str, "Only search files matching this pattern, e.g. '*.py'.", default="")
    ignore_case = Field(bool, "Case-insensitive search.", default=False)
    context = Field(int, "Lines of context around each match.", default=0, ge=0, le=5)


class Grep(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression. This is the fastest way "
        "to find where something is defined or used. Prefer it over reading files "
        "one by one."
    )
    Input = GrepInput

    async def run(self, args: GrepInput) -> ToolResult:
        target = self.resolve_path(args.path)
        if not target.exists():
            return ToolResult.failure(f"{self.relative(target)} does not exist.")
        try:
            flags = re.IGNORECASE if args.ignore_case else 0
            regex = re.compile(re.escape(args.pattern) if args.literal else args.pattern, flags)
        except re.error as exc:
            return ToolResult.failure(f"Invalid regex {args.pattern!r}: {exc}")

        hits, scanned = await asyncio.to_thread(self._scan, target, regex, args.glob, args.context)
        if not hits:
            where = f" in {args.glob}" if args.glob else ""
            return ToolResult.success(
                f"No matches for {args.pattern!r}{where} ({scanned} files searched)."
            )
        body, masked = self.shield.clean("\n".join(hits[:MAX_MATCHES]))
        if len(hits) > MAX_MATCHES:
            body += f"\n... and {len(hits) - MAX_MATCHES} more matches"
        if masked:
            body += (f"\n\n[{masked} credential"
                     f"{'s' if masked != 1 else ''} in these matches were "
                     f"masked before they reached you.]")
        return ToolResult.success(
            body,
            display=f"grep {args.pattern} -> {len(hits)} matches"
                    + (f", {masked} masked" if masked else ""),
            matches=len(hits), scanned=scanned, literal=args.literal,
            truncated=len(hits) > MAX_MATCHES,
        )

    def _scan(self, target: Path, regex: re.Pattern, glob: str, context: int):
        files = [target] if target.is_file() else self._candidates(target, glob)
        hits: list[str] = []
        scanned = 0
        for path in files:
            if len(hits) > MAX_MATCHES * 2 or scanned > MAX_FILES_SCANNED:
                break
            if _looks_binary(path):
                continue
            if self.shield.blocks(path):
                continue
            try:
                lines = _read_text(path).splitlines()
            except OSError:
                continue
            scanned += 1
            rel = _display_path(self, path)
            for i, line in enumerate(lines):
                if not regex.search(line):
                    continue
                if context:
                    lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                    for j in range(lo, hi):
                        marker = ":" if j == i else "-"
                        hits.append(f"{rel}{marker}{j + 1}{marker}{lines[j][:400]}")
                    hits.append("--")
                else:
                    hits.append(f"{rel}:{i + 1}:{line.strip()[:400]}")
        return hits, scanned

    def _candidates(self, root: Path, glob: str) -> list[Path]:
        matches = _matcher(glob.replace("\\", "/").lstrip("./")) if glob else None
        out = []
        for path in root.rglob("*"):
            try:
                is_file = path.is_file()
            except OSError:
                continue
            if not is_file:
                continue
            if any(part in IGNORED or part.startswith(".") for part in path.parts):
                continue
            if matches and not matches(_display_path(self, path)):
                continue
            out.append(path)
        return out
