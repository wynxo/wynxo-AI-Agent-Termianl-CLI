"""Finding things: by filename, and by content."""

from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult
from .files import IGNORED, _looks_binary, _read_text

MAX_MATCHES = 200
MAX_FILES_SCANNED = 20_000


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/test_*.ts'.")
    path: str = Field(".", description="Directory to search from.")


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
        out: list[str] = []
        for path in root.rglob("*"):
            if len(out) > MAX_MATCHES * 4:
                break
            if not path.is_file():
                continue
            if any(part in IGNORED or part.startswith(".") for part in path.parts):
                continue
            rel = self.relative(path)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
                out.append(rel)
        # Most-recently-modified first: when a model is hunting for the file it
        # just changed, that is nearly always the one it wants.
        out.sort(key=lambda r: -(root.joinpath(r).stat().st_mtime if (root / r).exists() else 0))
        return out


class GrepInput(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(".", description="Directory or file to search in.")
    glob: str = Field("", description="Only search files matching this pattern, e.g. '*.py'.")
    ignore_case: bool = Field(False)
    context: int = Field(0, ge=0, le=5, description="Lines of context around each match.")


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
            regex = re.compile(args.pattern, flags)
        except re.error as exc:
            return ToolResult.failure(f"Invalid regex {args.pattern!r}: {exc}")

        hits, scanned = await asyncio.to_thread(self._scan, target, regex, args.glob, args.context)
        if not hits:
            where = f" in {args.glob}" if args.glob else ""
            return ToolResult.success(
                f"No matches for {args.pattern!r}{where} ({scanned} files searched)."
            )
        body = "\n".join(hits[:MAX_MATCHES])
        if len(hits) > MAX_MATCHES:
            body += f"\n... and {len(hits) - MAX_MATCHES} more matches"
        return ToolResult.success(body, display=f"grep {args.pattern} -> {len(hits)} matches")

    def _scan(self, target: Path, regex: re.Pattern, glob: str, context: int):
        files = [target] if target.is_file() else self._candidates(target, glob)
        hits: list[str] = []
        scanned = 0
        for path in files:
            if len(hits) > MAX_MATCHES * 2 or scanned > MAX_FILES_SCANNED:
                break
            if _looks_binary(path):
                continue
            try:
                lines = _read_text(path).splitlines()
            except OSError:
                continue
            scanned += 1
            rel = self.relative(path)
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
        out = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED or part.startswith(".") for part in path.parts):
                continue
            if glob and not (fnmatch.fnmatch(path.name, glob) or fnmatch.fnmatch(self.relative(path), glob)):
                continue
            out.append(path)
        return out
