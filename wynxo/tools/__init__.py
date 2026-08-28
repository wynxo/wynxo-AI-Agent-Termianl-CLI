"""Tool registry."""

from __future__ import annotations

from pathlib import Path

from ..memory import Memory
from ..scope import Boundary
from .base import Tool, ToolResult
from .files import EditFile, ListDir, MultiEdit, ReadFile, WriteFile
from .fs_extra import FindFiles, ListDirectory, SearchText
from .dev import GitDiff, GitLog, GitStatus, RunTests
from ..secrets import Shield
from .memory_tool import Remember
from .search import Glob, Grep
from .shell import Shell
from .todo import TodoWrite
from .apps import OpenApplication

__all__ = ["Tool", "ToolResult", "Registry", "build_registry"]


class Registry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def ollama_schemas(self) -> list[dict]:
        return [t.ollama_schema() for t in self._tools.values()]

    def suggest(self, name: str) -> str | None:
        """A model that invents a tool name usually invents a near miss."""
        import difflib

        close = difflib.get_close_matches(name, self.names(), n=1, cutoff=0.6)
        return close[0] if close else None

    def describe(self) -> str:
        return "\n".join(f"  {t.signature()}\n      {t.description}" for t in self._tools.values())


def build_registry(
    workspace: Path,
    allow_shell: bool = True,
    boundary: Boundary | None = None,
    memory: Memory | None = None,
    shield: Shield | None = None,
) -> Registry:
    tools: list[Tool] = [
        ReadFile(workspace, boundary, shield),
        WriteFile(workspace, boundary, shield),
        EditFile(workspace, boundary, shield),
        MultiEdit(workspace, boundary, shield),
        ListDir(workspace, boundary, shield),
        Glob(workspace, boundary, shield),
        Grep(workspace, boundary, shield),
        TodoWrite(workspace, boundary, shield),
        OpenApplication(workspace, boundary, shield),
        Remember(workspace, boundary, memory, shield),
        ListDirectory(workspace, boundary, shield),
        FindFiles(workspace, boundary, shield),
        SearchText(workspace, boundary, shield),
        GitStatus(workspace, boundary, shield),
        GitDiff(workspace, boundary, shield),
        GitLog(workspace, boundary, shield),
        RunTests(workspace, boundary, shield),
    ]
    if allow_shell:
        tools.append(Shell(workspace, boundary, shield))
    return Registry(tools)
