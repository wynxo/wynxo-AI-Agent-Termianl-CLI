"""Tool registry."""

from __future__ import annotations

from pathlib import Path

from ..memory import Memory
from ..scope import Boundary
from .base import Tool, ToolResult
from .files import EditFile, ListDir, MultiEdit, ReadFile, WriteFile
from .dev import Git, RunTests
from ..secrets import Shield
from .memory_tool import Remember
from .search import Glob, Grep
from . import shell as _shell_module
from .shell import BackgroundPoll, Shell
from .shell_guard import hard_refusal as _portable_hard_refusal
from .todo import TodoWrite
from .apps import LaunchApplication, ListApplications
from .appcatalog import ApplicationCatalog
from .navigation_tool import NavigateSymbols
from .references_tool import FindReferences
from .github_tool import GitHubRead, GitHubWrite
from .web import WebSearch

# Shell.run resolves ``hard_refusal`` through its module globals at call time.
# Keep the execution implementation in shell.py while using one host-independent
# parser for the safety decision. This also means direct imports of
# ``wynxo.tools.shell.hard_refusal`` see the same guard on every platform.
_shell_module.hard_refusal = _portable_hard_refusal

__all__ = ["Tool", "ToolResult", "Registry", "build_registry"]


class Registry:
    """The tools the agent may call, and the ones it may not.

    Held apart rather than dropped: the agent is offered only what can
    work, so no request pays for a schema the model would be refused for
    using -- but /tools still lists what is missing and why, because "the
    GitHub tools are not there" is a confusing thing to discover from a
    model saying it cannot do something.
    """

    def __init__(self, tools: list[Tool]):
        self._tools = {}
        self.withheld: dict[str, str] = {}
        for tool in tools:
            if (reason := tool.unavailable()):
                self.withheld[tool.name] = reason
            else:
                self._tools[tool.name] = tool

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
    app_catalog: ApplicationCatalog | None = None,
    shell_max_output: int | None = None,
) -> Registry:
    app_catalog = app_catalog or ApplicationCatalog()

    tools: list[Tool] = [
        ReadFile(workspace, boundary, shield),
        WriteFile(workspace, boundary, shield),
        EditFile(workspace, boundary, shield),
        MultiEdit(workspace, boundary, shield),
        ListDir(workspace, boundary, shield),
        Glob(workspace, boundary, shield),
        Grep(workspace, boundary, shield),
        TodoWrite(workspace, boundary, shield),
        ListApplications(workspace, boundary, shield, catalog=app_catalog),
        LaunchApplication(workspace, boundary, shield, catalog=app_catalog),
        NavigateSymbols(workspace, boundary, shield),
        FindReferences(workspace, boundary, shield),
        GitHubRead(workspace, boundary, shield),
        GitHubWrite(workspace, boundary, shield),
        Remember(workspace, boundary, memory, shield),
        Git(workspace, boundary, shield),
        RunTests(workspace, boundary, shield),
        WebSearch(workspace, boundary, shield),
    ]
    if allow_shell:
        tools.append(BackgroundPoll(workspace, boundary, shield))
        kwargs = {}
        if shell_max_output:
            kwargs["max_output"] = shell_max_output
        tools.append(Shell(workspace, boundary, shield, **kwargs))
    return Registry(tools)
