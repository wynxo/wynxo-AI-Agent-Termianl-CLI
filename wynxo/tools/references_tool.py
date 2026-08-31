"""Relationships between parts of a project, as an agent tool.

``find_symbols`` answers where a name is *defined*. The questions left
over are all about how the pieces connect -- what calls this, what imports
this, what extends this, what tests cover this -- and the only tool for
them was grep, which answers with text matches. For a common name that is
hundreds of truncated lines whose relationship to the question is
unstated; for "what imports this module" it was measured at 5% signal.
"""

from __future__ import annotations

from ..navigation import (covering_tests, importers, references, subclasses)
from ..schema import Field, Schema
from .base import Tool, ToolResult

MAX_HITS = 30


class ReferencesInput(Schema):
    relation = Field(
        str,
        "What you want to know. 'callers': where a function or method is "
        "called. 'uses': every mention of a name, calls included -- wider, "
        "for when callers finds nothing. 'subclasses': classes extending a "
        "class. 'importers': files importing a module. 'tests': test files "
        "that cover a source file, found through imports rather than by "
        "name.",
        choices=("callers", "uses", "subclasses", "importers", "tests"),
        default="callers")
    name = Field(
        str,
        "The function, method or class to ask about. For 'importers' and "
        "'tests' give the module or file instead, e.g. 'wynxo/session.py'.",
        default="")


class FindReferences(Tool):
    name = "find_references"
    description = (
        "Answer how parts of the project relate: what calls a function, what "
        "subclasses a class, what imports a module, what tests cover a file. "
        "Use this instead of grepping a name and reading the matches -- grep "
        "returns text, this returns the relationship.")
    Input = ReferencesInput

    async def run(self, args: ReferencesInput) -> ToolResult:
        wanted = (args.name or "").strip()
        if not wanted:
            return ToolResult.failure(
                "find_references needs 'name': the function, class or module "
                "to ask about.")
        relation = (args.relation or "callers").strip() or "callers"
        handler = {
            "callers": self._callers,
            "uses": self._uses,
            "subclasses": self._subclasses,
            "importers": self._importers,
            "tests": self._tests,
        }.get(relation)
        if handler is None:
            return ToolResult.failure(
                f"Unknown relation {relation!r}. Use one of: callers, uses, "
                f"subclasses, importers, tests.")
        return handler(wanted)

    # -- the relations -----------------------------------------------------

    def _callers(self, wanted: str) -> ToolResult:
        hits, total, sampled = references(self.workspace, wanted,
                                          kinds=("call",), limit=MAX_HITS + 1)
        if not hits:
            # Falling back rather than answering "nothing" is the difference
            # between a useful tool and a misleading one: a class is never
            # "called", and neither is a constant.
            wider, wider_total, wider_sampled = references(
                self.workspace, wanted, limit=MAX_HITS + 1)
            if wider:
                return self._render(
                    wider, wider_total, wider_sampled, wanted,
                    f"Nothing calls '{wanted}' directly. It is referenced in "
                    f"{wider_total} place(s):")
            return self._nothing(wanted, "calls")
        return self._render(hits, total, sampled, wanted,
                            f"{total} call(s) of '{wanted}':")

    def _uses(self, wanted: str) -> ToolResult:
        hits, total, sampled = references(self.workspace, wanted,
                                          limit=MAX_HITS + 1)
        if not hits:
            return self._nothing(wanted, "references")
        return self._render(hits, total, sampled, wanted,
                            f"{total} reference(s) to '{wanted}':")

    def _subclasses(self, wanted: str) -> ToolResult:
        hits = subclasses(self.workspace, wanted)
        hits = [hit for hit in hits if not self._hidden(hit.path)]
        if not hits:
            return self._nothing(wanted, "subclasses")
        body = "\n".join(f"{hit.path}:{hit.line}  class {hit.context}({wanted})"
                         for hit in hits[:MAX_HITS])
        if len(hits) > MAX_HITS:
            body += f"\n... and {len(hits) - MAX_HITS} more"
        return ToolResult.success(
            body, display=f"{len(hits)} subclasses of {wanted}",
            relation="subclasses", hits=len(hits))

    def _importers(self, wanted: str) -> ToolResult:
        paths = [path for path in importers(self.workspace, wanted)
                 if not self._hidden(path)]
        if not paths:
            return self._nothing(wanted, "importers")
        body = "\n".join(paths[:MAX_HITS])
        if len(paths) > MAX_HITS:
            body += f"\n... and {len(paths) - MAX_HITS} more"
        return ToolResult.success(
            body, display=f"{len(paths)} files import {wanted}",
            relation="importers", hits=len(paths))

    def _tests(self, wanted: str) -> ToolResult:
        from pathlib import Path

        paths = covering_tests(self.workspace, [Path(wanted)])
        paths = [path for path in paths if not self._hidden(path)]
        if not paths:
            return ToolResult.success(
                f"No test file imports '{wanted}', directly or through "
                f"another module. Either it is untested, or its tests reach "
                f"it some way the import graph cannot see -- run the whole "
                f"suite rather than assuming it is covered.",
                display=f"no tests found for {wanted}",
                relation="tests", hits=0)
        body = "\n".join(paths[:MAX_HITS])
        if len(paths) > MAX_HITS:
            body += f"\n... and {len(paths) - MAX_HITS} more"
        return ToolResult.success(
            body, display=f"{len(paths)} test files cover {wanted}",
            relation="tests", hits=len(paths))

    # -- shared ------------------------------------------------------------

    def _hidden(self, path: str) -> bool:
        """A path the shield keeps out of the model's context."""
        return bool(self.shield.blocks(self.workspace / path))

    def _nothing(self, wanted: str, relation: str) -> ToolResult:
        return ToolResult.success(
            f"No {relation} of '{wanted}' in this project. Check the spelling "
            f"with find_symbols, or it may genuinely be unused.",
            display=f"no {relation} of {wanted}", relation=relation, hits=0)

    def _render(self, hits, total: int, sampled: bool, wanted: str,
                headline: str) -> ToolResult:
        hits = [hit for hit in hits if not self._hidden(hit.path)]
        shown = hits[:MAX_HITS]
        lines = [headline]
        lines.extend(f"{hit.path}:{hit.line}  in {hit.context or '<module>'}"
                     + ("" if hit.kind == "call" else f"  [{hit.kind}]")
                     for hit in shown)
        if total > len(shown):
            lines.append(
                f"... and {total - len(shown)} more."
                + (" The ones above are a sample, not the first few: this "
                   "name is used too widely to index in full, so read them "
                   "as examples and narrow the question."
                   if sampled else
                   " Narrow it with a path, or read the ones above."))
        body, masked = self.shield.clean("\n".join(lines))
        if masked:
            body += (f"\n\n[{masked} credential{'s' if masked != 1 else ''} "
                     f"were masked before this reached you.]")
        return ToolResult.success(
            body,
            display=(f"{wanted} used at {shown[0].path}:{shown[0].line}"
                     if len(shown) == 1 else f"{total} references to {wanted}"),
            relation="references", hits=len(shown), total=total)
