from __future__ import annotations

from ..navigation import Definition, find, symbols
from ..schema import Field, Schema
from .base import Tool, ToolResult

MAX_HITS = 25


class SymbolsInput(Schema):
    name = Field(
        str,
        "A definition to locate anywhere in the project -- a function, "
        "class, method or constant. Use this to answer 'where is X "
        "defined?' instead of grepping, which returns every call site as "
        "well. Write a method as 'ClassName.method' when several classes "
        "define the same one.",
        default="")
    path = Field(
        str,
        "A file to list the definitions of. Without 'name' this is an "
        "outline of the file; with it, the search is limited to this file.",
        default="")


class NavigateSymbols(Tool):
    name = "find_symbols"
    description = (
        "Locate where something is defined, or outline a file. Give 'name' "
        "to find a definition anywhere in the project without knowing which "
        "file it is in; give 'path' to list what one file defines. Prefer "
        "this over grep for definitions -- grep also returns every use.")
    Input = SymbolsInput

    async def run(self, args: SymbolsInput) -> ToolResult:
        wanted = (args.name or "").strip()
        where = (args.path or "").strip()
        if not wanted and not where:
            return ToolResult.failure(
                "find_symbols needs 'name' (what to look for, anywhere in "
                "the project) or 'path' (a file to outline), or both.")
        if wanted:
            return self._search(wanted, where)
        return self._outline(where)

    # -- the two questions -------------------------------------------------

    def _search(self, wanted: str, where: str) -> ToolResult:
        hits = find(self.workspace, wanted, limit=MAX_HITS + 1)
        if where:
            prefix = self.relative(self.resolve_path(where)).replace("\\", "/")
            hits = [h for h in hits if h.path == prefix or h.path.startswith(prefix + "/")]
        # An index is a read with extra steps. A signature carries default
        # values, and naming a definition inside a credentials file is a way
        # of reading it that nobody thinks to check.
        hits = [h for h in hits if not self.shield.blocks(self.workspace / h.path)]
        if not hits:
            # An empty result is a real answer, and saying what to do next
            # keeps the model from re-running the same search.
            scope = f" under {where}" if where else ""
            return ToolResult.success(
                f"No definition of '{wanted}'{scope} in this project. It may "
                f"be imported from a dependency, built in, or spelled "
                f"differently -- grep for it to see where it is used.",
                display=f"no definition of {wanted}", query=wanted, hits=0)

        more = len(hits) > MAX_HITS
        hits = hits[:MAX_HITS]
        body, masked = self.shield.clean("\n".join(hit.describe() for hit in hits))
        if masked:
            body += (f"\n\n[{masked} credential{'s' if masked != 1 else ''} in "
                     f"these signatures were masked before they reached you.]")
        if more:
            body += (f"\n... and more. '{wanted}' matches many definitions; "
                     f"narrow it with 'ClassName.{wanted}' or a path.")
        display = (f"{wanted} defined at {hits[0].path}:{hits[0].line}"
                   if len(hits) == 1 else
                   f"{len(hits)} definitions of {wanted}")
        return ToolResult.success(body, display=display, query=wanted,
                                  hits=len(hits),
                                  paths=[hit.path for hit in hits])

    def _outline(self, where: str) -> ToolResult:
        path = self.resolve_path(where)
        if not path.is_file():
            return ToolResult.failure(f"{self.relative(path)} is not a file.")
        if self.shield.blocks(path):
            return ToolResult.failure(
                f"{self.relative(path)} looks like a credentials file; its "
                f"contents are not available.")
        found = symbols(path)
        if not found:
            return ToolResult.success(
                f"No navigable Python symbols found in {self.relative(path)}.",
                path=self.relative(path), symbols=0)
        body = "\n".join(f"{item['kind']} {item['name']}:{item['line']}"
                         for item in found)
        return ToolResult.success(
            body, display=f"found {len(found)} symbols in {self.relative(path)}",
            path=self.relative(path), symbols=len(found))


__all__ = ["NavigateSymbols", "Definition"]
