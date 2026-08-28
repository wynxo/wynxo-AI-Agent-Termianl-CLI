from __future__ import annotations

from ..navigation import symbols
from ..schema import Field, Schema
from .base import Tool, ToolResult


class SymbolsInput(Schema):
    path = Field(str, "Python file to inspect for definitions.")


class NavigateSymbols(Tool):
    name = "find_symbols"
    description = "Find Python functions, classes, and methods without reading the whole file."
    Input = SymbolsInput

    async def run(self, args: SymbolsInput) -> ToolResult:
        path = self.resolve_path(args.path)
        if not path.is_file():
            return ToolResult.failure(f"{self.relative(path)} is not a file.")
        found = symbols(path)
        if not found:
            return ToolResult.success(f"No navigable Python symbols found in {self.relative(path)}.", path=self.relative(path), symbols=0)
        body = "\n".join(f"{item['kind']} {item['name']}:{item['line']}" for item in found)
        return ToolResult.success(body, display=f"found {len(found)} symbols in {self.relative(path)}", path=self.relative(path), symbols=len(found))
