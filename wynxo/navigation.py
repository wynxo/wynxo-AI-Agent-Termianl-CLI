"""Small, dependency-free code navigation helpers."""

from __future__ import annotations

import ast
from pathlib import Path


def symbols(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError, RecursionError):
        return []
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.append({"name": node.name, "kind": type(node).__name__, "line": node.lineno})
    return sorted(result, key=lambda item: int(item["line"]))


def affected_tests(changed: list[Path], tests_root: Path) -> list[Path]:
    """Rank likely tests without claiming dependency certainty."""
    if not tests_root.is_dir():
        return []
    names = {path.stem.lower(): path for path in tests_root.rglob("test_*.py") if path.is_file()}
    selected: list[Path] = []
    for source in changed:
        stem = source.stem.lower()
        for key, path in names.items():
            if stem in key or key.removeprefix("test_") in stem:
                if path not in selected:
                    selected.append(path)
    return selected
