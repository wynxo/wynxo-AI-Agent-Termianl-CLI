"""Code navigation: what a file defines, and where a name is defined.

"Where is X defined?" is the question a coding agent asks most often and
the one it was worst at. Grep answers it with every *use* of the name --
for a common one that is forty lines of call sites containing a single
definition, and the model has to read all forty to find the one. This
module answers it directly.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import projectmap

MAX_INDEX_FILES = 3000
"""Far above the map's own limit. The map is a summary that has to stay
readable; the index is a lookup table, and a definition missing from it is
a wrong answer rather than an abbreviated one."""

MAX_INDEX_BYTES = 1_000_000
"""Skip a file bigger than this. A megabyte of generated source is not
where anyone is looking for a definition, and parsing it is what makes an
index build slow enough to notice."""

CACHE_TTL = 2.0
"""Seconds an index may be reused without re-checking the tree. A single
turn asks several navigation questions in a row about a tree that is not
changing between them."""


def symbols(path: Path) -> list[dict[str, object]]:
    """Definitions in one Python file, oldest callers' shape kept."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found = [d for d in _python_definitions(text) if d.kind != "constant"]
    return sorted(
        ({"name": d.name, "kind": _LEGACY_KIND[d.kind], "line": d.line}
         for d in found),
        key=lambda item: int(item["line"]),
    )


_LEGACY_KIND = {"function": "FunctionDef", "async function": "AsyncFunctionDef",
                "class": "ClassDef", "method": "FunctionDef",
                "async method": "AsyncFunctionDef", "constant": "Assign"}


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


# -- definitions --------------------------------------------------------------


@dataclass(frozen=True)
class Definition:
    """One place a name is introduced."""

    name: str
    kind: str
    path: str
    """Relative to the root the index was built from, with forward slashes,
    so an answer reads the same on Windows as it does anywhere else."""

    line: int
    parent: str = ""
    """The enclosing class, for a method. Two classes can both define
    ``run``, and saying which is the difference between an answer and a
    list."""

    signature: str = ""

    def qualified(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def describe(self) -> str:
        """One line: where it is, and enough of what it is to often stop
        reading here. A signature already says ``def`` or ``class``, so the
        kind is only spelled out when there is no signature to say it."""
        what = self.signature or f"{self.kind} {self.qualified()}"
        return f"{self.path}:{self.line}  {what}"


def definitions_in(text: str, language: str) -> list[Definition]:
    """Every definition in one file's text. Never raises: a file that does
    not parse contributes nothing, which is better than failing a search
    across a tree that happens to contain one broken file."""
    try:
        if language == "python":
            return _python_definitions(text)
        return _regex_definitions(text, language)
    except Exception:
        return []


def _python_definitions(text: str) -> list[Definition]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    found: list[Definition] = []

    def visit(node: ast.AST, parent: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent else "function"
                if isinstance(child, ast.AsyncFunctionDef):
                    kind = "async " + kind
                found.append(Definition(
                    name=child.name, kind=kind, path="", line=child.lineno,
                    parent=parent, signature=_signature(child, parent)))
                visit(child, "")      # a closure belongs to no class
            elif isinstance(child, ast.ClassDef):
                found.append(Definition(
                    name=child.name, kind="class", path="", line=child.lineno,
                    parent=parent, signature=f"class {child.name}"))
                visit(child, child.name)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)) and not parent:
                for target in _targets(child):
                    if target.isupper() and len(target) > 1:
                        found.append(Definition(name=target, kind="constant",
                                                path="", line=child.lineno))
    visit(tree, "")
    return found


def _targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    raw = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [t.id for t in raw if isinstance(t, ast.Name)]


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, parent: str) -> str:
    """``name(args) -> returns``, so the model can often stop right here.

    Falls back to the bare name: a signature is a convenience, and losing
    it must never lose the definition.
    """
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    qualified = f"{parent}.{node.name}" if parent else node.name
    try:
        rendered = f"{prefix}{qualified}({ast.unparse(node.args)})"
        if node.returns is not None:
            rendered += f" -> {ast.unparse(node.returns)}"
    except Exception:
        return f"{prefix}{qualified}"
    # A long signature is worth truncating, not worth dropping.
    return rendered if len(rendered) <= 160 else rendered[:157] + "..."


_CLIKE_METHOD = re.compile(
    r"^[ \t]*(?:(?:public|private|protected|internal|static|final|readonly|"
    r"abstract|override|virtual|async|get|set|export|default)\s+)*"
    r"(?:[A-Za-z_$][\w$<>,\[\]]*\s+)?"
    r"([A-Za-z_$][\w$]*)\s*\([^)\n]*\)\s*(?::\s*[^{;\n]+)?\{",
    re.MULTILINE,
)
"""A method inside a class body -- ``total() {``, ``async load(x) {``,
``public int size() {``.

The declaration patterns the project map uses find only ``function`` and
``class``, which in a JavaScript or TypeScript project means every method
is missing. The map can afford that: it is a summary, and a method it
leaves out is an abbreviation. An index cannot -- a missing method makes
"where is total defined?" answer "nowhere in this project", which is a
confident wrong answer, and worse than the grep it replaced.

Regex here is a guess, so it is aimed at the failure that costs less: a
spurious entry is one extra line in a result, while a missed one is a lie.
"""

_NOT_A_NAME = frozenset({
    "if", "for", "while", "switch", "catch", "return", "function", "do",
    "else", "new", "typeof", "await", "with", "class", "try", "finally",
    "case", "delete", "in", "of", "yield", "throw", "void", "super",
})
"""Control flow reads exactly like a method declaration."""


_KINDS = {
    "go": ("func", projectmap._GO),
    "rust": ("definition", projectmap._RUST),
    "ruby": ("definition", projectmap._RUBY),
    "shell": ("function", projectmap._SHELL),
}


def _regex_definitions(text: str, language: str) -> list[Definition]:
    """Non-Python languages, via the patterns the project map already uses.

    Regex is a guess where ``ast`` is exact, but a guess that points at a
    line is still a better answer than forty call sites.
    """
    if language in _KINDS:
        kind, patterns = _KINDS[language][0], [_KINDS[language][1]]
    else:
        kind, patterns = "definition", [projectmap._CLIKE,
                                        projectmap._CLIKE_CONST_FN,
                                        _CLIKE_METHOD]
    found: list[Definition] = []
    seen: set[tuple[str, int]] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = match.group(1)
            if not name or name in _NOT_A_NAME:
                continue
            # The name's own position, not the match's: these patterns all
            # begin with ``^\s*``, and ``\s`` eats the newline before the
            # line they matched. Anchoring on the match start reported every
            # declaration that follows a blank line one line too early --
            # a wrong line number being exactly the confident wrong answer
            # this tool exists to avoid.
            line = text.count("\n", 0, match.start(1)) + 1
            if (name, line) in seen:
                continue        # a class both patterns matched
            seen.add((name, line))
            signature = match.group(0).strip().rstrip("{").strip()
            found.append(Definition(name=name, kind=kind, path="", line=line,
                                    signature=signature[:160]))
    return sorted(found, key=lambda d: d.line)


# -- the repository-wide index ------------------------------------------------


_cache: dict[str, tuple[float, float, int, list[Definition]]] = {}


def index(root: Path, *, refresh: bool = False) -> list[Definition]:
    """Every definition under ``root``.

    Cached against the newest source timestamp, so an edit invalidates it
    and a run of navigation questions in one turn does not re-parse the
    tree five times.
    """
    key = str(root)
    now = time.time()
    cached = _cache.get(key)
    if cached and not refresh:
        built, stamp, count, entries = cached
        if now - built < CACHE_TTL:
            return entries
        try:
            if projectmap.newest_source(root) == stamp and _count(root) == count:
                _cache[key] = (now, stamp, count, entries)
                return entries
        except OSError:
            return entries

    entries: list[Definition] = []
    files = projectmap.walk(root, limit=MAX_INDEX_FILES)
    for path in files:
        language = projectmap.LANGUAGES.get(path.suffix)
        if not language:
            continue
        try:
            if path.stat().st_size > MAX_INDEX_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        relative = path.relative_to(root).as_posix()
        for found in definitions_in(text, language):
            entries.append(Definition(
                name=found.name, kind=found.kind, path=relative,
                line=found.line, parent=found.parent,
                signature=found.signature))
    try:
        stamp = projectmap.newest_source(root)
    except OSError:
        stamp = 0.0
    _cache[key] = (now, stamp, len(files), entries)
    return entries


def _count(root: Path) -> int:
    return len(projectmap.walk(root, limit=MAX_INDEX_FILES))


def forget(root: Path | None = None) -> None:
    """Drop the cache. A test that writes a file and looks for it in the
    same second needs this; so does anything that moves the tree."""
    if root is None:
        _cache.clear()
    else:
        _cache.pop(str(root), None)


def find(root: Path, name: str, *, limit: int = 40,
         refresh: bool = False) -> list[Definition]:
    """Where ``name`` is defined, best match first.

    An exact match is what was asked for and comes first. A case-different
    or partial match is a guess, and offering it beats answering "not
    found" for a name the user half-remembered -- but never at the cost of
    burying the exact one.
    """
    wanted = (name or "").strip()
    if not wanted:
        return []
    # "Class.method" is how anyone writes a method, and it is the only way
    # to ask for one of several same-named methods.
    parent, _, bare = wanted.rpartition(".")
    entries = index(root, refresh=refresh)

    def rank(definition: Definition) -> int | None:
        if parent and definition.parent != parent:
            if definition.parent.lower() != parent.lower():
                return None
        if definition.name == bare:
            return 0
        if definition.name.lower() == bare.lower():
            return 1
        if not parent and bare.lower() in definition.name.lower():
            return 2
        return None

    scored = []
    for definition in entries:
        score = rank(definition)
        if score is None:
            continue
        scored.append((score, _kind_rank(definition), definition.path,
                       definition.line, definition))
    scored.sort(key=lambda row: row[:4])
    best = scored[0][0] if scored else 0
    # Once there is an exact match, a substring match is noise.
    return [row[-1] for row in scored if row[0] <= max(best, 1)][:limit]


def _kind_rank(definition: Definition) -> int:
    """A class or a function is more likely to be the thing being looked
    for than a method of the same name on some unrelated class."""
    return {"class": 0, "function": 1, "async function": 1}.get(definition.kind, 2)
