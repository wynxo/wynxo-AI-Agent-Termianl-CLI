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
from dataclasses import dataclass, field
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

MAX_REFERENCES_PER_NAME = 200
"""How many uses of one name to keep.

Unbounded, the reference map is the whole project written out again --
``self`` alone appears 4,600 times in this repository, and at the index's
file limit that is hundreds of megabytes to answer questions nobody asks.
The true total is counted separately, so a capped answer says "more than
200 uses, narrow it" rather than quietly reporting 200 as the number."""

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
    return _read_python(text).definitions


@dataclass
class FileFacts:
    """Everything one parse of one file can tell us.

    Definitions and references come out of the same walk deliberately.
    Parsing every file twice -- once to know what it defines and once to
    know what it uses -- doubles the cost of the index for no new
    information.
    """

    definitions: list[Definition] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    """Dotted module names this file imports, relative imports resolved
    later against the file's own package."""

    relative_imports: list[tuple[int, str]] = field(default_factory=list)
    """``(level, tail)`` for ``from ..pkg import x``, which cannot be
    resolved without knowing where the file sits."""


@dataclass(frozen=True)
class Reference:
    """One place a name is used, as opposed to defined."""

    name: str
    kind: str
    """``call``, ``use``, ``import`` or ``subclass``. A call is what people
    mean by "what calls this"; a use is a mention that is not a call --
    a decorator, an isinstance check, a default argument."""

    path: str
    line: int
    context: str = ""
    """The function or class the reference sits inside. "called from
    cli.py:1382" is a location; "in Repl.companion_state" is an answer."""

    def describe(self) -> str:
        where = f"{self.path}:{self.line}"
        return f"{where}  {self.context or '<module>'}" if self.context else where


class _PythonReader(ast.NodeVisitor):
    """One pass: what the file defines, what it uses, what it imports.

    The scope stack exists for the context line. Knowing that something is
    called at cli.py:1382 is a location; knowing it is called inside
    ``Repl._companion_state`` is most of the answer, and it costs one list.
    """

    def __init__(self) -> None:
        self.facts = FileFacts()
        self._scope: list[str] = []
        self._class: list[str] = []

    # -- definitions ---------------------------------------------------

    def _function(self, node, is_async: bool) -> None:
        parent = self._class[-1] if self._class else ""
        kind = "method" if parent else "function"
        if is_async:
            kind = "async " + kind
        self.facts.definitions.append(Definition(
            name=node.name, kind=kind, path="", line=node.lineno,
            parent=parent, signature=_signature(node, parent)))
        self._scope.append(node.name)
        # A closure belongs to no class: its own nested defs are not
        # methods of the class that happens to enclose it.
        self._class.append("")
        self.generic_visit(node)
        self._class.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node):
        self._function(node, False)

    def visit_AsyncFunctionDef(self, node):
        self._function(node, True)

    def visit_ClassDef(self, node):
        parent = self._class[-1] if self._class else ""
        self.facts.definitions.append(Definition(
            name=node.name, kind="class", path="", line=node.lineno,
            parent=parent, signature=f"class {node.name}"))
        for base in node.bases:
            name = _name_of(base)
            if name:
                # Recorded under the *base* name: "what subclasses Tool"
                # asks about Tool, so Tool is what has to be findable.
                self.facts.references.append(Reference(
                    name=name, kind="subclass", path="", line=node.lineno,
                    context=node.name))
        self._scope.append(node.name)
        self._class.append(node.name)
        self.generic_visit(node)
        self._class.pop()
        self._scope.pop()

    def visit_Assign(self, node):
        if not self._scope:
            for target in _targets(node):
                if target.isupper() and len(target) > 1:
                    self.facts.definitions.append(Definition(
                        name=target, kind="constant", path="",
                        line=node.lineno))
        self.generic_visit(node)

    visit_AnnAssign = visit_Assign

    # -- references ----------------------------------------------------

    def visit_Call(self, node):
        name = _name_of(node.func)
        if name:
            self.facts.references.append(Reference(
                name=name, kind="call", path="", line=node.lineno,
                context=self._context()))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in _NEVER_ASKED:
            self.facts.references.append(Reference(
                name=node.id, kind="use", path="", line=node.lineno,
                context=self._context()))

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self.facts.references.append(Reference(
                name=node.attr, kind="use", path="", line=node.lineno,
                context=self._context()))
        self.generic_visit(node)

    # -- imports -------------------------------------------------------

    def visit_Import(self, node):
        for alias in node.names:
            self.facts.imports.add(alias.name)
            self.facts.references.append(Reference(
                name=alias.name.rpartition(".")[2], kind="import", path="",
                line=node.lineno, context=self._context()))

    def visit_ImportFrom(self, node):
        module = node.module or ""
        if node.level:
            self.facts.relative_imports.append((node.level, module))
            for alias in node.names:
                # ``from . import companion`` names the module in the alias,
                # not the module field. Dropping it made every sibling
                # import in the package invisible.
                tail = f"{module}.{alias.name}" if module else alias.name
                self.facts.relative_imports.append((node.level, tail))
        elif module:
            self.facts.imports.add(module)
        for alias in node.names:
            # ``from .x import y`` may be importing a module or a symbol.
            # Recording both readings costs one string and means neither
            # question comes back empty.
            self.facts.references.append(Reference(
                name=alias.name, kind="import", path="", line=node.lineno,
                context=self._context()))
            if module and not node.level:
                self.facts.imports.add(f"{module}.{alias.name}")

    def _context(self) -> str:
        return ".".join(self._scope)


_NEVER_ASKED = frozenset({"self", "cls"})
"""Names that are never the subject of "where is this used?". Recording
them is the single largest cost in the reference map and buys nothing."""


def _name_of(node: ast.AST) -> str:
    """The last name in an expression: ``a.b.c()`` is a call to ``c``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _read_python(text: str) -> FileFacts:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return FileFacts()
    reader = _PythonReader()
    try:
        reader.visit(tree)
    except RecursionError:
        # A pathologically nested file. What was collected before the wall
        # is still true, and is better than nothing.
        pass
    return reader.facts


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


_GO_FUNC = re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)", re.MULTILINE)
"""Every Go function and method, exported or not.

The project map matches only capitalised names, which is the right filter
for a summary of a package's public surface. For an index it is a bug: an
unexported ``func helper()`` is exactly the kind of thing someone asks
where to find, and answering "nowhere in this project" is worse than the
grep this replaced.
"""

_GO_TYPE = re.compile(
    r"^\s*type\s+(\w+)\s+(?:struct|interface|func|map|\[|\*|\w)",
    re.MULTILINE)
"""``type Cart struct``. Nothing matched these at all, so in a Go project
every struct, interface and named type was missing from the index."""

_KINDS = {
    "go": ("definition", (_GO_TYPE, _GO_FUNC)),
    "rust": ("definition", (projectmap._RUST,)),
    "ruby": ("definition", (projectmap._RUBY,)),
    "shell": ("function", (projectmap._SHELL,)),
}


def _regex_definitions(text: str, language: str) -> list[Definition]:
    """Non-Python languages, via the patterns the project map already uses.

    Regex is a guess where ``ast`` is exact, but a guess that points at a
    line is still a better answer than forty call sites.
    """
    if language in _KINDS:
        kind, patterns = _KINDS[language][0], list(_KINDS[language][1])
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


@dataclass
class Graph:
    """What the project defines, and how its parts refer to each other."""

    definitions: list[Definition] = field(default_factory=list)
    references: dict[str, list[Reference]] = field(default_factory=dict)
    """Keyed by name, because every question asked of it is "where is this
    one name used". A flat list would mean scanning tens of thousands of
    entries per question."""

    imports: dict[str, set[str]] = field(default_factory=dict)
    """Source path -> the dotted module names it imports."""

    modules: dict[str, str] = field(default_factory=dict)
    """Dotted module name -> the source path that defines it."""

    reference_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    """name -> kind -> how many there really are, including the ones past
    the cap. Counted per kind because the stored sample is capped across
    all kinds at once: filtering 200 mixed references down to the calls
    among them and reporting that as the number of calls understates it,
    and a wrong number stated confidently is worse than no number."""

    capped: set[str] = field(default_factory=set)
    """Names whose stored sample hit the cap, so the list shown is a sample
    of the references rather than all of them."""

    files: int = 0

    def importers_of(self, module: str) -> set[str]:
        """Paths that import ``module`` directly."""
        return {path for path, names in self.imports.items()
                if module in names}


_cache: dict[str, tuple[float, float, int, Graph]] = {}


def graph(root: Path, *, refresh: bool = False) -> Graph:
    """The whole picture, built once and cached against the tree.

    Cached against the newest source timestamp, so an edit invalidates it
    and a run of navigation questions in one turn does not re-parse the
    tree five times.
    """
    key = str(root)
    now = time.time()
    cached = _cache.get(key)
    if cached and not refresh:
        built, stamp, count, built_graph = cached
        if now - built < CACHE_TTL:
            return built_graph
        try:
            if projectmap.newest_source(root) == stamp and _count(root) == count:
                _cache[key] = (now, stamp, count, built_graph)
                return built_graph
        except OSError:
            return built_graph

    found = Graph()
    files = projectmap.walk(root, limit=MAX_INDEX_FILES)
    found.files = len(files)
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
        _absorb(found, relative, text, language)
    _resolve_relative_imports(found)
    try:
        stamp = projectmap.newest_source(root)
    except OSError:
        stamp = 0.0
    _cache[key] = (now, stamp, len(files), found)
    return found


def index(root: Path, *, refresh: bool = False) -> list[Definition]:
    """Every definition under ``root``."""
    return graph(root, refresh=refresh).definitions


_PENDING: dict[str, list[tuple[int, str]]] = {}
"""Relative imports waiting for the whole tree, keyed by source path. A
``from ..gh import X`` cannot be resolved until we know what packages
exist, so it is held here until the walk finishes."""


def _absorb(found: Graph, relative: str, text: str, language: str) -> None:
    if language == "python":
        facts = _read_python(text)
        found.modules[_module_name(relative)] = relative
        if facts.relative_imports:
            _PENDING[relative] = facts.relative_imports
    else:
        facts = FileFacts(definitions=_regex_definitions(text, language),
                          references=_regex_references(text))
    for definition in facts.definitions:
        found.definitions.append(Definition(
            name=definition.name, kind=definition.kind, path=relative,
            line=definition.line, parent=definition.parent,
            signature=definition.signature))
    for reference in facts.references:
        counts = found.reference_counts.setdefault(reference.name, {})
        counts[reference.kind] = counts.get(reference.kind, 0) + 1
        kept = found.references.setdefault(reference.name, [])
        if len(kept) >= MAX_REFERENCES_PER_NAME:
            found.capped.add(reference.name)
            continue
        kept.append(Reference(
            name=reference.name, kind=reference.kind, path=relative,
            line=reference.line, context=reference.context))
    if facts.imports:
        found.imports.setdefault(relative, set()).update(facts.imports)


def _module_name(relative: str) -> str:
    stem = relative[:-3] if relative.endswith(".py") else relative
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def _resolve_relative_imports(found: Graph) -> None:
    """``from ..gh import X`` in wynxo/tools/x.py means ``wynxo.gh``.

    Left unresolved, every intra-package import in the project would be
    invisible -- which for a package that imports itself the way this one
    does is most of the graph.
    """
    for relative, pending in _PENDING.items():
        package = _module_name(relative).split(".")
        for level, tail in pending:
            base = package[: max(0, len(package) - level)]
            dotted = ".".join([*base, tail] if tail else base)
            if dotted:
                found.imports.setdefault(relative, set()).add(dotted)
    _PENDING.clear()


_CALL = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")


def _regex_references(text: str) -> list[Reference]:
    """Calls in a language we do not parse. Textual, and honest about it:
    a name inside a string or a comment looks the same to a regex."""
    found: list[Reference] = []
    for match in _CALL.finditer(text):
        name = match.group(1)
        if name in _NOT_A_NAME:
            continue
        found.append(Reference(name=name, kind="call", path="",
                               line=text.count("\n", 0, match.start(1)) + 1))
    return found


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


# -- relationships ------------------------------------------------------------


def references(root: Path, name: str, *, kinds: tuple[str, ...] = (),
               limit: int = 40, refresh: bool = False
               ) -> tuple[list[Reference], int, bool]:
    """Where ``name`` is used: the hits, the true total, and whether the
    hits are only a sample of them.

    Calls come first: "what calls this" is the question, and a decorator or
    an isinstance check is a weaker answer than a call site. The definition
    itself is not a use of the name and is left out -- ``find`` is for that.
    """
    wanted = (name or "").strip().rpartition(".")[2]
    if not wanted:
        return [], 0, False
    found = graph(root, refresh=refresh)
    hits = found.references.get(wanted, [])
    counts = found.reference_counts.get(wanted, {})
    if kinds:
        hits = [hit for hit in hits if hit.kind in kinds]
        total = sum(counts.get(kind, 0) for kind in kinds)
    else:
        total = sum(counts.values())
    order = {"call": 0, "subclass": 0, "import": 1, "use": 2}
    ranked = sorted(hits, key=lambda hit: (order.get(hit.kind, 3),
                                           hit.path, hit.line))
    return ranked[:limit], total, wanted in found.capped


def subclasses(root: Path, name: str, *, refresh: bool = False) -> list[Reference]:
    """Classes that list ``name`` among their bases.

    Exact where a grep for ``(Name)`` is a guess about formatting: a base
    written across two lines, or as ``module.Name``, is the same base.
    """
    hits, _, _ = references(root, name, kinds=("subclass",), limit=200,
                            refresh=refresh)
    return hits


def importers(root: Path, module: str, *, refresh: bool = False) -> list[str]:
    """Files that import ``module``, given as a path or a dotted name."""
    found = graph(root, refresh=refresh)
    dotted = _as_module(module)
    if not dotted:
        return []
    # ``wynxo.gh`` is imported both as itself and as ``wynxo.gh.something``
    # by ``from wynxo.gh import Blob``. Both are importers of it.
    prefix = dotted + "."
    return sorted(path for path, names in found.imports.items()
                  if any(name == dotted or name.startswith(prefix)
                         for name in names))


def _as_module(raw: str) -> str:
    value = (raw or "").strip().replace("\\", "/")
    if not value:
        return ""
    return _module_name(value) if "/" in value or value.endswith(".py") else value


def covering_tests(root: Path, changed: list[Path], *,
                   refresh: bool = False) -> list[str]:
    """Test files worth running for a change, best relation first.

    Selecting tests by filename similarity -- the previous rule -- averaged
    about 30% recall on this project, and picked nothing at all for modules
    whose name appears in no test file's name. Importing the module is the
    real relation.

    Direct importers only, by default. Following imports transitively is
    more *correct* and completely useless: in a project whose CLI imports
    everything, every test reaches every module, and the "focused" run
    becomes the whole suite with extra steps. Transitive reachability is
    the fallback for when nothing imports the file directly, and even then
    a selection that covers most of the suite is discarded -- running
    everything is what the caller already does when this returns nothing,
    and it does it without pretending to be focused.
    """
    found = graph(root, refresh=refresh)
    targets: set[str] = set()
    for path in changed:
        targets.add(_module_name(_relative_to(root, path)))
    if not targets:
        return []

    reverse = _reverse_imports(found)
    tests = [path for path in found.modules.values() if _is_test(path)]
    if not tests:
        tests = [path for path in found.imports if _is_test(path)]

    direct = {path for target in targets
              for path in reverse.get(target, ()) if _is_test(path)}
    selected = direct | {_relative_to(root, path)
                         for path in _tests_named_after(root, changed)}
    if not selected:
        selected = {path for path in _reachable(targets, reverse)
                    if _is_test(path)}
    if not selected:
        return []
    # Only worth discarding a wide selection when the suite is big enough
    # for "focused" to mean anything. In a project with three test files,
    # running all three *is* the focused run, and refusing to name them
    # just sends the caller the long way round to the same commands.
    if len(tests) >= MIN_SUITE_FOR_FRACTION and \
            len(selected) > len(tests) * MAX_FOCUSED_FRACTION:
        return []
    return sorted(selected)


MIN_SUITE_FOR_FRACTION = 8
"""Below this many test files, never discard a selection for being wide."""

MAX_FOCUSED_FRACTION = 0.5
"""Past this share of the suite, a focused run is the full run with a
longer command line. The caller falls back to the whole suite anyway, so
saying "nothing focused" is both cheaper and more honest."""


def _relative_to(root: Path, path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


def _reverse_imports(found: Graph) -> dict[str, set[str]]:
    """Module -> the files that import it."""
    # ``from wynxo.gh import Blob`` is recorded as both ``wynxo.gh`` and
    # ``wynxo.gh.Blob`` when it is read, so the module is already a key
    # here and there is nothing to derive.
    reverse: dict[str, set[str]] = {}
    for path, names in found.imports.items():
        for name in names:
            reverse.setdefault(name, set()).add(path)
    return reverse


def _reachable(targets: set[str], reverse: dict[str, set[str]]) -> set[str]:
    reached: set[str] = set()
    frontier = set(targets)
    while frontier:
        module = frontier.pop()
        for path in reverse.get(module, ()):
            if path not in reached:
                reached.add(path)
                frontier.add(_module_name(path))
    return reached


def _tests_named_after(root: Path, changed: list[Path]) -> list[Path]:
    """The old filename heuristic, kept as a second signal.

    Its recall was poor but its precision was not: when test_gh.py exists,
    it is a test for gh.py whether or not the import graph says so.
    """
    tests_root = root / "tests" if (root / "tests").is_dir() else root
    return affected_tests(list(changed), tests_root)


def _is_test(path: str) -> bool:
    name = path.rpartition("/")[2]
    return name.startswith("test_") or name.endswith("_test.py")
