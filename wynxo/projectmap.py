"""A one-page map of the codebase, so the model knows where things are.

Local models explore badly. Asked to "fix the retry", a 7B will grep for
"retry", get forty hits, read three wrong files and then guess -- each step
a slow round trip on hardware that has none to spare. It is not that the
model is stupid; it is that it starts every session blind in a house it has
never been in.

So the layout is worked out once, cheaply, without the model: which files
exist and what each one defines. That fits in a few hundred tokens, goes in
the system prompt, and turns "find the retry" into "open upload.py".

Deliberately not an index and not embeddings. It is regenerated when the
files change, from their mtimes, and the whole thing is a markdown file you
can read. Anything cleverer would cost more than it returns at this size.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

MAP_FILE = "map.md"
MAX_FILES = 400
MAX_SYMBOLS_PER_FILE = 12
MAX_CHARS = 6_000
MAX_FILE_BYTES = 400_000

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
    "build", "target", ".next", ".nuxt", ".idea", ".vscode", ".wynxo",
    "vendor", "coverage", ".gradle", "Pods", ".terraform",
}

# Languages worth pulling symbols out of. Everything else is listed by name.
LANGUAGES = {
    ".py": "python",
    ".js": "clike", ".jsx": "clike", ".ts": "clike", ".tsx": "clike",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".java": "clike",
    ".c": "clike", ".h": "clike", ".cpp": "clike", ".cs": "clike",
    ".sh": "shell",
}

_CLIKE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?"
    r"(?:async\s+)?(?:public\s+|private\s+|protected\s+|static\s+)*"
    r"(?:function|class|interface|type|enum|struct)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_CLIKE_CONST_FN = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
    re.MULTILINE,
)
_GO = re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Z][\w]*)", re.MULTILINE)
_RUST = re.compile(r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait)\s+(\w+)", re.MULTILINE)
_RUBY = re.compile(r"^\s*(?:def|class|module)\s+([\w.:]+)", re.MULTILINE)
_SHELL = re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\s*\)\s*\{", re.MULTILINE)


@dataclass
class Entry:
    path: str
    symbols: list[str] = field(default_factory=list)


def symbols_in(text: str, language: str) -> list[str]:
    """The names a file defines. Never raises -- a file that cannot be
    parsed is listed by name alone, which is still worth having."""
    try:
        if language == "python":
            found = _python_symbols(text)
        elif language == "go":
            found = _GO.findall(text)
        elif language == "rust":
            found = _RUST.findall(text)
        elif language == "ruby":
            found = _RUBY.findall(text)
        elif language == "shell":
            found = _SHELL.findall(text)
        else:
            found = _CLIKE.findall(text) + _CLIKE_CONST_FN.findall(text)
    except Exception:
        return []
    seen: list[str] = []
    for name in found:
        if name and not name.startswith("_") and name not in seen:
            seen.append(name)
    return seen[:MAX_SYMBOLS_PER_FILE]


def _python_symbols(text: str) -> list[str]:
    """Top-level definitions, via ast -- exact where a regex would guess.

    Falls back to nothing on a syntax error, which is the honest answer for
    a file that does not parse.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.append(node.name)
    return out[:MAX_SYMBOLS_PER_FILE]


def _is_junction(path: Path) -> bool:
    """Windows junctions, which is_symlink() reports as False."""
    try:
        return bool(path.is_junction())
    except AttributeError:      # Python before 3.12
        return False


def walk(root: Path, limit: int = MAX_FILES) -> list[Path]:
    """Source files under ``root``, skipping the noise directories.

    Symlinks are not descended into at all. A link can point anywhere --
    outside the project, or back at a parent -- and following it is how a
    map either reads things it should not or loops forever on a self
    reference (a junction to its own directory never empties the stack and
    never grows the file list). The map is the real tree; links are not
    part of it.
    """
    found: list[Path] = []
    stack = [root]
    seen: set[Path] = set()
    while stack and len(found) < limit:
        directory = stack.pop()
        try:
            key = directory.resolve()
        except (OSError, RuntimeError):
            key = directory
        if key in seen:
            continue
        seen.add(key)
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") and entry.name != ".github":
                continue
            # Links are not part of the map. is_symlink() catches POSIX
            # links; Windows junctions need their own check because
            # is_symlink() reports False for them. A junction to the
            # project's own parent used to loop forever, and one pointing
            # outside pulled unrelated files into the map.
            if entry.is_symlink() or _is_junction(entry):
                continue
            if entry.is_dir():
                if entry.name not in SKIP_DIRS:
                    stack.append(entry)
            elif entry.suffix in LANGUAGES:
                found.append(entry)
                if len(found) >= limit:
                    break
    return sorted(found)


def build(root: Path) -> str:
    """Render the map. Cheap enough to run on every start."""
    entries: list[Entry] = []
    for path in walk(root):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        entries.append(Entry(rel, symbols_in(text, LANGUAGES[path.suffix])))

    if not entries:
        return ""

    # Fit the budget by giving up detail, never by dropping files. Knowing a
    # file exists is most of the value, and a map that silently omits half
    # the project would send the model looking in the wrong place with
    # confidence -- worse than a plainer map, or than no map at all.
    for per_file in (MAX_SYMBOLS_PER_FILE, 6, 3, 1, 0):
        out = _render(entries, per_file)
        if len(out) <= MAX_CHARS:
            return out
    # Every file listed, with nothing said about any of them, still does not
    # fit. Cutting the text there is the one thing not to do: it ends
    # mid-directory while the header goes on claiming a total, so the model
    # is told there are three hundred files and shown the first two hundred
    # -- and looks for the rest with confidence in the wrong place. A
    # summary by directory says less about more, and says so.
    return _by_directory(entries)


def _by_directory(entries: list[Entry]) -> str:
    """One line per directory, for a project too large to list."""
    folders: dict[str, list[Entry]] = {}
    for entry in entries:
        folder = entry.path.rsplit("/", 1)[0] if "/" in entry.path else "."
        folders.setdefault(folder, []).append(entry)

    width = min(38, max((len(f) for f in folders), default=10) + 2)
    lines = [
        "# Project map", "",
        f"{len(entries)} source files in {len(folders)} directories -- too "
        "many to list one by one,",
        "so this is by directory. Use glob or list_dir for the rest.",
        "",
    ]
    body = []
    for folder in sorted(folders):
        names = [e.path.rsplit("/", 1)[-1] for e in folders[folder]]
        shown = ", ".join(names[:3])
        if len(names) > 3:
            shown += f", +{len(names) - 3}"
        body.append(f"{folder.ljust(width)} {len(names):>4} files   {shown}")

    kept, used = [], len("\n".join(lines)) + 60
    for line in body:
        if used + len(line) + 1 > MAX_CHARS:
            kept.append(f"... and {len(body) - len(kept)} more directories")
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(lines + kept) + "\n"


def _render(entries: list[Entry], per_file: int) -> str:
    width = min(38, max((len(e.path) for e in entries), default=10) + 2)
    header = [f"# Project map", "",
              f"{len(entries)} source files."]
    if per_file:
        header.append("Each line is a file and the names it defines, so you "
                      "can open the right one")
        header.append("instead of searching for it.")
    header.append("")

    lines = list(header)
    for entry in entries:
        names = ", ".join(entry.symbols[:per_file]) if per_file else ""
        more = len(entry.symbols) - per_file
        if per_file and more > 0:
            names += f", +{more}"
        lines.append(f"{entry.path.ljust(width)} {names}".rstrip()
                     if names else entry.path)
    return "\n".join(lines)


def cache_path(root: Path) -> Path:
    return root / ".wynxo" / MAP_FILE


def newest_source(root: Path) -> float:
    """Latest mtime among the files the map covers."""
    latest = 0.0
    for path in walk(root):
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def load(root: Path, max_age: float = 0.0) -> str:
    """The cached map, rebuilt when the sources have moved on.

    Compared against the newest source file rather than a clock: a project
    nobody has touched for a week does not need remapping, and one edited a
    second ago does.
    """
    path = cache_path(root)
    try:
        cached = path.read_text(encoding="utf-8")
        stamp = path.stat().st_mtime
    except OSError:
        cached, stamp = "", 0.0

    if cached and stamp >= newest_source(root):
        return cached

    fresh = build(root)
    if fresh:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(fresh, encoding="utf-8")
        except OSError:
            pass          # an unwritable project still gets the map in-prompt
    return fresh


_COUNT = re.compile(r"^(\d+) source files", re.MULTILINE)


def summarise(text: str) -> str:
    """One line about the map, for the start-up line.

    Reads the count the map already states rather than counting lines: the
    header has prose in it, and guessing from line shapes counted that prose
    as files.
    """
    match = _COUNT.search(text or "")
    if not match:
        return ""
    total = int(match.group(1))
    return f"{total} file{'s' if total != 1 else ''} mapped"


def timestamp() -> float:
    return time.time()
