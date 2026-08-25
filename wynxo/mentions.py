"""``@path`` in a message means "read this first".

Typing ``why does @src/auth.py reject my token?`` puts the file in front of
the model before it starts, instead of it guessing which file you meant or
spending a tool call finding out. The mention stays in the sentence -- the
model still sees the words you wrote -- and the contents arrive alongside.

Everything here goes through the same Boundary the tools use. A mention is
user input, and ``@../../etc/passwd`` has to be refused for the same reason
the read_file tool refuses it.
"""

from __future__ import annotations

import re
from pathlib import Path

MENTION = re.compile(r"(?<![\w@])@([\w./\\~-]*[\w/\\-])")
"""``@`` followed by a path. The lookbehind keeps it from firing inside an
email address, and the last character cannot be a dot so that a mention at
the end of a sentence does not swallow the full stop."""

MAX_FILE_BYTES = 60_000
MAX_FILES = 10
MAX_TOTAL_BYTES = 150_000

# Directories nobody means when they type a path prefix.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
    "build", "target", ".next", ".nuxt", ".idea", ".vscode", ".wynxo",
}


def find(text: str) -> list[str]:
    """Every ``@path`` in the message, in order, without duplicates."""
    seen: list[str] = []
    for match in MENTION.finditer(text):
        raw = match.group(1)
        if raw not in seen:
            seen.append(raw)
    return seen


def candidates(workspace: Path, prefix: str = "", limit: int = 200) -> list[str]:
    """Paths under ``workspace`` that could follow an ``@``, for completion.

    Directories come back with a trailing separator so a second Tab can walk
    into them, which is the behaviour every shell has trained people to
    expect.
    """
    prefix = prefix.replace("\\", "/")
    # Only walk the directory the prefix names, not the whole tree: a big
    # repository would otherwise take long enough to feel broken.
    head, _, tail = prefix.rpartition("/")
    base = (workspace / head) if head else workspace
    try:
        base = base.resolve()
        if not base.is_dir() or not _within(base, workspace):
            return []
        entries = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return []

    out: list[str] = []
    for entry in entries:
        if entry.name.startswith(".") and not tail.startswith("."):
            continue
        if entry.is_dir() and entry.name in SKIP_DIRS:
            continue
        if tail and not entry.name.lower().startswith(tail.lower()):
            continue
        shown = f"{head}/{entry.name}" if head else entry.name
        out.append(shown + "/" if entry.is_dir() else shown)
        if len(out) >= limit:
            break
    return out


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def expand(text: str, workspace: Path, boundary=None,
           shield=None) -> tuple[str, list[str]]:
    """Return ``(message, problems)`` with mentioned files inlined.

    The original sentence is kept verbatim at the top -- the mention is part
    of what the user said, and stripping it would leave the model reading a
    file with no idea why. Anything unreadable comes back in ``problems``
    rather than being silently dropped, because a mention that quietly did
    nothing is worse than one that says it could not.

    The shield applies here for the same reason it applies to read_file: a
    mention puts the file on the wire to a model that is often on another
    machine. Without it the same file behaved two different ways depending
    on how it was named -- read_file refused a .env and masked the key in a
    settings module, while "@.env" sent the lot.
    """
    paths = find(text)
    if not paths:
        return text, []

    blocks: list[str] = []
    problems: list[str] = []
    budget = MAX_TOTAL_BYTES

    for raw in paths[:MAX_FILES]:
        candidate = Path(raw).expanduser()
        full = candidate if candidate.is_absolute() else (workspace / candidate)
        try:
            full = full.resolve()
        except OSError as exc:
            problems.append(f"@{raw}: {exc}")
            continue

        if boundary is not None and not boundary.contains(full):
            problems.append(f"@{raw} is outside the current scope")
            continue
        if not full.exists():
            problems.append(f"@{raw} does not exist")
            continue
        if full.is_dir():
            listing = _listing(full)
            blocks.append(f"### {raw} (directory)\n{listing}")
            continue

        try:
            if full.stat().st_size > MAX_FILE_BYTES:
                problems.append(
                    f"@{raw} is too large to inline "
                    f"({full.stat().st_size // 1024}KB); ask me to read part of it")
                continue
            body = full.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            problems.append(f"@{raw}: {exc}")
            continue

        if shield is not None:
            if shield.blocks(full):
                problems.append(
                    f"@{raw} holds credentials, so it was not read. "
                    "/secrets allow it if that is wrong.")
                continue
            body, masked = shield.clean(body)
            if masked:
                problems.append(
                    f"@{raw}: {masked} value(s) masked before sending")

        if len(body) > budget:
            problems.append(f"@{raw} skipped: no room left in this message")
            continue
        budget -= len(body)
        blocks.append(f"### {raw}\n```\n{body.rstrip()}\n```")

    if len(paths) > MAX_FILES:
        problems.append(f"only the first {MAX_FILES} mentions were read")
    if not blocks:
        return text, problems

    joined = "\n\n".join(blocks)
    message = (
        f"{text}\n\n"
        f"---\nFiles referenced above, already read for you:\n\n{joined}"
    )
    return message, problems


def _listing(directory: Path, limit: int = 60) -> str:
    try:
        entries = sorted(directory.iterdir(),
                         key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return f"(could not list: {exc})"
    names = [f"{e.name}/" if e.is_dir() else e.name
             for e in entries if e.name not in SKIP_DIRS]
    shown = names[:limit]
    if len(names) > limit:
        shown.append(f"... and {len(names) - limit} more")
    return "\n".join(shown) or "(empty)"
