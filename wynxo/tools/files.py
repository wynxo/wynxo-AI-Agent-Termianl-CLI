"""Reading and changing files."""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import NamedTuple

from ..schema import Field, Schema
from ..session import estimate_tokens
from .base import Tool, ToolResult

MAX_READ_BYTES = 400_000
READ_SHARE = 0.33
"""How much of the remaining context one read may take. The rest has to
cover the reply, the next few tool results, and the turn's own history."""

MIN_READ_TOKENS = 500
"""Never trim below this. A read cut to nothing teaches the model only that
reading does not work, and it will try something worse."""
BINARY_SNIFF = 8_000


_BOMS = (
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)


def _looks_binary(path: Path) -> bool:
    """Whether this is a file that would be ruined by being treated as text.

    A byte-order mark settles it before the NUL test gets a say. UTF-16 is
    half NUL bytes by construction, so without this every UTF-16 file --
    which on Windows means most things PowerShell wrote -- was called binary
    and refused, and the UTF-16 branch of the decoder could never run.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(BINARY_SNIFF)
    except OSError:
        return False
    if any(head.startswith(bom) for bom, _ in _BOMS):
        return False
    return b"\0" in head


class Decoded(NamedTuple):
    """A file's text, and what it takes to write it back unchanged."""

    text: str
    encoding: str
    bom: bytes


def _decode(path: Path) -> Decoded:
    """Read with a forgiving encoding ladder, remembering which rung.

    Files written on Windows are routinely cp1252 or UTF-16, and a decode
    error here would look to the model like the file does not exist.

    The encoding comes back because writing is not symmetric with reading.
    An edit that changed one line but saved the whole file as UTF-8 rewrote
    every other byte in it: a UTF-16 PowerShell script became UTF-8 without
    its byte-order mark, a cp1252 file's accented characters were re-encoded
    from end to end, and a BOM the file was relying on simply vanished.
    """
    raw = path.read_bytes()
    for bom, encoding in _BOMS:
        if raw.startswith(bom):
            return Decoded(raw[len(bom):].decode(encoding, "replace"),
                           encoding, bom)
    for encoding in ("utf-8", "cp1252"):
        try:
            return Decoded(raw.decode(encoding), encoding, b"")
        except UnicodeDecodeError:
            continue
    return Decoded(raw.decode("utf-8", "replace"), "utf-8", b"")


def _read_text(path: Path) -> str:
    return _decode(path).text


def _write_back(path: Path, text: str, source: "Decoded | None" = None) -> str:
    """Save text the way the file it came from was stored.

    Bytes rather than a text handle, so nothing translates the line endings:
    a repo full of CRLF files must not quietly become LF (or the reverse) as
    a side effect of one edit.

    Returns a note for the model when the encoding had to change, which
    happens when the new text needs characters the old encoding cannot
    store -- a fact worth telling it about rather than a "?" to discover.
    """
    encoding = source.encoding if source else "utf-8"
    bom = source.bom if source else b""
    try:
        path.write_bytes(bom + text.encode(encoding))
        return ""
    except UnicodeEncodeError:
        path.write_bytes(text.encode("utf-8"))
        if encoding == "utf-8":
            return ""
        return (f" (saved as UTF-8: the new text needs characters "
                f"{encoding} cannot store)")


def make_diff(before: str, after: str, path: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)


class ReadInput(Schema):
    path = Field(str, "File path, relative to the project root.")
    offset = Field(int, "First line to read (0-indexed).", default=0, ge=0)
    limit = Field(int, "How many lines to read.", default=2000, gt=0, le=5000)


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file from the project. Returns numbered lines so you can "
        "refer to them precisely. Always read a file before editing it."
    )
    Input = ReadInput

    async def run(self, args: ReadInput) -> ToolResult:
        path = self.resolve_path(args.path)
        rel = self.relative(path)
        # Checked before the existence test on purpose: "that file holds
        # credentials" and "that file is not there" are different answers,
        # and confirming which secrets exist is itself worth withholding.
        if refused := self.shield.blocks(path):
            return ToolResult.failure(refused)
        if not path.exists():
            near = self._suggest(path)
            hint = f" Did you mean {near}?" if near else ""
            return ToolResult.failure(f"{rel} does not exist.{hint}")
        if path.is_dir():
            # Answer rather than refuse. A weaker model that asks to read a
            # directory will ask again, and again, if all it gets back is a
            # complaint -- it has no new information to act on. Handing it the
            # listing ends the loop and is what it wanted anyway.
            listing = await ListDir(self.workspace, self.boundary).run(
                ListInput(path=args.path))
            return ToolResult.success(
                f"{rel} is a directory, so here is what is in it "
                f"(use read_file on one of these files):\n\n{listing.output}",
                display=f"listed {rel} (asked to read a directory)",
                path=rel,
                was_directory=True,
            )
        if _looks_binary(path):
            size = path.stat().st_size
            return ToolResult.failure(f"{rel} is a binary file ({size} bytes).")
        if path.stat().st_size > MAX_READ_BYTES:
            return ToolResult.failure(
                f"{rel} is {path.stat().st_size} bytes, too large to read whole. "
                "Use grep to find the part you need, then read with offset/limit."
            )

        lines = _read_text(path).splitlines()
        if args.offset >= len(lines) and lines:
            return ToolResult.failure(
                f"offset {args.offset} is past the end of {rel} ({len(lines)} lines)."
            )
        window = lines[args.offset : args.offset + args.limit]
        width = len(str(args.offset + len(window)))
        body = "\n".join(
            f"{str(i).rjust(width)}\t{line}"
            for i, line in enumerate(window, start=args.offset + 1)
        )
        truncated = args.offset + len(window) < len(lines)
        note = (
            f"\n\n[showing lines {args.offset + 1}-{args.offset + len(window)} "
            f"of {len(lines)}]"
            if truncated or args.offset
            else ""
        )
        body, note, window = self._fit_context(body, note, window, args, len(lines))
        body, masked = self.shield.clean(body)
        if masked:
            # Said out loud rather than done quietly: the model is about to
            # reason about a file, and it should know that what it is looking
            # at is not exactly what is on disk.
            note += (f"\n\n[{masked} credential{'s' if masked != 1 else ''} "
                     f"in this file were masked before it reached you. The "
                     f"code is unchanged on disk.]")
        return ToolResult.success(
            body + note,
            display=f"read {rel} ({len(window)} lines)"
                    + (f", {masked} masked" if masked else ""),
            path=rel,
            lines=len(lines),
            masked=masked,
        )

    def _suggest(self, path: Path) -> str | None:
        parent = path.parent
        if not parent.is_dir():
            return None
        names = [p.name for p in parent.iterdir()]
        close = difflib.get_close_matches(path.name, names, n=1, cutoff=0.7)
        return close[0] if close else None


    def _fit_context(self, body: str, note: str, window: list, args,
                     total: int) -> tuple[str, str, list]:
        """Trim a read that would not fit in what is left of the context.

        A 2000-line file is roughly 22k tokens, which is more than the whole
        budget at low effort. Nothing used to notice: the read succeeded, the
        oldest messages fell out of the window, and the model got quietly
        stupid halfway through a task with no error anywhere.

        Trimmed rather than refused, and told exactly how to get the rest --
        a refusal leaves a weaker model with nowhere to go, and it will just
        ask for the same file again.
        """
        if self.context_left <= 0 or not window:
            return body, note, window

        # A single read should not eat more than a third of what is left.
        # The rest has to cover the reply, the next few tool results, and
        # whatever else the turn still needs.
        allowance = max(MIN_READ_TOKENS, int(self.context_left * READ_SHARE))
        if estimate_tokens(body) <= allowance:
            return body, note, window

        per_line = max(1, estimate_tokens(body) // len(window))
        keep = max(1, min(len(window), allowance // per_line))
        window = window[:keep]
        width = len(str(args.offset + keep))
        body = "\n".join(
            f"{str(i).rjust(width)}\t{line}"
            for i, line in enumerate(window, start=args.offset + 1)
        )
        last = args.offset + keep
        note = (
            f"\n\n[showing lines {args.offset + 1}-{last} of {total}. "
            f"The rest was left out because it would not fit in the context "
            f"still free this turn. Read on with offset={last}, or use grep "
            f"to jump to the part you need.]"
        )
        return body, note, window

class WriteInput(Schema):
    path = Field(str, "File path, relative to the project root.")
    content = Field(str, "Full contents to write.")


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Create a new file, or completely replace an existing one. For a small "
        "change to an existing file use edit_file instead -- it is far cheaper "
        "and cannot accidentally drop the rest of the file."
    )
    Input = WriteInput
    mutating = True
    concurrency_safe = False

    async def run(self, args: WriteInput) -> ToolResult:
        path = self.resolve_path(args.path)
        rel = self.relative(path)
        if path.is_dir():
            return ToolResult.failure(
                f"{rel} is a directory, not a file. Give the path of a file "
                "inside it, for example {rel}/notes.md.".replace("{rel}", rel))
        existed = path.exists()
        # Replacing the contents is not a reason to change how the file is
        # stored: a UTF-16 script rewritten as UTF-8 is a different file to
        # everything that reads it.
        source = _decode(path) if existed and not _looks_binary(path) else None
        before = source.text if source else ""

        path.parent.mkdir(parents=True, exist_ok=True)
        note = _write_back(path, args.content, source)

        n = len(args.content.splitlines())
        verb = "updated" if existed else "created"
        return ToolResult.success(
            f"{verb} {rel} ({n} lines){note}",
            display=make_diff(before, args.content, rel) if existed else "",
            path=rel,
            created=not existed,
        )


class EditInput(Schema):
    path = Field(str, "File path, relative to the project root.")
    expected_hash = Field(str, "Optional SHA-256 of the file as last read; rejects external drift.", default="")
    old_text = Field(str, "Exact text to replace, including indentation. Must appear "
                          "in the file exactly once unless replace_all is true.")
    new_text = Field(str, "Replacement text.")
    replace_all = Field(bool, "Replace every occurrence.", default=False)


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact span of text in a file. Read the file first so "
        "old_text matches byte for byte, whitespace included."
    )
    Input = EditInput
    mutating = True
    concurrency_safe = False

    async def run(self, args: EditInput) -> ToolResult:
        path = self.resolve_path(args.path)
        rel = self.relative(path)
        if path.is_dir():
            return ToolResult.failure(
                f"{rel} is a directory, not a file. Use list_dir to see what is "
                "in it, then edit one of the files.")
        if not path.exists():
            return ToolResult.failure(f"{rel} does not exist. Use write_file to create it.")
        if args.old_text == args.new_text:
            return ToolResult.failure("old_text and new_text are identical; nothing to do.")
        if not args.old_text:
            # Empty matches between every character, so the count that would
            # otherwise be reported ("appears 7 times") describes nothing and
            # tells the model nothing about what to do differently.
            return ToolResult.failure(
                "old_text is empty. Give the exact text to replace, or use "
                "write_file if you mean to create or replace the whole file.")

        if _looks_binary(path):
            return ToolResult.failure(
                f"{rel} is a binary file. Editing it as text would corrupt it.")

        source = _decode(path)
        before = source.text
        if args.expected_hash:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual.lower() != args.expected_hash.strip().lower():
                return ToolResult.failure(
                    f"{rel} changed since it was read (expected {args.expected_hash}, found {actual}). "
                    "Re-read the file before editing; no changes were applied.",
                    stale=True, expected_hash=args.expected_hash, actual_hash=actual,
                )
        count = before.count(args.old_text)

        if count == 0:
            return ToolResult.failure(
                f"old_text not found in {rel}. {self._near_miss(before, args.old_text)}"
            )
        if count > 1 and not args.replace_all:
            return ToolResult.failure(
                f"old_text appears {count} times in {rel}. Include surrounding "
                "lines to make it unique, or set replace_all=true."
            )

        after = (
            before.replace(args.old_text, args.new_text)
            if args.replace_all
            else before.replace(args.old_text, args.new_text, 1)
        )
        note = _write_back(path, after, source)

        return ToolResult.success(
            f"edited {rel} "
            f"({count if args.replace_all else 1} replacement(s)){note}",
            display=make_diff(before, after, rel),
            path=rel,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    @staticmethod
    def _near_miss(haystack: str, needle: str) -> str:
        """Tell the model *why* the match failed. Whitespace, almost always."""
        if needle.strip() and needle.strip() in haystack:
            return "A version with different leading/trailing whitespace does exist -- match the indentation exactly."
        first = needle.strip().splitlines()[0] if needle.strip() else ""
        if first and first in haystack:
            return f"The first line ({first[:60]!r}) is present, so the mismatch is further down. Re-read the file."
        return "Re-read the file; it may have changed since you last saw it."


class ListInput(Schema):
    path = Field(str, "Directory to list, relative to the project root.", default=".")
    depth = Field(int, "How many levels deep to descend.", default=2, ge=1, le=5)


IGNORED = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", "target", ".ruff_cache",
    ".idea", ".vscode", ".tox", ".DS_Store", "vendor", ".gradle",
}
MAX_ENTRIES = 400


class ListDir(Tool):
    name = "list_dir"
    description = "List a directory as a tree, skipping build output and vcs noise."
    Input = ListInput

    async def run(self, args: ListInput) -> ToolResult:
        root = self.resolve_path(args.path)
        rel = self.relative(root)
        if not root.exists():
            return ToolResult.failure(f"{rel} does not exist.")
        if not root.is_dir():
            return ToolResult.failure(f"{rel} is a file, not a directory.")

        lines: list[str] = []
        truncated = self._walk(root, args.depth, "", lines)
        if not lines:
            return ToolResult.success(f"{rel}/ is empty")
        body = f"{rel.rstrip('/')}/\n" + "\n".join(lines)
        if truncated:
            body += f"\n... (truncated at {MAX_ENTRIES} entries)"
        return ToolResult.success(body, display=f"listed {rel} ({len(lines)} entries)")

    def _walk(self, directory: Path, depth: int, prefix: str, out: list[str]) -> bool:
        if depth <= 0 or len(out) >= MAX_ENTRIES:
            return len(out) >= MAX_ENTRIES
        try:
            entries = sorted(
                (e for e in directory.iterdir() if e.name not in IGNORED and not e.name.startswith(".")),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except PermissionError:
            out.append(f"{prefix}[permission denied]")
            return False
        for i, entry in enumerate(entries):
            if len(out) >= MAX_ENTRIES:
                return True
            last = i == len(entries) - 1
            branch = "`-- " if last else "|-- "
            # A symlinked directory is shown as what it is and not walked
            # into. Following them turns a project with two directories
            # pointing at each other into a tree of itself repeating -- five
            # levels of a/to_b/to_a/to_b -- and a link back to the root
            # duplicates the whole listing under a name that is not where
            # anything lives.
            if entry.is_symlink():
                out.append(f"{prefix}{branch}{entry.name}"
                           f"{'/' if entry.is_dir() else ''} -> "
                           f"{self._target(entry)}")
                continue
            out.append(f"{prefix}{branch}{entry.name}"
                       f"{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                if self._walk(entry, depth - 1, prefix + ("    " if last else "|   "), out):
                    return True
        return False

    @staticmethod
    def _target(entry: Path) -> str:
        """Where a link points, said briefly and without following it."""
        try:
            return str(entry.readlink())
        except OSError:
            return "?"


class EditOp(Schema):
    old_text = Field(str, "Exact text to replace, including indentation.")
    new_text = Field(str, "Replacement text.")
    replace_all = Field(bool, "Replace every occurrence of this one.", default=False)


class MultiEditInput(Schema):
    path = Field(str, "File path, relative to the project root.")
    edits = Field(list, "Edits to apply in order, each an exact-match replacement.",
                  item_type=EditOp, default_factory=list)


class MultiEdit(Tool):
    name = "multi_edit"
    description = (
        "Apply several exact-match replacements to one file in a single pass. "
        "Use this instead of calling edit_file repeatedly on the same file: it "
        "is one round trip, one diff, and either every edit applies or none do."
    )
    Input = MultiEditInput
    mutating = True
    concurrency_safe = False

    async def run(self, args: MultiEditInput) -> ToolResult:
        path = self.resolve_path(args.path)
        rel = self.relative(path)
        if path.is_dir():
            return ToolResult.failure(f"{rel} is a directory, not a file.")
        if not path.exists():
            return ToolResult.failure(f"{rel} does not exist. Use write_file to create it.")
        if not args.edits:
            return ToolResult.failure("No edits given.")

        if _looks_binary(path):
            return ToolResult.failure(
                f"{rel} is a binary file. Editing it as text would corrupt it.")

        source = _decode(path)
        before = source.text
        text = before

        # Validate every edit against the running text first. All-or-nothing:
        # a half-applied batch leaves the file in a state nobody planned.
        for i, edit in enumerate(args.edits, 1):
            if edit.old_text == edit.new_text:
                return ToolResult.failure(f"edit {i}: old_text and new_text are identical.")
            if not edit.old_text:
                return ToolResult.failure(
                    f"edit {i}: old_text is empty. Give the exact text to "
                    "replace.")
            count = text.count(edit.old_text)
            if count == 0:
                hint = EditFile._near_miss(text, edit.old_text)
                return ToolResult.failure(
                    f"edit {i}: old_text not found in {rel}. {hint} "
                    "No edits were applied."
                )
            if count > 1 and not edit.replace_all:
                return ToolResult.failure(
                    f"edit {i}: old_text appears {count} times. Include surrounding "
                    "lines to make it unique, or set replace_all on that edit. "
                    "No edits were applied."
                )
            text = (text.replace(edit.old_text, edit.new_text)
                    if edit.replace_all
                    else text.replace(edit.old_text, edit.new_text, 1))

        note = _write_back(path, text, source)

        return ToolResult.success(
            f"applied {len(args.edits)} edit(s) to {rel}{note}",
            display=make_diff(before, text, rel),
            path=rel,
            edits=len(args.edits),
        )
