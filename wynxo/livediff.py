"""The edit that is happening, while it is happening.

Fed by the real incremental stream: the provider's tool-call argument
fragments, decoded into the file's contents as they are generated. Nothing
here reveals a finished string slowly. If the stream stalls, the card
stalls; if the provider sends the whole edit in one message (Ollama's native
tool calls do), the card fills in one step, because that is what actually
happened.

Where it draws matters as much as what it draws. A long edit streamed
straight into the transcript would leave a thousand lines of somebody else's
file in the conversation forever, and the transcript is append-only -- once
written, it cannot be taken back and compacted. So the live body goes into
an overlay, which is a layer over the conversation rather than part of it,
and what lands in the transcript when the edit finishes is one line:

    ✓ edit_file · wynxo/agent.py · +42 −17

The full diff stays retrievable (Ctrl-D) rather than being pushed at
everybody by default.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path

LIVE = "live"
DONE = "done"
FAILED = "failed"

EDIT_TOOLS = ("write_file", "edit_file", "multi_edit")
"""Tools whose argument stream is a file's contents. Anything else gets the
ordinary one-line tool result; a card for `list_dir` would be noise."""

MAX_LIVE_ROWS = 12
"""How much of the tail to show while streaming. A card that grows without
limit is a card that eventually owns the screen."""


@dataclass
class DiffCard:
    """One edit, from the first fragment to the final count."""

    tool: str
    path: str = ""
    before: str = ""
    """The file as it was, when it existed. Without it there is nothing to
    diff against and every line counts as an addition, which is the honest
    answer for a new file."""
    streamed: str = ""
    """Exactly what has arrived so far. Never padded, never predicted."""
    state: str = LIVE
    error: str = ""
    _committed: list[str] = field(default_factory=list, repr=False)

    # -- the stream --------------------------------------------------------

    def feed(self, delta: str) -> None:
        if self.state != LIVE or not delta:
            return
        self.streamed += delta

    def finish(self, ok: bool = True, error: str = "",
               settled: str = "") -> None:
        """Close the card. ``settled`` is the file as it actually ended up.

        Providers that send a tool call's arguments in one message -- Ollama's
        native tool_calls do -- stream nothing, so there is no content to
        count and the card would report +0 -0 on an edit that plainly changed
        the file. Reading the result back is still real data: it is what is
        on disk, not a guess about what the model meant. It is only ever used
        when nothing was streamed, so a streamed card still shows exactly
        what arrived.
        """
        self.state = DONE if ok else FAILED
        self.error = error
        if settled and not self.streamed:
            self.streamed = settled

    @property
    def live(self) -> bool:
        return self.state == LIVE

    # -- what changed ------------------------------------------------------

    def counts(self) -> tuple[int, int]:
        """(added, removed), by comparing what arrived with what was there.

        Computed from the content itself rather than reported by the tool, so
        a card cannot claim a change the file does not contain.
        """
        if not self.streamed:
            return (0, 0)
        old = self.before.splitlines()
        new = self.streamed.splitlines()
        added = removed = 0
        for line in difflib.unified_diff(old, new, lineterm="", n=0):
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return (added, removed)

    def diff_lines(self) -> list[str]:
        """The unified diff, as rows. Empty while nothing has arrived."""
        if not self.streamed:
            return []
        old = self.before.splitlines()
        new = self.streamed.splitlines()
        name = self.path or self.tool
        return [
            line for line in difflib.unified_diff(
                old, new, fromfile=name, tofile=name, lineterm="", n=2)
        ]

    # -- how it looks ------------------------------------------------------

    def title(self, glyphs) -> str:
        mark = {LIVE: glyphs.arrow, DONE: glyphs.tick,
                FAILED: glyphs.cross}[self.state]
        where = self.path or "(unnamed)"
        return f"{mark} {self.tool} {glyphs.dot} {where}"

    def summary(self, glyphs) -> str:
        """The one line that goes into the transcript when the edit ends."""
        if self.state == FAILED:
            detail = self.error.splitlines()[0][:80] if self.error else "failed"
            return f"{self.title(glyphs)} {glyphs.dot} {detail}"
        added, removed = self.counts()
        return f"{self.title(glyphs)} {glyphs.dot} +{added} -{removed}"

    def body(self, width: int, rows: int = MAX_LIVE_ROWS) -> list[str]:
        """The diff, trimmed to what fits.

        While streaming this is the *tail*: the interesting end of a file
        being written is the part just written. Finished, it is the head,
        because then the question is what the edit did rather than where it
        has got to.
        """
        lines = self.diff_lines()
        if not lines:
            return []
        room = max(8, width - 4)
        clipped = [fit(line, room) for line in lines]
        if len(clipped) <= rows:
            return clipped
        if self.live:
            return clipped[-rows:]
        return clipped[:rows - 1] + [f"... {len(clipped) - rows + 1} more lines"]

    def render(self, glyphs, width: int, expanded: bool = False) -> list[str]:
        """The whole card: a framed title, the diff, and a state line."""
        inner = max(20, min(width, 100)) - 2
        top = f"{glyphs.tl}{glyphs.hbar}{glyphs.hbar} "
        from rich.cells import cell_len

        title = fit(self.title(glyphs), inner - 4)
        rule = glyphs.hbar * max(0, inner - cell_len(title) - 4)
        out = [f"{top}{title} {rule}{glyphs.tr}"]
        rows = 10_000 if expanded else MAX_LIVE_ROWS
        for line in self.body(inner, rows):
            out.append(f"{glyphs.vbar} {line}")
        if self.live:
            out.append(f"{glyphs.vbar} streaming{glyphs.ellipsis}")
        else:
            added, removed = self.counts()
            out.append(f"{glyphs.vbar} +{added} -{removed}"
                       + (f"  {self.error.splitlines()[0][:40]}"
                          if self.error else ""))
        out.append(f"{glyphs.bl}{glyphs.hbar * max(0, inner)}{glyphs.br}")
        return out


def fit(text: str, cells: int) -> str:
    """``text`` trimmed to ``cells`` *display columns*, not codepoints.

    A CJK character occupies two columns and a combining accent none, so
    slicing by len() overflowed the card's border by up to double on a
    Japanese filename and produced a box that wrapped onto the next row.
    """
    from rich.cells import cell_len

    if cell_len(text) <= cells:
        return text
    out, used = [], 0
    for char in text:
        width = cell_len(char)
        if used + width > cells:
            break
        out.append(char)
        used += width
    return "".join(out)


def is_edit(tool: str) -> bool:
    return (tool or "").strip().lower() in EDIT_TOOLS


def read_before(workspace: Path, raw_path: str, boundary=None) -> str:
    """The file as it stands, for the diff to be against something.

    The path comes from the model's tool arguments, so it is checked against
    the same boundary the tools use before anything is opened. The tool would
    refuse to *write* outside the workspace, but this reads independently of
    it, and without the check a call naming ``../../../etc/shadow`` had its
    contents read and drawn into the card -- the write refused, the file
    disclosed anyway.

    Best effort otherwise, by design: a new file, an unreadable one or one
    outside the boundary all mean "nothing to compare with", which makes
    every line an addition -- the honest reading in each case.
    """
    if not raw_path:
        return ""
    try:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        candidate = Path(os.path.normpath(str(candidate)))
        if boundary is not None:
            if not boundary.contains(candidate):
                return ""
        elif not _within(workspace, candidate):
            return ""
        return candidate.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _within(workspace: Path, candidate: Path) -> bool:
    """Fallback containment check for callers with no boundary to hand.

    Resolves first: a symlink pointing out of the workspace is outside it,
    whatever its name says.
    """
    try:
        candidate.resolve().relative_to(workspace.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False
