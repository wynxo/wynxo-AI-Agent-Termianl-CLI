"""What the agent is allowed to touch, and how much it asks first.

Two independent dials, deliberately kept apart:

* **Scope** is the boundary. Where may tools read and write at all? This is
  enforced on every path, cannot be waived by a permission mode, and is the
  thing standing between a confused model and your home directory.

* **Mode** is the friction inside that boundary. How much does it ask before
  acting? A wide scope with careful prompts is very different from a narrow
  scope with none, and people want both combinations.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Scope(Enum):
    FOLDER = "folder"
    """Only the directory wynxo was started in. The default."""

    REPO = "repo"
    """The whole git repository, found by walking up for a .git. Lets the
    agent see a sibling package when you started in a subdirectory."""

    MACHINE = "machine"
    """No path restriction at all. Everything the user account can reach."""

    @classmethod
    def parse(cls, value: str) -> "Scope":
        key = value.strip().lower()
        aliases = {"dir": "folder", "cwd": "folder", "project": "folder",
                   "git": "repo", "repository": "repo",
                   "all": "machine", "system": "machine", "pc": "machine",
                   "everything": "machine"}
        key = aliases.get(key, key)
        try:
            return cls(key)
        except ValueError:
            raise KeyError(
                f"unknown scope {value!r}; choose folder, repo or machine"
            ) from None


class Mode(Enum):
    PLAN = "plan"
    """Read-only. No writes, no commands. It investigates and proposes."""

    MANUAL = "manual"
    """Ask before every write or command. The default."""

    AUTO = "auto"
    """Edit files inside scope without asking; still ask for shell commands
    and anything that reaches off the machine."""

    REVIEW = "review"
    """Edit freely, then show every change together at the end of the turn.

    The middle ground between manual and auto: manual interrupts a ten-file
    refactor ten times, and auto never shows you the shape of what happened.
    This lets the work finish, then puts the whole diff in front of you with
    one decision to make."""

    YOLO = "yolo"
    """Never ask. For a container or a scratch checkout."""

    @classmethod
    def parse(cls, value: str) -> "Mode":
        key = value.strip().lower()
        aliases = {"readonly": "plan", "read-only": "plan", "safe": "plan",
                   "ask": "manual", "default": "manual", "careful": "manual",
                   "edit": "auto", "acceptedits": "auto", "accept-edits": "auto",
                   "auto-accept": "auto",
                   "batch": "review", "diff": "review", "after": "review",
                   "all": "yolo", "never-ask": "yolo", "bypass": "yolo"}
        key = aliases.get(key, key)
        try:
            return cls(key)
        except ValueError:
            raise KeyError(
                f"unknown mode {value!r}; choose plan, manual, auto or yolo"
            ) from None

    def describe(self) -> str:
        return {
            Mode.PLAN: "read-only -- investigates and proposes, never writes",
            Mode.MANUAL: "asks before every write and command",
            Mode.AUTO: "edits freely in scope, still asks to run commands",
            Mode.REVIEW: "edits freely, shows you every change together at the end",
            Mode.YOLO: "never asks",
        }[self]


def git_root(start: Path) -> Path | None:
    """The repository containing ``start``, if any."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None


@dataclass
class Boundary:
    """The resolved answer to 'where may tools go?'."""

    scope: Scope
    root: Path
    """The directory tools are confined to. Meaningless when unrestricted."""

    unrestricted: bool = False

    def describe(self) -> str:
        if self.unrestricted:
            return "the whole machine"
        if self.scope is Scope.REPO:
            return f"the repository at {self.root}"
        return str(self.root)

    def contains(self, path: Path) -> bool:
        if self.unrestricted:
            return True
        try:
            Path(os.path.normpath(str(path))).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def reject(self, raw: str) -> str:
        """The message a tool shows when a path falls outside."""
        if self.scope is Scope.FOLDER:
            return (
                f"{raw!r} is outside the project directory ({self.root}). "
                "Start wynxo with --scope repo to reach the whole repository, "
                "or --scope machine to lift the restriction entirely."
            )
        if self.scope is Scope.REPO:
            return (
                f"{raw!r} is outside the repository ({self.root}). "
                "Start wynxo with --scope machine to lift the restriction."
            )
        return f"{raw!r} is not reachable."


def resolve(workspace: Path, scope: Scope) -> Boundary:
    """Turn a scope choice into a concrete boundary."""
    workspace = workspace.resolve()

    if scope is Scope.MACHINE:
        return Boundary(scope=scope, root=workspace, unrestricted=True)

    if scope is Scope.REPO:
        root = git_root(workspace)
        if root is None:
            # No repository to widen to; fall back rather than silently
            # granting more than was asked for.
            return Boundary(scope=Scope.FOLDER, root=workspace)
        return Boundary(scope=scope, root=root.resolve())

    return Boundary(scope=Scope.FOLDER, root=workspace)
