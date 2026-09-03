"""What the agent is allowed to touch, and how much it asks first."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Scope(Enum):
    FOLDER = "folder"
    REPO = "repo"
    MACHINE = "machine"

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
    MANUAL = "manual"
    AUTO = "auto"
    REVIEW = "review"
    YOLO = "yolo"

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
                "unknown mode %r; choose plan, manual, auto, review or yolo" % value
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
    scope: Scope
    root: Path
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
        except (OSError, RuntimeError):
            return False

    def reject(self, raw: str) -> str:
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
    workspace = workspace.resolve()
    if scope is Scope.MACHINE:
        return Boundary(scope=scope, root=workspace, unrestricted=True)
    if scope is Scope.REPO:
        root = git_root(workspace)
        if root is None:
            return Boundary(scope=Scope.FOLDER, root=workspace)
        return Boundary(scope=scope, root=root.resolve())
    return Boundary(scope=Scope.FOLDER, root=workspace)
