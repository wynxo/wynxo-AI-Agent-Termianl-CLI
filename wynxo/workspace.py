"""Execution/workspace identity kept separate from model providers.

Wynxo can talk to Ollama while operating on a local checkout or on a
GitHub-backed workspace.  These are intentionally independent concepts:
selecting GitHub does not imply a remote shell, and selecting Ollama does not
choose where edits happen.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    provider: str = "local"
    root: Path | None = None
    repository: str = ""
    branch: str = ""
    execution: str = "local"
    api_only: bool = False

    @property
    def label(self) -> str:
        if self.provider == "github":
            repo = self.repository or "repository"
            suffix = f"#{self.branch}" if self.branch else ""
            return f"github:{repo}{suffix}"
        if self.root is None:
            return "local"
        try:
            return str(self.root.expanduser().resolve())
        except OSError:
            return str(self.root)

    @property
    def prompt(self) -> str:
        if self.provider == "github":
            if self.api_only:
                return (
                    f"Workspace: GitHub {self.repository or 'repository'} "
                    f"(API-only; branch {self.branch or 'default'}). "
                    "Do not claim to have run local commands or edited a local checkout."
                )
            return (
                f"Workspace: GitHub {self.repository or 'repository'} "
                f"(local execution; branch {self.branch or 'default'}). "
                "Keep remote repository context distinct from the local checkout."
            )
        return f"Workspace: local {self.label}"

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "root": str(self.root) if self.root else "",
            "repository": self.repository,
            "branch": self.branch,
            "execution": self.execution,
            "api_only": self.api_only,
        }

    @classmethod
    def from_dict(cls, data: dict | None, fallback: Path | None = None) -> "Workspace":
        data = data if isinstance(data, dict) else {}
        root = data.get("root") or (str(fallback) if fallback else "")
        return cls(
            provider=str(data.get("provider") or "local"),
            root=Path(root) if root else None,
            repository=str(data.get("repository") or ""),
            branch=str(data.get("branch") or ""),
            execution=str(data.get("execution") or "local"),
            api_only=bool(data.get("api_only", False)),
        )
