"""Tool contract.

Each tool is atomic: one Pydantic input schema, one typed result, no shared
state, no knowledge of the agent loop. That is what lets the registry render
them for three different transports (Ollama native tools, Hermes prompted
tool calls, and the /tools help screen) from a single definition.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Type

from pydantic import BaseModel, ValidationError


@dataclass
class ToolResult:
    ok: bool
    output: str
    """What the model sees. Keep it terse; context is the scarce resource."""

    display: str = ""
    """What the user sees, when it should differ (e.g. a rendered diff)."""

    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, output: str, display: str = "", **meta) -> "ToolResult":
        return cls(ok=True, output=output, display=display, metadata=meta)

    @classmethod
    def failure(cls, error: str, **meta) -> "ToolResult":
        # The error goes in `output` too: the model only reads that field, and
        # a tool failing silently is how agents get stuck in loops.
        return cls(ok=False, output=f"ERROR: {error}", error=error, metadata=meta)


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[Type[BaseModel]]

    mutating: ClassVar[bool] = False
    """Whether this tool changes the world. Drives the permission prompt."""

    concurrency_safe: ClassVar[bool] = True
    """Whether several calls to this tool may run at once. Read-only tools
    can; anything that writes must not."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    @abstractmethod
    async def run(self, args: BaseModel) -> ToolResult: ...

    # -- schema rendering --------------------------------------------------

    def json_schema(self) -> dict:
        schema = self.Input.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def signature(self) -> str:
        """One-line form used in the Hermes prompt and the /tools screen."""
        props = self.json_schema().get("properties", {})
        required = set(self.json_schema().get("required", []))
        parts = []
        for key, spec in props.items():
            kind = spec.get("type", "any")
            parts.append(f"{key}: {kind}" if key in required else f"{key}?: {kind}")
        return f"{self.name}({', '.join(parts)})"

    # -- invocation --------------------------------------------------------

    def validate(self, raw: dict) -> BaseModel:
        return self.Input.model_validate(raw)

    async def invoke(self, raw: dict, timeout: float = 120.0) -> ToolResult:
        """Validate, run, and turn every failure into a result the model can act on."""
        try:
            args = self.validate(raw)
        except ValidationError as exc:
            return ToolResult.failure(
                f"Invalid arguments for {self.name}. {_explain_validation(exc)}\n"
                f"Expected: {self.signature()}"
            )
        try:
            return await asyncio.wait_for(self.run(args), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult.failure(f"{self.name} timed out after {timeout:.0f}s")
        except Exception as exc:  # a crashing tool must not kill the session
            return ToolResult.failure(f"{self.name} raised {type(exc).__name__}: {exc}")

    # -- path safety -------------------------------------------------------

    def resolve_path(self, raw: str) -> Path:
        """Resolve a model-supplied path, refusing to escape the workspace.

        The model is not adversarial, but it is frequently confused, and a
        confused agent writing to ``../../etc`` is the same problem as a
        malicious one.
        """
        candidate = Path(raw).expanduser()
        full = candidate if candidate.is_absolute() else (self.workspace / candidate)
        full = Path(os.path.normpath(str(full)))
        try:
            full.resolve().relative_to(self.workspace)
        except ValueError:
            raise PermissionError(
                f"{raw!r} is outside the workspace ({self.workspace}). "
                "Tools may only touch files in the project directory."
            ) from None
        return full

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.workspace))
        except ValueError:
            return str(path)


def _explain_validation(exc: ValidationError) -> str:
    bits = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        bits.append(f"{loc}: {err['msg']}")
    return "; ".join(bits)
