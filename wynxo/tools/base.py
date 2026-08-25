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

from ..schema import Schema, ValidationError
from ..scope import Boundary, Scope
from ..secrets import Shield


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
    Input: ClassVar[Type[Schema]]

    mutating: ClassVar[bool] = False
    """Whether this tool changes the world. Drives the permission prompt."""

    internal: ClassVar[bool] = False
    """Whether the only thing it changes is wynxo's own state -- its memory,
    its plan -- rather than the user's files. Internal writes are not worth a
    permission prompt, and blocking them in plan mode would mean a read-only
    session could never write down what it learned."""

    concurrency_safe: ClassVar[bool] = True
    """Whether several calls to this tool may run at once. Read-only tools
    can; anything that writes must not."""

    DEFAULT_TIMEOUT: ClassVar[float] = 120.0
    """How long a tool that says nothing about it may take."""

    TIMEOUT_GRACE: ClassVar[float] = 30.0
    """Slack over a tool's own timeout, so its handling runs first: a tool
    that knows it has timed out reports what it saw, and an outer cap firing
    at the same moment would replace that with a bare "timed out"."""

    def __init__(self, workspace: Path, boundary: Boundary | None = None,
                 shield: "Shield | None" = None):
        self.workspace = workspace.resolve()
        # Without an explicit boundary, confine to the workspace -- the safe
        # reading of "no scope was chosen".
        self.boundary = boundary or Boundary(scope=Scope.FOLDER, root=self.workspace)
        # Likewise: no shield given means the protective one, not none. A
        # tool built in a test or by future code should not leak by default.
        self.shield = shield if shield is not None else Shield(self.workspace)
        self.on_output = None
        """Optional async hook the agent sets so a long-running tool can show
        its output while it is still running, instead of only at the end."""
        self.context_left = 0
        """Tokens of context still free, set by the agent before each call.
        Zero means unknown, and every check treats that as "do not
        interfere" -- a guard that fires on missing information would be
        worse than no guard."""

    @abstractmethod
    async def run(self, args: Schema) -> ToolResult: ...

    # -- schema rendering --------------------------------------------------

    def json_schema(self) -> dict:
        return self.Input.json_schema()

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

    def validate(self, raw: dict) -> Schema:
        return self.Input.validate(raw)

    def timeout_for(self, args: Schema) -> float:
        """How long this particular call may take.

        A tool whose own input names a timeout is the authority on it. The
        shell accepts up to nine hundred seconds, and a flat two-minute cap
        out here made that a lie: a five-minute test suite was killed at two
        minutes, and the model was told "shell timed out after 120s" without
        the output that would have said how far it got.
        """
        own = getattr(args, "timeout", None)
        if isinstance(own, (int, float)) and not isinstance(own, bool) and own > 0:
            return float(own) + self.TIMEOUT_GRACE
        return self.DEFAULT_TIMEOUT

    async def invoke(self, raw: dict, timeout: float | None = None) -> ToolResult:
        """Validate, run, and turn every failure into a result the model can act on.

        ``timeout`` overrides what the tool would choose for itself; leaving
        it out is the usual case and asks the tool.
        """
        try:
            args = self.validate(raw)
        except ValidationError as exc:
            return ToolResult.failure(
                f"Invalid arguments for {self.name}. {_explain_validation(exc)}\n"
                f"Expected: {self.signature()}"
            )
        limit = self.timeout_for(args) if timeout is None else timeout
        try:
            return await asyncio.wait_for(self.run(args), timeout=limit)
        except asyncio.TimeoutError:
            return ToolResult.failure(f"{self.name} timed out after {limit:.0f}s")
        except PermissionError as exc:
            # An expected answer, not a crash. The boundary's message already
            # says what was refused and how to widen it; announcing it as
            # "read_file raised PermissionError" reads like wynxo broke, and
            # buries the sentence the model needs in order to do better.
            return ToolResult.failure(str(exc))
        except Exception as exc:  # a crashing tool must not kill the session
            return ToolResult.failure(f"{self.name} raised {type(exc).__name__}: {exc}")

    # -- path safety -------------------------------------------------------

    def resolve_path(self, raw: str) -> Path:
        """Resolve a model-supplied path, refusing to leave the boundary.

        The model is not adversarial, but it is frequently confused, and a
        confused agent writing to ``../../etc`` is the same problem as a
        malicious one. This check is not waivable by a permission mode:
        scope is the wall, mode is only how often it knocks.
        """
        candidate = Path(raw).expanduser()
        full = candidate if candidate.is_absolute() else (self.workspace / candidate)
        full = Path(os.path.normpath(str(full)))
        if not self.boundary.contains(full):
            raise PermissionError(self.boundary.reject(raw)) from None
        return full

    def relative(self, path: Path) -> str:
        """A short display path: relative to the workspace when it can be.

        The workspace itself comes back as its own directory name rather than
        ".", because "." is a directory' reads as a bug report about nothing.
        """
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            return str(path)      # unresolvable: show what was asked for
        for base in (self.workspace, self.boundary.root):
            try:
                relative = resolved.relative_to(base)
            except ValueError:
                continue
            text = str(relative)
            return f"{resolved.name}/" if text == "." else text
        return str(path)


def _explain_validation(exc: ValidationError) -> str:
    bits = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        bits.append(f"{loc}: {err['msg']}")
    return "; ".join(bits)
