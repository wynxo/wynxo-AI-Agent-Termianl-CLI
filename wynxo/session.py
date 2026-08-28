"""Conversation state, token accounting and compaction."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .coerce import as_float, as_int, as_list, as_text
from .config import atomic_write, data_dir


MIN_WORTH_COMPACTING = 400
"""Tokens the older half must be worth before summarising it earns its
model call. Below this, compacting costs more than it recovers."""


def estimate_tokens(text: str) -> int:
    """Rough token count without pulling in a tokenizer.

    Deliberately pessimistic: ~3.6 chars/token rather than the usual 4, since
    code tokenizes worse than prose and the cost of *under*-estimating is a
    silently truncated context, which is the failure this whole module exists
    to prevent.
    """
    return max(1, int(len(text) / 3.6))


def message_tokens(message: dict) -> int:
    total = estimate_tokens(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        total += estimate_tokens(json.dumps(call, default=str))
    return total + 4  # role and framing overhead


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    generation_seconds: float = 0.0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def tokens_per_second(self) -> float:
        if self.generation_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.generation_seconds

    def add_chunk(self, prompt: int, completion: int, duration_ns: int) -> None:
        # Ollama reports cumulative counts for the whole request in the final
        # chunk, so these are assignments per request, not running sums.
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.generation_seconds += duration_ns / 1e9
        self.requests += 1


@dataclass
class Session:
    workspace: Path
    system_prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    compactions: int = 0

    # -- building the wire format -----------------------------------------

    def wire(self) -> list[dict]:
        """The message list as Ollama wants it."""
        out = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return out + self.messages

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: list[dict] | None = None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_result(self, name: str, content: str, call_id: str = "") -> None:
        # Ollama's Message type names this field `tool_name` (api/types.go).
        # `name` is silently ignored, so the model cannot tell which tool a
        # result came from -- which matters as soon as a turn calls two.
        message: dict[str, Any] = {"role": "tool", "content": content, "tool_name": name}
        if call_id:
            message["tool_call_id"] = call_id
        self.messages.append(message)

    # -- size --------------------------------------------------------------

    def token_estimate(self) -> int:
        base = estimate_tokens(self.system_prompt)
        return base + sum(message_tokens(m) for m in self.messages)

    def should_compact(self, budget: int, num_ctx: int) -> bool:
        limit = min([n for n in (budget, num_ctx) if n and n > 0] or [8000])
        if self.token_estimate() <= limit * 0.75:
            return False
        older, _ = self.slice_for_summary()
        if not older:
            return False
        removable = sum(message_tokens(m) for m in older)
        return removable > max(MIN_WORTH_COMPACTING,
                               self.token_estimate() * 0.15)

    # -- compaction --------------------------------------------------------

    def slice_for_summary(self, keep_recent: int = 6) -> tuple[list[dict], list[dict]]:
        """Split into (to summarise, to keep verbatim).

        The tail is kept whole because the model needs exact recent state --
        a summarised tool result is worse than useless when it was the thing
        the next step depends on. A complete assistant-tool exchange is also
        atomic, even when the chosen `keep_recent` boundary falls immediately
        before or inside that exchange.
        """
        if len(self.messages) <= keep_recent:
            return [], self.messages

        cut = len(self.messages) - keep_recent

        # If the boundary lands on an assistant tool call, keep that complete
        # exchange. If it lands on a tool result, walk backwards to its owning
        # assistant. Likewise, if the boundary is immediately after a tool
        # result, move it backwards so that result is never left in `older`
        # without its assistant call. Keeping a few extra messages is safer
        # than producing a structurally invalid Ollama history.
        exchange_start = None
        if cut < len(self.messages):
            candidate = self.messages[cut]
            if candidate.get("role") == "assistant" and candidate.get("tool_calls"):
                exchange_start = cut
            elif candidate.get("role") == "tool":
                cursor = cut - 1
                while cursor >= 0 and self.messages[cursor].get("role") == "tool":
                    cursor -= 1
                if (cursor >= 0
                        and self.messages[cursor].get("role") == "assistant"
                        and self.messages[cursor].get("tool_calls")):
                    exchange_start = cursor
            elif cut > 0 and self.messages[cut - 1].get("role") == "tool":
                cursor = cut - 1
                while cursor >= 0 and self.messages[cursor].get("role") == "tool":
                    cursor -= 1
                if (cursor >= 0
                        and self.messages[cursor].get("role") == "assistant"
                        and self.messages[cursor].get("tool_calls")):
                    exchange_start = cursor

        if exchange_start is not None:
            cut = exchange_start

        return self.messages[:cut], self.messages[cut:]

    def apply_compaction(self, summary: str, kept: list[dict]) -> None:
        # Preserve the objective and a compact audit trail even when the model
        # returns an overly terse summary. Recent exact messages remain intact.
        summary = summary.strip() or "No summary was returned; preserve the recent conversation below."
        self.messages = [
            {
                "role": "user",
                "content": (
                    "[Earlier conversation, condensed to save context. Treat this "
                    "as established fact.]\n\n" + summary
                ),
            },
            {"role": "assistant", "content": "Understood. Continuing."},
            *kept,
        ]
        self.compactions += 1

    # -- persistence -------------------------------------------------------

    def path(self) -> Path:
        return data_dir() / "sessions" / f"{self.session_id}.json"

    def save(self) -> Path | None:
        try:
            path = self.path()
            atomic_write(
                path,
                json.dumps(
                    {
                        "session_id": self.session_id,
                        "workspace": str(self.workspace),
                        "created_at": self.created_at,
                        "updated_at": time.time(),
                        "compactions": self.compactions,
                        "messages": self.messages,
                        "usage": {
                            "prompt_tokens": self.usage.prompt_tokens,
                            "completion_tokens": self.usage.completion_tokens,
                            "requests": self.usage.requests,
                            "tool_calls": self.usage.tool_calls,
                        },
                    },
                    indent=2,
                    default=str,
                ),
            )
            return path
        except OSError:
            # Never let a full disk or a read-only home directory end a session.
            return None

    @classmethod
    def load(cls, session_id: str, workspace: Path) -> "Session | None":
        path = data_dir() / "sessions" / f"{session_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None

        session = cls(
            workspace=workspace,
            session_id=as_text(data.get("session_id")) or session_id,
            created_at=as_float(data.get("created_at"), time.time()),
            compactions=as_int(data.get("compactions")),
        )
        session.messages = [m for m in as_list(data.get("messages"))
                            if isinstance(m, dict)]
        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        session.usage = Usage(
            prompt_tokens=as_int(usage.get("prompt_tokens")),
            completion_tokens=as_int(usage.get("completion_tokens")),
            requests=as_int(usage.get("requests")),
            tool_calls=as_int(usage.get("tool_calls")),
        )
        return session

    @staticmethod
    def recent(limit: int = 20) -> list[dict]:
        directory = data_dir() / "sessions"
        if not directory.is_dir():
            return []
        def when(path: Path) -> float:
            try:
                return -path.stat().st_mtime
            except OSError:
                return 0.0

        out = []
        for path in sorted(directory.glob("*.json"), key=when)[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                messages = [m for m in as_list(data.get("messages"))
                            if isinstance(m, dict)]
                first = next((as_text(m.get("content")) for m in messages
                              if m.get("role") == "user"), "")
                out.append({
                    "session_id": as_text(data.get("session_id")) or path.stem,
                    "workspace": as_text(data.get("workspace")) or "?",
                    "updated_at": as_float(data.get("updated_at"), 0.0),
                    "messages": len(messages),
                    "preview": first[:70].replace("\n", " "),
                })
            except (OSError, json.JSONDecodeError, ValueError):
                continue
        return out
