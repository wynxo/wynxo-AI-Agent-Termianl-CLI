"""Conversation state, token accounting and compaction."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import data_dir


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
        message: dict[str, Any] = {"role": "tool", "content": content, "name": name}
        if call_id:
            message["tool_call_id"] = call_id
        self.messages.append(message)

    # -- size --------------------------------------------------------------

    def token_estimate(self) -> int:
        base = estimate_tokens(self.system_prompt)
        return base + sum(message_tokens(m) for m in self.messages)

    def should_compact(self, budget: int, num_ctx: int) -> bool:
        limit = budget if budget > 0 else num_ctx
        # Compact at 75%: leave room for the reply itself plus the next few
        # tool results, or compaction triggers only once it is already too late.
        return self.token_estimate() > limit * 0.75

    # -- compaction --------------------------------------------------------

    def slice_for_summary(self, keep_recent: int = 6) -> tuple[list[dict], list[dict]]:
        """Split into (to summarise, to keep verbatim).

        The tail is kept whole because the model needs exact recent state --
        a summarised tool result is worse than useless when it was the thing
        the next step depends on.
        """
        if len(self.messages) <= keep_recent:
            return [], self.messages

        cut = len(self.messages) - keep_recent
        # Never split an assistant turn from the tool results that answer it;
        # a dangling tool message with no matching call confuses every model.
        while cut < len(self.messages) and self.messages[cut].get("role") == "tool":
            cut += 1
        return self.messages[:cut], self.messages[cut:]

    def apply_compaction(self, summary: str, kept: list[dict]) -> None:
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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
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
                encoding="utf-8",
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
        session = cls(
            workspace=workspace,
            session_id=data.get("session_id", session_id),
            created_at=data.get("created_at", time.time()),
            compactions=data.get("compactions", 0),
        )
        session.messages = data.get("messages", [])
        usage = data.get("usage") or {}
        session.usage = Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            requests=usage.get("requests", 0),
            tool_calls=usage.get("tool_calls", 0),
        )
        return session

    @staticmethod
    def recent(limit: int = 20) -> list[dict]:
        directory = data_dir() / "sessions"
        if not directory.is_dir():
            return []
        out = []
        for path in sorted(directory.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            first = next(
                (m["content"] for m in data.get("messages", []) if m.get("role") == "user"),
                "",
            )
            out.append({
                "session_id": data.get("session_id", path.stem),
                "workspace": data.get("workspace", "?"),
                "updated_at": data.get("updated_at", 0),
                "messages": len(data.get("messages", [])),
                "preview": str(first)[:70].replace("\n", " "),
            })
        return out
