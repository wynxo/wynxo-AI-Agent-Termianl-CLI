"""Extracting structure from what a local model actually emits.

Three separate problems, all of which bite hard on local models:

1.  **Thinking.** Qwen3 and the R1 distills wrap reasoning in ``<think>``
    tags inside ``content`` when the server is older than Ollama's native
    ``thinking`` field. It must be split out or it ends up quoted back to
    the user as if it were an answer.

2.  **Hermes tool calls.** Qwen3, Hermes and most tool-tuned open models were
    trained on ``<tool_call>{json}</tool_call>``. When a model's Ollama
    template does not wire that into the native ``tool_calls`` field, the
    calls arrive as plain text and are invisible unless parsed here.

3.  **Broken JSON.** Local models emit trailing commas, single quotes,
    Python literals and unescaped newlines at a rate that would be a
    scandal in a hosted API. Repairing what is obviously repairable is the
    single highest-leverage thing in this whole file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
OPEN_THINK = re.compile(r"<(think|thinking|reasoning)>(.*)$", re.DOTALL | re.IGNORECASE)
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
# Some checkpoints drop the closing tag when they hit the token limit.
UNCLOSED_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*)$", re.DOTALL | re.IGNORECASE)
FENCED_JSON = re.compile(r"```(?:json|tool_code)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw: str = ""
    """The original text, kept so a repair prompt can quote it back."""


@dataclass
class ParsedTurn:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    """Fragments that looked like tool calls but could not be salvaged."""

    @property
    def has_work(self) -> bool:
        return bool(self.tool_calls)


_HIDDEN_TAGS = ("think", "thinking", "reasoning", "tool_call")


class LiveContentFilter:
    """Strips <think>/<thinking>/<reasoning>/<tool_call> blocks out of a
    live token stream, without ever showing a partial tag.

    parse_turn() already strips these out of the *final* assembled text --
    but a model with no native ``thinking`` or ``tools`` support writes them
    straight into plain content instead, and streaming that unfiltered means
    every tool call and every thought shows up as raw protocol markup in the
    middle of the answer instead of prose. This applies the identical rule
    live, one chunk at a time, so what the user watches stream and what
    parse_turn() later extracts can never disagree.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.open_tag: str | None = None

    def feed(self, text: str) -> str:
        """Consume one chunk of raw content; return what is safe to show."""
        self.buffer += text
        out: list[str] = []
        while True:
            if self.open_tag is None:
                found = self._find_open_tag()
                if found is None:
                    break
                start, tag, marker_len = found
                out.append(self.buffer[:start])
                self.buffer = self.buffer[start + marker_len:]
                self.open_tag = tag
                continue
            close = f"</{self.open_tag}>"
            end = self.buffer.lower().find(close)
            if end == -1:
                break
            self.buffer = self.buffer[end + len(close):]
            self.open_tag = None
        if self.open_tag is None:
            safe = self._safe_emit_length()
            out.append(self.buffer[:safe])
            self.buffer = self.buffer[safe:]
        return "".join(out)

    def finish(self) -> str:
        """Flush whatever is left once the stream ends.

        Text still waiting inside an unterminated tag is a truncated thought
        or tool call, not an answer -- discarded, the same way parse_turn()
        drops an unclosed block rather than showing it.
        """
        if self.open_tag is not None:
            self.buffer = ""
            self.open_tag = None
            return ""
        remainder, self.buffer = self.buffer, ""
        return remainder

    def _find_open_tag(self) -> tuple[int, str, int] | None:
        """The earliest complete opening tag in the buffer, if any."""
        lowered = self.buffer.lower()
        best: tuple[int, str, int] | None = None
        for tag in _HIDDEN_TAGS:
            marker = f"<{tag}>"
            idx = lowered.find(marker)
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, tag, len(marker))
        return best

    def _safe_emit_length(self) -> int:
        """How much of the buffer is safe to emit now.

        A trailing ``<`` that could still grow into a recognised opening tag
        is held back -- otherwise a tag split across two chunks (``<tool_c``
        then ``all>``) would leak its first half before the second arrives.
        """
        idx = self.buffer.rfind("<")
        if idx == -1:
            return len(self.buffer)
        tail = self.buffer[idx:].lower()
        for tag in _HIDDEN_TAGS:
            if f"<{tag}>".startswith(tail):
                return idx
        return len(self.buffer)


def split_thinking(text: str) -> tuple[str, str]:
    """Return ``(visible_content, thinking)``."""
    thoughts: list[str] = []

    def take(match: re.Match) -> str:
        thoughts.append(match.group(2).strip())
        return ""

    content = THINK_BLOCK.sub(take, text)

    # An unterminated block means the model was cut off mid-thought. Everything
    # after the opening tag is reasoning, not an answer.
    if open_match := OPEN_THINK.search(content):
        thoughts.append(open_match.group(2).strip())
        content = content[: open_match.start()]

    return content.strip(), "\n\n".join(t for t in thoughts if t).strip()


def repair_json(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of near-JSON. Returns None if it is truly hopeless."""
    text = raw.strip()
    if not text:
        return None

    # Strip a code fence the model wrapped around the object.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z_]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    attempts = [text]

    # Trailing commas before a closing brace or bracket.
    attempts.append(re.sub(r",(\s*[}\]])", r"\1", text))

    # Python literals leaking through from a model that thinks in Python.
    pythonish = re.sub(r"\bTrue\b", "true", text)
    pythonish = re.sub(r"\bFalse\b", "false", pythonish)
    pythonish = re.sub(r"\bNone\b", "null", pythonish)
    attempts.append(pythonish)
    attempts.append(re.sub(r",(\s*[}\]])", r"\1", pythonish))

    # Single-quoted keys and values, but only when there are no double quotes
    # to destroy -- otherwise this mangles legitimate apostrophes in strings.
    if "'" in text and '"' not in text:
        attempts.append(text.replace("'", '"'))

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed

    # Literal newlines inside a JSON string are the most common failure of all
    # (a model writing file contents). Escape them and retry.
    escaped = _escape_newlines_in_strings(text)
    if escaped != text:
        try:
            parsed = json.loads(escaped)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # Truncated output: close whatever is still open and see if it parses.
    if closed := _close_unbalanced(text):
        try:
            parsed = json.loads(closed)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _escape_newlines_in_strings(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch == "\t":
            out.append("\\t")
            continue
        if in_string and ch == "\r":
            continue
        out.append(ch)
    return "".join(out)


def _close_unbalanced(text: str) -> str | None:
    """Close a truncated JSON object so it can at least be inspected."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if not stack and not in_string:
        return None
    tail = '"' if in_string else ""
    for opener in reversed(stack):
        tail += "}" if opener == "{" else "]"
    return text + tail


def _normalise_call(payload: dict, raw: str, index: int) -> ToolCall | None:
    """Accept the several shapes models use to name a tool and its args."""
    # OpenAI-ish nesting: {"function": {"name": ..., "arguments": ...}}
    if "function" in payload and isinstance(payload["function"], dict):
        inner = payload["function"]
        name = inner.get("name")
        args = inner.get("arguments", inner.get("parameters", {}))
        call_id = payload.get("id", "")
    else:
        name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
        args = payload.get("arguments")
        if args is None:
            args = payload.get("parameters")
        if args is None:
            args = payload.get("args")
        if args is None:
            args = payload.get("input", {})
        call_id = payload.get("id", "")

    if not name or not isinstance(name, str):
        return None

    # Arguments sometimes arrive as a JSON string rather than an object.
    if isinstance(args, str):
        args = repair_json(args) or {}
    if not isinstance(args, dict):
        args = {}

    return ToolCall(
        name=name.strip(),
        arguments=args,
        call_id=call_id or f"call_{index}",
        raw=raw,
    )


def parse_turn(
    content: str,
    thinking: str = "",
    native_tool_calls: list[dict] | None = None,
) -> ParsedTurn:
    """Turn a raw assistant response into structured work.

    ``native_tool_calls`` are Ollama's own parsed calls, which are trusted
    when present. Text parsing runs regardless, because a model will
    occasionally emit one call natively and another as text in the same turn.
    """
    turn = ParsedTurn()

    visible, inline_thinking = split_thinking(content)
    turn.thinking = "\n\n".join(t for t in (thinking.strip(), inline_thinking) if t)

    index = 0
    seen: set[str] = set()

    for native in native_tool_calls or []:
        call = _normalise_call(native, json.dumps(native), index)
        if call:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
            if key not in seen:
                seen.add(key)
                turn.tool_calls.append(call)
                index += 1

    # Hermes-style blocks in the visible text.
    blocks = TOOL_CALL_BLOCK.findall(visible)
    if not blocks:
        if unclosed := UNCLOSED_TOOL_CALL.search(visible):
            blocks = [unclosed.group(1)]

    for block in blocks:
        payload = repair_json(block)
        if payload is None:
            turn.malformed.append(block.strip())
            continue
        call = _normalise_call(payload, block, index)
        if call is None:
            turn.malformed.append(block.strip())
            continue
        key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
        if key in seen:
            continue
        seen.add(key)
        turn.tool_calls.append(call)
        index += 1

    visible = TOOL_CALL_BLOCK.sub("", visible)
    visible = UNCLOSED_TOOL_CALL.sub("", visible)

    turn.content = visible.strip()
    return turn


def strip_soft_switches(text: str) -> str:
    """Remove Qwen3's ``/think`` and ``/no_think`` markers from displayed text."""
    return re.sub(r"\s*/(no_?think|think)\b", "", text, flags=re.IGNORECASE).strip()
