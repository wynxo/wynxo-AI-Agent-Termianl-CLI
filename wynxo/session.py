"""Conversation state, token accounting and compaction."""

from __future__ import annotations

import contextlib
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
    """Time the model spent generating, per the server's own measurement."""
    wall_seconds: float = 0.0
    """Time the requests took altogether: load, prompt and generation."""

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def tokens_per_second(self) -> float:
        if self.generation_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.generation_seconds

    def add_chunk(self, prompt: int, completion: int, duration_ns: int,
                  eval_ns: int = 0) -> None:
        # Ollama reports cumulative counts for the whole request in the final
        # chunk, so these are assignments per request, not running sums.
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.wall_seconds += duration_ns / 1e9
        # Generation only where the server says so. total_duration includes
        # reading the weights off disk and reading the prompt, and on a
        # machine where most of the model is on the CPU those dwarf the
        # generating -- so a speed computed from it reads as a fraction of
        # the truth, which is worse than no speed at all when somebody is
        # using it to judge a change they just made.
        self.generation_seconds += (eval_ns or duration_ns) / 1e9
        self.requests += 1


_SUPERSEDE_MIN = 400
"""Below this a result is not worth collapsing. A one-line "ok" costs less
than the note explaining that it was dropped, and a conversation littered
with supersede notes is harder to read than one with a few short results
in it."""


@dataclass
class Session:
    workspace: Path
    system_prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    compactions: int = 0
    superseded_chars: int = 0
    """How much stale tool output has been collapsed, for /stats."""

    overhead: int = 0
    """Tokens the request carries that are neither a message nor the system
    prompt: the tool schemas.

    Native tool calling sends the schemas in their own wire field, not in
    the prompt, so they were invisible to every count here -- and they are
    not small. Seventeen tools is around 3,900 tokens, which at
    num_ctx=8192 is half the window that nothing was accounting for:
    wynxo reported 52% used while actually sending 8,301 tokens into 8,192,
    and Ollama truncates the front of an overflowing prompt, which is the
    system prompt. The model quietly loses its instructions and its tools,
    and nothing anywhere says so.

    Zero on the prompted path, where the tools are described in the system
    prompt and therefore already counted.
    """

    autosave: bool = False
    """Whether every message is written to disk as it is recorded.

    Off by default so that constructing a Session -- in a test, in a tool,
    anywhere -- never touches the user's session store. The agent turns it
    on for the conversation it owns, which is the one that has to survive
    being interrupted.
    """

    _pruned: bool = False
    """Whether this process has already swept the session store. Autosave
    writes after every message rather than once per turn, and globbing plus
    stat-ing the whole directory on each of those is a lot of syscalls to
    answer a question whose answer does not change within a turn."""

    # -- building the wire format -----------------------------------------

    def wire(self) -> list[dict]:
        """The message list as Ollama wants it."""
        out = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return out + self.messages

    def _recorded(self) -> None:
        """One message went in. Write the conversation out if asked to.

        Called from every method that appends, rather than from the turn
        loop, so that no future call site has to remember: if it is in the
        conversation, it is on disk. This is what makes Ctrl-C survivable --
        a turn can spend minutes across a dozen tool calls, and saving only
        when one finishes meant an interrupt threw away everything since the
        last one, the request included.
        """
        if self.autosave:
            self.save()

    def add_user(self, content: str, images: list[str] | None = None) -> None:
        """One user message, optionally carrying pictures.

        ``images`` are base64 strings, which is what Ollama's chat API takes
        directly; the OpenAI translation turns them into content parts. They
        ride on a user message rather than on the tool result that produced
        them because that is the shape every vision model is trained on, and
        the one both protocols agree about.
        """
        message: dict[str, Any] = {"role": "user", "content": content}
        if images:
            message["images"] = list(images)
        self.messages.append(message)
        self._recorded()

    def drop_images(self, keep_last: int = 1) -> int:
        """Forget all but the most recent pictures. Returns how many went.

        A screenshot is worth about a thousand tokens for as long as it
        stays in the conversation, and two screenshots of the same desktop
        taken a minute apart cannot both be current -- the newer one is
        what the screen looks like, and the older is a description of what
        it looked like. Same rule as two reads of one file, and for the
        same reason: on a small window it is the difference between
        finishing and compacting halfway.
        """
        carrying = [m for m in self.messages if m.get("images")]
        dropped = 0
        for message in carrying[:max(0, len(carrying) - keep_last)]:
            message.pop("images", None)
            message["content"] = (
                (message.get("content") or "")
                + "\n(The screenshot that was here has been dropped; it is "
                  "no longer current. Look again if you need to see.)")
            dropped += 1
        return dropped

    def add_assistant(self, content: str, tool_calls: list[dict] | None = None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)
        self._recorded()

    def add_tool_result(self, name: str, content: str, call_id: str = "",
                        subject: str = "") -> None:
        """Record what a tool returned.

        ``subject`` is what the result is *about* -- a file path, for a tool
        that returns a file's contents. Two results about the same subject
        cannot both be true: the newer one is what the file says now, and
        the older is a description of a file that has since changed or been
        re-read for a reason. Keeping both in full is how a read-edit-verify
        loop -- the ordinary shape of a coding turn -- paid for the same
        file three times over, and on an 8k window that is the difference
        between finishing a task and compacting in the middle of it.
        """
        # Ollama's Message type names this field `tool_name` (api/types.go).
        # `name` is silently ignored, so the model cannot tell which tool a
        # result came from -- which matters as soon as a turn calls two.
        message: dict[str, Any] = {"role": "tool", "content": content, "tool_name": name}
        if call_id:
            message["tool_call_id"] = call_id
        if subject:
            message["subject"] = subject
            self._supersede(subject, len(content))
        self.messages.append(message)
        self._recorded()

    def _supersede(self, subject: str, incoming: int) -> None:
        """Collapse older results about the same subject to a one-line note.

        Replaced rather than deleted: the message has to stay where it is,
        because a tool result answers a specific tool call and removing it
        would leave that call unanswered -- which is a malformed
        conversation, not a smaller one. What is left says plainly that the
        content moved further down, so the model does not read the note as
        "the file is empty".
        """
        for message in self.messages:
            if message.get("role") != "tool" or message.get("subject") != subject:
                continue
            if message.get("superseded"):
                continue
            was = len(str(message.get("content") or ""))
            if was <= _SUPERSEDE_MIN and incoming <= _SUPERSEDE_MIN:
                continue        # nothing worth reclaiming
            message["content"] = (
                f"[superseded: {subject} was read again later in this "
                f"conversation, and the current contents are further down. "
                f"{was} characters dropped from here.]")
            message["superseded"] = True
            self.superseded_chars += was - len(message["content"])

    INTERRUPTED_RESULT = ("[not run: the turn was interrupted before this "
                          "tool call was executed]")
    """What stands in for a call that never got an answer. Says plainly that
    nothing happened, so the model does not read an empty result as a tool
    that ran and returned nothing."""

    def close_open_tool_calls(self, note: str = "") -> int:
        """Give every announced tool call an answer. Returns how many it added.

        A tool call is a question the conversation has to answer: the
        assistant message announces N calls and the next N ``tool`` messages
        answer them, one each, in order -- which is the same positional
        convention `_openai_messages` uses to rebuild the ids the Ollama
        shape does not carry. _supersede() is careful never to break that
        pairing. Unwinding was not: a turn cancelled between announcing its
        calls and running them left them unanswered for good, because the
        announcement is written first. The user was told the conversation was
        intact while it was malformed, and an OpenAI-compatible server
        rejects that shape outright -- so one Ctrl-C at the wrong moment made
        every later request in the session fail.

        Called on the way out of a turn, however it ends. Idempotent: a
        conversation that is already well-formed is left alone.
        """
        added = 0
        index = 0
        while index < len(self.messages):
            message = self.messages[index]
            index += 1
            calls = (message.get("tool_calls")
                     if message.get("role") == "assistant" else None)
            if not isinstance(calls, list) or not calls:
                continue
            answered = 0
            while (index < len(self.messages)
                   and self.messages[index].get("role") == "tool"):
                answered += 1
                index += 1
            for position in range(answered, len(calls)):
                call = calls[position]
                name = str((call.get("function") or {}).get("name")
                           or call.get("name") or "tool")
                self.messages.insert(index, {
                    "role": "tool",
                    "tool_name": name,
                    # The same id the translation would invent for the call
                    # at this position, so the pair lines up on the wire.
                    "tool_call_id": str(call.get("id") or f"call_{position}"),
                    "content": note or self.INTERRUPTED_RESULT,
                })
                index += 1
                added += 1
        if added:
            # This runs on the way out of an interrupted turn, which is the
            # one moment the on-disk copy most needs to be the repaired
            # shape rather than the malformed one.
            self._recorded()
        return added

    # -- size --------------------------------------------------------------

    def token_estimate(self) -> int:
        return (estimate_tokens(self.system_prompt) + self.overhead
                + sum(message_tokens(m) for m in self.messages))

    def should_compact(self, budget: int, num_ctx: int) -> bool:
        limit = min([n for n in (budget, num_ctx) if n and n > 0] or [8000])
        if self.token_estimate() <= limit * 0.75:
            return False
        older, _ = self.slice_for_summary()
        if not older:
            return False
        removable = sum(message_tokens(m) for m in older)
        # Against the conversation, not against everything. "Is the older
        # half worth summarising" is a question about the messages; the
        # system prompt and the tool schemas are neither summarised nor
        # reclaimed, so counting them here sets the bar with the weight of
        # the one thing compaction cannot move. At a small window, where
        # that fixed part is most of the request, it made the bar
        # unreachable: over the window every turn and never compacting,
        # because the reason it was over was also the reason it would not.
        conversation = sum(message_tokens(m) for m in self.messages)
        return removable > max(MIN_WORTH_COMPACTING, conversation * 0.15)

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
        self._recorded()

    # -- persistence -------------------------------------------------------

    def path(self) -> Path:
        return data_dir() / "sessions" / f"{self.session_id}.json"

    def title(self) -> str:
        """A one-line name for this conversation, taken from what was asked.

        Written into the file so /session can list conversations without
        parsing every message of every one of them, and so a resumed
        conversation is recognisable by what it was about rather than by
        twelve hex digits.
        """
        for message in self.messages:
            if message.get("role") != "user":
                continue
            text = " ".join(str(message.get("content") or "").split())
            # The plan note and the compaction preamble are wynxo's own
            # words wearing the user role. Neither names the conversation.
            if not text or text.startswith(("(", "[")):
                continue
            return text[:70]
        return ""

    def save(self) -> Path | None:
        """Write the conversation to disk. Cheap enough to call often.

        Called after every message that changes the conversation, not once
        per turn: a turn can run for minutes across a dozen tool calls, and
        Ctrl-C during one of them used to throw away everything since the
        last completed turn -- the request included, so the conversation on
        disk did not even record having been asked.
        """
        try:
            path = self.path()
            atomic_write(
                path,
                json.dumps(
                    {
                        "session_id": self.session_id,
                        "workspace": str(self.workspace),
                        "title": self.title(),
                        "created_at": self.created_at,
                        "updated_at": time.time(),
                        "compactions": self.compactions,
                        "superseded_chars": self.superseded_chars,
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
            if not self._pruned:
                self._pruned = True
                self.prune()
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
            superseded_chars=as_int(data.get("superseded_chars")),
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

    def prune(self, keep: int = 30) -> None:
        """Keep the newest sessions; delete the rest.

        Sessions are written after every turn, so a store that never sheds
        old files grows without bound -- a few hundred turns of one project
        leaves hundreds of JSON files the /sessions list has to read on
        every resume. Called after each save: the store stays bounded at
        ``keep`` sessions, newest by file mtime first.
        """
        directory = data_dir() / "sessions"
        if not directory.is_dir():
            return

        def mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        old = sorted(directory.glob("*.json"), key=mtime, reverse=True)[keep:]
        for path in old:
            with contextlib.suppress(OSError):
                path.unlink()

    @staticmethod
    def recent(limit: int = 20, exclude: str = "") -> list[dict]:
        """Saved conversations, newest first, from every workspace.

        Deliberately not filtered to the current directory: the point of
        resuming is often that the conversation happened somewhere else --
        you were in one project, learned something, and want to carry on
        talking about it from another. Each row carries its workspace so the
        list can say where it came from, and ``exclude`` drops the
        conversation you are already in, which is never something you resume.
        """
        directory = data_dir() / "sessions"
        if not directory.is_dir():
            return []
        def when(path: Path) -> float:
            try:
                return -path.stat().st_mtime
            except OSError:
                return 0.0

        out = []
        for path in sorted(directory.glob("*.json"), key=when):
            if len(out) >= limit:
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                session_id = as_text(data.get("session_id")) or path.stem
                if exclude and session_id == exclude:
                    continue
                messages = [m for m in as_list(data.get("messages"))
                            if isinstance(m, dict)]
                # Files written before titles were stored still list, by
                # falling back to the same first-message rule that produced
                # them.
                preview = as_text(data.get("title"))
                if not preview:
                    preview = next(
                        (as_text(m.get("content")) for m in messages
                         if m.get("role") == "user"), "")
                out.append({
                    "session_id": session_id,
                    "workspace": as_text(data.get("workspace")) or "?",
                    "updated_at": as_float(data.get("updated_at"), 0.0),
                    "messages": len(messages),
                    "preview": " ".join(preview.split())[:70],
                })
            except (OSError, json.JSONDecodeError, ValueError):
                continue
        return out
