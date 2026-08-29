"""The agent loop.

The shape of a turn is set entirely by the effort policy:

    plan  ->  critique  ->  act (tool loop)  ->  verify (repeat)

At ``low`` the first, second and fourth stages are skipped and this is a
plain tool loop. At ``ultra`` every stage runs, the plan is sampled several
times and reconciled, and verification repeats until a review pass comes
back clean. Same code path; the policy decides which parts execute.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .checkpoints import Checkpoints
from ._agent_hardening import _stream_test_output
from .config import Config
from .effort import EffortPolicy, override
from .memory import Memory
from .parsing import (LiveContentFilter, ParsedTurn, parse_turn,
                      partial_string_value)
from .permissions import Decision, PermissionStore, summarise_call
from .prompts import (
    TESTS_FAILED_PROMPT,
    TESTS_PASSED_NOTE,
    CONSENSUS_PROMPT,
    CRITIQUE_PROMPT,
    PLAN_PROMPT,
    REPAIR,
    VERIFY_EXTRA_TESTS,
    VERIFY_PROMPT,
    build_chat_prompt,
    build_system_prompt,
)
from . import intent as intent_mod
from . import safety
from .intent import Intent
from .provider import OllamaClient, ProviderError
from .model import ModelBackend, OllamaBackend
from .scope import Boundary, Mode
from .session import Session
from . import testing
from .secrets import Shield
from .tools import Registry, build_registry
from .tools.base import ToolResult
from .events import ToolEvent
from .task_state import TaskState, TaskStateMachine

VERIFIED = "VERIFIED"


def _fingerprint(call) -> str:
    """A canonical identity for a tool call, stable under formatting noise.

    A model rarely repeats its JSON byte-for-byte: key order, spacing and
    quoting all drift between calls. Fingerprinting the raw string would
    miss the repeats that matter and flag the ones that do not.
    """
    args = call.arguments
    if isinstance(args, dict):
        try:
            args = json.dumps(args, sort_keys=True, default=str,
                              separators=(",", ":"))
        except (TypeError, ValueError):
            args = str(sorted(args.items()))
    return f"{call.name}:{args}"

# Shown to the model once when it answers with nothing at all -- no text, no
# tool call. The single retry costs one model call, not a loop, and turns
# what would otherwise be a dead turn into a second chance.
EMPTY_ANSWER_NUDGE = (
    "Your previous reply came back empty -- no text and no tool call. "
    "Answer the user's request directly, or call a tool if the task needs "
    "one. If the conversation no longer fits your context window, say so."
)

# Things people say that are not work. Anchored and whole-string: "hi" is a
# greeting, "hi, now fix the parser" is a task with a greeting stuck on it.
_SMALL_TALK = re.compile(
    r"^\s*(?:"
    r"h[ei]y?|hey+|hi+|hel{1,2}o+|yo|sup|hiya|howdy|heya|"
    r"good\s*(?:morning|afternoon|evening|night)|"
    r"thanks?|thank\s*you|ty|thx|cheers|"
    r"ok(?:ay)?|k|cool|nice|great|awesome|lol|lmao|haha+|hmm+|"
    r"bye|goodbye|gn|cya|see\s*ya|"
    r"who\s+are\s+you|what\s+are\s+you(?:\s+(?:doing|up\s+to|working\s+on))?|what'?s?\s+your\s+name|"
    r"how\s+are\s+you(?:\s+(?:doing|today))?|how'?s\s+it\s+going|"
    r"what'?s?\s+up|wyd|how\s+are\s+things|"
    r"are\s+you\s+(?:there|awake|ready|alive)|test(?:ing)?|"
    r"y(?:es|eah|ep)?|n(?:o|ope|ah)?|sure|alright|fine|good|right|wow|aha|"
    r"help|need\s+help|i\s+need\s+help|i'?m?\s+(?:really\s+)?stuck|"
    r"can\s+you\s+help\s+me|could\s+you\s+help\s+me|"
    r"what'?s?|how'?s?\s+the\s+weather|what\s+is\s+the\s+weather|weather|"
    r"bro|bruh|brb|fr|damn|dang|dammit|oof|rip|yikes|geez|darn|literally|"
    r"cmon|for\s+real|morning|afternoon|evening|night|my\s+bad"
    # \b matters more than it looks: alternation takes the first branch that
    # matches, so h[ei]y? claimed the "he" of "hello" and left "llo" behind,
    # which is not small talk -- so "hello there" was treated as a task.
    r")\b"
    r"[\s!.?~,:;)（）\-]*$",
    re.IGNORECASE,
)

# Feedback aimed at wynxo itself, and past-tense "I fixed it" win reports.
# Both are conversation, not a request for work. The task-signal check runs
# first, so a report with a real task attached ("i fixed it, now run the
# tests") is still routed to work -- these only catch pure conversation.
_FEEDBACK = re.compile(
    r"^\s*(?:"
    r"why\s+(?:are|r|is|am|ur)?\s*(?:you|u|ur)?\s+(?:so|this)\s+"
    r"(?:dry|cold|robotic|stiff|boring|formal|serious)"
    r"|you\s+(?:are|r)\s+(?:so|too)\s+(?:dry|cold|robotic|stiff|formal)"
    r"|that\s+was\s+(?:crazy|wild|insane|nuts|sick|cool|dope|lit|real|funny)"
    r")\s*$",
    re.IGNORECASE,
)

_DONE_REPORT = re.compile(
    r"^\s*(?:(?:bro|yo|hey|bruh|dude|man|fam)\s+)?"
    r"(?:i|we|it|that|stuff)\s+(?:finally|just|literally|actually)?\s*"
    r"(?:fixed|found|solved|got|did|made|wrote|landed|finished|completed|"
    r"added|removed|cracked|beat|works|passed)\b"
    r"(?:\s+[\w.'/-]+)*\s*$",
    re.IGNORECASE,
)

# Anything that means real work, regardless of how short the message is.
_TASK_SIGNAL = re.compile(
    r"(?:"
    r"\.(?:py|js|ts|tsx|jsx|go|rs|rb|java|c|h|cpp|cs|sh|md|json|ya?ml|toml|txt|html|css)\b"
    r"|[/\\][\w.-]"                 # a path
    r"|```"                          # a code block
    r"|\b(?:fix|add|write|create|make|build|run|test|refactor|implement|"
    r"remove|delete|rename|update|change|edit|debug|explain|review|check|"
    r"find|search|install|deploy|commit|push|merge|read|open|show\s+me)\b"
    r")",
    re.IGNORECASE,
)


# Telling the agent who you are is conversation, not a task -- and it is the
# one kind of conversation it should be paying attention to.
#
# Deliberately tight: each alternative must consume the whole clause. An
# earlier version allowed "i'm <anything>", which swallowed "im going to need
# a retry helper" -- a real task, silently demoted to chatter.
_PERSONAL = re.compile(
    r"^\s*(?:"
    r"(?:my\s+name\s+is|call\s+me|i'?m|i\s+am)\s+[A-Za-z][\w'-]{0,30}"
    r"|(?:i'?m|i\s+am)\s+\d{1,3}(?:\s*(?:years?\s*old|yo))?"
    r"|my\s+(?:name|age)\s+is\s+[\w'-]{1,30}"
    r"|i\s+am\s+\d{1,3}"
    r")\s*$",
    re.IGNORECASE,
)

# A greeting can lead into another conversational phrase with no punctuation
# between them: "hey whats your name", "hi there".
_GREETING_LEAD = re.compile(
    r"^\s*(?:h[ei]y?|hey+|hi+|hello+|yo|sup|hiya|howdy|heya|"
    r"good\s*(?:morning|afternoon|evening|night))\b"
    r"(?:\s+(?:there|again|friend|buddy|mate))?\s*",
    re.IGNORECASE,
)

_CLAUSE = re.compile(r"[,.!?;~]+")


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE.split(text) if c.strip()]


def _is_chatter(clause: str) -> bool:
    """One clause, with a leading greeting allowed to run into the rest."""
    if (_SMALL_TALK.match(clause) or _PERSONAL.match(clause)
            or _FEEDBACK.match(clause) or _DONE_REPORT.match(clause)
            or _CHAT_EXTRA.match(clause)):
        return True
    trimmed = _GREETING_LEAD.sub("", clause, count=1)
    if trimmed != clause:
        # "hey" on its own leaves nothing behind, which is still a greeting.
        return not trimmed or bool(
            _SMALL_TALK.match(trimmed) or _PERSONAL.match(trimmed)
            or _CHAT_EXTRA.match(trimmed))
    return False


def is_small_talk(request: str) -> bool:
    """Whether this is conversation rather than work.

    At high effort a turn is plan -> execute -> verify, and the plan prompt
    asks for "a plan for this task". Handed "hello", a model does as it is
    told: it invents a task, and the next message tells it to carry the plan
    out. That is how saying hello ended up creating hello_world.py.

    Conservative in the direction that matters. A task misread as chat still
    runs -- just as a plain turn, without the planning scaffold. Chat misread
    as a task is the bug, so any hint of real work wins.
    """
    text = request.strip()
    if not text or len(text) > 120:
        return False
    # A question asking for suggestions about future work is conversation even
    # though it contains a work verb ("what should we build next"): it is a
    # question, not a command, and the planner would otherwise invent a task.
    if _SUGGESTION_QUESTION.match(text):
        return True
    if _TASK_SIGNAL.search(text):
        return False
    if _PERSONAL.match(text):
        return True
    # People greet in pieces: "hi, how are you?" is two conversational
    # clauses, and matching the whole string against one pattern missed
    # every compound form -- which is how a greeting reached the planner.
    parts = _clauses(text)
    if not parts:
        return False
    return all(_is_chatter(part) for part in parts)


# More conversation that is not work, and that the plan prompt would otherwise
# invent a task for. General classes, not a phrase database: a feeling, a
# request for something to talk about, a reaction to something the model said,
# a question about shared history. The task-signal check runs first, so a
# real task attached to any of these still routes to work.
_CHAT_EXTRA = re.compile(
    r"^\s*(?:"
    r"i'?m?\s+(?:bored|tired|sleepy|happy|sad|lonely|excited|good|great|ok(?:ay)?|fine|sad)"
    r"|i\s+feel(?:\s+(?:really|so|very|a\s+bit))?\s+[a-z-]{2,20}"
    r"|tell\s+me\s+(?:something(?:\s+[a-z-]+)?|a\s+joke|a\s+story|about\s+you(?:rself)?|more|anything)\b"
    r"|what\s+should\s+we\s+(?:build|make|do|work\s+on|try)\s+next"
    r"|(?:do\s+you\s+)?remember\s+(?:when|what|that|the\s+time)\b"
    r"(?:\s+(?!and\s+)(?:[\w'/.-]+\s*))*"
    r"|that'?s?\s+(?:actually|really|so|pretty|genuinely|freaking|lowkey|hella)?\s*"
    r"(?:crazy|wild|insane|nuts|sick|cool|dope|lit|awesome|amazing|interesting|funny|cursed|real|weird|unreal|wack)"
    r"|that\s+makes\s+sense|makes\s+sense|i\s+see|huh|interesting|noted|wow|nice|ohh?|mhm|mm"
    r")\s*$",
    re.IGNORECASE,
)

# Questions that ask for suggestions about future work. Checked before the
# task-signal guard in is_small_talk: they contain a work verb ("build",
# "make") but are questions, not commands -- "what should we build next?"
# wants a conversation, while "build me a script" wants work. Anchored and
# narrow on purpose; everything else still respects the task-signal guard.
_SUGGESTION_QUESTION = re.compile(
    r"^\s*what\s+should\s+we\s+(?:build|make|do|work\s+on|try)\s+next\s*[?]?\s*$",
    re.IGNORECASE,
)


# A runtime safety boundary, not a canned-response router: self-directed
# distress short-circuits the whole turn -- no plan, no tools, no memory, no
# repo work -- and routes to the model with a serious, persona-free prompt.
# Deliberately tight and self-directed: "this code is killing me" and "kill
# the process" are idioms and must not trip it. False positives land in the
# caring path, which is the safe direction.
_DISTRESS = re.compile(
    r"(?:"
    r"\bsuicid\w*\b"
    r"|\bkill(?:ing)?\s+(?:myself|my\s+self)\b"
    r"|\b(end|ending|take|taking)\s+my\s+(?:own\s+)?life\b"
    r"|\bhurt(?:ing)?\s+(?:myself|my\s+self)\b"
    r"|\bwant\w*\s+to\s+(?:kill\s+myself|die|end\s+it)\b"
    r"(?!\s+(?:laugh\w*|try\w*|hard|anyway|for|inside|in\s+my|again))"
    r"|\bdon'?t\s+want\s+to\s+(?:live|be\s+here|go\s+on|exist)\b"
    r"|\bcan'?t\s+(?:take|handle)\s+(?:this|it)?\s*anymore\b"
    r"|\bno\s+reason\s+to\s+live\b"
    r"|\bending\s+it\s+all\b"
    r")",
    re.IGNORECASE,
)


def is_distress(request: str) -> bool:
    """Whether this turn needs the serious, tool-free path."""
    return bool(_DISTRESS.search(request.strip()))


class Interrupted(Exception):
    """The user pressed Ctrl-C during a turn."""


class Stuck(Exception):
    """The same action has repeated without any progress event between
    repeats. Raised by the tool loop; _act() turns the first occurrence
    into a model-visible recovery prompt and a second into a clean stop."""


@dataclass
class TurnResult:
    content: str
    iterations: int = 0
    tool_calls: int = 0
    verify_rounds: int = 0
    elapsed: float = 0.0
    interrupted: bool = False
    compacted: bool = False
    empty_retried: bool = False
    recovered: bool = False
    terminal_action: bool = False
    """A system action (e.g. an application launch) succeeded and the turn
    ended there, by design, instead of continuing into coding work."""
    """The loop hit the no-progress repeat cap and gave the model a
    structured recovery prompt before it kept going."""
    errors: list[str] = field(default_factory=list)


class Callbacks:
    """What the agent tells the UI. Overridden by the CLI; no-ops in tests."""

    async def on_thinking(self, text: str) -> None: ...
    async def on_content(self, text: str) -> None: ...
    async def on_stage(self, name: str, detail: str = "") -> None: ...
    async def on_tool_start(self, name: str, summary: str, event: ToolEvent | None = None) -> None: ...
    async def on_tool_result(self, name: str, ok: bool, display: str, output: str, event: ToolEvent | None = None) -> None: ...
    async def on_tool_output(self, name: str, line: str) -> None: ...
    async def on_code(self, text: str) -> None: ...
    async def on_todos(self, rendered: str) -> None: ...
    async def on_warning(self, message: str) -> None: ...

    async def ask_permission(self, name: str, summary: str, preview: str) -> Decision:
        return Decision.ALLOW


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        config: Config,
        policy: EffortPolicy,
        workspace: Path,
        callbacks: Callbacks | None = None,
        registry: Registry | None = None,
        boundary: Boundary | None = None,
        memory: Memory | None = None,
    ):
        self.client = client
        self.backend: ModelBackend = OllamaBackend(client)
        self.config = config
        self.policy = policy
        self.workspace = workspace
        self.cb = callbacks or Callbacks()
        self.boundary = boundary
        self.memory = memory or Memory(workspace)
        self._turn_mark = 0
        self._warned_over_window = False
        """Said once per turn: it is the same news on every iteration."""
        self._failure_signatures: list[tuple] = []
        """Failure signatures from test runs, so the same failure twice in a
        task is flagged as no-progress instead of silently re-attempted."""
        self._recovery_inserted = False
        """The no-progress recovery prompt has been shown once this turn;
        the next repeat-cap trip stops the loop instead of nudging again."""
        self._terminal_action = False
        """A system action succeeded and must end the turn. Reset per turn
        in the run loop; initialized here so `_run_tool_calls` is safe on a
        bare agent (and state never lingers from a previous turn)."""
        self.shield = Shield(workspace, enabled=config.protect_secrets)
        self.tools = registry or build_registry(
            workspace, allow_shell=config.allow_shell,
            boundary=boundary, memory=self.memory, shield=self.shield,
            shell_max_output=config.max_command_output_chars)
        self.permissions = PermissionStore()
        self.permissions.preapprove(config.auto_approve)

        self.native_tools = True
        """Set false when the model's template cannot do tool calls, in which
        case tools are described in the prompt in Hermes format instead."""

        self.project_map = ""
        """A one-page layout of the codebase, refreshed by the CLI. Local
        models explore badly, and this is what stops them starting every
        session blind."""

        self._template_prefills_think = False
        """Set once a closing </think> arrives with nothing having opened it.

        The chat template put the opening tag in the prompt, so generation
        starts inside the block. Remembering it lets every turn after the
        first stream the answer cleanly instead of the reasoning."""

        self.task_state = TaskStateMachine()
        self._model_info = None
        """Cached from the last detect_capabilities() call, so set_effort()
        can re-apply the same thinking downgrade without a network round
        trip -- otherwise /effort on a non-thinking model would silently
        turn native `think` back on and undo what detect_capabilities()
        already established."""

        self.session = Session(workspace=workspace)
        # The undo stack rides with the session: it is created after the
        # session so it can persist under the same id, and a restarted
        # session keeps its undo history.
        self.checkpoints = Checkpoints(session_id=self.session.session_id)
        self.refresh_system_prompt()

    # -- setup -------------------------------------------------------------

    def refresh_system_prompt(self) -> None:
        self.session.system_prompt = build_system_prompt(
            self.workspace,
            self.policy,
            tools_description=self.tools.describe(),
            native_tools=self.native_tools,
            memory=self.memory.prompt_section(),
            boundary=self.boundary,
            mode=self.permissions.mode,
            voice=self.config.voice,
            project_map=self.project_map,
        )

    def set_effort(self, policy: EffortPolicy) -> None:
        self.policy = self._apply_capability_limits(policy)
        self.refresh_system_prompt()

    def _apply_capability_limits(self, policy: EffortPolicy) -> EffortPolicy:
        """Downgrade a freshly-resolved policy to what the current model can
        actually do, using the capabilities detect_capabilities() cached.

        Picking a named level (`/effort high`, `-e xhigh`) always starts from
        the policy's own defaults, thinking included -- so without this, an
        effort change after detect_capabilities() already turned thinking off
        for a model that does not support it would silently turn it back on,
        and the model would reject `think` the same way it did before the
        first downgrade.
        """
        info = self._model_info
        if policy.thinking and info is not None and info.capabilities_known \
                and not info.supports_thinking:
            return override(policy, thinking=False)
        return policy

    async def detect_capabilities(self) -> None:
        """Ask the server what the model can do, and adapt.

        Always sets native_tools rather than only ever clearing it, so
        switching from a model with no native tool support to one that has
        it turns Hermes-style prompted calls back off. Without the reset, a
        model switch could only ever downgrade the tool-calling path, never
        upgrade it, and the better path would stay silently unused.
        """
        try:
            info = await self.client.show(self.config.model)
        except ProviderError:
            return
        self._model_info = info
        self.native_tools = not info.capabilities_known or info.supports_tools
        if info.capabilities_known and not info.supports_tools:
            await self.cb.on_warning(
                f"{self.config.model} has no native tool support; using Hermes-style "
                "prompted tool calls. A tool-tuned model will work noticeably better."
            )
        self.policy = self._apply_capability_limits(self.policy)
        self.refresh_system_prompt()

    # -- one model call ----------------------------------------------------

    async def _call_model(
        self,
        *,
        messages: list[dict] | None = None,
        use_tools: bool = True,
        stream_content: bool = True,
        temperature: float | None = None,
        num_predict: int | None = None,
        silent: bool = False,
    ) -> ParsedTurn:
        """``silent`` suppresses the thinking channel as well as content.

        For infrastructure calls -- the intent router, compaction -- whose
        reasoning is not part of the user's conversation and should not land
        in the thinking record they can open with Ctrl-O.
        """
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        native_calls: list[dict] = []
        # A model with no native `thinking`/`tools` support writes
        # <think>...</think> and <tool_call>...</tool_call> straight into
        # plain content. The filter hides those from the live view; the raw
        # chunks still go to content_parts below, for parse_turn() to act on.
        live_filter = LiveContentFilter(
            start_in_thinking=self._template_prefills_think)
        # A native tool call's arguments arrive as JSON fragments on
        # providers that stream them. Accumulated here so the file's contents
        # can be read out of the half-written object and shown as they are
        # generated -- the same trick LiveContentFilter uses for the
        # text-mode form of a tool call, applied to the native one.
        argument_buffer = ""
        argument_shown = 0
        stream = self.backend.chat(
            messages if messages is not None else self.session.wire(),
            model=self.config.model,
            tools=self.tools.ollama_schemas() if (use_tools and self.native_tools) else None,
            think=self._think_value(),
            temperature=self.policy.temperature if temperature is None else temperature,
            num_predict=self.policy.num_predict if num_predict is None else num_predict,
            stream=self.config.stream,
        )

        async for chunk in stream:
            if chunk.thinking:
                thinking_parts.append(chunk.thinking)
                if not silent:
                    await self.cb.on_thinking(chunk.thinking)
            if chunk.content:
                content_parts.append(chunk.content)
                if stream_content:
                    if visible := live_filter.feed(chunk.content):
                        await self.cb.on_content(visible)
                    # A file being written is worth watching. Without this the
                    # screen shows nothing at all between "the model started a
                    # tool call" and "the file exists", which on a slow local
                    # model is long enough to wonder whether it is working.
                    if code := live_filter.code_delta():
                        await self.cb.on_code(code)
            if chunk.arguments_delta and stream_content:
                argument_buffer += chunk.arguments_delta
                whole = partial_string_value(argument_buffer)
                if len(whole) > argument_shown:
                    await self.cb.on_code(whole[argument_shown:])
                    argument_shown = len(whole)
            if chunk.tool_calls:
                native_calls.extend(chunk.tool_calls)
            if chunk.done:
                self.session.usage.add_chunk(
                    chunk.prompt_tokens, chunk.completion_tokens, chunk.total_duration_ns
                )

        if stream_content and (trailing := live_filter.finish()):
            await self.cb.on_content(trailing)
        if live_filter.saw_dangling_close:
            self._template_prefills_think = True

        turn = parse_turn("".join(content_parts), "".join(thinking_parts),
                          native_calls)

        # Last line of defence, whatever the cause: a turn that streamed
        # nothing while the parsed result does have an answer means the live
        # filter and parse_turn disagreed. They can, because the filter may
        # be started inside a think block that the raw text never mentions.
        # An answer nobody saw is the one outcome worth any amount of care.
        if stream_content and turn.content and not live_filter.emitted_any:
            await self.cb.on_content(turn.content)

        return turn

    def _think_value(self) -> bool | str | None:
        """What to send as Ollama's ``think``.

        A string level where the policy has one, so models that support the
        native dial get it; a plain ``True`` otherwise; and nothing at all when
        thinking is off, so models without the capability are unaffected.
        """
        if not self.policy.thinking:
            return None
        return self.policy.think_level or True

    # -- stages ------------------------------------------------------------

    async def _plan(self, request: str) -> str:
        """Produce a plan, optionally by sampling several and reconciling."""
        if self.policy.plan in ("none", "inline"):
            return ""

        self.task_state.transition(TaskState.PLANNING)
        await self.cb.on_stage("planning")

        base = self.session.wire() + [{"role": "user", "content": PLAN_PROMPT}]

        if self.policy.parallel_samples > 1:
            # Independent samples at a raised temperature, then reconciliation.
            # Different framings of the same problem surface assumptions that a
            # single pass takes for granted.
            tasks = [
                self._call_model(
                    messages=base,
                    use_tools=False,
                    stream_content=False,
                    temperature=min(1.0, self.policy.temperature + 0.25),
                )
                for _ in range(self.policy.parallel_samples)
            ]
            try:
                turns = await asyncio.gather(*tasks)
            except ProviderError as exc:
                await self.cb.on_warning(f"Plan sampling failed ({exc}); using a single plan.")
                turns = [await self._call_model(messages=base, use_tools=False, stream_content=False)]

            drafts = [t.content.strip() for t in turns if t.content.strip()]
            if len(drafts) > 1:
                await self.cb.on_stage("reconciling", f"{len(drafts)} plans")
                joined = "\n\n".join(f"--- plan {i + 1} ---\n{d}" for i, d in enumerate(drafts))
                merged = await self._call_model(
                    messages=base + [{
                        "role": "user",
                        "content": CONSENSUS_PROMPT.format(n=len(drafts), plans=joined),
                    }],
                    use_tools=False,
                    stream_content=False,
                )
                plan = merged.content.strip() or drafts[0]
            else:
                plan = drafts[0] if drafts else ""
        else:
            turn = await self._call_model(messages=base, use_tools=False, stream_content=False)
            plan = turn.content.strip()

        if plan and self.policy.plan == "critique":
            await self.cb.on_stage("critiquing plan")
            critique = await self._call_model(
                messages=base + [
                    {"role": "assistant", "content": plan},
                    {"role": "user", "content": CRITIQUE_PROMPT},
                ],
                use_tools=False,
                stream_content=False,
            )
            if critique.content.strip():
                plan = f"{plan}\n\nAfter self-critique:\n{critique.content.strip()}"

        return plan

    def _needs_routing(self, request: str) -> bool:
        """Whether this turn is worth one classification call.

        Two shapes need it, and nothing else does:

        *A planning policy.* Where ``_plan()`` runs before ``_act()``, "open
        the text editor" reaches the planner and comes back as a coding plan
        with a repository scan attached, because nothing has yet established
        what the request is. Where there is no planning phase the tool loop
        starts immediately and its terminal-action guard already ends the
        turn on the launch.

        *An ambiguous message.* A request carrying a real work signal -- a
        path, a code fence, "fix", "add", "run the tests" -- is work, and
        asking about it would be a round-trip to confirm the obvious. What is
        left is the middle: "nice one", "i hate this bug", "do you think rust
        is better than go", "im so tired today". Those have no work signal
        and no greeting shape, so the heuristic sent every one of them into
        the coding loop, where the engineering system prompt is what made the
        replies sound like a support bot. They are the turns this call is
        actually for.
        """
        if self.policy.plan in ("explicit", "critique"):
            return True
        return not _TASK_SIGNAL.search(request or "")

    async def _classify_call(self, prompt: str) -> str:
        """One cheap, tool-free model call for the intent router.

        Deliberately off the session: the classifier must not see the
        conversation and must not add to it. Low temperature and a tight
        token budget because the answer is one small JSON object, and a
        router that costs as much as the turn it routes is not worth having.
        """
        turn = await self._call_model(
            messages=[{"role": "user", "content": prompt}],
            use_tools=False, stream_content=False, silent=True,
            temperature=0.0, num_predict=64)
        return turn.content or ""

    async def _system_action(self, request: str, decision,
                             started: float) -> TurnResult:
        """Launch what was asked for, then stop.

        The whole point of routing before planning: this path never plans,
        never verifies, never reads the repository and offers exactly one
        tool. A system action has a terminal state, and reaching it ends the
        turn -- there is nothing for the agent loop to add.
        """
        await self.cb.on_stage("launching")
        self.session.add_user(request)
        launcher = self.tools.get("launch_application")
        if launcher is None:
            # No launcher on this build; the request is still not coding.
            self.session.add_assistant(
                "I cannot launch applications in this session.")
            return TurnResult(content="I cannot launch applications in this "
                                      "session.", iterations=1,
                              elapsed=time.monotonic() - started)

        outcomes = []
        for target in decision.targets or (request,):
            result = await launcher.invoke({"query": target})
            self._note_terminal(result)
            await self.cb.on_tool_result(
                launcher.name, result.ok, result.display or target,
                result.output)
            self.session.add_tool_result(launcher.name, result.output)
            outcomes.append(result)

        content = "\n".join(r.output for r in outcomes if r.output).strip()
        self.session.add_assistant(content)
        self.session.save()
        result = TurnResult(content=content, iterations=1,
                            elapsed=time.monotonic() - started,
                            errors=[r.error for r in outcomes
                                    if not r.ok and r.error])
        result.terminal_action = True
        if decision.then_coding:
            # A genuinely combined request. The action is done; the rest of
            # the message is ordinary work, so it goes through the normal
            # loop rather than being dropped on the floor.
            self._terminal_action = False
            result.terminal_action = False
            follow = await self._act(first_turn=None)
            follow.content = f"{content}\n\n{follow.content}".strip()
            return follow
        return result

    async def _run_tool_calls(self, turn: ParsedTurn) -> bool:
        """Execute a turn's tool calls. Returns False if the user aborted.

        Read-only calls that need no decision from the user are run together;
        everything else is serialised, so two writes to one file cannot
        interleave and two permission prompts cannot arrive at once.
        """
        pending = list(turn.tool_calls)
        while pending:
            if self._terminal_action:
                # A system action succeeded: the launch (or similar) *is*
                # the answer, so the remaining calls in this step are
                # invented work the user did not ask for -- "open vscode"
                # must never be followed by a write_file the model dreamed
                # up in the same breath. The already-run results are in the
                # session record; the rest are simply dropped, and the main
                # loop finishes the turn instead of letting the model keep
                # going.
                return True
            batch = self._parallel_batch(pending)
            if batch:
                await self._run_together(batch)
                if self._terminal_action:
                    return True
                del pending[:len(batch)]
                continue
            call = pending.pop(0)
            if not await self._run_one(call):
                return False
        return True

    def _parallel_batch(self, pending: list) -> list:
        """The leading run of calls that may safely go at once.

        A call qualifies only if it changes nothing, its tool allows
        concurrency, and it needs no answer from the user -- so the batch can
        never reorder a write or stack up prompts. Fewer than two is not
        worth the machinery, and reads straight through the ordinary path.
        """
        batch = []
        for call in pending:
            tool = self.tools.get(call.name)
            if tool is None or tool.mutating or not tool.concurrency_safe:
                break
            if self.permissions.blocked(call.name, tool.mutating, tool.internal):
                break
            if self.permissions.needs_prompt(call.name, tool.mutating,
                                             call.arguments, tool.internal):
                break
            batch.append(call)
        return batch if len(batch) > 1 else []

    def _note_terminal(self, result) -> None:
        if isinstance(result, ToolResult) and result.terminal:
            self._terminal_action = True

    async def _run_together(self, batch: list) -> None:
        """Run a batch of read-only calls at once.

        Started together, but reported strictly in the order the model asked
        for -- both on screen and in the conversation. Results that arrived
        in a race-dependent order would make the same turn read differently
        each time it ran, and the model's next step depends on that order.
        """
        for call in batch:
            await self.cb.on_tool_start(
                call.name, summarise_call(call.name, call.arguments, self.workspace))

        context_share = max(1, self._context_left() // len(batch))

        async def one(call):
            tool = self.tools.get(call.name)
            # Each gets a slice of the budget rather than all of it: they are
            # about to land in the same context together, and three reads
            # each sized against the whole of it would overflow between them.
            tool = copy.copy(tool)
            tool.context_left = context_share
            tool.on_output = None
            return await tool.invoke(call.arguments)

        results = await asyncio.gather(*(one(c) for c in batch),
                                       return_exceptions=True)

        for call, result in zip(batch, results):
            if isinstance(result, BaseException):
                if isinstance(result, (asyncio.CancelledError, Interrupted)):
                    raise result
                result = ToolResult.failure(
                    f"{call.name} raised {type(result).__name__}: {result}")
            self.session.usage.tool_calls += 1
            event = ToolEvent(call.name, summarise_call(call.name, call.arguments, self.workspace))
            event.start()
            if isinstance(result, ToolResult):
                self._note_terminal(result)
                event.finish(result.ok, output=result.output, error=result.error,
                             display=result.display, **result.metadata)
                await self.cb.on_tool_result(call.name, result.ok, result.display,
                                             result.output, event=event)
            else:
                event.finish(False, error=str(result))
                await self.cb.on_tool_result(call.name, False, str(result), str(result), event=event)
            self.session.add_tool_result(
                call.name, self._trim_output(result.output), call.call_id)

    def _trim_output(self, output: str) -> str:
        keep = min(self.policy.max_tool_output, self.config.max_tool_result_chars)
        if len(output) <= keep:
            return output
        return (output[: keep // 2]
                + f"\n\n... [{len(output) - keep} characters truncated] ...\n\n"
                + output[-keep // 2:])

    async def _run_one(self, call) -> bool:
        """One tool call, with any permission question it needs.

        Returns False only when the user aborted the whole turn; a call that
        was refused or declined returns True, because the turn carries on
        with that answer in hand.
        """
        tool = self.tools.get(call.name)
        if tool is None:
            suggestion = self.tools.suggest(call.name)
            hint = f" Did you mean {suggestion}?" if suggestion else ""
            message = (
                f"No tool named {call.name!r}.{hint} "
                f"Available: {', '.join(self.tools.names())}"
            )
            await self.cb.on_tool_result(call.name, False, message, message)
            self.session.add_tool_result(call.name, f"ERROR: {message}", call.call_id)
            return True

        summary = summarise_call(call.name, call.arguments, self.workspace)
        repeats = self.task_state.record_action(_fingerprint(call))
        repeated = repeats >= 1
        if repeats >= self.config.max_action_repeats:
            # A hard limit, not another warning: the same action has come
            # back this many times with no edit, passing check or plan
            # change between repeats. _act() decides between a recovery
            # prompt and a clean stop; the call itself is not executed.
            raise Stuck(f"repeated the same action {repeats + 1} times "
                        f"({summary}) with no progress in between")
        if repeated:
            await self.cb.on_warning(
                f"Repeated tool action detected: {summary}. Reconsider the approach.")
        path = call.arguments.get("path") if isinstance(call.arguments, dict) else None
        if path:
            self.task_state.add_file(str(path), changed=tool.mutating)
        event = ToolEvent(call.name, summary)
        event.start()
        # Keep the execution identity in the model-visible metadata too. This
        # makes parallel/sequential calls distinguishable in journals and
        # lets a late UI callback be ignored by consumers that track events.
        execution_id = event.execution_id

        if refusal := self.permissions.blocked(call.name, tool.mutating, tool.internal):
            event.finish(False, error=refusal)
            await self.cb.on_tool_result(call.name, False, refusal, refusal, event=event)
            self.session.add_tool_result(call.name, f"ERROR: {refusal}", call.call_id)
            return True

        if self.permissions.needs_prompt(
                call.name, tool.mutating, call.arguments, tool.internal):
            preview = await self._permission_preview(call.name, call.arguments)
            decision = await self.cb.ask_permission(call.name, summary, preview)
            if decision is Decision.ABORT:
                self.session.add_tool_result(
                    call.name, "User stopped the agent here.", call.call_id
                )
                return False
            if decision is Decision.DENY:
                self.permissions.record_denial(call.name, summary)
                self.session.add_tool_result(
                    call.name,
                    "The user declined this action. Do not retry it. "
                    "Continue with a different approach, or ask what they want instead.",
                    call.call_id,
                )
                return True
            if decision is Decision.ALLOW_ALWAYS:
                self.permissions.remember(call.name, call.arguments)

        try:
            await self.cb.on_tool_start(call.name, summary, event=event)
        except TypeError:
            await self.cb.on_tool_start(call.name, summary)
        # Progress means a file actually changed, not that a mutating-capable
        # tool succeeded -- a shell echo is "may mutate", not progress. The
        # checkpoint mark taken before this call tells us exactly that.
        pre_call = self.checkpoints.mark()
        self._checkpoint(tool, call)
        # Long-running tools report as they go. Cleared afterwards so a
        # tool object reused for a later call cannot write into a line
        # that has already been closed.
        tool.on_output = lambda line, _n=call.name: self.cb.on_tool_output(_n, line)
        tool.context_left = self._context_left()
        try:
            result = await tool.invoke(call.arguments)
        except asyncio.CancelledError:
            event.cancel("cancelled")
            raise
        finally:
            tool.on_output = None
            tool.context_left = 0
        self.session.usage.tool_calls += 1

        self._note_terminal(result)
        event.finish(result.ok, output=result.output, error=result.error,
                     display=result.display, execution_id=execution_id,
                     **result.metadata)
        try:
            await self.cb.on_tool_result(call.name, result.ok, result.display, result.output, event=event)
        except TypeError:
            await self.cb.on_tool_result(call.name, result.ok, result.display, result.output)
        model_output = self._trim_output(result.output)
        if repeated:
            # The UI warning alone reaches only the user. The model is the
            # one stuck in the loop, so the repeat has to be visible in the
            # conversation it reads next.
            model_output = (
                "\u26a0 This exact action was already performed earlier in "
                "this task and is being flagged as a repeat. If it gave you "
                "no new information, change strategy instead of doing it "
                "again.\n\n" + model_output
            )
        self.session.add_tool_result(call.name, model_output, call.call_id)
        if result.ok:
            self.task_state.record_success(f"{call.name}: {result.display or 'completed'}")
            if call.name == "read_file" and path:
                self.task_state.add_file(str(path), changed=False)
            if call.name in ("write_file", "edit_file", "multi_edit") and path:
                # What the agent left on disk, so /undo can tell it apart
                # from a change the user makes afterwards. Resolved exactly
                # as _checkpoint resolved it, so the snapshot is found.
                try:
                    resolved = tool.resolve_path(str(path))
                except (PermissionError, OSError, ValueError):
                    resolved = None
                if resolved is not None:
                    self.checkpoints.mark_expected(resolved)
            if call.name == "todo_write" or self.checkpoints.changes_since(pre_call):
                # A file changed or the plan moved: measurable progress, so
                # repeat counts reset. A re-read after that is a fresh
                # action, not a stuck loop; a shell echo changes nothing and
                # stays counted.
                self.task_state.mark_progress()
        else:
            self.task_state.record_failure(f"{call.name}: {result.error or result.display or 'failed'}")

        if call.name == "todo_write" and result.ok:
            await self.cb.on_todos(result.display)
        return True

    def _checkpoint(self, tool, call) -> None:
        """Snapshot a file before a tool changes it, so /undo can put it back."""
        if call.name not in ("write_file", "edit_file", "multi_edit"):
            return
        raw = str(call.arguments.get("path", ""))
        if not raw:
            return
        try:
            path = tool.resolve_path(raw)
        except (PermissionError, OSError):
            return
        self.checkpoints.capture(path, call.name, label=tool.relative(path))

    async def _permission_preview(self, name: str, args: dict) -> str:
        """Show the user what a write would actually do, before they approve it."""
        if name == "write_file":
            from .tools.files import _read_text, make_diff

            try:
                path = self.tools.get(name).resolve_path(str(args.get("path", "")))
            except (PermissionError, AttributeError):
                return ""
            content = str(args.get("content", ""))
            if path.exists():
                try:
                    return make_diff(_read_text(path), content, str(args.get("path")))
                except OSError:
                    return ""
            lines = content.splitlines()
            head = "\n".join(lines[:30])
            return head + (f"\n... (+{len(lines) - 30} more lines)" if len(lines) > 30 else "")
        if name == "edit_file":
            old, new = str(args.get("old_text", "")), str(args.get("new_text", ""))
            return "\n".join(
                [*(f"-{line}" for line in old.splitlines()[:20]),
                 *(f"+{line}" for line in new.splitlines()[:20])]
            )
        return ""

    async def _repair_tool_calls(self, turn: ParsedTurn) -> ParsedTurn | None:
        """Hand malformed tool calls back for another attempt."""
        for attempt in range(self.policy.repair_attempts):
            await self.cb.on_stage(
                "repairing tool call", f"attempt {attempt + 1}/{self.policy.repair_attempts}"
            )
            raw = "\n\n".join(turn.malformed[:2])
            self.session.add_assistant(turn.content or "")
            self.session.add_user(
                REPAIR.format(
                    raw=raw[:2000],
                    reason="It is not valid JSON. Check the quotes, commas and escaping.",
                )
            )
            repaired = await self._call_model(stream_content=False)
            if repaired.tool_calls:
                return repaired
            turn = repaired
            if not turn.malformed:
                return None
        await self.cb.on_warning("Could not get a valid tool call after repair attempts.")
        return None

    async def _verify(self, request: str) -> int:
        """Self-review rounds. Returns how many ran.

        The test run comes first. Asking a model to review its own work is
        asking the author whether the author was right, and a 7B answers yes;
        a failing test is the one thing in this loop that does not come from
        the model.
        """
        rounds = await self._verify_with_tests()

        # Do not run model-based verification when nothing changed: conversation,
        # questions and simple responses need no review.
        if not self.checkpoints.changes_since(self._turn_mark):
            return rounds

        if self.policy.verify_rounds == 0:
            return rounds

        limit = (
            self.policy.max_verify_rounds
            if self.policy.verify_rounds < 0
            else self.policy.verify_rounds
        )
        extra = VERIFY_EXTRA_TESTS if self.policy.name in ("max", "ultra") else ""

        for i in range(limit):
            await self.cb.on_stage("verifying", f"round {i + 1}/{limit}")
            self.session.add_user(VERIFY_PROMPT.format(extra=extra))

            turn = await self._call_model(stream_content=False)
            rounds += 1

            body = turn.content.strip()
            self.session.add_assistant(body, _wire_calls(turn))

            if turn.tool_calls:
                # The reviewer found something and is fixing it.
                if not await self._run_tool_calls(turn):
                    break
                # Let it finish the fix before judging again.
                follow = await self._act(max_iterations=min(12, self.policy.max_iterations))
                if follow.interrupted:
                    break
                continue

            if VERIFIED in body.upper() and len(body) < 400:
                break

            if self.policy.verify_rounds > 0 and rounds >= self.policy.verify_rounds:
                break

        return rounds

    def _context_left(self) -> int:
        """Tokens still free in the window, as best we can tell.

        Measured against the effort policy's budget rather than the model's
        full num_ctx: the budget is what the rest of the turn was planned
        around, and filling the window to its brim is how a session ends up
        compacting mid-task.

        Whichever is smaller, though. A budget is a ceiling an effort level
        puts on itself, never a licence to exceed the window the model
        actually has.
        """
        limit = min([n for n in (self.policy.context_budget,
                                 self.config.num_ctx) if n and n > 0]
                    or [8000])
        return max(0, limit - self.session.token_estimate())

    async def _verify_with_tests(self) -> int:
        """Run the project's own tests and hand back any failures.

        Runs only when this turn actually changed a file. After a question,
        or a turn that only read things, there is nothing new to break and a
        test run would be a slow way to learn that.
        """
        if not self.config.verify_with_tests:
            return 0
        if self.permissions.mode is Mode.PLAN:
            return 0        # read-only means read-only, tests included
        changed = self.checkpoints.changes_since(self._turn_mark)
        if not changed:
            return 0

        runner = testing.detect(self.workspace)
        if runner is None:
            # No test runner the project asked for. A cheap syntax gate is
            # still worth it when Python files changed -- it catches a
            # broken edit before the user does -- and nothing at all for
            # non-Python changes.
            py_changed = [snapshot.path for snapshot in changed
                          if str(snapshot.path).endswith(".py")]
            if not py_changed:
                return 0
            shell = self.tools.get("shell")
            if shell is None:
                return 0
            return await self._syntax_gate(py_changed, shell)

        shell = self.tools.get("shell")
        if shell is None:
            return 0        # the user disabled it, so this is not ours to do

        async def run_tests(command: str):
            """One test run with output forwarded, returning the result."""
            self.task_state.transition(TaskState.TESTING)
            await self.cb.on_stage("testing", command)
            previous_output = shell.on_output

            async def forward_output(line: str) -> None:
                await _stream_test_output(self.cb.on_tool_output, line)

            shell.on_output = forward_output
            try:
                return await shell.invoke(
                    {"command": command, "timeout": testing.DEFAULT_TIMEOUT},
                    timeout=testing.DEFAULT_TIMEOUT + 30,
                )
            finally:
                shell.on_output = previous_output

        # Focused first: the tests most likely to know whether this change
        # broke something, before spending minutes on the whole suite. A
        # focused failure is the news worth reporting -- the full suite would
        # only add noise around it.
        focused = testing.focused_command(
            self.workspace, [snapshot.path for snapshot in changed])
        if focused:
            result = await run_tests(focused)
            if not result.ok:
                return await self._report_test_failure(focused, result, run_tests)

        result = await run_tests(runner.command)
        if result.ok:
            self.task_state.record_success(f"tests passed: {runner.command}")
            self.task_state.record_verification(runner.command)
            # A passing run is measurable progress: whatever the model does
            # next starts from a clean repeat counter. And it means a check
            # that failed earlier in the turn is fixed, not still broken.
            self.task_state.mark_progress()
            self.task_state.clear_blocking_failures()
            await self.cb.on_tool_result(
                "tests", True, TESTS_PASSED_NOTE.format(command=runner.command), "")
            return 0

        return await self._report_test_failure(runner.command, result, run_tests)

    async def _syntax_gate(self, py_changed, shell) -> int:
        """Cheap validation when the project has no test runner: a Python
        edit that does not parse is caught here rather than by the user."""
        command = (f"{testing.python_command(self.workspace)} -m compileall -q "
                   + " ".join(testing.quote_arg(str(p.relative_to(self.workspace)))
                              for p in py_changed[:12]))
        self.task_state.transition(TaskState.TESTING)
        await self.cb.on_stage("syntax check", "compileall")
        previous_output = shell.on_output

        async def forward_output(line: str) -> None:
            await _stream_test_output(self.cb.on_tool_output, line)

        shell.on_output = forward_output
        try:
            result = await shell.invoke(
                {"command": command, "timeout": testing.DEFAULT_TIMEOUT},
                timeout=testing.DEFAULT_TIMEOUT + 30,
            )
        finally:
            shell.on_output = previous_output
        if result.ok:
            self.task_state.record_success("syntax check passed")
            self.task_state.mark_progress()
            self.task_state.clear_blocking_failures()
            await self.cb.on_tool_result(
                "tests", True, "syntax check passed (compileall)", "")
            return 0
        body = testing.summarise(result.output)
        self.task_state.record_failure(f"syntax check failed: {command}")
        await self.cb.on_tool_result("tests", False, "syntax check failed", body)
        self.session.add_user(TESTS_FAILED_PROMPT.format(
            command=command,
            code=result.metadata.get("exit_code", 1),
            output=body + testing.failure_report(result.output, self.workspace)))
        return 1

    async def _report_test_failure(self, command: str, result, retest) -> int:
        """Turn a failed test run into the model's next instruction.

        ``retest`` runs a command through the same harness that produced the
        failure, so the fix pass can be verified once before the turn ends.
        """
        code = result.metadata.get("exit_code", 0 if result.ok else 1)
        body = testing.summarise(result.output)
        body += testing.failure_report(result.output, self.workspace)

        signature = tuple((f.kind, f.file, f.line)
                          for f in testing.parse_failures(result.output))
        if signature and signature in self._failure_signatures[-2:]:
            body = (
                "\u26a0 No meaningful progress: this run failed with the same "
                "failure signature as an earlier run. Reconsider the approach "
                "-- repeating the same edit will produce the same failure.\n\n"
                + body
            )
        if signature:
            self._failure_signatures.append(signature)
            if len(self._failure_signatures) > 8:
                self._failure_signatures.pop(0)

        self.task_state.record_failure(f"tests failed: {command}")
        await self.cb.on_tool_result(
            "tests", False, f"tests failed ({command})", body)
        self.session.add_user(TESTS_FAILED_PROMPT.format(
            command=command, code=code, output=body))

        # One pass to fix what broke. Not a loop: a model that cannot fix it
        # in one go is usually making it worse, and the user is better served
        # by seeing the failure than by watching it thrash.
        follow = await self._act(max_iterations=min(12, self.policy.max_iterations))
        if follow.interrupted:
            return 0

        # The fix may have landed, and the turn is about to be declared done
        # with the failure still on record. Re-run the exact command that
        # failed, once: a passing retest clears the stale failure so the
        # completion report and the user see current state, not history.
        retest = await retest(command)
        if retest.ok:
            self.task_state.record_success(f"tests passed after fix: {command}")
            self.task_state.record_verification(f"{command} (after fix)")
            self.task_state.clear_blocking_failures()
            self.task_state.mark_progress()
            await self.cb.on_tool_result(
                "tests", True, TESTS_PASSED_NOTE.format(command=command), "")
            return 0
        return 1

    async def _act(self, max_iterations: int | None = None,
                   first_turn: ParsedTurn | None = None) -> TurnResult:
        """The tool loop proper.

        ``first_turn`` is a turn the caller already obtained -- used when a
        request routed as conversation comes back with tool calls, so those
        calls are executed rather than silently dropped.
        """
        limit = max_iterations or min(self.policy.max_iterations, self.config.max_tool_iterations)
        result = TurnResult(content="")
        self._terminal_action = False

        for iteration in range(limit):
            result.iterations = iteration + 1

            if self.session.should_compact(self.policy.context_budget, self.config.num_ctx):
                await self._compact()
                result.compacted = True
            await self._warn_if_over_the_window()

            if first_turn is not None:
                turn = first_turn
                first_turn = None
            else:
                try:
                    turn = await self._call_model()
                except ProviderError as exc:
                    result.errors.append(str(exc))
                    raise

            if turn.thinking and not self.config.stream:
                await self.cb.on_thinking(turn.thinking)

            if turn.malformed and not turn.tool_calls:
                repaired = await self._repair_tool_calls(turn)
                if repaired is None:
                    self.session.add_assistant(turn.content)
                    result.content = turn.content
                    return result
                turn = repaired

            self.session.add_assistant(turn.content, _wire_calls(turn))

            if not turn.tool_calls:
                if not turn.content.strip() and not result.empty_retried:
                    # Nothing at all came back. Local models drop replies
                    # for transient reasons -- a swallowed stop token, a
                    # one-off glitch -- and one explicit nudge fixes most
                    # of them. Guarded by the flag so it never loops.
                    result.empty_retried = True
                    self.session.add_user(EMPTY_ANSWER_NUDGE)
                    continue
                await self._warn_if_nothing_came_back(turn)
                result.content = turn.content
                return result

            result.tool_calls += len(turn.tool_calls)
            try:
                if not await self._run_tool_calls(turn):
                    result.interrupted = True
                    result.content = turn.content
                    return result
            except Stuck as stuck:
                if not self._recovery_inserted:
                    # First trip: hand the model a structured recovery
                    # block -- what repeated, what failed, the objective --
                    # and let it change strategy once. The block is new
                    # information, so repeat counts reset for a fair try.
                    self._recovery_inserted = True
                    result.recovered = True
                    self.task_state.transition(TaskState.RECOVERING)
                    self.task_state.mark_progress()
                    self.session.add_user(self.task_state.recovery_block())
                    continue
                await self.cb.on_warning(
                    f"{stuck}; stopping the tool loop.")
                result.content = f"(stopped: {stuck})"
                return result

            if self._terminal_action:
                # A system action succeeded: the user's request was to launch
                # an application (or similar), and launching it *is* the
                # answer. Stop the tool loop now -- further tool calls,
                # planning follow-through and verification would be invented
                # work the user did not ask for. A closing line is produced
                # only when the model gave none alongside its tool call.
                result.terminal_action = True
                if not turn.content.strip():
                    try:
                        closing = await self._call_model(
                            use_tools=False, stream_content=False)
                        if closing.content.strip():
                            result.content = closing.content.strip()
                            self.session.add_assistant(closing.content)
                    except ProviderError:
                        pass
                else:
                    result.content = turn.content
                return result

        await self.cb.on_warning(
            f"Hit the {limit}-iteration ceiling for effort '{self.policy.name}'. "
            "Raise the effort level with /effort if the task genuinely needs more steps."
        )
        result.content = "(stopped: iteration limit reached)"
        return result

    async def _warn_if_nothing_came_back(self, turn: ParsedTurn) -> None:
        """Say so when the model answers with nothing at all.

        A local model does this far more often than a hosted one: a chat
        template that swallows the reply, a window the conversation has just
        outgrown, a stop token emitted immediately. Nothing errors. The turn
        succeeds. The screen shows the question, and then silence -- no
        answer, no warning, no way to tell whether wynxo is still working or
        has quietly given up.

        Two neighbouring cases are already handled: an answer that ended up
        labelled as thought, and an answer that parsed but never streamed.
        This is the one where there really is nothing, and saying so is the
        entire fix.
        """
        if turn.content.strip() or turn.tool_calls:
            return
        await self.cb.on_warning(
            "The model sent back an empty answer. Usually that means its "
            "chat template does not fit the prompt wynxo builds, or the "
            "conversation has outgrown the context window. /doctor checks "
            "the first; /compact fixes the second."
        )

    async def _warn_if_over_the_window(self) -> None:
        """Say so when the conversation no longer fits the model's window.

        Ollama does not refuse an over-long prompt; it drops the far end of
        it and answers anyway. So the model stops being able to see the
        beginning of the task -- the instruction, usually -- and the only
        symptom is that it starts behaving as though it was never told.

        Compaction handles the common case. This is for the one it cannot:
        a single message, or a system prompt and one file, already larger
        than the window. Once per turn, because it is the same news every
        iteration.
        """
        window = self.config.num_ctx
        if window <= 0 or self._warned_over_window:
            return
        used = self.session.token_estimate()
        if used <= window:
            return
        self._warned_over_window = True
        await self.cb.on_warning(
            f"This conversation is about {used} tokens and the model's window "
            f"is {window}. Ollama drops the oldest part of an over-long "
            "prompt without saying so, so it can no longer see the start of "
            "the task. /compact summarises what is there, or restart with a "
            "larger --ctx."
        )

    async def _compact(self) -> None:
        """Summarise the older half of the conversation to reclaim context."""
        older, kept = self.session.slice_for_summary()
        if not older:
            return

        await self.cb.on_stage("compacting context")

        transcript = []
        for message in older:
            role = message.get("role", "?")
            content = str(message.get("content") or "")[:2000]
            transcript.append(f"[{role}] {content}")

        todo = self.tools.get("todo_write")
        outstanding = todo.outstanding() if todo and hasattr(todo, "outstanding") else []
        remaining = (
            "\n\nStill outstanding: " + "; ".join(outstanding) if outstanding else ""
        )

        prompt = (
            "Summarise this conversation so you can continue working without it.\n\n"
            "Keep: what the user asked for, decisions made and why, files read or "
            "changed and what is in them, commands run and their outcomes, what is "
            "still to do. Drop: pleasantries, superseded attempts, tool output you "
            "no longer need.\n\n"
            "Write it as notes to yourself, not prose.\n\n"
            + "\n\n".join(transcript)[:30_000]
            + remaining
        )

        try:
            turn = await self._call_model(
                messages=[{"role": "user", "content": prompt}],
                use_tools=False,
                stream_content=False,
                temperature=0.2,
            )
        except ProviderError:
            # Better a blunt truncation than a dead session.
            self.session.messages = kept
            return

        self.session.apply_compaction(turn.content.strip() or "(summary unavailable)", kept)

    # -- entry point -------------------------------------------------------

    async def run(self, request: str) -> TurnResult:
        started = time.monotonic()
        self.task_state.begin(request)
        # Where this turn began, so the test pass can tell whether anything
        # was actually changed. A turn that only answered a question has
        # nothing new to break.
        self._turn_mark = self.checkpoints.mark()
        self._warned_over_window = False
        self._recovery_inserted = False

        # What kind of turn is this? Asked once, before anything commits to a
        # shape.
        #
        # Distress is decided first and locally, because a safety boundary
        # that depends on a model call is a boundary that opens whenever the
        # provider is slow, unreachable or talked around. The router is never
        # consulted for these turns.
        distress = is_distress(request)
        chatting = is_small_talk(request)
        if distress or chatting:
            # Already settled, and settled locally. Neither needs a model
            # call to tell us what it is.
            decision = Intent(kind=intent_mod.CONVERSATION, source="fallback")
        elif self._needs_routing(request):
            decision = await intent_mod.classify(
                self._classify_call, request, chatting=chatting)
        else:
            # A clear work request on a policy that does not plan first.
            # Both questions the router answers are already settled.
            decision = Intent(kind=intent_mod.CODING, source="fallback")
        self.last_intent = decision
        # Application launching used to be left for the model to reach on its
        # own somewhere inside the tool loop. At medium effort and above
        # _plan() runs first, so "open the text editor" was handed to the
        # planner, which produced a coding plan and scanned the repository
        # before anyone had established that this was a launch at all. The
        # terminal-action guard further down was answering "should this turn
        # stop now?", which is only decidable after a tool has run; "what is
        # this turn?" has to be answered before the first one does.
        if decision.is_system_action:
            return await self._system_action(request, decision, started)
        chatting = chatting or decision.is_conversation
        initial: ParsedTurn | None = None
        """A turn already obtained (conversation that grew tool calls); the
        tool loop executes it before asking the model again. Currently only
        reachable on the coding path -- conversation never grows tool calls."""
        if chatting or distress:
            self.session.add_user(request)
            # Conversation gets its own minimal system prompt, not the coding
            # mega-prompt: engineering context (tools, effort, scope, project
            # map) is exactly what turns a conversational model robotic, and
            # the raw model -- the user's own baseline -- is already natural
            # once the prompt sandwich is gone. Sampled hot so a canned
            # "How can I help you today?" is not the auto-picked token.
            # Tools are never advertised on a conversation turn: the observed
            # interference is a coding model reaching for tools on chat
            # ("remember what we were building?" burned six tool calls), and
            # conversation is the one mode that must be tool-free. A distress
            # turn is the safety boundary: a serious, persona-free prompt and
            # tools hard-disabled -- the model cannot plan, edit, launch or
            # remember anything on this turn, whatever it tries.
            chat_prompt = build_chat_prompt(
                voice=self.config.voice,
                memory=self.memory.prompt_section(),
                serious=distress,
            )
            chat_wire = [{"role": "system", "content": chat_prompt},
                         *self.session.messages]
            try:
                turn = await self._call_model(
                    messages=chat_wire, temperature=0.95, num_predict=200,
                    use_tools=False,
                    # A safety-sensitive turn is completed before any of it
                    # is shown. Streaming is the hole in an output boundary:
                    # by the time a procedural answer is recognisable it is
                    # already on the user's terminal and cannot be taken
                    # back. Ordinary conversation still streams.
                    stream_content=not distress)
            except ProviderError as exc:
                return TurnResult(content="", elapsed=time.monotonic() - started,
                                  errors=[str(exc)])
            except asyncio.CancelledError:
                raise Interrupted from None
            if turn.tool_calls:
                # Belt and braces: tools were never offered, so a tool call
                # here is the model hallucinating one. Never run it -- the
                # turn is conversation, not work.
                turn = ParsedTurn(content=turn.content or "", tool_calls=[])
            if not turn.content.strip():
                # Same one-chance recovery as the tool loop: a greeting that
                # comes back empty should not die on the spot.
                self.session.add_user(EMPTY_ANSWER_NUDGE)
                try:
                    turn = await self._call_model(
                        messages=[{"role": "system", "content": chat_prompt},
                                  *self.session.messages],
                        temperature=0.9, num_predict=200,
                        use_tools=False)
                except ProviderError as exc:
                    return TurnResult(content="",
                                      elapsed=time.monotonic() - started,
                                      errors=[str(exc)])
                except asyncio.CancelledError:
                    raise Interrupted from None
                if turn.tool_calls:
                    turn = ParsedTurn(content=turn.content or "", tool_calls=[])
            await self._warn_if_nothing_came_back(turn)
            content = turn.content
            if distress:
                # Nothing was streamed, so the whole reply is emitted here --
                # screened first. The runtime decides what reaches the screen
                # on this turn, not the prompt and not the model's
                # disposition. Usually that is the model's own words
                # unchanged; when it is not, it is because they were
                # procedural.
                content = safety.screen(content)
                if content:
                    await self.cb.on_content(content)
            self.session.add_assistant(content)
            self.session.save()
            return TurnResult(content=content, iterations=1,
                              elapsed=time.monotonic() - started)

        if self.policy.plan == "inline" and initial is None:
            request = (
                f"{request}\n\n"
                "(If this takes more than a couple of steps, open with a one-line "
                "plan, then carry it out.)"
            )

        plan = ""
        if initial is None and self.policy.plan in ("explicit", "critique"):
            self.session.add_user(request)
            try:
                plan = await self._plan(request)
            except ProviderError as exc:
                # Planning is a convenience; failing it should not lose the
                # turn. Carry on without one and let _act() report if the
                # same problem is still there.
                await self.cb.on_warning(f"planning failed: {exc}")
                plan = ""
            except asyncio.CancelledError:
                raise Interrupted from None
            # The plan prompt is allowed to say there is nothing to plan.
            # Belt and braces with is_small_talk(): the heuristic catches the
            # common phrasings, this catches the ones it does not.
            if plan.strip().upper().startswith("NO PLAN NEEDED"):
                plan = ""
            if plan:
                self.session.add_assistant(plan)
                self.session.add_user(
                    "Now carry out that plan. Use tools; do not ask for confirmation "
                    "to begin."
                )
                await self.cb.on_stage("executing")
        elif initial is None:
            self.session.add_user(request)

        try:
            result = await self._act(first_turn=initial)

            if not result.interrupted and result.content \
                    and not result.terminal_action:
                # Inside the same guard as _act(). A provider error during
                # verification used to escape run() entirely and take the
                # whole REPL down with it -- the work was already done, and
                # the session was lost anyway.
                result.verify_rounds = await self._verify(request)
                if result.verify_rounds:
                    # The last substantive assistant message is the real
                    # answer; a verification turn saying "VERIFIED" is not.
                    final = self._last_substantive()
                    if final:
                        result.content = final
        except ProviderError as exc:
            self.task_state.transition(TaskState.FAILED)
            partial = self._last_substantive()
            return TurnResult(
                content=partial,
                elapsed=time.monotonic() - started,
                errors=[str(exc)],
            )
        except asyncio.CancelledError:
            self.task_state.transition(TaskState.CANCELLED)
            raise Interrupted from None

        self.task_state.transition(TaskState.COMPLETED)
        result.elapsed = time.monotonic() - started
        self.session.save()
        return result

    def _last_substantive(self) -> str:
        for message in reversed(self.session.messages):
            if message.get("role") != "assistant":
                continue
            body = str(message.get("content") or "").strip()
            if not body or message.get("tool_calls"):
                continue
            if VERIFIED in body.upper() and len(body) < 400:
                continue
            return body
        return ""


def _wire_calls(turn: ParsedTurn) -> list[dict]:
    """Tool calls in the shape Ollama expects them echoed back."""
    return [
        {"function": {"name": c.name, "arguments": c.arguments}}
        for c in turn.tool_calls
    ] or []
