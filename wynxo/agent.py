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
from .parsing import LiveContentFilter, ParsedTurn, parse_turn
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
    build_system_prompt,
)
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
    r"who\s+are\s+you|what\s+are\s+you|what'?s?\s+your\s+name|"
    r"how\s+are\s+you(?:\s+(?:doing|today))?|how'?s\s+it\s+going|"
    r"what'?s?\s+up|wyd|how\s+are\s+things|"
    r"are\s+you\s+(?:there|awake|ready|alive)|test(?:ing)?"
    # \b matters more than it looks: alternation takes the first branch that
    # matches, so h[ei]y? claimed the "he" of "hello" and left "llo" behind,
    # which is not small talk -- so "hello there" was treated as a task.
    r")\b"
    r"[\s!.?~,:;)（）\-]*$",
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
    if _SMALL_TALK.match(clause) or _PERSONAL.match(clause):
        return True
    trimmed = _GREETING_LEAD.sub("", clause, count=1)
    if trimmed != clause:
        # "hey" on its own leaves nothing behind, which is still a greeting.
        return not trimmed or bool(
            _SMALL_TALK.match(trimmed) or _PERSONAL.match(trimmed))
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


class Interrupted(Exception):
    """The user pressed Ctrl-C during a turn."""


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
        self.checkpoints = Checkpoints()
        self._turn_mark = 0
        self._warned_over_window = False
        """Said once per turn: it is the same news on every iteration."""
        self.shield = Shield(workspace, enabled=config.protect_secrets)
        self.tools = registry or build_registry(
            workspace, allow_shell=config.allow_shell,
            boundary=boundary, memory=self.memory, shield=self.shield)
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
    ) -> ParsedTurn:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        native_calls: list[dict] = []
        # A model with no native `thinking`/`tools` support writes
        # <think>...</think> and <tool_call>...</tool_call> straight into
        # plain content. The filter hides those from the live view; the raw
        # chunks still go to content_parts below, for parse_turn() to act on.
        live_filter = LiveContentFilter(
            start_in_thinking=self._template_prefills_think)
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

    async def _run_tool_calls(self, turn: ParsedTurn) -> bool:
        """Execute a turn's tool calls. Returns False if the user aborted.

        Read-only calls that need no decision from the user are run together;
        everything else is serialised, so two writes to one file cannot
        interleave and two permission prompts cannot arrive at once.
        """
        pending = list(turn.tool_calls)
        while pending:
            batch = self._parallel_batch(pending)
            if batch:
                await self._run_together(batch)
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
        repeated = not self.task_state.record_action(_fingerprint(call))
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
        if not self.checkpoints.changes_since(self._turn_mark):
            return 0

        runner = testing.detect(self.workspace)
        if runner is None:
            return 0

        shell = self.tools.get("shell")
        if shell is None:
            return 0        # the user disabled it, so this is not ours to do

        self.task_state.transition(TaskState.TESTING)
        await self.cb.on_stage("testing", runner.command)
        previous_output = shell.on_output

        async def forward_output(line: str) -> None:
            await _stream_test_output(self.cb.on_tool_output, line)

        shell.on_output = forward_output
        try:
            result = await shell.invoke(
                {"command": runner.command, "timeout": testing.DEFAULT_TIMEOUT},
                timeout=testing.DEFAULT_TIMEOUT + 30,
            )
        finally:
            shell.on_output = previous_output
        code = result.metadata.get("exit_code", 0 if result.ok else 1)

        if result.ok:
            self.task_state.record_success(f"tests passed: {runner.command}")
            self.task_state.record_verification(runner.command)
            await self.cb.on_tool_result(
                "tests", True, TESTS_PASSED_NOTE.format(command=runner.command), "")
            return 0

        body = testing.summarise(result.output)
        self.task_state.record_failure(f"tests failed: {runner.command}")
        await self.cb.on_tool_result(
            "tests", False, f"tests failed ({runner.command})", body)
        self.session.add_user(TESTS_FAILED_PROMPT.format(
            command=runner.command, code=code, output=body))

        # One pass to fix what broke. Not a loop: a model that cannot fix it
        # in one go is usually making it worse, and the user is better served
        # by seeing the failure than by watching it thrash.
        follow = await self._act(max_iterations=min(12, self.policy.max_iterations))
        return 0 if follow.interrupted else 1

    async def _act(self, max_iterations: int | None = None) -> TurnResult:
        """The tool loop proper."""
        limit = max_iterations or min(self.policy.max_iterations, self.config.max_tool_iterations)
        result = TurnResult(content="")

        for iteration in range(limit):
            result.iterations = iteration + 1

            if self.session.should_compact(self.policy.context_budget, self.config.num_ctx):
                await self._compact()
                result.compacted = True
            await self._warn_if_over_the_window()

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
            if not await self._run_tool_calls(turn):
                result.interrupted = True
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

        # "hello" is not a task. Without this the planning scaffold at high
        # effort turns a greeting into invented work -- see is_small_talk().
        # Application launching is deliberately NOT short-circuited here: the
        # model reads the intent ("open vscode" vs "fix the vscode
        # integration") and calls launch_application with the user's own
        # words; the OS catalog decides what actually exists.
        chatting = is_small_talk(request)
        if chatting:
            self.session.add_user(request)
            try:
                turn = await self._call_model()
            except ProviderError as exc:
                return TurnResult(content="", elapsed=time.monotonic() - started,
                                  errors=[str(exc)])
            except asyncio.CancelledError:
                raise Interrupted from None
            if not turn.content.strip():
                # Same one-chance recovery as the tool loop: a greeting that
                # comes back empty should not die on the spot.
                self.session.add_user(EMPTY_ANSWER_NUDGE)
                try:
                    turn = await self._call_model()
                except ProviderError as exc:
                    return TurnResult(content="",
                                      elapsed=time.monotonic() - started,
                                      errors=[str(exc)])
                except asyncio.CancelledError:
                    raise Interrupted from None
            await self._warn_if_nothing_came_back(turn)
            self.session.add_assistant(turn.content)
            self.session.save()
            return TurnResult(content=turn.content, iterations=1,
                              elapsed=time.monotonic() - started)

        if self.policy.plan == "inline":
            request = (
                f"{request}\n\n"
                "(If this takes more than a couple of steps, open with a one-line "
                "plan, then carry it out.)"
            )

        plan = ""
        if self.policy.plan in ("explicit", "critique"):
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
        else:
            self.session.add_user(request)

        try:
            result = await self._act()

            if not result.interrupted and result.content:
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
