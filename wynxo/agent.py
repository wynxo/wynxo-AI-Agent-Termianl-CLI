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
import time
from dataclasses import dataclass, field
from pathlib import Path

from .checkpoints import Checkpoints
from .config import Config
from .effort import EffortPolicy, override
from .memory import Memory
from .parsing import LiveContentFilter, ParsedTurn, parse_turn
from .permissions import Decision, PermissionStore, summarise_call
from .prompts import (
    CONSENSUS_PROMPT,
    CRITIQUE_PROMPT,
    PLAN_PROMPT,
    REPAIR,
    VERIFY_EXTRA_TESTS,
    VERIFY_PROMPT,
    build_system_prompt,
)
from .provider import OllamaClient, ProviderError
from .scope import Boundary
from .session import Session
from .tools import Registry, build_registry

VERIFIED = "VERIFIED"


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
    errors: list[str] = field(default_factory=list)


class Callbacks:
    """What the agent tells the UI. Overridden by the CLI; no-ops in tests."""

    async def on_thinking(self, text: str) -> None: ...
    async def on_content(self, text: str) -> None: ...
    async def on_stage(self, name: str, detail: str = "") -> None: ...
    async def on_tool_start(self, name: str, summary: str) -> None: ...
    async def on_tool_result(self, name: str, ok: bool, display: str, output: str) -> None: ...
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
        self.config = config
        self.policy = policy
        self.workspace = workspace
        self.cb = callbacks or Callbacks()
        self.boundary = boundary
        self.memory = memory or Memory(workspace)
        self.checkpoints = Checkpoints()
        self.tools = registry or build_registry(
            workspace, allow_shell=config.allow_shell,
            boundary=boundary, memory=self.memory)
        self.permissions = PermissionStore()
        self.permissions.preapprove(config.auto_approve)

        self.native_tools = True
        """Set false when the model's template cannot do tool calls, in which
        case tools are described in the prompt in Hermes format instead."""

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
        live_filter = LiveContentFilter()
        stream = self.client.chat(
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
            if chunk.tool_calls:
                native_calls.extend(chunk.tool_calls)
            if chunk.done:
                self.session.usage.add_chunk(
                    chunk.prompt_tokens, chunk.completion_tokens, chunk.total_duration_ns
                )

        if stream_content and (trailing := live_filter.finish()):
            await self.cb.on_content(trailing)

        return parse_turn("".join(content_parts), "".join(thinking_parts), native_calls)

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
        """Execute a turn's tool calls. Returns False if the user aborted."""
        # Read-only calls in a turn can run together; anything mutating is
        # serialised so two writes to one file cannot interleave.
        for call in turn.tool_calls:
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
                continue

            summary = summarise_call(call.name, call.arguments, self.workspace)

            if refusal := self.permissions.blocked(call.name, tool.mutating, tool.internal):
                await self.cb.on_tool_result(call.name, False, refusal, refusal)
                self.session.add_tool_result(call.name, f"ERROR: {refusal}", call.call_id)
                continue

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
                    continue
                if decision is Decision.ALLOW_ALWAYS:
                    self.permissions.remember(call.name, call.arguments)

            await self.cb.on_tool_start(call.name, summary)
            self._checkpoint(tool, call)
            result = await tool.invoke(call.arguments)
            self.session.usage.tool_calls += 1

            output = result.output
            if len(output) > self.policy.max_tool_output:
                keep = self.policy.max_tool_output
                output = (
                    output[: keep // 2]
                    + f"\n\n... [{len(result.output) - keep} characters truncated] ...\n\n"
                    + output[-keep // 2 :]
                )

            await self.cb.on_tool_result(call.name, result.ok, result.display, result.output)
            self.session.add_tool_result(call.name, output, call.call_id)

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
        """Self-review rounds. Returns how many ran."""
        if self.policy.verify_rounds == 0:
            return 0

        limit = (
            self.policy.max_verify_rounds
            if self.policy.verify_rounds < 0
            else self.policy.verify_rounds
        )
        extra = VERIFY_EXTRA_TESTS if self.policy.name in ("max", "ultra") else ""

        rounds = 0
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

    async def _act(self, max_iterations: int | None = None) -> TurnResult:
        """The tool loop proper."""
        limit = max_iterations or self.policy.max_iterations
        result = TurnResult(content="")

        for iteration in range(limit):
            result.iterations = iteration + 1

            if self.session.should_compact(self.policy.context_budget, self.config.num_ctx):
                await self._compact()
                result.compacted = True

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

        if self.policy.plan == "inline":
            request = (
                f"{request}\n\n"
                "(If this takes more than a couple of steps, open with a one-line "
                "plan, then carry it out.)"
            )

        plan = ""
        if self.policy.plan in ("explicit", "critique"):
            self.session.add_user(request)
            plan = await self._plan(request)
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
        except ProviderError as exc:
            return TurnResult(
                content="", elapsed=time.monotonic() - started, errors=[str(exc)]
            )
        except asyncio.CancelledError:
            raise Interrupted from None

        if not result.interrupted and result.content:
            result.verify_rounds = await self._verify(request)
            if result.verify_rounds:
                # The last substantive assistant message is the real answer;
                # a verification turn saying "VERIFIED" is not.
                final = self._last_substantive()
                if final:
                    result.content = final

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
