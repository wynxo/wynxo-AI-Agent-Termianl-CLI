"""Pre-flight checks against a real Ollama server.

Everything wynxo assumes about a server and a model, checked one assumption
at a time, with a concrete fix for each failure. Run it once after pointing
wynxo at a new server or switching model:

    wynxo --doctor

The checks that matter most are the last two: whether the model will actually
emit a tool call, and whether it does so through Ollama's native ``tools``
field or as Hermes-style text. Everything else can look perfect while those
two decide whether the agent works at all.
"""

from __future__ import annotations

import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .config import MIN_USABLE_CONTEXT, Config
from .parsing import parse_turn
from .provider import (OllamaClient, ProviderError, fits_on_gpu,
                       same_model)
from .ui import ACCENT, BAD, GOOD, MUTED, WARN, UI


class Status(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Check:
    name: str
    status: Status
    detail: str = ""
    fix: str = ""
    facts: list[str] = field(default_factory=list)


PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_file_size",
        "description": "Return the size in bytes of a file in the project.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."}
            },
            "required": ["path"],
        },
    },
}
PROBE_PROMPT = "How big is the file README.md? Use the tool."


class Doctor:
    def __init__(self, client: OllamaClient, config: Config, ui: UI,
                 workspace: Path | None = None):
        self.client = client
        self.config = config
        self.ui = ui
        self.workspace = workspace
        self.checks: list[Check] = []

    async def run(self) -> int:
        """Run every check. Returns a process exit code."""
        # One line, at column zero, in the same shape as the session banner:
        # what is being checked and where. It was a title on one row and a
        # subtitle indented under it on the next, which is a page header --
        # and the header of a page is not what a terminal command should
        # open with.
        from rich.text import Text

        head = Text()
        head.append("doctor", style=f"bold {ACCENT}")
        head.append(f"  {self.ui.g.dot}  ", style=MUTED)
        head.append(self.config.model, style=MUTED)
        head.append(f"  {self.ui.g.dot}  ", style=MUTED)
        head.append(str(self.client.base_url), style=MUTED)
        self.ui.console.print()
        self.ui.console.print(head, overflow="ellipsis", no_wrap=True)

        if not await self.check_server():
            # Keep the provider failure as the sole recorded check: callers
            # use this to distinguish a connectivity failure from a later
            # model/configuration failure. Environment facts are shown in the
            # header and do not need to masquerade as completed checks.
            self.checks = [self.checks[-1]]
            self.report()
            return 1
        self.checks.extend([
            Check("git", Status.PASS if shutil.which("git") else Status.WARN,
                  shutil.which("git") or "not found"),
            Check("installation", Status.PASS, str(Path(__file__).resolve().parent)),
        ])
        if self.workspace:
            self.checks.extend(await self._python_checks())
        if not await self.check_model_present():
            self.report()
            return 1

        info = await self.check_capabilities()
        await self.check_context(info)
        await self.check_generation()
        await self.check_memory()
        await self.check_thinking(info)
        await self.check_tool_calling(info)

        return self.report()

    # -- python environment checks ----------------------------------------

    async def _python_checks(self) -> list[Check]:
        """What the active Python environment looks like for this project.

        The interpreter Wynxo would actually run tests under -- not the one
        running Wynxo, which may be a different environment entirely -- plus
        whether the tooling the project's tests need is installed there.
        """
        from . import testing
        env = testing.environment_info(self.workspace)
        checks: list[Check] = []

        interpreter = env.interpreter
        if "WindowsApps" in interpreter:
            checks.append(Check(
                "python interpreter", Status.FAIL, interpreter,
                fix="The resolved interpreter is the Microsoft Store alias, "
                    "which opens the Store instead of running. Activate a "
                    "real environment (e.g. .venv) and restart wynxo from it."))
        elif "virtualenv" in env.environment or "conda" in env.environment:
            checks.append(Check("python interpreter", Status.PASS, interpreter))
        else:
            checks.append(Check(
                "python interpreter", Status.WARN,
                f"{interpreter} ({env.environment})",
                fix="No virtualenv detected for this project. A venv keeps "
                    "dependencies where the project can find them."))

        checks.append(Check(
            "python version",
            Status.PASS if env.version else Status.WARN,
            env.version or "unknown"))
        checks.append(Check("python environment", Status.PASS, env.environment))
        checks.append(Check("package manager", Status.PASS, env.package_manager))

        if env.pytest_installed is False:
            checks.append(Check(
                "pytest", Status.FAIL,
                "not installed in the active environment",
                fix=f"pip: {testing.pip_command(self.workspace)} install pytest"))
        else:
            checks.append(Check(
                "pytest",
                Status.PASS if env.pytest_installed else Status.WARN,
                "installed" if env.pytest_installed else "not checked",
                facts=(["tests run with: " + env.test_runner]
                       if env.test_runner else [])))

        if env.async_tests:
            if env.pytest_asyncio_installed is False:
                checks.append(Check(
                    "pytest-asyncio", Status.FAIL,
                    "async tests present but the plugin is not installed",
                    fix=f"pip: {testing.pip_command(self.workspace)} "
                        "install pytest-asyncio"))
            else:
                checks.append(Check(
                    "pytest-asyncio",
                    Status.PASS if env.pytest_asyncio_installed else Status.WARN,
                    "installed" if env.pytest_asyncio_installed else "not checked"))

        checks.append(Check(
            "project config",
            Status.PASS if env.config_files else Status.WARN,
            ", ".join(env.config_files) or "none found"))
        return checks

    # -- individual checks -------------------------------------------------

    async def check_server(self) -> bool:
        try:
            with self.ui.status("reaching the server..."):
                version = await self.client.ping()
        except ProviderError as exc:
            self.checks.append(Check(
                "server reachable", Status.FAIL, str(exc).splitlines()[0],
                fix="Start Ollama there. A server on another machine needs\n"
                    "OLLAMA_HOST=0.0.0.0:11434 -- it binds loopback only by default.",
            ))
            return False
        self.checks.append(Check(
            "server reachable", Status.PASS, f"ollama {version} at {self.client.base_url}"))
        return True

    async def check_model_present(self) -> bool:
        try:
            with self.ui.status("listing models..."):
                models = await self.client.list_models()
        except ProviderError as exc:
            self.checks.append(Check("model installed", Status.FAIL, str(exc)))
            return False

        names = [m.name for m in models]
        if self.config.model in names:
            model = next(m for m in models if m.name == self.config.model)
            self.checks.append(Check(
                "model installed", Status.PASS,
                f"{model.name}  {model.human_size()}  {model.parameter_size} "
                f"{model.quantization}".strip()))
            return True

        near = [n for n in names if n.split(":")[0] == self.config.model.split(":")[0]]
        self.checks.append(Check(
            "model installed", Status.FAIL,
            f"{self.config.model} is not on this server",
            fix=(f"Pick a tag it has: {', '.join(near)}" if near
                 else f"ollama pull {self.config.model}"),
            facts=[f"installed: {', '.join(names[:8])}" if names else "no models installed"],
        ))
        return False

    async def check_capabilities(self):
        try:
            info = await self.client.show(self.config.model)
        except ProviderError as exc:
            self.checks.append(Check("model capabilities", Status.WARN, str(exc)))
            return None

        if not info.capabilities_known:
            self.checks.append(Check(
                "model capabilities", Status.WARN,
                "this server does not report capabilities",
                fix="Upgrade Ollama. wynxo will assume native tool support and "
                    "fall back if a request is rejected.",
            ))
            return info

        caps = ", ".join(info.capabilities or []) or "none"
        if info.supports_tools:
            self.checks.append(Check("model capabilities", Status.PASS, caps))
        else:
            self.checks.append(Check(
                "model capabilities", Status.WARN,
                f"{caps} -- no native tool support",
                fix="wynxo will use Hermes-style prompted tool calls, which work "
                    "but less reliably. A tool-tuned model (qwen3-coder, devstral) "
                    "is a large upgrade for agent work.",
            ))
        return info

    async def check_context(self, info) -> None:
        native = info.context_length if info else 0
        configured = self.config.num_ctx

        facts = [f"configured num_ctx: {configured}"]
        if native:
            facts.append(f"model's native window: {native}")

        if configured < MIN_USABLE_CONTEXT:
            self.checks.append(Check(
                "context window", Status.FAIL,
                f"num_ctx {configured} is too small for agent work",
                fix=f"Raise it to at least {MIN_USABLE_CONTEXT}: wynxo --ctx 32768, "
                    "or /ctx 32768 in the REPL.",
                facts=facts,
            ))
            return
        if native and configured > native:
            self.checks.append(Check(
                "context window", Status.WARN,
                f"num_ctx {configured} exceeds the model's native {native}",
                fix=f"Lower it to {native} or below; quality degrades past the "
                    "window the model was trained for.",
                facts=facts,
            ))
            return
        self.checks.append(Check("context window", Status.PASS, f"{configured} tokens", facts=facts))

    async def check_memory(self) -> None:
        """Where the server actually put the model.

        Run after generation, because that is what loads it -- asking before
        the first request finds nothing resident and reports nothing.

        This is the check that answers "why is wynxo slower than `ollama
        run`". `ollama run` uses the model's own default window, usually
        4096; wynxo asks for num_ctx, which defaults to 32768. The KV cache
        scales with that window, so a model that fits entirely on the GPU
        under `ollama run` can have a third of its layers pushed onto the
        CPU under wynxo -- same model, same machine, several times slower,
        and nothing anywhere says why.
        """
        loaded = await self.client.running()
        mine = next((m for m in loaded
                     if same_model(m.name, self.config.model)), None)
        if mine is None:
            # An older server with no /api/ps, or a model that unloaded
            # between the request and this check. Neither is a problem, and
            # a check that reports "unknown" on every old server is noise.
            return

        def gigabytes(n: int) -> str:
            return f"{n / 1e9:.1f} GB"

        facts = [f"{gigabytes(mine.size)} total"]
        if mine.context_length:
            facts.append(f"window {mine.context_length:,}")

        if not mine.split:
            # Either all on the GPU, or a machine with no GPU at all. There
            # is nothing to fix in either case.
            where = ("all on the GPU" if mine.size_vram
                     else "on the CPU (no GPU offload)")
            self.checks.append(Check("model in memory", Status.PASS, where,
                                     facts=facts))
            return

        share = mine.on_gpu
        facts.append(f"{gigabytes(mine.size_vram)} on the GPU")
        # Worked out rather than guessed at. The weights do not grow with
        # the window, so everything the model needs above its file size is
        # KV cache, and the KV cache is linear in tokens -- which makes the
        # window that would fit a division, not a halving repeated until it
        # stops complaining.
        weights = await self._weights_of(self.config.model)
        fitting = mine.context_that_fits(weights, self.config.num_ctx)
        fix = [
            "Every token is generated at the speed of the slowest part, so "
            "this is the main reason generation feels slow: a split model "
            "is typically several times slower than one that fits entirely "
            "on the GPU.",
        ]
        if fitting:
            fix.append(
                f"The KV cache grows with the context window. wynxo asks "
                f"for num_ctx={self.config.num_ctx:,}; `ollama run` uses "
                f"the model's own default, which is usually far smaller -- "
                f"that is the whole difference.")
            fix.append(
                f"About {fitting:,} tokens would fit entirely on the GPU: "
                f"try `/ctx {fitting}`, then `/doctor` again.")
            if fitting < MIN_USABLE_CONTEXT:
                fix.append(
                    f"That is below the {MIN_USABLE_CONTEXT:,} an agent "
                    f"really wants, so expect more compaction mid-task. A "
                    f"smaller quantisation of {self.config.model} buys the "
                    f"window back.")
        else:
            # No window is small enough, so recommending one is advice that
            # cannot work. This used to fall back to halving num_ctx --
            # which is the same instruction the user has already followed
            # as far as it goes, offered again with no more reason.
            fix.append(
                f"The weights alone need more than this card has (about "
                f"{gigabytes(mine.size_vram)} of it usable), so no context "
                f"window is small enough. A smaller model, or a tighter "
                f"quantisation of this one, is the only thing that helps.")
        if (smaller := await self._smaller_model(mine.size_vram)) is not None:
            fix.append(
                f"{smaller.name} is already installed at "
                f"{gigabytes(smaller.size)} and would fit whole: "
                f"`/model {smaller.name}`.")
        self.checks.append(Check(
            "model in memory", Status.WARN,
            f"{share:.0%} on the GPU, the rest on the CPU",
            fix="\n".join(fix),
            facts=facts,
        ))

    async def _smaller_model(self, vram: int):
        """The largest installed model that would run entirely on this card.

        "A smaller model is the fix" is advice, not help: it leaves the one
        question that matters -- which one -- to somebody who has just been
        told their setup is slow.
        """
        try:
            installed = await self.client.list_models()
        except ProviderError:
            return None
        for candidate in fits_on_gpu(installed, vram):
            if not same_model(candidate.name, self.config.model):
                return candidate
        return None

    async def _weights_of(self, model: str) -> int:
        """The model's size on disk, which is what its weights cost loaded.

        Zero when it cannot be told, which every caller treats as "say
        nothing" rather than as a number to compute with.
        """
        try:
            for entry in await self.client.list_models():
                if same_model(entry.name, model):
                    return entry.size
        except ProviderError:
            pass
        return 0

    async def check_generation(self) -> None:
        """A real streamed request, which also measures speed."""
        started = time.monotonic()
        chunks = 0
        text = []
        tokens = 0
        try:
            with self.ui.status("generating (this loads the model, so it may take a while)..."):
                async for chunk in self.client.chat(
                    [{"role": "user", "content":
                      "Reply with exactly: OK. Then count from one to ten in words."}],
                    think=None, temperature=0.0, num_predict=80, stream=True,
                ):
                    if chunk.content:
                        chunks += 1
                        text.append(chunk.content)
                    if chunk.done:
                        tokens = chunk.completion_tokens
        except ProviderError as exc:
            self.checks.append(Check(
                "generation", Status.FAIL, str(exc).splitlines()[0],
                fix="\n".join(str(exc).splitlines()[1:]) or "",
            ))
            return

        elapsed = time.monotonic() - started
        speed = tokens / elapsed if elapsed > 0 and tokens else 0
        facts = [f"{tokens} tokens in {elapsed:.1f}s"]
        if speed:
            facts.append(f"{speed:.1f} tok/s (includes model load)")

        body = "".join(text)
        # A short answer legitimately fits in one chunk; that is not evidence
        # of anything. Only a long single-chunk response means no streaming.
        if chunks <= 1 and len(body) > 40:
            self.checks.append(Check(
                "generation", Status.WARN,
                "the response arrived in one piece, not streamed",
                fix="Output will appear all at once rather than as it is written. "
                    "Harmless, but check the server is not behind a buffering proxy.",
                facts=facts,
            ))
            return
        if not body.strip():
            self.checks.append(Check(
                "generation", Status.FAIL,
                "the model returned an empty response",
                fix="Check the server logs. A corrupt model file or an out-of-memory "
                    "condition usually shows up this way.",
                facts=facts,
            ))
            return

        self.checks.append(Check(
            "generation", Status.PASS,
            f"streamed {chunks} chunk(s): {body.strip()[:40]!r}", facts=facts))

    async def check_thinking(self, info) -> None:
        if info is not None and info.capabilities_known and not info.supports_thinking:
            self.checks.append(Check(
                "thinking mode", Status.WARN,
                "this model does not support thinking",
                fix="Effort levels still work -- they drive planning, verification "
                    "and iteration budgets, which is most of what they do. Only the "
                    "native reasoning dial is unavailable.",
            ))
            return

        try:
            got_thinking = False
            async for chunk in self.client.chat(
                [{"role": "user", "content": "What is 17 * 23? Think it through."}],
                think="medium", temperature=0.0, num_predict=200, stream=True,
            ):
                if chunk.thinking:
                    got_thinking = True
        except ProviderError as exc:
            self.checks.append(Check(
                "thinking mode", Status.WARN, str(exc).splitlines()[0],
                fix="wynxo falls back to think:true automatically.",
            ))
            return

        if not self.client.think_levels_supported:
            self.checks.append(Check(
                "thinking mode", Status.WARN,
                "this server only accepts think as a boolean",
                fix="Upgrade Ollama for graduated think levels "
                    '("low"/"medium"/"high"/"max"). wynxo downgrades automatically, '
                    "so nothing breaks.",
            ))
            return

        if got_thinking:
            self.checks.append(Check(
                "thinking mode", Status.PASS, 'think levels accepted, reasoning returned'))
        else:
            self.checks.append(Check(
                "thinking mode", Status.WARN,
                "think level accepted but no reasoning came back",
                fix="Effort levels still drive planning and verification.",
            ))

    async def check_tool_calling(self, info) -> None:
        """The check that decides whether the agent can work at all."""
        native_ok = False
        text_ok = False
        raw = []

        use_native = info is None or not info.capabilities_known or info.supports_tools

        try:
            async for chunk in self.client.chat(
                [{"role": "user", "content": PROBE_PROMPT}],
                tools=[PROBE_TOOL] if use_native else None,
                think=None, temperature=0.0, num_predict=300, stream=True,
            ):
                if chunk.tool_calls:
                    native_ok = True
                if chunk.content:
                    raw.append(chunk.content)
        except ProviderError as exc:
            self.checks.append(Check(
                "tool calling", Status.FAIL, str(exc).splitlines()[0],
                fix="\n".join(str(exc).splitlines()[1:]) or
                    "Try a tool-tuned model: qwen3-coder, devstral, or gpt-oss.",
            ))
            return

        body = "".join(raw)
        turn = parse_turn(body)
        text_ok = bool(turn.tool_calls)

        if native_ok:
            self.checks.append(Check(
                "tool calling", Status.PASS,
                "the model calls tools through Ollama's native tools field"))
            return
        if text_ok:
            self.checks.append(Check(
                "tool calling", Status.PASS,
                "the model emits Hermes-style <tool_call> text, which wynxo parses",
                facts=["wynxo will run in Hermes mode for this model"]))
            return
        if turn.malformed:
            self.checks.append(Check(
                "tool calling", Status.WARN,
                "the model tried to call a tool but the JSON was malformed",
                fix="wynxo repairs and retries these, so it will usually still work. "
                    "A tool-tuned model would do this far less often.",
                facts=[f"raw: {turn.malformed[0][:120]}"]))
            return

        self.checks.append(Check(
            "tool calling", Status.FAIL,
            "the model did not attempt a tool call when asked to",
            fix="This model cannot drive an agent loop. Use a tool-tuned model:\n"
                "  ollama pull qwen3-coder:30b     (best all-rounder)\n"
                "  ollama pull devstral:24b        (built for agent loops)\n"
                "  ollama pull qwen3:8b            (if VRAM is tight)",
            facts=[f"it replied: {body.strip()[:150]!r}"] if body.strip() else []))

    # -- output ------------------------------------------------------------

    def report(self) -> int:
        """The findings, in the shape the rest of the transcript uses.

        Heads at column zero and their evidence at two, drawn through
        detail_line so it wraps inside its own column. It was printed raw at
        two and six, so it was both the one block indented differently from
        everything else and the one that could run off the edge -- the
        longest text in the report, a fix worth several sentences, was
        exactly the text that fell back to column zero on its second line.
        """
        from rich.text import Text

        marks = {
            Status.PASS: (self.ui.g.tick, GOOD),
            Status.WARN: ("!", WARN),
            Status.FAIL: (self.ui.g.cross, BAD),
        }
        self.ui.console.print()
        for check in self.checks:
            mark, style = marks[check.status]
            head = Text()
            head.append(f"{mark} ", style=style)
            head.append(check.name, style="bold" if check.status is Status.PASS
                        else f"bold {style}")
            if check.detail:
                head.append(f"  {check.detail}", style=MUTED)
            self.ui.console.print(head, overflow="ellipsis", no_wrap=True)
            for fact in check.facts:
                self.ui.detail_line(fact, MUTED)
            if check.fix:
                # A blank row before advice that runs to several sentences,
                # so it reads as a paragraph under the finding rather than
                # as one more fact about it.
                if check.facts and len(check.fix) > 120:
                    self.ui.console.print()
                self.ui.detail_line(check.fix, WARN)

        failures = sum(1 for c in self.checks if c.status is Status.FAIL)
        warnings = sum(1 for c in self.checks if c.status is Status.WARN)
        self.ui.console.print()

        if failures:
            self.ui.console.print(Text(
                f"{failures} problem{'' if failures == 1 else 's'} will stop "
                f"wynxo working. Fix those first.", style=f"bold {BAD}"))
            return 1
        if warnings:
            self.ui.console.print(Text(
                f"Usable, with {warnings} caveat{'' if warnings == 1 else 's'} "
                f"noted above.", style=f"bold {WARN}"))
            return 0
        self.ui.console.print(
            Text("Everything checks out. You are good to go.",
                 style=f"bold {GOOD}"))
        return 0


async def run_doctor(config: Config, ui: UI) -> int:
    client = OllamaClient(config)
    try:
        return await Doctor(client, config, ui).run()
    finally:
        await client.aclose()
