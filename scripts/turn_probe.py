#!/usr/bin/env python3
"""Run one real coding turn against a real model and report what happened.

This exists because the failure it looks for -- a tool succeeds, the next
model call comes back empty, and the agent goes quiet -- depends on the
model's own chat template, so a fake server cannot reproduce it. Point this
at a genuine Ollama and it will say whether the continuation works, and if
it does not, what the server actually reported rather than what wynxo
guessed.

    python scripts/turn_probe.py --endpoint http://192.168.178.29:11434
    python scripts/turn_probe.py --endpoint ... --model qwen3-coder:30b

It writes into a temporary directory and deletes it afterwards. Nothing in
your project is touched.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wynxo.agent import Agent, Callbacks          # noqa: E402
from wynxo.config import Config, Endpoint         # noqa: E402
from wynxo.effort import resolve                  # noqa: E402
from wynxo.provider import OllamaClient           # noqa: E402
from wynxo.scope import Boundary, Scope           # noqa: E402
from wynxo.tools import build_registry            # noqa: E402

REQUEST = ("Create a file called hello.py containing a function greet(name) "
           "that returns a greeting string. Use the write_file tool.")


class Probe(Callbacks):
    """Records the turn as the user would experience it, with timings."""

    def __init__(self):
        self.started = time.monotonic()
        self.events: list[tuple[float, str, str]] = []
        self._streaming = False

    def _at(self, kind: str, text: str) -> None:
        self.events.append((time.monotonic() - self.started, kind, text))

    async def on_stage(self, name, detail=""):
        self._at("stage", f"{name} {detail}".strip())

    async def on_tool_start(self, name, summary, event=None):
        self._streaming = False        # a new answer follows each tool
        self._at("tool", f"{name}  {summary}")

    async def on_tool_result(self, name, ok, display, output, event=None):
        mark = "ok" if ok else "FAILED"
        self._at("result", f"{name} {mark}: {(output or display or '')[:70]}")

    async def on_warning(self, message):
        self._at("warning", message)

    async def on_content(self, text):
        # First token only. What matters here is *when* text started
        # arriving relative to the tool, not every fragment of it -- logging
        # each one buried the timeline under the answer.
        if text.strip() and not self._streaming:
            self._streaming = True
            self._at("content", "first token of the answer")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True,
                    help="e.g. http://192.168.178.29:11434")
    ap.add_argument("--model", default="", help="defaults to the first installed")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--effort", default="low")
    args = ap.parse_args()

    config = Config(
        endpoints=[Endpoint(name="probe", url=args.endpoint.rstrip("/"))],
        active_endpoint="probe", num_ctx=args.num_ctx,
        verify_with_tests=False, allow_shell=False, auto_approve=["*"],
        log=False,
    )
    client = OllamaClient(config)

    print(f"server   {args.endpoint}")
    try:
        models = await client.list_models()
    except Exception as exc:
        print(f"\nCannot reach it: {type(exc).__name__}: {exc}")
        await client.aclose()
        return 1
    names = [m.name for m in models]
    if not names:
        print("\nThe server is up but has no models installed.")
        await client.aclose()
        return 1
    config.model = args.model or names[0]
    print(f"models   {', '.join(names[:6])}{' ...' if len(names) > 6 else ''}")
    print(f"using    {config.model}")

    # What the model says it can do. A model with no `tools` capability
    # cannot make a native tool call at all, which is worth knowing before
    # concluding anything about the continuation.
    try:
        info = await client.show(config.model)
        caps = info.capabilities
        print(f"caps     {', '.join(caps) if caps else '(none reported)'}"
              f"{'  <- no tool support' if caps is not None and 'tools' not in caps else ''}")
        if info.context_length:
            print(f"ctx      {info.context_length} (model)   {args.num_ctx} (requested)")
    except Exception as exc:
        print(f"caps     could not read: {exc}")

    workspace = Path(tempfile.mkdtemp(prefix="wynxo-probe-"))
    probe = Probe()
    agent = Agent(client, config, resolve(args.effort), workspace, probe,
                  registry=build_registry(workspace),
                  boundary=Boundary(scope=Scope.FOLDER, root=workspace))

    print(f"\nasking:  {REQUEST[:66]}...\n")
    try:
        result = await agent.run(REQUEST)
    except Exception as exc:
        print(f"the turn raised {type(exc).__name__}: {exc}")
        shutil.rmtree(workspace, ignore_errors=True)
        await client.aclose()
        return 1

    print("timeline")
    for when, kind, text in probe.events:
        print(f"  {when:6.1f}s  {kind:8} {text}")
    if not probe.events:
        print("  (nothing at all -- no stage, no tool, no content)")

    ev = getattr(agent, "_last_evidence", None)
    wrote = (workspace / "hello.py").exists()
    announced = sum(len(m["tool_calls"]) for m in agent.session.messages
                    if m.get("role") == "assistant" and m.get("tool_calls"))
    answered = sum(1 for m in agent.session.messages if m.get("role") == "tool")

    print("\nverdict")
    print(f"  answered ............ {result.answered}")
    print(f"  final text .......... {result.content[:60]!r}")
    print(f"  tool calls made ..... {result.tool_calls}")
    print(f"  hello.py written .... {wrote}")
    print(f"  tool pairing ........ {announced} announced / {answered} answered"
          f"{'  <- MALFORMED' if announced != answered else ''}")
    print(f"  model round trips ... {result.iterations}")
    print(f"  empty-answer retry .. {result.empty_retried}")
    print(f"  errors .............. {result.errors or 'none'}")

    print("\nlast model call, as the server reported it")
    if ev is None:
        print("  (no evidence recorded)")
    else:
        print(f"  stop_reason ......... {ev.stop_reason or '(none sent)'}")
        print(f"  prompt tokens ....... {ev.prompt_tokens}")
        print(f"  completion tokens ... {ev.completion_tokens}")
        print(f"  chunks received ..... {ev.chunks}")
        print(f"  stream truncated .... {ev.truncated}")
        print(f"  sent any thinking ... {ev.had_thinking}")

    if result.answered and wrote and announced == answered:
        print("\n=> The tool -> model continuation works with this model.")
    elif wrote and not result.answered:
        print("\n=> THIS IS THE REPORTED FAILURE: the tool ran and the file")
        print("   was written, but the model never produced an answer.")
        print("   The diagnosis above is built from the server's own numbers.")
    elif not result.tool_calls:
        print("\n=> The model never called a tool. Check the `caps` line: a")
        print("   model without `tools` cannot, and wynxo falls back to")
        print("   describing tools in the prompt instead.")
    else:
        print("\n=> Something else. The timeline above is the evidence.")

    print("\nPaste everything from 'server' down.")
    shutil.rmtree(workspace, ignore_errors=True)
    await client.aclose()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
