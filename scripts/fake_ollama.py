#!/usr/bin/env python3
"""A stand-in Ollama server, for when you do not have one yet.

It speaks Ollama's real wire protocol -- ``/api/version``, ``/api/tags``,
``/api/show`` and streaming NDJSON from ``/api/chat``, including native
``tool_calls`` -- but there is no model behind it. Responses come from simple
pattern matching, so the agent loop, the tools, the permission prompts, the
diffs and the effort machinery all run for real against real files.

Use it to try the interface, to demo, or to develop wynxo without a GPU.

    python scripts/fake_ollama.py &
    wynxo --endpoint localhost:11435 --doctor
    wynxo --endpoint localhost:11435

Swap in your real server whenever you have one; nothing else changes.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [
    ("qwen3-coder:30b", 18_600_000_000, "30.5B", "Q4_K_M"),
    ("qwen3:8b", 5_200_000_000, "8.2B", "Q4_K_M"),
]
CONTEXT_LENGTH = 262_144


def plan_for(request: str) -> str:
    return (
        "1. Find the relevant files with glob and grep.\n"
        "2. Read them before changing anything.\n"
        "3. Make the smallest change that does the job.\n"
        "4. Re-read the diff and check it against the request."
    )


def _routing_answer(text: str) -> dict | None:
    """The intent router's question, answered.

    Recognised by the shape of the router's prompt rather than by any phrase
    in the user's message, so this stays a stand-in for a model reading the
    request rather than a keyword table pretending to be one.
    """
    if "Classify the user's message" not in text:
        return None
    message = text.rsplit("Message:", 1)[-1].strip().lower()
    if re.search(r"\b(open|launch|start|run)\b", message):
        target = re.sub(r".*\b(?:open|launch|start|run)\s+", "", message).strip(" .?!")
        combined = bool(re.search(r"\b(inspect|review|check|fix|read)\b", message))
        if combined:
            target = target.split(" and ")[0].strip()
        return {"content": json.dumps({"kind": "system_action",
                                       "targets": [target or "editor"],
                                       "then_coding": combined})}
    return {"content": json.dumps({"kind": "coding", "targets": []})}


_WYNXO_ASIDE = "(If this takes more than a couple of steps,"
"""wynxo's own inline-plan note, which it sends as a message of its own.

A real model reads the whole conversation and knows an aside from a
request. This harness reads one message, so without skipping the aside it
would answer wynxo's parenthetical instead of the user -- which is a
property of reading only the last line, not something a model does.
"""


def _last_request(messages: list[dict]) -> dict:
    for message in reversed(messages or []):
        if str(message.get("content") or "").startswith(_WYNXO_ASIDE):
            continue
        return message
    return {}


def decide(messages: list[dict], tools: list | None) -> dict:
    """Pick a reply. Deliberately simple -- this is a harness, not a model."""
    last = _last_request(messages)
    role = last.get("role", "")
    text = str(last.get("content") or "")
    low = text.lower()

    # A tool just answered: comment on it and stop.
    if role == "tool":
        name = last.get("tool_name") or last.get("name") or "the tool"
        body = text.strip()
        if body.startswith("ERROR"):
            return {"content": f"`{name}` failed: {body[7:120]}. I will stop here rather than guess."}
        first = body.splitlines()[0][:120] if body else "(no output)"
        return {"content": f"`{name}` returned: {first}\n\nThat answers it."}

    if "produce a short plan" in low or "plan for this task" in low:
        return {"content": plan_for(text), "thinking": "Sketching the steps before touching anything."}
    if "attack that plan" in low:
        return {"content": "The plan assumes the files are where I expect. Otherwise it is sound."}
    if "reconcile them" in low:
        return {"content": plan_for(text)}
    if "review the work you just did" in low:
        return {"content": "VERIFIED"}
    if "summarise this conversation" in low:
        return {"content": "- User asked for a change.\n- Files were read and edited.\n- Nothing outstanding."}

    # A tool probe, e.g. from `wynxo --doctor`.
    if tools and ("readme.md" in low or "use the tool" in low):
        name = tools[0]["function"]["name"]
        args = {}
        if "path" in tools[0]["function"]["parameters"].get("properties", {}):
            args["path"] = "README.md"
        return {"tool_calls": [{"function": {"name": name, "arguments": args}}]}

    if (routed := _routing_answer(text)) is not None:
        return routed
    if re.search(r"\b(reply with exactly|say ok)\b", low):
        return {"content": "OK"}
    if re.search(r"\d+\s*[*x]\s*\d+", text):
        return {"content": "391.", "thinking": "17 * 23 = 17*20 + 17*3 = 340 + 51 = 391."}

    # An actual request: look at the project first.
    if tools and re.search(r"\b(read|open|show|what|explain|look|find|fix|add|list|plan|build)\b", low):
        names = {t["function"]["name"] for t in tools}
        # A plan on request, so the todo panel and its overlay can be driven
        # without a real model. The layout tests need a plan on screen to
        # prove an overlay cannot move the composer.
        # A launch request, so the system-action path can be driven without
        # a real model deciding to call it.
        if re.search(r"\b(open|launch|start|run)\b", low) and "launch_application" in names:
            target = re.sub(r".*\b(?:open|launch|start|run)\s+", "", text).strip(" .?!")
            return {"tool_calls": [{"function": {
                "name": "launch_application",
                "arguments": {"query": target or "editor"}}}]}
        # A real edit, so the live diff card can be driven end to end.
        if re.search(r"\bfix\b", low) and "write_file" in names:
            return {"tool_calls": [{"function": {
                "name": "write_file",
                "arguments": {"path": "demo.py",
                              "content": "def average(values):\n"
                                         "    total = 0\n"
                                         "    for value in values:\n"
                                         "        total += value\n"
                                         "    return total / len(values)\n"}}}]}
        if "plan" in low and "todo_write" in names:
            return {"tool_calls": [{"function": {
                "name": "todo_write",
                "arguments": {"items": [
                    {"task": "read the parser", "status": "done"},
                    {"task": "add the retry path", "status": "in_progress"},
                    {"task": "run the tests", "status": "pending"},
                    {"task": "write it up", "status": "pending"},
                ]}}}]}
        quoted = re.search(r"[\w/\\.-]+\.\w{1,5}", text)
        if quoted and "read_file" in names:
            return {"tool_calls": [{"function": {"name": "read_file",
                                                 "arguments": {"path": quoted.group(0)}}}]}
        if "list_dir" in names:
            return {"tool_calls": [{"function": {"name": "list_dir",
                                                 "arguments": {"path": "."}}}]}

    return {"content":
            "This is the fake server, so there is no model thinking about that. "
            "The agent loop, tools and effort machinery around it are real -- "
            "point wynxo at a genuine Ollama to get real answers."}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.server.verbose:
            print(f"  {self.command} {self.path}", flush=True)

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send({"version": "0.12.0-fake"})
        if self.path.startswith("/api/tags"):
            return self._send({"models": [
                {"name": n, "model": n, "size": size,
                 "digest": "0" * 64, "modified_at": "2026-01-01T00:00:00Z",
                 "details": {"family": "qwen3", "parameter_size": params,
                             "quantization_level": quant, "format": "gguf"}}
                for n, size, params, quant in MODELS]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send({"error": "invalid json"}, 400)

        if self.path == "/api/show":
            model = payload.get("model") or payload.get("name") or ""
            if model not in {m[0] for m in MODELS}:
                return self._send({"error": f'model "{model}" not found'}, 404)
            return self._send({
                "capabilities": ["completion", "tools", "thinking"],
                "details": {"family": "qwen3", "parameter_size": "30.5B",
                            "quantization_level": "Q4_K_M", "format": "gguf"},
                "model_info": {"general.architecture": "qwen3",
                               "qwen3.context_length": CONTEXT_LENGTH},
                "modelfile": "# fake", "template": "{{ .Prompt }}"})

        if self.path == "/api/chat":
            return self._chat(payload)

        self._send({"error": "not found"}, 404)

    def _chat(self, payload):
        model = payload.get("model", "")
        if model not in {m[0] for m in MODELS}:
            return self._send({"error": f'model "{model}" not found, try pulling it first'}, 404)

        think = payload.get("think")
        if isinstance(think, str) and think not in ("low", "medium", "high", "max"):
            return self._send({"error": f'invalid think value: "{think}"'}, 400)

        reply = decide(payload.get("messages", []), payload.get("tools"))
        content = reply.get("content", "")
        thinking = reply.get("thinking", "") if think else ""
        tool_calls = reply.get("tool_calls")

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(obj):
            line = (json.dumps(obj) + "\n").encode()
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()

        started = time.time()
        if thinking:
            for piece in _chunks(thinking):
                emit({"model": model, "message": {"role": "assistant", "content": "",
                                                  "thinking": piece}, "done": False})
                time.sleep(self.server.delay)
        if tool_calls:
            emit({"model": model, "message": {"role": "assistant", "content": "",
                                              "tool_calls": tool_calls}, "done": False})
        for piece in _chunks(content):
            emit({"model": model, "message": {"role": "assistant", "content": piece},
                  "done": False})
            time.sleep(self.server.delay)

        emit({"model": model, "message": {"role": "assistant", "content": ""},
              "done": True, "done_reason": "stop",
              "prompt_eval_count": sum(len(str(m.get("content", ""))) for m in
                                       payload.get("messages", [])) // 4,
              "eval_count": max(1, len(content) // 4),
              "total_duration": int((time.time() - started) * 1e9),
              "load_duration": 1_000_000})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def _chunks(text: str, size: int = 12):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--delay", type=float, default=0.01,
                        help="seconds between chunks, to imitate generation speed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.delay = args.delay
    server.verbose = args.verbose
    print(f"fake ollama on http://{args.host}:{args.port}")
    print(f"  wynxo --endpoint {args.host}:{args.port} --doctor")
    print(f"  wynxo --endpoint {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
