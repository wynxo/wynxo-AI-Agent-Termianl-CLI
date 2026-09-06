"""What the user actually asked for, decided before any work begins.

The turn used to find out what kind of request it had by attempting it. At
medium effort and above ``_plan()`` runs before ``_act()``, so "open the text
editor" was handed to the planner, which did what a planner does: produced a
coding plan, scanned the repository, and wrote a todo list about a parser
nobody mentioned. The launch happened -- if it happened -- several steps
later, after the terminal-action guard finally fired somewhere down in the
tool loop. By then the invented work had already been done and shown.

The guard was in the right place for the wrong question. "Should this turn
stop now?" is only answerable after a tool has run. "What is this turn?" has
to be answered before anything runs at all:

    request -> intent -> execute -> result -> end

So this module asks the model, once, cheaply, before the turn commits to a
shape. It returns structure, not prose, and it is deliberately the only place
that decides:

    conversation    talk back; no tools, no planning, no repository
    system_action   launch what was named, then stop
    coding          the full agent loop
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

CONVERSATION = "conversation"
SYSTEM_ACTION = "system_action"
CODING = "coding"
KINDS = (CONVERSATION, SYSTEM_ACTION, CODING)


@dataclass(frozen=True)
class Intent:
    kind: str
    targets: tuple[str, ...] = ()
    command: str = ""
    then_coding: bool = False
    source: str = "model"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown intent {self.kind!r}")

    @property
    def is_conversation(self) -> bool:
        return self.kind == CONVERSATION

    @property
    def is_system_action(self) -> bool:
        return self.kind == SYSTEM_ACTION

    @property
    def is_coding(self) -> bool:
        return self.kind == CODING


PROMPT = """\
Classify the user's message. Answer with one JSON object and nothing else.

{"kind": "conversation" | "system_action" | "coding",
 "targets": [],
 "command": "",
 "then_coding": false}

conversation   chat, greetings, reactions, opinions, questions about you,
               or anything that wants an answer rather than work.
system_action  the user wants an application or program opened, launched,
               started or run on their machine. Put what they called it in
               "targets", in their own words. Do not translate it into a
               filename.
               Several are allowed, in the order asked for: "open the
               calculator then the browser" -> ["the calculator", "the
               browser"].
               If they also say what should run inside a terminal, put that
               in "command": "open the terminal and run main.py" ->
               targets ["the terminal"], command "main.py". Leave
               "command" empty otherwise.
coding         the user wants something done to code, files, tests or this
               project: read, find, explain, fix, add, refactor, run tests.

Set "then_coding" to true only when the message asks for BOTH an application
and work on the project. Do not add work the user did not ask for.

Message:
{request}
"""


def _first_json_object(text: str) -> dict | None:
    """Decode the first JSON object embedded in model prose in bounded time.

    The old ``\{.*\}`` DOTALL regex backtracked quadratically on an answer
    containing an opening brace but no closing one. Intent routing runs before
    every task, so one malformed small-model response could pin a CPU core.
    ``JSONDecoder.raw_decode`` already knows where a JSON value ends. Try only
    the first few object starts; model replies are capped upstream and a reply
    with sixteen stray opening braces before its answer is already nonsense.
    """
    decoder = json.JSONDecoder()
    start = 0
    attempts = 0
    while attempts < 16:
        start = text.find("{", start)
        if start < 0:
            return None
        attempts += 1
        try:
            value, _end = decoder.raw_decode(text[start:])
        except (ValueError, TypeError):
            start += 1
            continue
        if isinstance(value, dict):
            return value
        start += 1
    return None


def parse(raw: str) -> Intent | None:
    """The model's answer, or None when it did not give a usable one."""
    text = (raw or "").strip()
    if not text:
        return None

    data = _first_json_object(text)
    if isinstance(data, dict):
        kind = str(data.get("kind", "")).strip().lower()
        if kind in KINDS:
            raw_targets = data.get("targets") or []
            if isinstance(raw_targets, str):
                raw_targets = [raw_targets]
            targets = tuple(
                str(t).strip() for t in raw_targets
                if isinstance(t, (str, int, float)) and str(t).strip()
            )[:4]
            command = data.get("command")
            command = (str(command).strip()
                       if isinstance(command, (str, int, float)) else "")
            return Intent(
                kind=kind,
                targets=targets,
                command=command if kind == SYSTEM_ACTION else "",
                then_coding=bool(data.get("then_coding")),
            )

    lowered = text.lower()
    for kind in KINDS:
        if re.fullmatch(rf"\W*{kind}\W*", lowered):
            return Intent(kind=kind)
    return None


def fallback(request: str, *, chatting: bool) -> Intent:
    return Intent(kind=CONVERSATION if chatting else CODING, source="fallback")


async def classify(call, request: str, *, chatting: bool) -> Intent:
    request = (request or "").strip()
    if not request:
        return Intent(kind=CONVERSATION, source="fallback")
    try:
        raw = await call(PROMPT.replace("{request}", request[:2000]))
    except Exception:
        return fallback(request, chatting=chatting)
    decided = parse(raw)
    if decided is None:
        return fallback(request, chatting=chatting)
    if decided.is_system_action and not decided.targets:
        decided = Intent(kind=SYSTEM_ACTION, targets=(request,),
                         command=decided.command,
                         then_coding=decided.then_coding)
    return decided
