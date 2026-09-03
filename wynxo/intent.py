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

Two things it is careful *not* to be:

* It holds no table of application names. "vscode -> code.exe" is a promise
  that rots on the first machine that spells it differently; the model reads
  the user's words and the OS catalog decides what exists.
* It holds no list of chat phrases as its primary answer. The regex
  heuristics in agent.py remain as the fallback for when the model is
  unreachable or answers with nonsense, because a router that fails closed
  into "coding" on a greeting is the behaviour this module exists to remove.
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
    """The decision, and where it came from."""

    kind: str
    targets: tuple[str, ...] = ()
    """For a system action: what to launch, in the user's own words. Passed
    to the catalog as-is -- resolving it is the OS's job, not ours."""
    command: str = ""
    """A command to run inside the last target, when it is a terminal.

    "open a terminal and run main.py" is one request, not two: the terminal
    and what it should be running arrive together, and splitting them opens
    an empty window and calls it done. It applies to the last target
    because that is where the sentence puts it -- "the calculator, then the
    browser, then a terminal running main.py"."""
    then_coding: bool = False
    """A genuinely combined request ("open the editor and inspect this
    repo"): do the action, then carry on into the agent loop. Only ever set
    because the model said so; a turn must not invent the second half."""
    source: str = "model"
    """``model`` or ``fallback``, so the runtime can tell a real decision
    from a guess when something goes wrong."""

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

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse(raw: str) -> Intent | None:
    """The model's answer, or None when it did not give a usable one.

    Small local models wrap JSON in prose, in fences, or hand back a bare
    word. All three are recoverable; anything else is not, and the caller
    falls back rather than guessing.
    """
    text = (raw or "").strip()
    if not text:
        return None

    match = _JSON.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            data = None
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
                return Intent(kind=kind, targets=targets,
                              # Only where it can mean anything. A command
                              # on a conversation or a coding turn is the
                              # model filling in a field it was shown, and
                              # acting on it would run something nobody
                              # asked for.
                              command=command if kind == SYSTEM_ACTION else "",
                              then_coding=bool(data.get("then_coding")))

    # A bare word, which is what a very small model tends to answer with.
    lowered = text.lower()
    for kind in KINDS:
        if re.fullmatch(rf"\W*{kind}\W*", lowered):
            return Intent(kind=kind)
    return None


def fallback(request: str, *, chatting: bool) -> Intent:
    """The answer when the model could not give one.

    ``chatting`` is agent.is_small_talk(), the heuristic that was doing this
    job alone before. Keeping it here means an unreachable or incoherent
    provider degrades to the previous behaviour instead of routing every
    greeting into the coding loop.
    """
    return Intent(kind=CONVERSATION if chatting else CODING, source="fallback")


async def classify(call, request: str, *, chatting: bool) -> Intent:
    """Ask the model what this is. ``call`` is an async callable taking a
    prompt and returning text; the agent supplies one bound to its client.

    Never raises: a router that can fail is a router that takes the turn
    down with it, and every failure here has a defined answer.
    """
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
        # A launch with nothing to launch is not actionable. The words are
        # the user's own; hand the whole message to the catalog and let it
        # decide whether anything matches.
        decided = Intent(kind=SYSTEM_ACTION, targets=(request,),
                         command=decided.command,
                         then_coding=decided.then_coding)
    return decided
