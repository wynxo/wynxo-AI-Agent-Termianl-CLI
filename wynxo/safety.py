"""The safety boundary, enforced by the runtime rather than by the prompt.

A prompt is a request. A boundary is a thing the code will not do. Everything
here is the second kind, because the model on the other end of this agent is
whatever the user pulled from Ollama -- possibly small, possibly tuned to be
agreeable, possibly abliterated. Asking it nicely is not a control.

Two halves, and the second is the one that was missing:

* **Input.** A turn that looks like a personal crisis never reaches the
  planner, the tool loop, the shell or memory. It is answered on a
  conversation path with tools hard-disabled, so there is nothing for a
  model to reach for even if it tries.

* **Output.** That turn's reply is not streamed to the screen as it
  arrives. It is completed, screened, and only then shown. Streaming is the
  hole: by the time a procedural answer is recognisable it is already on
  the user's terminal, and nothing can take it back.

The detector is deliberately narrow. It looks for *procedure* -- means,
methods, doses, instructions -- in a turn already established as
safety-sensitive. It is not a topic filter: talking about how someone feels,
or about grief, or about a bug that is making them miserable, is not what it
fires on, and a boundary that swallowed those would push people away from
the one moment it exists for.
"""

from __future__ import annotations

import re

# Only consulted on a turn already flagged as distress, so these can be
# direct without catching ordinary conversation. Each looks for the shape of
# instruction rather than for a subject.
_PROCEDURAL = re.compile(
    r"(?:"
    r"\b(?:how|ways?|methods?|steps?|instructions?)\s+to\s+"
    r"(?:kill|end|hurt|harm|overdose|suffocate|hang|poison)\b"
    r"|\b(?:lethal|fatal|deadly|toxic)\s+(?:dose|doses|amount|amounts|level)"
    r"|\b(?:overdose|OD)\s+(?:on|with)\s+\w+"
    r"|\bhow\s+(?:much|many)\s+\w+\s+(?:would|will|to)\s+"
    r"(?:kill|be\s+lethal|be\s+fatal)"
    r"|\b(?:take|swallow|ingest)\s+\d+\s*(?:mg|g|pills?|tablets?)"
    r"|\bmost\s+(?:effective|painless|reliable)\s+way\s+to\s+"
    r"(?:die|kill|end)\b"
    r")",
    re.IGNORECASE,
)

REFUSAL = (
    "I'm not going to help with that part, but I'm not going anywhere "
    "either.\n\n"
    "If you're in danger right now, please contact your local emergency "
    "number, or reach a crisis line: in the US and Canada call or text 988; "
    "in the UK and Ireland call 116 123; elsewhere, findahelpline.com lists "
    "one for your country.\n\n"
    "If you want to keep talking, I'm here."
)


def unsafe_output(text: str) -> bool:
    """Whether a reply on a safety-sensitive turn must not be shown as-is."""
    return bool(_PROCEDURAL.search(text or ""))


def screen(text: str) -> str:
    """The reply to actually show on a safety-sensitive turn.

    Returns the model's own words when they are fine -- which is the usual
    case, and the point: this replaces a specific failure, not the model's
    ability to respond warmly to someone having a bad night.
    """
    return REFUSAL if unsafe_output(text) else text


def may_persist(text: str, *, sensitive: bool) -> bool:
    """Whether something said this turn may go into durable user memory.

    Never, on a sensitive turn. A crisis is a moment, not a fact about
    somebody, and writing it into a file that is loaded into every future
    system prompt would make the worst night someone had into a permanent
    part of how the agent greets them.
    """
    return not sensitive and bool((text or "").strip())
