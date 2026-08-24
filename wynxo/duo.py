"""Two models: one that talks to you, one that writes the code.

A 30B coder is slow to first token and writes like a commit message. A 1B
chat model answers instantly and sounds like a person, but cannot be trusted
near your files. Running both plays each to its strength:

    you  ->  talker (small, fast, no tools)  ->  you        the conversation
                      |
                      v
             coder (big, tool-capable)                      the actual work

The talker never touches a file. It has no tools at all -- not restricted
tools, *no tools* -- so there is no path by which the small model can edit
anything. It only reads what the coder did and says it back in its own
voice. That separation is the whole point: the fast, chatty, easily-confused
model is kept structurally incapable of doing damage.

This also makes speech worth listening to. The coder's answer is paths and
diffs; the talker's is two sentences. It is the second one you want read out
loud, which is why Speaker is fed from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .provider import OllamaClient, ProviderError

OPENER_PROMPT = """You are the voice of a coding assistant. A separate \
coding model is about to start work on the user's request.

Say one short line -- at most 15 words -- acknowledging what they asked for, \
in your own voice. Present tense, as if you are about to start.

Do not claim it is finished. Do not describe how you would do it. Do not use \
code, backticks, file paths or lists. Just the one line."""

REPORT_PROMPT = """You are the voice of a coding assistant. The coding model \
has finished and its report is below.

Say what happened, in your own voice, in at most two short sentences. Keep \
every fact exactly as given -- if it says something failed, say it failed. \
Never claim something worked when the report does not say so.

No code, no backticks, no file paths, no lists. This is going to be read out \
loud, so write it to be heard."""


@dataclass
class DuoConfig:
    talker: str = ""
    """Model tag for the voice. Empty disables the whole thing."""

    coder: str = ""
    """Model tag for the work. Empty means "whatever /model is set to"."""

    @property
    def enabled(self) -> bool:
        return bool(self.talker)


class Talker:
    """The small model that speaks. Has no tools and never gets any."""

    def __init__(self, client: OllamaClient, model: str, voice_block: str = "",
                 num_predict: int = 96) -> None:
        self.client = client
        self.model = model
        self.voice_block = voice_block
        self.num_predict = num_predict
        self.last_error = ""

    def _system(self, instruction: str) -> str:
        # The voice block is the same one the main agent uses, so "kawaii"
        # sounds like the same character in both halves of the program.
        return f"{instruction}\n{self.voice_block}".strip()

    async def _say(self, instruction: str, content: str) -> str:
        messages = [
            {"role": "system", "content": self._system(instruction)},
            {"role": "user", "content": content},
        ]
        parts: list[str] = []
        try:
            async for chunk in self.client.chat(
                messages,
                model=self.model,
                tools=None,          # never, under any configuration
                think=None,          # a one-liner does not need reasoning
                temperature=0.8,     # this half is allowed to have some life
                num_predict=self.num_predict,
                stream=True,
            ):
                if chunk.content:
                    parts.append(chunk.content)
        except ProviderError as exc:
            self.last_error = str(exc)
            return ""
        return _tidy("".join(parts))

    async def opening(self, request: str) -> str:
        """One line, before the coder starts."""
        return await self._say(OPENER_PROMPT, request)

    async def report(self, request: str, result: str, failed: bool = False) -> str:
        """Two sentences, after the coder stops."""
        if failed:
            body = (f"The user asked: {request}\n\n"
                    f"The coding model FAILED. What went wrong:\n{result}")
        else:
            body = (f"The user asked: {request}\n\n"
                    f"The coding model's report:\n{result}")
        return await self._say(REPORT_PROMPT, body)


def _tidy(text: str) -> str:
    """Small models pad. Take the padding off.

    Also strips the markup the prompt asked it not to use, because asking is
    not the same as it complying -- especially at 1B.
    """
    import re

    out = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", " ",
                 text, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"```.*?```", " ", out, flags=re.DOTALL)
    out = out.replace("`", "")
    # A leading role label ("Assistant:", "Voice:") is a classic small-model tic.
    out = re.sub(r"^\s*(assistant|voice|ai|reply|response)\s*:\s*", "",
                 out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    # Surrounding quotes, when it decided the line was a quotation.
    if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    return out
