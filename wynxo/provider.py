"""Ollama client.

Talks to Ollama's native ``/api/chat`` rather than the OpenAI-compatible
shim, because the native endpoint is the one that exposes ``think``,
``keep_alive`` and per-request ``options`` -- all three of which matter for
running a 30B locally.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from .coerce import as_int, as_list, as_text, loads as json_object
from .config import Config, MIN_USABLE_CONTEXT

LARGE_CONTEXT = 131_072


class ProviderError(RuntimeError):
    """Anything that went wrong talking to the server, phrased for a human."""


@dataclass
class Loaded:
    """A model the server currently has in memory, and where it put it.

    ``size_vram`` is the part that fits on the GPU and ``size`` is the whole
    thing, so anything less than the whole is running partly on the CPU.
    That is the single most useful number about local generation speed and
    nothing in wynxo was reading it: a model that would run at forty tokens
    a second entirely on the GPU runs at five with a few layers spilled, and
    the only symptom is that everything feels slow for no stated reason.
    """

    name: str
    size: int = 0
    size_vram: int = 0
    context_length: int = 0

    @property
    def on_gpu(self) -> float:
        """How much of the model is on the GPU, 0.0 to 1.0.

        1.0 for a CPU-only machine too: with no GPU to spill off, nothing
        has gone wrong and there is nothing to report. The number this
        exists to catch is a *partial* offload -- a machine that could be
        fast and is not.
        """
        if self.size <= 0:
            return 1.0
        return min(1.0, self.size_vram / self.size)

    @property
    def split(self) -> bool:
        """Whether the model is spread across GPU and CPU."""
        return 0 < self.size_vram < self.size

    def context_that_fits(self, weights: int, num_ctx: int) -> int:
        """Roughly the largest context window that would stay on the GPU.

        This is the number that answers "why is wynxo slower than `ollama
        run`", and it can be worked out from what the server already
        reports rather than guessed at.

        ``size`` is what the model needs in memory at the window it was
        loaded under, and the weights do not change with the window, so
        everything above them is the KV cache -- and the KV cache is linear
        in the number of tokens. Divide by the window and you have the cost
        per token; the room left on the card once the weights are there,
        divided by that, is how many tokens would fit.

        ``size_vram`` stands in for how much VRAM there is. Ollama places as
        many layers as it can, so what it managed to place is a fair
        estimate of the card, and an underestimate rather than an over one:
        being told a slightly smaller window than would truly fit costs
        some context, where the other way round costs the speed this exists
        to recover.

        Zero when the answer is not worth acting on -- an unknown weight, a
        window that is not really the KV cache's, or a card too small for
        the weights alone, where no window is small enough and the fix is a
        smaller model or a tighter quantisation instead.
        """
        return self.context_within(weights, num_ctx, self.size_vram)

    def context_within(self, weights: int, num_ctx: int, budget: int) -> int:
        """The largest window whose whole model stays inside ``budget``.

        The same arithmetic against any ceiling, because the card is not
        the only one that matters. A model needs its weights and its cache
        resident *somewhere*, and when that comes to more than the card and
        the machine's RAM together the rest is read off disk on every
        token -- which is slow in a way no amount of CPU makes up for, and
        is the one condition where a smaller window helps a machine with no
        GPU room left to gain.
        """
        if weights <= 0 or num_ctx <= 0 or self.size <= weights:
            return 0
        per_token = (self.size - weights) / num_ctx
        room = budget - weights
        if per_token <= 0 or room <= 0:
            return 0
        # Down to a round number: this is an estimate, and a recommendation
        # of "13,417" claims a precision it does not have.
        return int(room / per_token) // 1024 * 1024

    def share_at(self, weights: int, num_ctx: int, want: int) -> float:
        """How much of this model would sit on the GPU at ``want`` tokens.

        Layers are placed whole, each with its slice of the KV cache, so
        what fits is ``vram / (weights + kv)`` -- and shrinking the window
        shrinks only the kv half of that. Where the weights already do not
        fit there is almost nothing to win: 17.7 GB of weights against 2.4
        GB of cache means going from a 32k window to a 2k one moves
        eighteen percent of the model onto the card instead of twenty, and
        telling somebody to halve their context there sends them to change
        a setting for two points.

        Checked against a real machine: 17.7 GB of weights, 20.1 GB in
        memory at 32,768 tokens, 3.6 GB of it on the card. This predicts
        17.9%; Ollama reported 18%.
        """
        if weights <= 0 or num_ctx <= 0 or self.size <= weights:
            return self.on_gpu
        per_token = (self.size - weights) / num_ctx
        need = weights + per_token * max(0, want)
        return min(1.0, self.size_vram / need) if need > 0 else 1.0


FAST_ENOUGH = 0.8
"""The share on the GPU past which a model counts as fast.

Not a fits-or-does-not line. Generation runs at the speed of the slowest
part, and a fifth of a model on the CPU is a modest tax where four fifths
of it there is the difference between forty tokens a second and five."""


def gpu_share(weights: int, vram: int) -> float:
    """Roughly how much of a model of this size would sit on the GPU.

    Weights only, which makes it an upper bound and says so: a model also
    needs a KV cache, and how big that is depends on its shape. What the
    number is for is comparing candidates, and for that it is enough --
    generation runs at the speed of the slowest part, so this is very
    nearly the speed.
    """
    if weights <= 0:
        return 0.0
    return min(1.0, max(0, vram) / weights)


def faster_on_gpu(models: list["ModelInfo"], vram: int,
                  better_than: float) -> list["ModelInfo"]:
    """Installed models that would run more of themselves on this card.

    Not a fits-or-does-not test. A 4.4 GB model on a 3.6 GB card runs about
    four fifths on the GPU, and four fifths is transformative next to the
    fifth a 17.7 GB model manages -- so ruling it out for not fitting whole
    would be withholding the answer for being imperfect.

    ``vram`` is estimated from what Ollama managed to place on the card for
    the model it has loaded. It fills the GPU as far as it can, so that is a
    fair reading of the card's capacity, and an underestimate rather than an
    over one.
    """
    if vram <= 0:
        return []
    scored = [(gpu_share(m.size, vram), m) for m in models if m.size > 0]
    good = [(share, m) for share, m in scored if share > better_than + 0.2]
    # Above the bar, the largest wins; below it, the one with most of
    # itself on the card. Ranked on share alone, a 3B that fits whole beat
    # a 7B at four fifths -- and at four fifths the 7B is already fast, so
    # that is trading away most of the model to win a few percent of a
    # speed you already had. Clamping the share is what makes "fast enough"
    # a bar rather than a race.
    good.sort(key=lambda pair: (min(pair[0], FAST_ENOUGH), pair[1].size),
              reverse=True)
    return [m for _, m in good]


def same_model(a: str, b: str) -> bool:
    """Whether two model names are the same model.

    ``/api/ps`` answers with the tag the server loaded, which is not always
    the string configured here: ``qwen3-coder:30b`` and ``qwen3-coder:latest``
    are the same weights under two names, and a check that misses that
    reports nothing at all rather than reporting the wrong thing -- which is
    the failure mode nobody notices.
    """
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and (a == b or a.split(":")[0] == b.split(":")[0])


@dataclass
class ModelInfo:
    name: str
    size: int = 0
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""
    context_length: int = 0
    capabilities: list[str] | None = None
    """None means the server did not report capabilities at all (older Ollama),
    which is different from reporting that the model has none. Callers must
    not treat the two as the same: the first is 'unknown, assume it works',
    the second is 'definitely not supported'."""

    @property
    def capabilities_known(self) -> bool:
        return self.capabilities is not None

    @property
    def supports_tools(self) -> bool:
        return "tools" in (self.capabilities or [])

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in (self.capabilities or [])

    def human_size(self) -> str:
        if not self.size:
            return "?"
        gb = self.size / 1_000_000_000
        return f"{gb:.1f}GB"


@dataclass
class Chunk:
    """One streamed piece of an assistant turn."""

    content: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    arguments_delta: str = ""
    """A fragment of a tool call's arguments, as it is generated.

    This is what makes an edit visible while it is being written rather than
    when it is finished. Providers that stream tool calls delta-by-delta
    (the OpenAI-compatible wire format) send the arguments JSON in pieces,
    and those pieces were being accumulated into a list and yielded only
    once the stream ended -- so the real incremental data existed and was
    thrown away, and a 200-line edit appeared all at once after the wait.

    Ollama's native tool_calls arrive with their arguments complete in one
    message, so nothing is emitted here for them. There is no partial data
    to show, and inventing some by revealing a finished string slowly would
    be an animation pretending to be a stream.
    """
    done: bool = False
    truncated: bool = False
    """The stream ended without the provider ever saying it had finished.

    A server that dies, is killed, or unloads a model mid-generation closes
    its connection cleanly, so from the client's side that is indistinguish-
    able from a well-formed stream except for the missing end marker -- and
    without checking for one, half an answer was handed back as a whole one
    and the agent went on to act on it. Local models make this ordinary
    rather than exotic: an OOM during generation looks exactly like this.
    """
    stop_reason: str = ""
    """Why generation ended: Ollama's ``done_reason``, OpenAI's
    ``finish_reason``. "stop", "length", "load", "tool_calls".

    It was being dropped, which is why an empty answer had no evidence
    behind it and every one of them got the same guess. "length" with no
    tokens generated is a num_predict problem; "stop" with tokens generated
    but no text is a template problem; neither is a context problem, and
    the user was told it might be all three."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    eval_duration_ns: int = 0
    """Time spent generating, with the load and the prompt left out.

    The one number that answers "how fast is this model". total_duration
    covers reading the weights off disk and reading the prompt as well, and
    on a machine where most of the model is on the CPU those dwarf the
    generation -- so a speed computed from it is a speed nobody can use to
    judge a change they just made."""


def _payload(response, what: str, where: str) -> dict:
    """The response body as a mapping, or a ProviderError worth reading.

    Every one of these calls used to assume a 200 meant JSON. A wrong port,
    a proxy's HTML error page, a captive-portal login redirect, or a server
    that answers 200 with nothing at all all produced a bare
    JSONDecodeError, uncaught, on the start-up path -- so pointing wynxo at
    the wrong address crashed it with "Expecting value: line 1 column 1"
    instead of telling anybody what was wrong.

    A list is accepted and reported as such rather than silently coerced:
    some shims answer /v1/models with a bare array, and treating that as an
    empty mapping would report "no models" for a server that has plenty.
    """
    try:
        data = response.json()
    except ValueError as exc:
        body = (response.text or "").strip()
        looks_like_html = body[:1] == "<"
        detail = ("an HTML page" if looks_like_html
                  else "an empty body" if not body
                  else f"{body[:60]!r}")
        raise ProviderError(
            f"{where} answered {what} with {detail} rather than JSON. "
            "That address is reachable but is not the API wynxo expects -- "
            "check the port, and whether something (a proxy, a login page) "
            "is sitting in front of it."
        ) from exc
    if isinstance(data, list):
        return {"_list": data}
    if not isinstance(data, dict):
        raise ProviderError(
            f"{where} answered {what} with {type(data).__name__} rather than "
            "an object. That address is reachable but is not the API wynxo "
            "expects."
        )
    return data


_TRANSIENT = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)
"""Failures that are about the connection rather than the request, and so
may well succeed on a second try."""

CONNECT_ATTEMPTS = 3
RETRY_BACKOFF = 0.75


def _is_template_parse_error(low: str) -> bool:
    """Whether the server failed parsing what the model wrote, not the request.

    Ollama renders tool calls through the model's own chat template. Several
    tool-tuned models emit XML-ish calls, and when the model closes a tag
    wrongly the template's parser is what complains -- so the error names XML
    or a template rather than anything the user did.
    """
    if "syntax error" in low and ("xml" in low or "element" in low):
        return True
    return ("template" in low and "error" in low) or "closed by" in low


class OllamaClient:
    def __init__(self, config: Config):
        self.config = config
        self.think_levels_supported = True
        """Ollama gained string think levels ("low"/"medium"/"high"/"max")
        after the plain boolean. An older server rejects the string outright,
        so the first rejection downgrades this for the rest of the session."""
        ep = config.endpoint()
        self.base_url = ep.url
        headers = {"Content-Type": "application/json"}
        if ep.api_key:
            headers["Authorization"] = f"Bearer {ep.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=10.0,
                read=config.request_timeout,
                write=30.0,
                pool=10.0,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # -- discovery ---------------------------------------------------------

    async def ping(self) -> str:
        """Return the server version, or raise with a message worth reading."""
        try:
            r = await self._client.get("/api/version", timeout=10.0)
            r.raise_for_status()
            return _payload(r, "/api/version", self.base_url).get(
                "version", "unknown")
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot reach an Ollama server at {self.base_url}.\n"
                "  - Is `ollama serve` running on that machine?\n"
                "  - If it is a homelab box, it must be started with\n"
                "    OLLAMA_HOST=0.0.0.0:11434 or it only listens on loopback.\n"
                "  - Check a firewall is not eating port 11434."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{self.base_url} answered {exc.response.status_code}. "
                "That address is reachable but does not look like Ollama."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach {self.base_url}: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            r = await self._client.get("/api/tags", timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not list models: {exc}") from exc
        out = []
        payload = _payload(r, "/api/tags", self.base_url)
        for m in payload.get("models", payload.get("_list", [])):
            if not isinstance(m, dict):
                continue
            details = m.get("details") or {}
            out.append(
                ModelInfo(
                    name=m.get("name", "?"),
                    size=m.get("size", 0),
                    parameter_size=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    family=details.get("family", ""),
                )
            )
        out.sort(key=lambda m: m.name)
        return out

    async def show(self, model: str) -> ModelInfo:
        """Fetch capabilities and the model's real context length."""
        try:
            r = await self._client.post("/api/show", json={"model": model}, timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not inspect {model!r}: {exc}") from exc
        data = _payload(r, "/api/show", self.base_url)
        details = data.get("details") or {}
        info = ModelInfo(
            name=model,
            parameter_size=details.get("parameter_size", ""),
            quantization=details.get("quantization_level", ""),
            family=details.get("family", ""),
            capabilities=data.get("capabilities"),
        )
        # The context length key is namespaced by architecture, e.g.
        # "qwen3.context_length". Find whichever one is present.
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                info.context_length = value
                break
        return info

    async def running(self) -> list[Loaded]:
        """What the server has in memory right now, from /api/ps.

        Never raises: this only ever informs a diagnostic, and a server too
        old to have the endpoint, or one that answers something unexpected,
        must not be able to break a turn. An empty list means "nothing
        loaded, or could not tell", and every caller treats those the same.
        """
        try:
            r = await self._client.get("/api/ps", timeout=10.0)
            r.raise_for_status()
            payload = _payload(r, "/api/ps", self.base_url)
        except (httpx.HTTPError, ProviderError):
            return []
        out = []
        for entry in as_list(payload.get("models")):
            if not isinstance(entry, dict):
                continue
            details = entry.get("details") or {}
            out.append(Loaded(
                name=as_text(entry.get("name")) or as_text(entry.get("model")) or "?",
                size=as_int(entry.get("size")),
                size_vram=as_int(entry.get("size_vram")),
                context_length=as_int(entry.get("context_length")
                                      or details.get("context_length")),
            ))
        return out

    async def warm(self, model: str = "", num_ctx: int = 0,
                   messages: list[dict] | None = None,
                   tools: list[dict] | None = None) -> bool:
        """Ask the server to load the model, and read the prompt into it.

        An empty message list is Ollama's documented way to say "load this
        and hold it": the model is read from disk and the KV cache is
        allocated, and nothing is generated.

        This is most of why `ollama run` feels quicker. It loads the model
        while you are still typing your first message; wynxo asked nothing
        of the server until you pressed enter, so the first question of
        every session paid for a cold load -- tens of seconds for a 30B --
        behind a status line that said "thinking", which is not what was
        happening.

        Given ``messages``, it goes one step further and has the model
        *read* them, one token of generation to make sure the prompt is
        actually evaluated. That matters because loading the weights is
        only half the wait. wynxo's system prompt and tool schemas come to
        somewhere north of five thousand tokens, and a local model reads
        every one of them before it writes a word -- where `ollama run`
        sends your message and nothing else. Ollama keeps the KV cache
        between requests and reuses whatever prefix matches, so a first
        question that arrives after this shares all of it and pays only for
        itself.

        The prompt has to be the one the first real request will send, or
        the prefix does not match and the work is thrown away. Nothing here
        can tell whether the server honoured it; the cost of being wrong is
        one request's worth of arithmetic done early, in the background,
        while somebody types.

        The options must be the ones every later request will carry too.
        Loading under a different num_ctx would be worse than not loading at
        all: Ollama would evict and reload the model on the first real
        question, so the wait would be paid twice.

        Never raises. A server too old for this, a model that is not there,
        a machine with no room -- each is the first request's problem to
        report properly, and none of them is a reason to fail a start-up.
        """
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages or [],
            "keep_alive": self.config.keep_alive,
            "options": {"num_ctx": num_ctx or self.config.num_ctx},
        }
        if messages:
            # One token, discarded. Zero is accepted by some builds as "load
            # only" and by others as "evaluate nothing", and a warm that
            # quietly skipped the prefill would look exactly like one that
            # worked.
            payload["options"]["num_predict"] = 1
            payload["stream"] = False
            if tools:
                payload["tools"] = tools
        try:
            r = await self._client.post("/api/chat", json=payload,
                                        timeout=self.config.request_timeout)
            return r.status_code < 400
        except (httpx.HTTPError, ValueError):
            return False

    async def pull(self, model: str) -> AsyncIterator[str]:
        """Stream human-readable progress while a model downloads."""
        payload = {"model": model, "stream": True}
        async with self._client.stream(
            "POST", "/api/pull", json=payload, timeout=None
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise ProviderError(f"Pull failed: {response.text.strip()}")
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if (data := json_object(line)) is None:
                    continue
                if err := data.get("error"):
                    raise ProviderError(f"Pull failed: {err}")
                status = data.get("status", "")
                total, completed = data.get("total"), data.get("completed")
                if total and completed:
                    pct = 100 * completed / total
                    yield f"{status} {pct:.0f}%"
                elif status:
                    yield status

    # -- generation --------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        think: bool | str | None = None,
        temperature: float = 0.4,
        num_predict: int = -1,
        num_ctx: int | None = None,
        stream: bool = True,
        extra_options: dict[str, Any] | None = None,
    ) -> AsyncIterator[Chunk]:
        """Stream one assistant turn."""
        options: dict[str, Any] = {
            "num_ctx": num_ctx or self.config.num_ctx,
            "temperature": temperature,
        }
        if num_predict and num_predict > 0:
            options["num_predict"] = num_predict
        if extra_options:
            options.update(extra_options)

        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.config.keep_alive,
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            if isinstance(think, str) and not self.think_levels_supported:
                think = True
            payload["think"] = think

        emitted = False
        for attempt in range(CONNECT_ATTEMPTS):
            try:
                async for chunk in self._stream_chat(payload):
                    emitted = True
                    yield chunk
                return
            except httpx.ReadTimeout as exc:
                raise ProviderError(
                    f"The model did not respond within "
                    f"{self.config.request_timeout:.0f}s. On CPU or a loaded "
                    "GPU this can be normal for a 30B -- raise "
                    "`request_timeout` in your config, or use a smaller model."
                ) from exc
            except _TRANSIENT as exc:
                # Ollama drops connections while it loads a model, and
                # loading a 30B from cold takes long enough that the first
                # request of a session is the one most likely to be hit.
                #
                # Retried only while nothing has been emitted. Once tokens
                # have reached the user, a second attempt would replay the
                # answer from the top and print it twice -- worse than the
                # error it was trying to hide.
                if emitted or attempt == CONNECT_ATTEMPTS - 1:
                    raise ProviderError(
                        self._explain_transient(exc, emitted)) from exc
                await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"Request to {self.base_url} failed: {exc}") from exc

    async def _stream_chat(self, payload: dict) -> AsyncIterator[Chunk]:
        """Issue the request, retrying once without a string think level.

        The retry is safe because a rejected request fails on the status line,
        before any chunk has been yielded -- nothing has been emitted that a
        second attempt could duplicate.
        """
        timeout = self.config.request_timeout
        async with self._client.stream(
            "POST", "/api/chat", json=payload, timeout=timeout
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                if self._is_think_level_rejection(body, payload):
                    self.think_levels_supported = False
                    payload = {**payload, "think": True}
                else:
                    raise ProviderError(
                        self._explain_error(response.status_code, body, payload)
                    )
            else:
                finished = False
                produced = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if (data := json_object(line)) is None:
                        continue
                    if err := data.get("error"):
                        # Mid-stream errors used to be re-raised verbatim,
                        # so a template parse failure reached the user as
                        # "XML syntax error on line 6" with no hint that it
                        # was the model's output and not their machine.
                        raise ProviderError(
                            self._explain_error(200, str(err), payload))
                    chunk = self._to_chunk(data)
                    finished = finished or chunk.done
                    produced = produced or bool(
                        chunk.content or chunk.thinking or chunk.tool_calls)
                    yield chunk
                if produced and not finished:
                    # The socket stopped without Ollama ever sending
                    # `done: true`: the model was still generating. Reported
                    # rather than raised, because half an answer is still
                    # worth reading -- it just must not be mistaken for a
                    # whole one.
                    yield Chunk(done=True, truncated=True)
                return

        # Only reached after a think-level downgrade.
        async with self._client.stream(
            "POST", "/api/chat", json=payload, timeout=timeout
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise ProviderError(self._explain_error(response.status_code, body, payload))
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if (data := json_object(line)) is None:
                    continue
                if err := data.get("error"):
                    raise ProviderError(
                        self._explain_error(200, str(err), payload))
                yield self._to_chunk(data)

    @staticmethod
    def _is_think_level_rejection(body: str, payload: dict) -> bool:
        """Did this fail only because `think` was a string rather than a bool?"""
        if not isinstance(payload.get("think"), str):
            return False
        return "think" in body.lower()

    @staticmethod
    def _to_chunk(data: dict) -> Chunk:
        # Everything below is coerced rather than trusted. Ollama itself is
        # well behaved, but wynxo also talks to llama.cpp's server, LM Studio
        # and assorted compat shims, and those disagree about shapes: some
        # send `reasoning` as an object, some send counts as strings. A
        # mistyped field used to travel inland and die as an AttributeError
        # halfway through a turn, which reads to the user as wynxo crashing
        # rather than as a peculiar server.
        message = data.get("message")
        if not isinstance(message, dict):
            message = {}
        return Chunk(
            content=as_text(message.get("content")),
            # Ollama has used both keys across versions.
            thinking=as_text(message.get("thinking"))
                     or as_text(message.get("reasoning")),
            tool_calls=[c for c in as_list(message.get("tool_calls"))
                        if isinstance(c, dict)],
            done=bool(data.get("done")),
            stop_reason=as_text(data.get("done_reason")),
            prompt_tokens=as_int(data.get("prompt_eval_count")),
            completion_tokens=as_int(data.get("eval_count")),
            total_duration_ns=as_int(data.get("total_duration")),
            load_duration_ns=as_int(data.get("load_duration")),
            eval_duration_ns=as_int(data.get("eval_duration")),
        )

    def _explain_transient(self, exc: Exception, emitted: bool) -> str:
        """Why a connection failed, once retrying has stopped helping."""
        if emitted:
            return (
                f"The connection to {self.base_url} dropped while the model "
                "was still answering, so the reply above is incomplete. "
                "wynxo did not retry: it would have started the answer again "
                "from the top rather than continuing it."
            )
        return (
            f"Could not hold a connection to {self.base_url} after "
            f"{CONNECT_ATTEMPTS} attempts ({type(exc).__name__}).\n"
            "  - If Ollama is loading a large model, it can refuse "
            "connections for a while; try again once it has settled.\n"
            "  - Check `ollama serve` is still running and did not run out "
            "of memory.\n"
            "  - On a remote box, check the network and that it is started "
            "with OLLAMA_HOST=0.0.0.0:11434."
        )

    def _explain_error(self, status: int, body: str, payload: dict) -> str:
        text = body.strip()
        # A server is free to answer an error with a bare JSON string, a
        # list, a number, or nothing that parses at all. Reaching straight
        # for .get raised AttributeError out of the one function whose job
        # is to explain a failure, replacing the server's own diagnosis
        # with a traceback about explaining it.
        if reported := as_text((json_object(body) or {}).get("error")).strip():
            text = reported
        low = text.lower()
        model = payload.get("model", "?")
        if status == 404 or "not found" in low:
            return (
                f"The server does not have {model!r}.\n"
                f"  Pull it first:  ollama pull {model}\n"
                f"  Or run /model inside wynxo to pick one it does have."
            )
        if _is_template_parse_error(low):
            return (
                f"{model!r} produced a tool call its own Ollama template "
                "could not parse.\n"
                f"  ({text})\n"
                "  This is the model writing malformed output, not a problem "
                "with your setup.\n"
                "  It usually clears on a retry. If it keeps happening:\n"
                "    - lower the effort level, which shortens each reply\n"
                "    - update the model:  ollama pull " + str(model) + "\n"
                "    - or try a different tool-tuned model"
            )
        if "does not support tools" in low:
            return (
                f"{model!r} has no tool-calling support in its template.\n"
                "  wynxo will fall back to Hermes-style prompted tool calls, "
                "but a tool-tuned model (qwen3-coder, devstral) works far better."
            )
        if "memory" in low or "out of memory" in low:
            return (
                f"{model!r} does not fit in available memory at num_ctx="
                f"{payload.get('options', {}).get('num_ctx')}.\n"
                "  Try a lower num_ctx (/ctx 16384), a smaller quant, or set\n"
                "  OLLAMA_KV_CACHE_TYPE=q8_0 on the server to halve KV memory."
            )
        return f"Ollama returned {status}: {text}"


class OpenAIClient:
    """An OpenAI-compatible ``/v1/chat/completions`` server.

    Same surface as :class:`OllamaClient` -- ``ping``, ``list_models``,
    ``show``, ``chat``, ``aclose`` -- so the agent does not care which it
    is talking to. This is what lets wynxo work against any OpenAI style
    endpoint: a real OpenAI account, a self-hosted gateway, or Ollama's
    own OpenAI shim at ``localhost:11434/v1``.
    """

    def __init__(self, config: Config):
        self.config = config
        self.think_levels_supported = True   # attribute parity with Ollama
        ep = config.endpoint()
        self.base_url = ep.url
        headers = {"Content-Type": "application/json"}
        if ep.api_key:
            headers["Authorization"] = f"Bearer {ep.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=10.0, read=config.request_timeout, write=60.0,
                pool=10.0,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAIClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def running(self) -> list["Loaded"]:
        """Nothing. The protocol has no notion of a resident model.

        Present so callers do not have to ask which provider they are
        holding: an empty list already means "could not tell", which is
        exactly the truth here, and every caller treats it as "say
        nothing".
        """
        return []

    async def warm(self, model: str = "", num_ctx: int = 0,
                   messages: list[dict] | None = None,
                   tools: list[dict] | None = None) -> bool:
        """Nothing to warm. Loading is the server's business, not ours.

        The arguments are the Ollama client's and are ignored, but they
        have to be accepted: start-up calls this without knowing which
        provider it is holding, and a signature that does not match is a
        TypeError on the first line of every session against an
        OpenAI-compatible server.
        """
        return False

    async def ping(self) -> str:
        """Return a label or raise something readable if unreachable."""
        try:
            r = await self._client.get(_OpenAI_MODELS_PATH, timeout=10.0)
            r.raise_for_status()
            return "openai-compatible"
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Cannot reach an OpenAI-compatible server at {self.base_url}\n"
                f"  ({exc})"
            ) from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            r = await self._client.get(_OpenAI_MODELS_PATH, timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not list models: {exc}") from exc
        payload = _payload(r, _OpenAI_MODELS_PATH, self.base_url)
        out = [
            ModelInfo(name=m.get("id", "?"))
            for m in payload.get("data", payload.get("_list", []))
            if isinstance(m, dict)
        ]
        out.sort(key=lambda m: m.name)
        return out

    async def show(self, model: str) -> ModelInfo:
        """The OpenAI protocol exposes no capability metadata, so capabilities
        are 'unknown' -- callers treat that as assume-it-works, which is the
        right default for a server with no introspection endpoint."""
        return ModelInfo(name=model, capabilities=None)

    async def pull(self, model: str) -> AsyncIterator[str]:
        raise ProviderError(
            f"{self.base_url} is OpenAI-compatible; it has no pull."
            " Install the model with the provider's own tooling."
        )

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        think: bool | str | None = None,
        temperature: float = 0.4,
        num_predict: int = -1,
        num_ctx: int | None = None,
        stream: bool = True,
        extra_options: dict[str, Any] | None = None,
    ) -> AsyncIterator[Chunk]:
        """Stream one assistant turn over ``/chat/completions``.

        Ollama-specific options (``think``, ``keep_alive``, ``num_ctx``
        under ``options``) are accepted and ignored -- the OpenAI protocol
        has no direct equivalents, and reasoning is often enabled on the
        model's side already.
        """
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": _openai_messages(messages),
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            # The registry's tool schemas are already the OpenAI shape
            # ({type, function:{name, description, parameters}}).
            payload["tools"] = tools
        if num_predict and num_predict > 0:
            payload["max_completion_tokens"] = num_predict
        if extra_options:
            for key, value in extra_options.items():
                if key in ("keep_alive", "num_ctx"):
                    continue      # Ollama-only
                payload[key] = value

        async def body() -> AsyncIterator[Chunk]:
            calls: dict[int, dict] = {}
            prompt_tokens = 0
            completion_tokens = 0
            stop_reason = ""
            finished = False
            """Whether the provider said it was done, rather than the socket
            simply stopping. Either [DONE] or a finish_reason counts; shims
            differ about which they send, and some send both."""
            produced = False
            async with self._client.stream(
                "POST", _OpenAI_CHAT_PATH, json=payload,
                timeout=self.config.request_timeout,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        self._explain_error(response.status_code, body, payload))
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        finished = True
                        break
                    if (obj := json_object(data)) is None:
                        continue
                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = as_int(usage.get("prompt_tokens"))
                        completion_tokens = as_int(usage.get("completion_tokens"))
                    choices = obj.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    if reason := as_text(choices[0].get("finish_reason")):
                        finished = True
                        stop_reason = reason
                    delta = choices[0].get("delta") or {}
                    if content := as_text(delta.get("content")):
                        produced = True
                        yield Chunk(content=content)
                    if thinking := as_text(delta.get("reasoning")) \
                          or as_text(delta.get("thinking")):
                        yield Chunk(thinking=thinking)
                    for call in delta.get("tool_calls") or []:
                        index = call.get("index", 0)
                        acc = calls.setdefault(
                            index, {"id": "", "name": "", "arguments": []})
                        if call.get("id"):
                            acc["id"] = call["id"]
                        fn = call.get("function") or {}
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        if fn.get("arguments"):
                            produced = True
                            acc["arguments"].append(fn["arguments"])
                            # Emitted as it arrives, as well as accumulated
                            # for the finished call below.
                            yield Chunk(arguments_delta=fn["arguments"])
                # Flush whatever accumulated before [DONE] / stream end.
                # A plain turn never touches ``calls``, so emitting it
            # unfiltered is safe either way -- only genuine delta tool_calls
            # populate it, and they must survive the trailing [DONE].
            tool_calls: list[dict] = []
            for acc in calls.values():
                if not acc["name"]:
                    continue
                tool_calls.append({
                    "id": acc["id"] or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": acc["name"],
                        "arguments": "".join(acc["arguments"]),
                    },
                })
            yield Chunk(
                tool_calls=tool_calls,
                done=True,
                stop_reason=stop_reason,
                # Only when something was actually being generated. A stream
                # that produced nothing at all is the empty-answer case, which
                # already has its own handling and its own message.
                truncated=produced and not finished,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        # A single attempt: OpenAI is a network round trip without Ollama's
        # model-loading drops, so the transient-retry loop is unnecessary.
        try:
            async for chunk in body():
                yield chunk
        except httpx.ReadTimeout as exc:
            raise ProviderError(
                f"The model did not respond within "
                f"{self.config.request_timeout:.0f}s (raise `request_timeout` "
                "in your config, or use a smaller model)."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Request to {self.base_url} failed: {exc}") from exc

    def _explain_error(self, status: int, body: str, payload: dict) -> str:
        text = body.strip()
        # A server is free to answer an error with a bare JSON string, a
        # list, a number, or nothing that parses at all. Reaching straight
        # for .get raised AttributeError out of the one function whose job
        # is to explain a failure, replacing the server's own diagnosis
        # with a traceback about explaining it.
        if reported := as_text((json_object(body) or {}).get("error")).strip():
            text = reported
        if status == 404 or "model not found" in text.lower():
            return (
                f"The server does not have {payload.get('model', '?')!r}.\n"
                "  Install it with the provider's own tooling, then run /model."
            )
        if status == 401:
            return (
                f"{self.base_url} rejected the request with 401.\n"
                "  Check the endpoint's API key / authentication."
            )
        return f"The server returned {status}: {text}"


def _openai_messages(messages: list[dict]) -> list[dict]:
    """Translate wynxo's Ollama-shaped conversation into OpenAI wire format.

    wynxo stores assistant tool calls and ``role: tool`` results; the OpenAI
    protocol wants ``type: function`` on every call, ``function.arguments``
    as a JSON string, and a matching ``tool_call_id`` on each tool result -
    the call id is carried through so the pair lines up.

    Ollama's wire shape carries no ids at all, so most conversations reach
    here without them and both sides have to invent the same ones. The
    announcing side numbered its calls by position within the message and
    the answering side did not: every result with no stored id fell back to
    a flat "call_0". A turn that called two tools therefore sent two answers
    both claiming to answer the first, and left the second call unanswered
    -- which a strict server rejects outright, and a lenient one acts on
    with the results attributed to the wrong calls. Both sides count
    positions now, which is the convention ``close_open_tool_calls`` was
    already written against.
    """
    out: list[dict] = []
    answered = 0
    """How many results have followed the current assistant message. The
    nth answers the nth call."""
    for message in messages:
        role = message.get("role", "user")
        if role == "assistant":
            answered = 0
            item: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or None,
            }
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                openai_calls = []
                for index, call in enumerate(calls):
                    function = call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, default=str)
                    openai_calls.append({
                        "id": call.get("id") or f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": function.get("name", "?"),
                            "arguments": arguments,
                        },
                    })
                item["tool_calls"] = openai_calls
            out.append(item)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": message.get("tool_call_id")
                               or message.get("id") or f"call_{answered}",
                "content": message.get("content") or "",
            })
            answered += 1
        else:
            answered = 0
            out.append({"role": role, "content": message.get("content") or ""})
    return out


_OpenAI_CHAT_PATH = "/v1/chat/completions"
_OpenAI_MODELS_PATH = "/v1/models"
"""The OpenAI protocol paths. ``normalise_url`` strips a trailing /v1 as
an Ollama artefact, so the client adds it back on to the bare host."""


def make_client(config: Config) -> "OllamaClient | OpenAIClient":
    """Pick the right client for the configured endpoint.

    ``endpoint.kind`` decides: 'openai' for any OpenAI-compatible /v1 server,
    'ollama' or the 'auto' default for native Ollama (the richer endpoint).
    """
    if config.endpoint().kind == "openai":
        return OpenAIClient(config)
    return OllamaClient(config)


async def inspect_all(client: "OllamaClient", models: list[ModelInfo],
                      timeout: float = 20.0) -> list[ModelInfo]:
    """Fill in capabilities and context length for every model, concurrently.

    A server with a dozen models would take a dozen round trips in sequence.
    Failures are left as-is rather than raised: an unknown capability is a
    blank column, not a reason to fail the whole picker.
    """
    import asyncio

    async def one(model: ModelInfo) -> ModelInfo:
        try:
            detail = await asyncio.wait_for(client.show(model.name), timeout=timeout)
        except (ProviderError, asyncio.TimeoutError):
            return model
        model.capabilities = detail.capabilities
        model.context_length = detail.context_length
        if detail.parameter_size:
            model.parameter_size = detail.parameter_size
        if detail.quantization:
            model.quantization = detail.quantization
        return model

    return list(await asyncio.gather(*(one(m) for m in models)))


async def check_context(client: OllamaClient, config: Config) -> str | None:
    """Warn when the configured context is smaller than an agent needs.

    Returns a warning string, or None when everything is fine. This exists
    because the failure mode it catches is silent: the model simply forgets
    the earlier half of the task and nobody is told why.
    """
    if config.num_ctx < MIN_USABLE_CONTEXT:
        return (
            f"num_ctx is {config.num_ctx}, below the {MIN_USABLE_CONTEXT} an "
            "agent realistically needs. Long tasks will silently lose history. "
            "Raise it with /ctx."
        )
    try:
        info = await client.show(config.model)
    except ProviderError:
        return None
    if info.context_length and config.num_ctx > info.context_length:
        return (
            f"num_ctx {config.num_ctx} exceeds what {config.model} was trained "
            f"for ({info.context_length}). Ollama will accept it, but quality "
            "degrades past the native window."
        )
    if config.num_ctx > LARGE_CONTEXT:
        # KV cache grows linearly with the window and is allocated up front,
        # so this is where a model that used to fit stops fitting.
        return (
            f"num_ctx is {config.num_ctx}. The KV cache scales with it and is "
            "reserved up front, which on a 30B can be many gigabytes on top of "
            "the weights. If loading gets slow or fails, drop to 32768 -- and "
            "set OLLAMA_KV_CACHE_TYPE=q8_0 on the server to roughly halve it."
        )
    return None
