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

        ``size`` is what the model needs in memory at the window it was
        loaded under, and the weights do not change with the window, so
        everything above them is the KV cache -- and the KV cache is linear
        in the number of tokens. Divide by the window and you have the cost
        per token; the room left on the card once the weights are there,
        divided by that, is how many tokens would fit.
        """
        if weights <= 0 or num_ctx <= 0 or self.size <= weights:
            return 0
        per_token = (self.size - weights) / num_ctx
        if per_token <= 0:
            return 0
        room = self.size_vram - weights
        if room <= 0:
            return 0
        return int(room / per_token) // 1024 * 1024

    def context_within(self, weights: int, num_ctx: int, budget: int) -> int:
        """Estimate the largest context that fits within a memory budget."""
        if weights <= 0 or num_ctx <= 0 or budget <= weights or self.size <= weights:
            return 0
        per_token = (self.size - weights) / num_ctx
        if per_token <= 0:
            return 0
        room = budget - weights
        if room <= 0:
            return 0
        return int(room / per_token) // 1024 * 1024

    def share_at(self, weights: int, num_ctx: int, floor_ctx: int = 2048) -> float:
        """Estimate GPU share if the model is loaded at a smaller context."""
        if weights <= 0 or self.size <= 0 or num_ctx <= 0:
            return 0.0
        floor_ctx = max(1, min(floor_ctx, num_ctx))
        if self.size <= weights:
            return 0.0

        # Preserve the same linear KV-cache model used by context_that_fits.
        kv_at_ctx = self.size - weights
        per_token = kv_at_ctx / num_ctx
        needed = weights + per_token * floor_ctx
        if needed <= 0:
            return 0.0
        return min(1.0, self.size_vram / needed)


def same_model(a: str, b: str) -> bool:
    """Whether two model names are the same model."""
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and (a == b or a.split(":")[0] == b.split(":")[0])


def gpu_share(model_size: int, vram: int) -> float:
    """Estimate the fraction of a model that can live on the GPU."""
    if model_size <= 0 or vram <= 0:
        return 0.0
    return min(1.0, max(0.0, vram / model_size))


def faster_on_gpu(models: list["ModelInfo"], vram: int, current_share: float = 0.0) -> list["ModelInfo"]:
    """Return installed models ordered by the estimated GPU share.

    The caller filters out the currently selected model. Keeping the helper
    in the provider module avoids duplicating the simple memory heuristic in
    both the terminal UI and the doctor.
    """
    if vram <= 0:
        return list(models)
    return sorted(
        models,
        key=lambda model: (gpu_share(model.size, vram), -(model.size or 0)),
        reverse=True,
    )


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


def _payload(response, what: str, where: str) -> dict:
    """The response body as a mapping, or a ProviderError worth reading."""
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

CONNECT_ATTEMPTS = 3
RETRY_BACKOFF = 0.75


def _is_template_parse_error(low: str) -> bool:
    if "syntax error" in low and ("xml" in low or "element" in low):
        return True
    return ("template" in low and "error" in low) or "closed by" in low


class OllamaClient:
    def __init__(self, config: Config):
        self.config = config
        self.think_levels_supported = True
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

    async def ping(self) -> str:
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
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                info.context_length = value
                break
        return info

    async def running(self) -> list[Loaded]:
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

    async def warm(self, model: str = "", num_ctx: int = 0) -> bool:
        payload = {
            "model": model or self.config.model,
            "messages": [],
            "keep_alive": self.config.keep_alive,
            "options": {"num_ctx": num_ctx or self.config.num_ctx},
        }
        try:
            r = await self._client.post("/api/chat", json=payload,
                                        timeout=self.config.request_timeout)
            return r.status_code < 400
        except (httpx.HTTPError, ValueError):
            return False

    async def pull(self, model: str) -> AsyncIterator[str]:
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
                if emitted or attempt == CONNECT_ATTEMPTS - 1:
                    raise ProviderError(
                        self._explain_transient(exc, emitted)) from exc
                await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"Request to {self.base_url} failed: {exc}") from exc

    async def _stream_chat(self, payload: dict) -> AsyncIterator[Chunk]:
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
                        raise ProviderError(
                            self._explain_error(200, str(err), payload))
                    chunk = self._to_chunk(data)
                    finished = finished or chunk.done
                    produced = produced or bool(
                        chunk.content or chunk.thinking or chunk.tool_calls)
                    yield chunk
                if produced and not finished:
                    yield Chunk(done=True, truncated=True)
                return

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
        if not isinstance(payload.get("think"), str):
            return False
        return "think" in body.lower()

    @staticmethod
    def _to_chunk(data: dict) -> Chunk:
        message = data.get("message")
        if not isinstance(message, dict):
            message = {}
        return Chunk(
            content=as_text(message.get("content")),
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
        )

    def _explain_transient(self, exc: Exception, emitted: bool) -> str:
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
    """An OpenAI-compatible ``/v1/chat/completions`` server."""

    def __init__(self, config: Config):
        self.config = config
        self.think_levels_supported = True
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
        return []

    async def warm(self, model: str = "", num_ctx: int = 0) -> bool:
        return False

    async def ping(self) -> str:
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
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": _openai_messages(messages),
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if num_predict and num_predict > 0:
            payload["max_completion_tokens"] = num_predict
        if extra_options:
            for key, value in extra_options.items():
                if key in ("keep_alive", "num_ctx"):
                    continue
                payload[key] = value

        async def body() -> AsyncIterator[Chunk]:
            calls: dict[int, dict] = {}
            prompt_tokens = 0
            completion_tokens = 0
            stop_reason = ""
            finished = False
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
                    if thinking := as_text(delta.get("reasoning")) or as_text(delta.get("thinking")):
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
                            yield Chunk(arguments_delta=fn["arguments"])
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
                truncated=produced and not finished,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

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
    out: list[dict] = []
    answered = 0
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


def make_client(config: Config) -> "OllamaClient | OpenAIClient":
    if config.endpoint().kind == "openai":
        return OpenAIClient(config)
    return OllamaClient(config)


async def inspect_all(client: "OllamaClient", models: list[ModelInfo],
                      timeout: float = 20.0) -> list[ModelInfo]:
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
        return (
            f"num_ctx is {config.num_ctx}. The KV cache scales with it and is "
            "reserved up front, which on a 30B can be many gigabytes on top of "
            "the weights. If loading gets slow or fails, drop to 32768 -- and "
            "set OLLAMA_KV_CACHE_TYPE=q8_0 on the server to roughly halve it."
        )
    return None
