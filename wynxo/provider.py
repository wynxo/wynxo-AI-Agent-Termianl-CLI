"""Ollama client.

Talks to Ollama's native ``/api/chat`` rather than the OpenAI-compatible
shim, because the native endpoint is the one that exposes ``think``,
``keep_alive`` and per-request ``options``.
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
    name: str
    size: int = 0
    size_vram: int = 0
    context_length: int = 0
    digest: str = ""

    @property
    def on_gpu(self) -> float:
        if self.size <= 0:
            return 1.0
        return min(1.0, self.size_vram / self.size)

    @property
    def split(self) -> bool:
        return 0 < self.size_vram < self.size

    def context_that_fits(self, weights: int, num_ctx: int) -> int:
        return self.context_within(weights, num_ctx, self.size_vram)

    def context_within(self, weights: int, num_ctx: int, budget: int) -> int:
        if weights <= 0 or num_ctx <= 0 or self.size <= weights:
            return 0
        per_token = (self.size - weights) / num_ctx
        room = budget - weights
        if per_token <= 0 or room <= 0:
            return 0
        return int(room / per_token) // 1024 * 1024

    def share_at(self, weights: int, num_ctx: int, want: int) -> float:
        if weights <= 0 or num_ctx <= 0 or self.size <= weights:
            return self.on_gpu
        per_token = (self.size - weights) / num_ctx
        need = weights + per_token * max(0, want)
        return min(1.0, self.size_vram / need) if need > 0 else 1.0


FAST_ENOUGH = 0.8


def gpu_share(weights: int, vram: int) -> float:
    if weights <= 0:
        return 0.0
    return min(1.0, max(0, vram) / weights)


def faster_on_gpu(models: list["ModelInfo"], vram: int,
                  better_than: float) -> list["ModelInfo"]:
    if vram <= 0:
        return []
    scored = [(gpu_share(m.size, vram), m) for m in models if m.size > 0]
    good = [(share, m) for share, m in scored if share > better_than + 0.2]
    good.sort(key=lambda pair: (min(pair[0], FAST_ENOUGH), pair[1].size), reverse=True)
    return [m for _, m in good]


def _model_base(name: str) -> str:
    return name.strip().lower().split(":", 1)[0]


def same_model(a: str, b: str) -> bool:
    """Compare tags conservatively.

    Names with different tags are not assumed equivalent: qwen3:8b and
    qwen3:30b must never collapse into one diagnostic entry. Exact names are
    treated case-insensitively. Alias resolution belongs to the server/API,
    not to string-prefix guessing.
    """
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and a == b


@dataclass
class ModelInfo:
    name: str
    size: int = 0
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""
    context_length: int = 0
    capabilities: list[str] | None = None
    digest: str = ""

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
        return f"{self.size / 1_000_000_000:.1f}GB"


@dataclass
class Chunk:
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    arguments_delta: str = ""
    done: bool = False
    truncated: bool = False
    stop_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0


def _payload(response, what: str, where: str) -> dict:
    try:
        data = response.json()
    except ValueError as exc:
        body = (response.text or "").strip()
        looks_like_html = body[:1] == "<"
        detail = ("an HTML page" if looks_like_html else "an empty body" if not body else f"{body[:60]!r}")
        raise ProviderError(
            f"{where} answered {what} with {detail} rather than JSON. Check the port and API endpoint."
        ) from exc
    if isinstance(data, list):
        return {"_list": data}
    if not isinstance(data, dict):
        raise ProviderError(f"{where} answered {what} with {type(data).__name__} rather than an object.")
    return data


_TRANSIENT = (
    httpx.ConnectError, httpx.ReadError, httpx.WriteError,
    httpx.RemoteProtocolError, httpx.ConnectTimeout, httpx.PoolTimeout,
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
            timeout=httpx.Timeout(connect=10.0, read=config.request_timeout, write=30.0, pool=10.0),
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
            return _payload(r, "/api/version", self.base_url).get("version", "unknown")
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot reach an Ollama server at {self.base_url}.\n"
                "  - Is `ollama serve` running there?\n"
                "  - A remote server generally needs OLLAMA_HOST=0.0.0.0:11434.\n"
                "  - Check the firewall and port."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"{self.base_url} answered {exc.response.status_code}; it does not look like Ollama.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach {self.base_url}: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            r = await self._client.get("/api/tags", timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not list models: {exc}") from exc
        payload = _payload(r, "/api/tags", self.base_url)
        out = []
        for m in payload.get("models", payload.get("_list", [])):
            if not isinstance(m, dict):
                continue
            details = m.get("details") or {}
            out.append(ModelInfo(
                name=as_text(m.get("name")) or "?",
                size=as_int(m.get("size")),
                parameter_size=as_text(details.get("parameter_size")),
                quantization=as_text(details.get("quantization_level")),
                family=as_text(details.get("family")),
                digest=as_text(m.get("digest")),
            ))
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
            parameter_size=as_text(details.get("parameter_size")),
            quantization=as_text(details.get("quantization_level")),
            family=as_text(details.get("family")),
            capabilities=data.get("capabilities"),
            digest=as_text(data.get("digest")),
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
                context_length=as_int(entry.get("context_length") or details.get("context_length")),
                digest=as_text(entry.get("digest")),
            ))
        return out

    async def warm(self, model: str = "", num_ctx: int = 0,
                   messages: list[dict] | None = None,
                   tools: list[dict] | None = None) -> bool:
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages or [],
            "keep_alive": self.config.keep_alive,
            "options": {"num_ctx": num_ctx or self.config.num_ctx},
        }
        if messages:
            payload["options"]["num_predict"] = 1
            payload["stream"] = False
            if tools:
                payload["tools"] = tools
        try:
            r = await self._client.post("/api/chat", json=payload, timeout=self.config.request_timeout)
            return r.status_code < 400
        except (httpx.HTTPError, ValueError):
            return False

    async def pull(self, model: str) -> AsyncIterator[str]:
        payload = {"model": model, "stream": True}
        async with self._client.stream("POST", "/api/pull", json=payload, timeout=None) as response:
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
                    yield f"{status} {100 * completed / total:.0f}%"
                elif status:
                    yield status

    async def chat(self, messages: list[dict], *, model: str | None = None,
                   tools: list[dict] | None = None, think: bool | str | None = None,
                   temperature: float = 0.4, num_predict: int = -1,
                   num_ctx: int | None = None, stream: bool = True,
                   extra_options: dict[str, Any] | None = None) -> AsyncIterator[Chunk]:
        options: dict[str, Any] = {"num_ctx": num_ctx or self.config.num_ctx, "temperature": temperature}
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
                    f"The model did not respond within {self.config.request_timeout:.0f}s. "
                    "Raise request_timeout or use a smaller model."
                ) from exc
            except _TRANSIENT as exc:
                if emitted or attempt == CONNECT_ATTEMPTS - 1:
                    raise ProviderError(self._explain_transient(exc, emitted)) from exc
                await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
            except httpx.HTTPError as exc:
                raise ProviderError(f"Request to {self.base_url} failed: {exc}") from exc

    async def _stream_chat(self, payload: dict) -> AsyncIterator[Chunk]:
        timeout = self.config.request_timeout
        async with self._client.stream("POST", "/api/chat", json=payload, timeout=timeout) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                if self._is_think_level_rejection(body, payload):
                    self.think_levels_supported = False
                    retry = {**payload, "think": True}
                else:
                    raise ProviderError(self._explain_error(response.status_code, body, payload))
            else:
                finished = False
                produced = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if (data := json_object(line)) is None:
                        continue
                    if err := data.get("error"):
                        raise ProviderError(self._explain_error(200, str(err), payload))
                    chunk = self._to_chunk(data)
                    finished = finished or chunk.done
                    produced = produced or bool(chunk.content or chunk.thinking or chunk.tool_calls)
                    yield chunk
                if produced and not finished:
                    yield Chunk(done=True, truncated=True)
                return

        async with self._client.stream("POST", "/api/chat", json=retry, timeout=timeout) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise ProviderError(self._explain_error(response.status_code, body, retry))
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if (data := json_object(line)) is None:
                    continue
                if err := data.get("error"):
                    raise ProviderError(self._explain_error(200, str(err), retry))
                yield self._to_chunk(data)

    @staticmethod
    def _is_think_level_rejection(body: str, payload: dict) -> bool:
        return isinstance(payload.get("think"), str) and "think" in body.lower()

    @staticmethod
    def _to_chunk(data: dict) -> Chunk:
        message = data.get("message")
        if not isinstance(message, dict):
            message = {}
        return Chunk(
            content=as_text(message.get("content")),
            thinking=as_text(message.get("thinking")) or as_text(message.get("reasoning")),
            tool_calls=[c for c in as_list(message.get("tool_calls")) if isinstance(c, dict)],
            done=bool(data.get("done")),
            stop_reason=as_text(data.get("done_reason")),
            prompt_tokens=as_int(data.get("prompt_eval_count")),
            completion_tokens=as_int(data.get("eval_count")),
            total_duration_ns=as_int(data.get("total_duration")),
            load_duration_ns=as_int(data.get("load_duration")),
        )

    def _explain_transient(self, exc: Exception, emitted: bool) -> str:
        if emitted:
            return (f"The connection to {self.base_url} dropped while the model was answering; "
                    "the reply above is incomplete. wynxo did not replay it.")
        return (f"Could not hold a connection to {self.base_url} after {CONNECT_ATTEMPTS} attempts "
                f"({type(exc).__name__}). Check the server, memory and network.")

    def _explain_error(self, status: int, body: str, payload: dict) -> str:
        text = body.strip()
        parsed = json_object(body)
        if isinstance(parsed, dict):
            reported = as_text(parsed.get("error")).strip()
            if reported:
                text = reported
        low = text.lower()
        model = payload.get("model", "?")
        if status == 404 or "not found" in low:
            return f"The server does not have {model!r}. Pull it first or choose an installed model."
        if _is_template_parse_error(low):
            return f"{model!r} produced a tool call its template could not parse: {text}"
        return f"Ollama returned HTTP {status} for {model!r}: {text or 'no details'}"
