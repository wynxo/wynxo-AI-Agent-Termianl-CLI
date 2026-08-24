"""Ollama client.

Talks to Ollama's native ``/api/chat`` rather than the OpenAI-compatible
shim, because the native endpoint is the one that exposes ``think``,
``keep_alive`` and per-request ``options`` -- all three of which matter for
running a 30B locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from .config import Config, MIN_USABLE_CONTEXT

LARGE_CONTEXT = 131_072


class ProviderError(RuntimeError):
    """Anything that went wrong talking to the server, phrased for a human."""


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
    done: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0


def _as_text(value: object) -> str:
    """A wire field as a string, whatever the server actually sent.

    ``None`` and ``""`` both mean absent, so both come back empty. Anything
    else is stringified rather than dropped: a server that wraps reasoning in
    an object is being odd, but the text inside is still the model's thought
    and the user would rather see it than lose it.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        # The shapes seen in the wild all park the text under one of these.
        for key in ("text", "content", "thinking", "reasoning", "value"):
            if key in value:
                return _as_text(value[key])
        return ""
    return str(value)


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        # A single call sent unwrapped, which some shims do.
        return [value]
    return []


def _as_int(value: object) -> int:
    """A count as an int. Strings and floats appear in compat servers."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value and value not in (
            float("inf"), float("-inf")) else 0
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (ValueError, OverflowError):
            return 0
    return 0


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
            return r.json().get("version", "unknown")
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
        for m in r.json().get("models", []):
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
        data = r.json()
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
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
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

        try:
            async for chunk in self._stream_chat(payload):
                yield chunk
        except httpx.ReadTimeout as exc:
            raise ProviderError(
                f"The model did not respond within {self.config.request_timeout:.0f}s. "
                "On CPU or a loaded GPU this can be normal for a 30B -- raise "
                "`request_timeout` in your config, or use a smaller model."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Request to {self.base_url} failed: {exc}") from exc

    async def _stream_chat(self, payload: dict) -> AsyncIterator[Chunk]:
        """Issue the request, retrying once without a string think level.

        The retry is safe because a rejected request fails on the status line,
        before any chunk has been yielded -- nothing has been emitted that a
        second attempt could duplicate.
        """
        async with self._client.stream(
            "POST", "/api/chat", json=payload, timeout=None
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
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if err := data.get("error"):
                        # Mid-stream errors used to be re-raised verbatim,
                        # so a template parse failure reached the user as
                        # "XML syntax error on line 6" with no hint that it
                        # was the model's output and not their machine.
                        raise ProviderError(
                            self._explain_error(200, str(err), payload))
                    yield self._to_chunk(data)
                return

        # Only reached after a think-level downgrade.
        async with self._client.stream(
            "POST", "/api/chat", json=payload, timeout=None
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                raise ProviderError(self._explain_error(response.status_code, body, payload))
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
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
            content=_as_text(message.get("content")),
            # Ollama has used both keys across versions.
            thinking=_as_text(message.get("thinking"))
                     or _as_text(message.get("reasoning")),
            tool_calls=[c for c in _as_list(message.get("tool_calls"))
                        if isinstance(c, dict)],
            done=bool(data.get("done")),
            prompt_tokens=_as_int(data.get("prompt_eval_count")),
            completion_tokens=_as_int(data.get("eval_count")),
            total_duration_ns=_as_int(data.get("total_duration")),
            load_duration_ns=_as_int(data.get("load_duration")),
        )

    def _explain_error(self, status: int, body: str, payload: dict) -> str:
        text = body.strip()
        try:
            text = json.loads(body).get("error", text)
        except json.JSONDecodeError:
            pass
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
