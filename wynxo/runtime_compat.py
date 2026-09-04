"""Runtime compatibility shims for mixed-version WYNXO modules.

These shims keep the installed CLI usable when a newer caller is paired with
an older provider implementation. They are intentionally tiny and can be
removed once all callers and provider APIs are on the same signature.
"""
from __future__ import annotations

from typing import Any

from .provider import OllamaClient, OpenAIClient


_original_ollama_warm = OllamaClient.warm
_original_openai_warm = OpenAIClient.warm


async def _warm_compat(
    self: OllamaClient,
    model: str = "",
    num_ctx: int = 0,
    *,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> bool:
    """Accept the current warm-up arguments while preserving old callers."""
    if messages is None and tools is None:
        return await _original_ollama_warm(self, model=model, num_ctx=num_ctx)

    payload: dict[str, Any] = {
        "model": model or self.config.model,
        "messages": messages or [],
        "keep_alive": self.config.keep_alive,
        "options": {"num_ctx": num_ctx or self.config.num_ctx},
    }
    # One token forces Ollama to evaluate the supplied prefix instead of
    # merely loading the model and returning an empty generation.
    if messages:
        payload["options"]["num_predict"] = 1
    if tools:
        payload["tools"] = tools

    try:
        response = await self._client.post(
            "/api/chat",
            json=payload,
            timeout=self.config.request_timeout,
        )
        return response.status_code < 400
    except Exception:
        return False


async def _openai_warm_compat(
    self: OpenAIClient,
    model: str = "",
    num_ctx: int = 0,
    *,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> bool:
    """OpenAI-compatible endpoints do not expose a separate warm API."""
    return await _original_openai_warm(self, model=model, num_ctx=num_ctx)


# Install once, before cli.py imports the client.
OllamaClient.warm = _warm_compat
OpenAIClient.warm = _openai_warm_compat
