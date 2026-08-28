"""Provider-neutral model contracts used by the agent orchestration layer."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from .provider import Chunk


class ModelBackend(Protocol):
    """Minimum capability required by the generic coding agent."""

    def chat(self, messages: list[dict], **options) -> AsyncIterator[Chunk]: ...


class OllamaBackend:
    """Small adapter that keeps the agent independent of Ollama's class name."""

    def __init__(self, client):
        self.client = client

    def chat(self, messages: list[dict], **options) -> AsyncIterator[Chunk]:
        return self.client.chat(messages, **options)
