from __future__ import annotations

import json


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "done"}
    return False


def install() -> None:
    from .coerce import as_int, as_list, as_text
    from .provider import ModelInfo, OllamaClient, ProviderError

    original_chunk = OllamaClient._to_chunk
    if not getattr(original_chunk, "_wynxo_wire_hardened", False):
        @staticmethod
        def to_chunk(data):
            if not isinstance(data, dict):
                return original_chunk({})
            message = data.get("message")
            if not isinstance(message, dict):
                message = {}
            chunk = original_chunk(data)
            chunk.content = as_text(message.get("content"))
            chunk.thinking = as_text(message.get("thinking")) or as_text(message.get("reasoning"))
            chunk.tool_calls = [item for item in as_list(message.get("tool_calls")) if isinstance(item, dict)]
            chunk.done = _boolish(data.get("done"))
            chunk.prompt_tokens = as_int(data.get("prompt_eval_count"))
            chunk.completion_tokens = as_int(data.get("eval_count"))
            chunk.total_duration_ns = as_int(data.get("total_duration"))
            chunk.load_duration_ns = as_int(data.get("load_duration"))
            return chunk

        to_chunk._wynxo_wire_hardened = True
        OllamaClient._to_chunk = to_chunk

    original_list = OllamaClient.list_models
    if not getattr(original_list, "_wynxo_list_hardened", False):
        async def list_models(self):
            try:
                data = await original_list(self)
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    f"The Ollama model list at {self.base_url} had an invalid response ({type(exc).__name__})."
                ) from exc
            return data if isinstance(data, list) else []

        list_models._wynxo_list_hardened = True
        OllamaClient.list_models = list_models

    original_show = OllamaClient.show
    if not getattr(original_show, "_wynxo_show_hardened", False):
        async def show(self, model):
            try:
                info = await original_show(self, model)
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    f"The Ollama inspection response for {model!r} had an invalid shape ({type(exc).__name__})."
                ) from exc
            if not isinstance(info, ModelInfo):
                return ModelInfo(name=model)
            if info.capabilities is not None:
                info.capabilities = [as_text(value) for value in as_list(info.capabilities)]
            info.context_length = as_int(info.context_length)
            info.parameter_size = as_text(info.parameter_size)
            info.quantization = as_text(info.quantization)
            info.family = as_text(info.family)
            return info

        show._wynxo_show_hardened = True
        OllamaClient.show = show

    original_ping = OllamaClient.ping
    if not getattr(original_ping, "_wynxo_ping_hardened", False):
        async def ping(self):
            try:
                return await original_ping(self)
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    f"Ollama at {self.base_url} returned an invalid /api/version response ({type(exc).__name__})."
                ) from exc

        ping._wynxo_ping_hardened = True
        OllamaClient.ping = ping


install()
