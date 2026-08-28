from __future__ import annotations

import json


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
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
            return type(original_chunk({}))(
                content=as_text(message.get("content")),
                thinking=as_text(message.get("thinking")) or as_text(message.get("reasoning")),
                tool_calls=[c for c in as_list(message.get("tool_calls")) if isinstance(c, dict)],
                done=_as_bool(data.get("done")),
                prompt_tokens=as_int(data.get("prompt_eval_count")),
                completion_tokens=as_int(data.get("eval_count")),
                total_duration_ns=as_int(data.get("total_duration")),
                load_duration_ns=as_int(data.get("load_duration")),
            )
        to_chunk._wynxo_wire_hardened = True
        OllamaClient._to_chunk = to_chunk

    original_list = OllamaClient.list_models
    if not getattr(original_list, "_wynxo_discovery_hardened", False):
        async def list_models(self):
            try:
                return await original_list(self)
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    f"The Ollama model list at {self.base_url} had an unexpected response shape: {type(exc).__name__}."
                ) from exc
        list_models._wynxo_discovery_hardened = True
        OllamaClient.list_models = list_models

    original_show = OllamaClient.show
    if not getattr(original_show, "_wynxo_show_hardened", False):
        async def show(self, model):
            try:
                info = await original_show(self, model)
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    f"The Ollama inspection response for {model!r} had an unexpected shape: {type(exc).__name__}."
                ) from exc
            # Compat servers sometimes serialize capabilities/context as a
            # string or number. Normalize it at the boundary so the rest of
            # the agent can rely on ModelInfo's advertised types.
            if not isinstance(info, ModelInfo):
                return ModelInfo(name=model)
            if isinstance(info.capabilities, str):
                info.capabilities = [part.strip() for part in info.capabilities.split(",") if part.strip()]
            elif info.capabilities is not None:
                info.capabilities = [str(part) for part in info.capabilities]
            info.context_length = as_int(info.context_length)
            info.parameter_size = as_text(info.parameter_size)
            info.quantization = as_text(info.quantization)
            info.family = as_text(info.family)
            return info
        from .coerce import as_int
        show._wynxo_show_hardened = True
        OllamaClient.show = show


install()
