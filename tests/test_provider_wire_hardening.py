from __future__ import annotations

from wynxo.provider import OllamaClient


def test_string_done_false_is_not_truthy() -> None:
    chunk = OllamaClient._to_chunk({"message": {}, "done": "false"})
    assert chunk.done is False


def test_string_done_true_is_true() -> None:
    chunk = OllamaClient._to_chunk({"message": {}, "done": "true"})
    assert chunk.done is True
