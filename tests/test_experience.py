"""Regression coverage for the chat-first product experience."""

from __future__ import annotations

from prompt_toolkit.document import Document

from wynxo import cli, experience


def _complete(text: str) -> list[str]:
    completer_type = experience.command_completer_class(cli)
    completer = completer_type()
    return [item.text for item in completer.get_completions(Document(text), None)]


def test_affection_is_conversation_without_swallowing_work():
    assert experience.extra_conversation("i lov yuuu")
    assert experience.extra_conversation("ilyyy")
    assert experience.extra_conversation("I love you!!!")
    assert not experience.extra_conversation("I love Python, fix this parser")
    assert not experience.extra_conversation("fix the thing i love")


def test_chat_prompt_stays_with_the_human_moment():
    prompt = " ".join(experience.CHAT_PROMPT.lower().split())
    assert "local ai companion" in prompt
    assert "do not pivot to features, coding" in prompt
    assert "coding is one capability" in prompt
    assert "what next" in prompt


def test_bare_slash_prioritises_product_modes():
    suggestions = _complete("/")
    assert suggestions[:4] == ["/chat", "/code", "/github", "/help"]


def test_primary_modes_stay_first_for_shared_prefixes():
    suggestions = _complete("/c")
    assert suggestions[:2] == ["/chat", "/code"]


def test_typo_is_recovered_inside_completion_menu():
    assert "/help" in _complete("/hlep")


def test_enumerated_values_appear_immediately_after_space():
    suggestions = _complete("/effort ")
    assert suggestions == ["low", "medium", "high", "xhigh", "max", "ultra"]


def test_subcommand_completion_works_through_aliases():
    suggestions = _complete("/e h")
    assert suggestions == ["high"]


def test_trailing_space_never_replaces_the_wrong_character():
    assert _complete("/effort h ") == []
