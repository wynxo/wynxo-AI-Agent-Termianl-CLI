"""Regression coverage for the chat-first product experience."""

from __future__ import annotations

import asyncio

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
    assert "scratch work private" in prompt
    assert 'never print a "thinking" section' in prompt
    assert "what next" in prompt


def test_bare_slash_prioritises_product_modes():
    suggestions = _complete("/")
    assert suggestions[:4] == ["/chat", "/code", "/github", "/help"]


def test_primary_modes_stay_first_for_shared_prefixes():
    suggestions = _complete("/c")
    assert suggestions[:2] == ["/chat", "/code"]


def test_typo_is_recovered_inside_completion_menu():
    assert "/help" in _complete("/hlep")


def test_intent_words_find_the_command_without_becoming_aliases():
    assert _complete("/talk")[0] == "/chat"
    assert _complete("/agent")[0] == "/code"
    assert _complete("/repo")[0] == "/github"
    assert _complete("/llm")[0] == "/model"
    assert _complete("/bye")[0] == "/quit"
    assert cli.resolve_command("/repo") is None


def test_footer_and_completion_share_semantic_ranking():
    popup = _complete("/repo")
    footer = experience._product_command_hints(cli, "/repo")
    assert popup[:len(footer)] == footer
    assert footer[0] == "/github"


def test_enumerated_values_appear_immediately_after_space():
    suggestions = _complete("/effort ")
    assert suggestions == ["low", "medium", "high", "xhigh", "max", "ultra"]


def test_subcommand_completion_works_through_aliases():
    suggestions = _complete("/e h")
    assert suggestions == ["high"]


def test_trailing_space_never_replaces_the_wrong_character():
    assert _complete("/effort h ") == []


def test_chat_call_detection_requires_the_chat_system_prompt():
    chat = [{"role": "system", "content": experience.CHAT_PROMPT}]
    coding = [{"role": "system", "content": "You are a coding agent."}]
    assert experience._is_chat_messages(chat)
    assert not experience._is_chat_messages(coding)
    assert not experience._is_chat_messages(None)


def test_conversation_callbacks_hide_reasoning_but_keep_a_natural_activity():
    class Recorder:
        def __init__(self):
            self.events = []

        async def on_thinking(self, text):
            self.events.append(("thinking", text))

        async def on_stage(self, name, detail=""):
            self.events.append(("stage", name, detail))

    async def run():
        base = Recorder()
        callbacks = experience._ConversationCallbacks(base)
        await callbacks.on_thinking("internal scratch work")
        await callbacks.on_stage("thinking")
        await callbacks.on_stage("retrying", "model sent nothing")
        return base.events

    assert asyncio.run(run()) == [
        ("stage", "responding", ""),
        ("stage", "retrying", "model sent nothing"),
    ]
