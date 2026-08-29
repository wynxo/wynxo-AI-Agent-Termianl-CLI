"""Wynxo must talk like a person, not a support bot, and the prompt -- not a
canned-response map -- is what makes it do so.

These tests pin the *behaviour*, never an exact sentence: the system prompt
must actively forbid the support-bot closers, give the model felt examples
instead of word-for-word scripts, and route pure chat straight to an answer
with no tools and no plan. If wynxo starts sounding canned again, it is a
prompt regression and these fail.
"""

from __future__ import annotations

from pathlib import Path

from wynxo.agent import is_small_talk
from wynxo.effort import resolve
from wynxo.prompts import build_system_prompt


def _prompt(voice: str = "mommy", workspace: Path | None = None) -> str:
    return build_system_prompt(
        workspace or Path("/tmp/proj"), resolve("medium"),
        tools_description="read_file, edit_file, shell",
        voice=voice,
    )


# -- the support-bot closers are forbidden, not offered ------------------------


def test_every_voice_promises_no_canned_closer():
    canned = [
        "How can I help you today?",
        "What's on your mind?",
        "Let me know if you need anything else.",
        "How can I assist you?",
    ]
    for voice in ("plain", "warm", "mentor", "blunt", "kawaii", "mommy"):
        prompt = _prompt(voice)
        for closer in canned:
            idx = prompt.find(closer)
            if idx != -1:
                # The phrase may only appear as something NOT to say.
                window = prompt[max(0, idx - 400): idx + len(closer) + 60]
                assert ("NEVER" in window or "never" in window or "not" in window), \
                    f"voice '{voice}' presents '{closer}' as something to say:\n{window}"


def test_chat_style_is_injected_for_every_voice():
    prompt = _prompt("mommy")
    assert "## How to talk to a person" in prompt
    assert "responding to what they actually said" in prompt or \
        "Respond to what they actually said" in prompt
    plain = _prompt("plain")
    assert "## How to talk to a person" in plain


def test_the_rules_are_numbered_and_forbid_copying():
    prompt = _prompt()
    assert "never reuse an answer you already gave" in prompt
    assert "Never copy canned sentences" in prompt
    # No greeting example the weakest model could lift and repeat verbatim.
    assert "user: yo" not in prompt
    assert "yo yo" not in prompt


def test_chat_prompt_is_minimal_and_human():
    """The chat system prompt must be a stripped-down identity, not the
    engineering mega-prompt -- that is what makes a raw model robotic."""
    from wynxo.prompts import build_chat_prompt

    chat = build_chat_prompt("mommy")
    assert "You are wynxo" in chat
    assert "customer-service autopilot" in chat
    assert "How can I help you today?" in chat      # as a ban
    # No engineering context: no tools, no effort, no environment, no scope.
    for noise in ("## Environment", "## Effort", "## Voice", "## How you work",
                  "read_file", "edit_file", "tool", "workspace",
                  "Memory", "project map"):
        assert noise.lower() not in chat.lower(), f"chat prompt leaks {noise!r}"
    assert len(chat) < 1200, "chat prompt must stay compact"


def test_chat_prompt_carries_a_one_line_voice_tag():
    from wynxo.prompts import VOICE_TAG, build_chat_prompt
    assert set(VOICE_TAG) == {"plain", "warm", "mentor", "blunt",
                              "kawaii", "mommy"}
    for voice, tag in VOICE_TAG.items():
        assert tag.strip() and len(tag) < 160
    assert "goodboy" in build_chat_prompt("mommy")
    assert "doting" not in build_chat_prompt("plain").lower()


def test_serious_chat_prompt_drops_the_persona():
    """The safety gate's prompt: no personality, no jokes, no tools, no tasks
    -- a person, not a persona."""
    from wynxo.prompts import build_chat_prompt

    serious = build_chat_prompt("mommy", serious=True)
    assert "person, not a persona" in serious
    assert "no jokes, no tasks, no tools, no plans" in serious.lower()
    assert "calm, caring, and plain" in serious.lower()
    assert "goodboy" not in serious
    plain_serious = build_chat_prompt("kawaii", serious=True)
    assert "kaomoji" not in plain_serious
    # The normal variant keeps the persona but must yield when things get real.
    normal = build_chat_prompt("mommy")
    assert "drop the persona" in normal


def test_the_chat_path_is_hot_and_uses_the_minimal_prompt(tmp_path, monkeypatch):
    """A casual turn must be sampled hot (so the canned phrase isn't the
    auto-picked token) and must run against the minimal chat prompt, not the
    full coding system prompt."""
    import asyncio
    from types import SimpleNamespace

    from wynxo.agent import Agent, Callbacks
    from wynxo.config import Config
    from wynxo.effort import resolve
    from wynxo.prompts import build_chat_prompt

    fake = SimpleNamespace()

    async def aclose():
        return None
    fake.aclose = aclose
    agent = Agent(fake, Config(verify_with_tests=False),
                  resolve("low"), tmp_path, Callbacks())
    captured: dict = {}

    async def fake_call(self, *, messages=None, temperature=None, use_tools=True,
                        **kw):
        captured["messages"] = messages
        captured["temperature"] = temperature
        captured["use_tools"] = use_tools
        return SimpleNamespace(content="nice", tool_calls=[])

    monkeypatch.setattr(Agent, "_call_model", fake_call)
    asyncio.run(agent.run("yo"))
    assert captured["temperature"] == 0.95
    system = captured["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("You are wynxo")
    assert system["content"] != agent.session.system_prompt
    assert build_chat_prompt("mommy") == system["content"]
    # Conversation never advertises tools -- that is what made a coding model
    # burn six tool calls on "remember what we were building?".
    assert captured["use_tools"] is False
    # History is preserved: the user message follows the system prompt.
    assert captured["messages"][1] == {"role": "user", "content": "yo"}
    asyncio.run(fake.aclose())


# -- the engineering layer is untouched by personality -------------------------


def test_personality_never_softens_a_failure():
    prompt = _prompt()
    assert "never soften a failure" in prompt
    assert "never imply something worked" in prompt


def test_the_base_engineering_rules_survive():
    prompt = _prompt()
    assert "Investigate before you act" in prompt
    assert "do not widen the scope" in prompt or "widen the scope" in prompt


# -- casual chat is cheap and tool-free ----------------------------------------


def test_casual_messages_route_to_chat_not_tools():
    for msg in ("yo", "hey", "lol", "bro", "how are you", "what are you doing",
                "that was crazy", "why ur so dry", "bro I finally fixed it",
                "nah", "bruh", "fr", "damn", "what", "sup", "morning"):
        assert is_small_talk(msg) is True, f"{msg!r} must be chat"


def test_real_tasks_are_not_treated_as_chat():
    for msg in ("find the bug and fix it", "open an installed application",
                "add a /mommy command", "run the tests", "write text.py",
                "refactor the shell tool", "tell me the bug in foo.py"):
        assert is_small_talk(msg) is False, f"{msg!r} must be a task"


def test_more_conversation_classes_route_to_chat():
    """Feeling-state, requests to talk, reactions and shared-history questions
    are conversation -- the plan prompt would otherwise invent a task for
    them, which is exactly the interference the raw model does not have."""
    for msg in ("i'm bored", "im tired", "i feel stuck",
                "tell me something cool", "tell me a joke",
                "tell me about yourself", "that's actually sick",
                "that was crazy", "that makes sense", "makes sense",
                "i see", "do you remember that bug",
                "remember what we were building",
                "what should we build next"):
        assert is_small_talk(msg) is True, f"{msg!r} must be chat"


def test_conversation_classes_with_a_task_attached_still_route_to_work():
    for msg in ("i'm bored, make me a script",
                "tell me something and then fix the bug",
                "remember that bug and fix it"):
        assert is_small_talk(msg) is False, f"{msg!r} must be a task"


# -- the safety boundary ------------------------------------------------------


def test_self_directed_distress_is_detected():
    from wynxo.agent import is_distress
    for msg in ("i want to kill myself", "i want to die",
                "i'm thinking about suicide", "i can't take this anymore",
                "i don't want to be here", "i keep hurting myself",
                "i'm going to end my life"):
        assert is_distress(msg) is True, f"{msg!r} must be distress"


def test_idioms_and_tasks_are_not_distress():
    from wynxo.agent import is_distress
    for msg in ("this code is killing me", "kill the process",
                "that bug is annoying", "can't take this deadline",
                "fix the link in the readme",
                "i want to die laughing", "i can't handle the keyboard"):
        assert is_distress(msg) is False, f"{msg!r} must not be distress"


def test_a_distress_turn_is_serious_and_tool_free(tmp_path, monkeypatch):
    """The safety boundary end to end: a distress turn gets the serious
    prompt, tools hard-disabled (so no plan/edit/launch/remember is possible),
    and no tool calls ever run -- even if the model emits one anyway."""
    import asyncio
    from types import SimpleNamespace

    from wynxo.agent import Agent, Callbacks
    from wynxo.config import Config
    from wynxo.effort import resolve
    from wynxo.prompts import build_chat_prompt

    fake = SimpleNamespace()

    async def aclose():
        return None
    fake.aclose = aclose
    agent = Agent(fake, Config(verify_with_tests=False),
                  resolve("low"), tmp_path, Callbacks())
    captured: dict = {}

    async def fake_call(self, *, messages=None, use_tools=True, **kw):
        captured["messages"] = messages
        captured["use_tools"] = use_tools
        # The model still (incorrectly) tries a tool -- the runtime must drop it.
        return SimpleNamespace(content="I'm here. That sounds really heavy -- want to tell me more?",
                               tool_calls=[{"name": "remember"}])

    monkeypatch.setattr(Agent, "_call_model", fake_call)
    result = asyncio.run(agent.run("i want to kill myself"))
    assert captured["use_tools"] is False
    system = captured["messages"][0]
    assert system["content"] == build_chat_prompt("mommy", serious=True)
    # The tool call was dropped: the turn answered, nothing ran.
    assert result.tool_calls == 0
    assert "I'm here" in result.content
    assert all(m.get("role") != "tool" for m in agent.session.wire())
    # No plan scaffold, no "now carry out that plan".
    assert all("carry out" not in m.get("content", "") for m in agent.session.wire())
    asyncio.run(fake.aclose())


# -- voices are distinct but all inherit the mechanics ------------------------


def test_mommy_carries_personality_and_mechanics():
    prompt = _prompt("mommy")
    assert "goodboy" in prompt
    assert "## How to talk to a person" in prompt


def test_plain_is_human_not_blank():
    from wynxo.prompts import VOICES
    plain = VOICES["plain"]
    assert plain.strip(), "plain must carry a personality block now"
    assert len(plain) < 700, "keep the persona compact for small models"