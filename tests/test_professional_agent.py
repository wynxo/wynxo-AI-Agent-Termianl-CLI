from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.agent import Agent
from wynxo.config import Config
from wynxo.effort import resolve
from wynxo.parsing import ParsedTurn
from wynxo.tui import ChatUI
from wynxo.tools import build_registry


def test_config_exposes_agent_safety_limits():
    config = Config()
    assert config.max_tool_iterations > 0
    assert config.max_tool_result_chars >= 1000
    assert config.max_command_output_chars >= 1000


def test_agent_caps_tool_context_results(tmp_path: Path):
    config = Config(max_tool_result_chars=1000)
    agent = Agent.__new__(Agent)
    agent.policy = resolve("max")
    agent.config = config
    output = "x" * 5000
    trimmed = agent._trim_output(output)
    assert len(trimmed) <= 1100
    assert "truncated" in trimmed


def test_chat_tool_events_are_transient(tmp_path: Path):
    chat = ChatUI(status=lambda: "")
    # The production callback uses bar state; this verifies the layout itself
    # remains transcript-free while activity changes.
    chat.transcript.console.print("assistant")
    chat.flush()
    before = list(chat.transcript.lines)
    chat.set_todos("[>] inspect\n[ ] fix")
    assert chat.transcript.lines == before
