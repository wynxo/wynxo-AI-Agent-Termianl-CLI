from pathlib import Path

import pytest

from wynxo.agent import is_system_action, system_application
from wynxo.memory import Memory
from wynxo.tools.apps import OpenApplication
from wynxo.tools.memory_tool import Remember


def test_casual_text_cannot_create_user_memory(tmp_path: Path):
    memory = Memory(tmp_path, user_dir=tmp_path / "user")
    memory._agent_write = True
    added, message = memory.remember("The user's name is heio", "user")
    assert not added
    assert "explicit" in message
    assert memory.user.entries() == []


def test_explicit_memory_request_is_allowed(tmp_path: Path):
    memory = Memory(tmp_path, user_dir=tmp_path / "user")
    added, _ = memory.remember("The user's name is Heio", "user", explicit=True)
    assert added
    assert len(memory.user.entries()) == 1


def test_system_action_routing_distinguishes_file_fix():
    assert is_system_action("run calc on my pc")
    assert system_application("launch calc") == "calculator"
    assert system_application("open calculator") == "calculator"
    assert not is_system_action("fix calc.py")


def test_application_tool_schema_is_allowlisted():
    tool = OpenApplication(Path.cwd())
    result = __import__('asyncio').run(tool.invoke({"application": "unknown"}))
    assert not result.ok


@pytest.mark.asyncio
async def test_memory_tool_cannot_persist_user_fact(tmp_path: Path):
    result = await Remember(tmp_path, memory=Memory(tmp_path, user_dir=tmp_path / "user")).invoke(
        {"note": "The user's name is heio", "scope": "user"}
    )
    assert result.ok
    assert "explicit" in result.output
