from pathlib import Path

import pytest

from wynxo.memory import Memory
from wynxo.tools.appcatalog import ApplicationCatalog, Sources
from wynxo.tools.apps import LaunchApplication
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


def test_the_launch_tool_takes_a_query_not_an_allowlisted_name(tmp_path: Path):
    """The schema is a free-form query resolved against the machine's real
    applications -- there is no allowlist to fail a name against."""
    catalog = ApplicationCatalog(sources=Sources(
        shortcut_dirs=(tmp_path / "none",)))
    tool = LaunchApplication(tmp_path, catalog=catalog)
    assert "query" in tool.Input.json_schema()["properties"]
    assert "application" not in tool.Input.json_schema()["properties"]


def test_an_unknown_application_is_a_clean_miss(tmp_path: Path):
    catalog = ApplicationCatalog(sources=Sources(
        shortcut_dirs=(tmp_path / "none",)))
    tool = LaunchApplication(tmp_path, catalog=catalog)
    result = __import__('asyncio').run(tool.invoke({"query": "Zorg Editor 9000"}))
    assert not result.ok
    assert "Could not find" in result.error
    assert result.metadata.get("not_found") is True


def test_a_path_query_is_refused_without_touching_the_catalog(tmp_path: Path):
    """The model must never be able to smuggle an executable path through
    the launch tool; only catalog entries are launchable."""
    catalog = ApplicationCatalog(sources=Sources(
        shortcut_dirs=(tmp_path / "none",)))
    tool = LaunchApplication(tmp_path, catalog=catalog)
    result = __import__('asyncio').run(
        tool.invoke({"query": "C:\\tools\\something.exe"}))
    assert not result.ok
    assert "name" in result.error.lower()
    assert result.metadata.get("status") == "path_query"


@pytest.mark.asyncio
async def test_memory_tool_cannot_persist_user_fact(tmp_path: Path):
    result = await Remember(tmp_path, memory=Memory(tmp_path, user_dir=tmp_path / "user")).invoke(
        {"note": "The user's name is heio", "scope": "user"}
    )
    assert result.ok
    assert "explicit" in result.output
