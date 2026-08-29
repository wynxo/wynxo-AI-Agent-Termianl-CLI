"""The launch tool: resolve against the catalog, launch honestly, and never
substitute. Every test injects a fake catalog so nothing here depends on
what happens to be installed on the machine running the tests -- and nothing
here launches a real application.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


from wynxo.tools.appcatalog import ApplicationCatalog, Sources
from wynxo.tools import apps as apps_module
from wynxo.tools.apps import LaunchApplication


def catalog_with(tmp_path: Path, *names: str) -> ApplicationCatalog:
    programs = tmp_path / "Start Menu" / "Programs"
    programs.mkdir(parents=True, exist_ok=True)
    for name in names:
        (programs / f"{name}.lnk").write_text("", encoding="utf-8")
    return ApplicationCatalog(sources=Sources(
        shortcut_dirs=(programs,), use_app_paths=False))


def tool_with(catalog: ApplicationCatalog) -> LaunchApplication:
    return LaunchApplication(Path.cwd(), catalog=catalog)


def patch_startfile(monkeypatch, launched: list[str], exc: Exception | None = None):
    """Stand in for os.startfile, capturing what would be launched."""

    def fake(path: str, arg: str = "") -> None:
        if exc is not None:
            raise exc
        launched.append((path, arg))

    monkeypatch.setattr(apps_module, "_startfile", fake)


# -- the launch path ----------------------------------------------------------


def test_a_matched_application_is_launched(monkeypatch, tmp_path):
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Visual Studio Code"))
    result = asyncio.run(tool.invoke({"query": "vscode"}))
    assert result.ok
    assert result.metadata["status"] == "launched"
    assert result.metadata["application"] == "Visual Studio Code"
    assert result.metadata["source"] == "start_menu"
    assert len(launched) == 1
    assert launched[0][0].endswith("Visual Studio Code.lnk")


def test_a_successful_launch_is_a_terminal_action(monkeypatch, tmp_path):
    """Launching the application *is* the answer to the user's request; the
    agent must be able to stop the turn on it rather than inventing coding
    work after it."""
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Steam"))
    result = asyncio.run(tool.invoke({"query": "Steam"}))
    assert result.ok
    assert result.terminal is True


def test_a_miss_and_ambiguity_are_not_terminal(tmp_path, monkeypatch):
    """Only a real launch ends the turn. A miss or an ambiguous query leaves
    the model to tell the user and ask -- the turn is not over yet."""
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Alpha Editor", "Beta Editor"))
    ambiguous = asyncio.run(tool.invoke({"query": "editor"}))
    assert not ambiguous.ok
    assert ambiguous.terminal is False
    miss = asyncio.run(tool.invoke({"query": "No Such Thing"}))
    assert not miss.ok
    assert miss.terminal is False


def test_the_launch_message_says_what_was_launched(monkeypatch, tmp_path):
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Steam"))
    result = asyncio.run(tool.invoke({"query": "Steam"}))
    assert "Steam" in result.output
    assert "Steam" in result.display


def test_a_failed_launch_is_reported_as_a_failure(monkeypatch, tmp_path):
    patch_startfile(monkeypatch, [], exc=OSError("the shortcut's target is gone"))
    tool = tool_with(catalog_with(tmp_path, "Broken App"))
    result = asyncio.run(tool.invoke({"query": "Broken App"}))
    assert not result.ok
    assert result.metadata["status"] == "failed"
    assert "Broken App" in result.error
    assert "failed" in result.error or "launching" in result.error


def test_a_miss_triggers_one_refresh_then_honest_failure(monkeypatch, tmp_path):
    """An application installed a moment ago must be found, but a query
    nothing matches must come back as a miss -- never a substitution."""
    programs = tmp_path / "Start Menu" / "Programs"
    programs.mkdir(parents=True)
    catalog = ApplicationCatalog(sources=Sources(
        shortcut_dirs=(programs,), use_app_paths=False))
    assert catalog.resolve("Freshly Installed Thing").status == "not_found"

    refreshed = []
    real_refresh = catalog.refresh

    def counting_refresh():
        refreshed.append(1)
        (programs / "Freshly Installed Thing.lnk").write_text("", encoding="utf-8")
        return real_refresh()

    monkeypatch.setattr(catalog, "refresh", counting_refresh)
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog)
    result = asyncio.run(tool.invoke({"query": "Freshly Installed Thing"}))
    assert len(refreshed) == 1
    assert result.ok and result.metadata["status"] == "launched"
    assert len(launched) == 1


def test_an_ambiguous_query_fails_and_names_the_candidates(tmp_path, monkeypatch):
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Alpha Editor", "Beta Editor"))
    result = asyncio.run(tool.invoke({"query": "editor"}))
    assert not result.ok
    assert result.metadata.get("ambiguous") is True
    assert set(result.metadata["candidates"]) == {"Alpha Editor", "Beta Editor"}
    assert launched == []


# -- the behaviours the spec forbids ------------------------------------------


def test_explorer_is_never_a_fallback_for_something_else(tmp_path, monkeypatch):
    """'open vscode' on a machine without VS Code must fail honestly. It
    must never become File Explorer."""
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "File Explorer"))
    result = asyncio.run(tool.invoke({"query": "Visual Studio Code"}))
    assert not result.ok
    assert "Could not find" in result.error
    assert launched == []


def test_explorer_requested_explicitly_launches_explorer(tmp_path, monkeypatch):
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Visual Studio Code", "File Explorer"))
    result = asyncio.run(tool.invoke({"query": "explorer"}))
    assert result.ok
    assert result.metadata["application"] == "File Explorer"
    assert launched[0][0].endswith("File Explorer.lnk")


def test_an_executable_path_cannot_be_launched(tmp_path, monkeypatch):
    """No path from the model, however it is phrased, reaches a launcher."""
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Steam"))
    for query in (r"C:\Windows\System32\cmd.exe", "..\\..\\evil.exe",
                  "notevil.app", "thing.desktop"):
        result = asyncio.run(tool.invoke({"query": query}))
        assert not result.ok, query
    assert launched == []


def test_an_empty_query_is_refused(tmp_path):
    tool = tool_with(catalog_with(tmp_path, "Steam"))
    result = asyncio.run(tool.invoke({"query": "   "}))
    assert not result.ok


# -- opening a file in the launched application --------------------------------


def test_a_path_is_handed_to_the_launched_application(monkeypatch, tmp_path):
    """'create text.py and open it in vscode': the app is still resolved from
    the catalog, and the path rides along as an argument."""
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Visual Studio Code"))
    result = asyncio.run(tool.invoke({
        "query": "vscode",
        "path": r"C:\Users\elliot\Desktop\text.py"}))
    assert result.ok
    assert result.metadata["application"] == "Visual Studio Code"
    assert result.metadata["opened"] == r"C:\Users\elliot\Desktop\text.py"
    assert "text.py" in result.output
    assert len(launched) == 1
    assert launched[0][0].endswith("Visual Studio Code.lnk")
    assert launched[0][1] == r"C:\Users\elliot\Desktop\text.py"


def test_launch_without_a_path_passes_no_argument(monkeypatch, tmp_path):
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Steam"))
    result = asyncio.run(tool.invoke({"query": "Steam"}))
    assert result.ok
    assert "opened" not in result.metadata
    assert len(launched) == 1
    assert launched[0][1] == ""


def test_a_path_never_resolves_the_application(monkeypatch, tmp_path):
    """The path is an argument, not a launch target: a path alone must not
    launch anything, whatever it points at."""
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Steam"))
    result = asyncio.run(tool.invoke({
        "query": r"C:\Windows\System32\cmd.exe",
        "path": r"C:\Users\elliot\Desktop\text.py"}))
    assert not result.ok
    assert launched == []


def test_repeated_use_stays_consistent(monkeypatch, tmp_path):
    """Dictate-and-launch is a repeated pattern; the second call must behave
    exactly like the first, cache or no cache."""
    launched = []
    patch_startfile(monkeypatch, launched)
    tool = tool_with(catalog_with(tmp_path, "Discord"))
    first = asyncio.run(tool.invoke({"query": "discord"}))
    second = asyncio.run(tool.invoke({"query": "discord"}))
    assert first.ok and second.ok
    assert first.metadata["application"] == second.metadata["application"]
    assert len(launched) == 2


def test_the_tool_description_tells_the_model_the_rules():
    description = LaunchApplication.description.lower()
    assert "never" in description
    assert "substitute" in description
    assert "installed" in description


# -- through the real agent loop ----------------------------------------------


class ScriptedBackend:
    """One model turn that calls the tool, then one that answers."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        if self.calls == 1:
            return self._chunks("", [{"function": {
                "name": "launch_application",
                "arguments": {"query": "Visual Studio Code"}}}])
        return self._chunks("VS Code is opening.", [])

    async def _iter(self, content, calls):
        from wynxo.provider import Chunk
        yield Chunk(content=content, tool_calls=calls, done=True)

    def _chunks(self, content, calls):
        return self._iter(content, calls)


def test_open_request_flows_through_the_agent_to_the_tool(monkeypatch, tmp_path):
    """'open vscode' end to end: the model reads the intent, calls the tool
    with the user's words, the catalog resolves it, and the shortcut is
    launched -- with no repository tools involved."""
    from wynxo.agent import Agent, Callbacks
    from wynxo.config import Config
    from wynxo.effort import resolve
    from wynxo.tools import build_registry

    launched = []
    patch_startfile(monkeypatch, launched)

    class Events(Callbacks):
        def __init__(self):
            self.started = []

        async def on_tool_start(self, name, summary):
            self.started.append(name)

    backend = ScriptedBackend()
    agent = Agent(
        backend, Config(verify_with_tests=False, allow_shell=True,
                        auto_approve=["*"]),
        resolve("low"), tmp_path, Events(),
        registry=build_registry(
            tmp_path, app_catalog=catalog_with(tmp_path, "Visual Studio Code")),
    )
    agent.backend = backend
    result = asyncio.run(agent.run("open vscode"))
    assert "opening" in result.content.lower()
    assert len(launched) == 1
    assert launched[0][0].endswith("Visual Studio Code.lnk")
    assert backend.calls == 2          # tool call + final answer, nothing more
