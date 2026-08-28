from pathlib import Path

from wynxo.discovery import Discovery
from wynxo.events import ExecutionState, ToolEvent
from wynxo.ui import ActivityBar, UI


def test_discovery_detects_python_project_and_caches(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (tmp_path / "pytest.ini").write_text("")
    discovery = Discovery(tmp_path)
    first = discovery.scan()
    second = discovery.scan()
    assert first is second
    assert "Python" in first.languages
    assert "pyproject.toml" in first.markers
    assert "pytest" in first.test_frameworks


def test_discovery_invalidates_when_marker_changes(tmp_path: Path):
    discovery = Discovery(tmp_path)
    first = discovery.scan()
    (tmp_path / "package.json").write_text("{}")
    second = discovery.scan()
    assert second is not first
    assert "JavaScript/Node" in second.languages


def test_activity_bar_retains_only_recent_tool_events():
    bar = ActivityBar(UI(), "medium")
    for i in range(10):
        event = ToolEvent("read_file", f"file{i}.py")
        event.start()
        event.finish(True, display=f"read file{i}.py")
        bar.record_tool_event(event)
    assert len(bar.tool_events) == bar.max_tool_events
    assert bar.tool_events[0].summary == "file4.py"
    assert all(event.state is ExecutionState.SUCCESS for event in bar.tool_events)
