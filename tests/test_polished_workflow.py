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


def test_the_activity_bar_keeps_no_history_of_finished_calls():
    """The pinned region says what is true now; the record says what
    happened.

    It used to redraw the last six finished calls as "✓ read calc.py
    (5 lines)  0.00s" -- a third rendering of calls the transcript had
    already committed as blocks a couple of lines above, and those blocks
    are the ones that stay. Nothing in the live region is a record, so
    nothing that has finished belongs in it.
    """
    bar = ActivityBar(UI(), "medium")
    assert not hasattr(bar, "record_tool_event")
    assert not hasattr(bar, "tool_events")
    event = ToolEvent("read_file", "file0.py")
    event.start()
    event.finish(True, display="read file0.py")
    assert event.state is ExecutionState.SUCCESS
    # What it does show is the call in flight, once -- in the scene above
    # the strip, where the activity moved when the companion arrived.
    bar.update(activity="reading", detail="file0.py")
    assert "reading" in "".join(t.plain for t in bar._scene())
