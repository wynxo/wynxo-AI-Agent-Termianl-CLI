"""Regression coverage for session integrity and responsive terminal chrome."""
import io
import json

import pytest
from rich.cells import cell_len
from rich.console import Console

from wynxo import shell
from wynxo.session import Session, Usage
from wynxo.ui import UI, Glyphs


@pytest.mark.parametrize("unicode", [True, False])
def test_model_names_respect_cell_budgets(unicode):
    ui = UI()
    ui.g = Glyphs(unicode)
    for name in ("namespace/模型模型模型:大", "namespace/very-long-model:27b"):
        for room in range(0, 30):
            assert cell_len(ui.shorten_model(name, room)) <= room


def test_path_shortening_only_replaces_the_home_directory(monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda _: "/home/ann")
    ui = UI()
    ui.width = 180
    assert ui.shorten_path("/home/anna/project") == "/home/anna/project"
    assert ui.shorten_path("/home/ann/project") == "~/project"
    ui.width = 60
    assert cell_len(ui.shorten_path("/tmp/" + "项目" * 30)) <= 20


@pytest.mark.parametrize("width", [60, 80, 120])
@pytest.mark.parametrize("unicode", [True, False])
def test_short_terminal_welcome_leaves_room_for_composer(monkeypatch, width, unicode):
    monkeypatch.setattr(shell, "terminal_height", lambda: 24)
    ui = UI()
    ui.width = width
    ui.g = Glyphs(unicode)
    stream = io.StringIO()
    console = Console(file=stream, width=width, color_system=None)
    console.width = width
    console.print(shell.home(ui, model="qwen3:8b", version="v0.1.0",
                             workspace="/tmp/project"))
    lines = stream.getvalue().splitlines()
    assert len(lines) <= 13
    assert all(cell_len(line) <= width for line in lines)
    assert "/help" in stream.getvalue()
    assert "project" in stream.getvalue()
    if not unicode:
        assert stream.getvalue().isascii()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("wynxo.session.data_dir", lambda: tmp_path)
    (tmp_path / "sessions").mkdir()
    return tmp_path


def test_generation_rate_survives_resume(store):
    session = Session(workspace=store)
    session.usage = Usage(completion_tokens=100, generation_seconds=2.5)
    assert session.save()
    restored = Session.load(session.session_id, store)
    assert restored.usage.tokens_per_second() == 40


@pytest.mark.parametrize("session_id", ["../outside", "/tmp/session", "a/b", "a\\b", "..", ""])
def test_session_ids_cannot_escape_the_store(store, session_id):
    assert Session.load(session_id, store) is None
    assert Session(workspace=store, session_id=session_id).save() is None


def test_embedded_session_id_cannot_redirect_the_next_save(store):
    path = store / "sessions" / "safe.json"
    path.write_text(json.dumps({"session_id": "../outside", "messages": []}))
    loaded = Session.load("safe", store)
    assert loaded.session_id == "safe"
    assert loaded.save() == path
    assert Session.recent()[0]["session_id"] == "safe"
    assert not (store / "outside.json").exists()


def test_corrupt_utf8_session_does_not_break_resume(store):
    (store / "sessions" / "broken.json").write_bytes(b"\xff")
    assert Session.load("broken", store) is None
    assert Session.recent() == []


def test_closed_loop_shutdown_reaps_its_child():
    import asyncio
    import subprocess
    import sys
    from types import SimpleNamespace
    from wynxo.tools.shell import _reap_after_loop_close

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    loop = asyncio.new_event_loop()
    process = SimpleNamespace(_loop=loop, _transport=SimpleNamespace(_proc=child))
    try:
        assert not _reap_after_loop_close(process)
        loop.close()
        assert not _reap_after_loop_close(process)
        child.terminate()
        assert _reap_after_loop_close(process, timeout=5.0)
        assert child.returncode is not None
    finally:
        if not loop.is_closed():
            loop.close()
        if child.poll() is None:
            child.kill()
        child.wait()
