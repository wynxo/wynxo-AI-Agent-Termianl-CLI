from __future__ import annotations

from pathlib import Path

from wynxo.memory import MemoryFile, PROJECT_HEADER
from wynxo.session import Session, Usage


def test_memory_forget_requires_nonempty_pattern(tmp_path: Path):
    memory = MemoryFile(tmp_path / "memory.md", PROJECT_HEADER, 8000)
    memory.append("keep pytest enabled")

    count, message = memory.forget("   ")

    assert count == 0
    assert "non-empty" in message
    assert len(memory.entries()) == 1


def test_session_save_and_load_preserve_generation_seconds(tmp_path: Path, monkeypatch):
    from wynxo import session as session_module

    monkeypatch.setattr(session_module, "data_dir", lambda: tmp_path)
    session = Session(tmp_path / "workspace")
    session.usage = Usage(completion_tokens=42, generation_seconds=12.5)

    path = session.save()
    assert path is not None

    loaded = Session.load(session.session_id, session.workspace)
    assert loaded is not None
    assert loaded.usage.generation_seconds == 12.5
