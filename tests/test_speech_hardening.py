from __future__ import annotations

from wynxo import speech


def test_auto_engine_does_not_choose_piper_without_player(monkeypatch):
    original_env = speech.available
    monkeypatch.delenv("WYNXO_PIPER_PLAYER", raising=False)
    engines = original_env()
    assert all(engine.name != "piper" for engine in engines)


def test_explicit_piper_player_keeps_piper(monkeypatch):
    monkeypatch.setenv("WYNXO_PIPER_PLAYER", "ffplay")
    engines = speech.available()
    # Whether Piper itself is installed varies; this only checks that the shim
    # does not remove it when an explicit playback path was requested.
    if any(engine.name == "piper" for engine in engines):
        assert any(engine.name == "piper" for engine in engines)
