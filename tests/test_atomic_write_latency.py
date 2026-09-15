"""Persistence must stay atomic without synchronously flushing hardware.

Session autosave uses the same helper as configuration writes and runs after
almost every conversational event. On Windows, ``os.fsync`` maps to a real
FlushFileBuffers call and can stall for seconds, turning a long agent turn
into hundreds of serial disk waits. Atomic rename protects against partial
JSON without requiring that hardware-level flush.
"""

from wynxo import config


def test_atomic_write_does_not_fsync_each_save(tmp_path, monkeypatch):
    target = tmp_path / "session.json"

    def blocked(*_args, **_kwargs):
        raise AssertionError("atomic_write must not synchronously fsync")

    monkeypatch.setattr(config.os, "fsync", blocked)
    config.atomic_write(target, '{"state":"ready"}')

    assert target.read_text(encoding="utf-8") == '{"state":"ready"}'
    assert not (tmp_path / ".session.json.new").exists()


def test_atomic_write_replaces_complete_contents(tmp_path):
    target = tmp_path / "session.json"
    config.atomic_write(target, '{"turn":1}')
    config.atomic_write(target, '{"turn":2,"complete":true}')

    assert target.read_text(encoding="utf-8") == '{"turn":2,"complete":true}'
