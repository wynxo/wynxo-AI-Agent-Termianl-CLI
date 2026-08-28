from __future__ import annotations

from pathlib import Path

import pytest

from wynxo import discovery, journal


@pytest.mark.asyncio
async def test_malformed_discovery_json_is_ignored(monkeypatch):
    async def broken(*args, **kwargs):
        raise AttributeError("list has no attribute get")
    monkeypatch.setattr(discovery, "verify", broken)
    assert await discovery.verify("http://127.0.0.1:11434") is None


def test_journal_log_is_private(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(journal, "data_dir", lambda: tmp_path)
    item = journal.Journal.open("privacy-test")
    assert item.path is not None
    assert item.path.stat().st_mode & 0o777 == 0o600
