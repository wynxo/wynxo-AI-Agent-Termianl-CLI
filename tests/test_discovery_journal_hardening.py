from __future__ import annotations

import os
from pathlib import Path

import pytest

from wynxo import discovery, journal


@pytest.mark.asyncio
async def test_malformed_discovery_json_is_ignored(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return Response()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client())

    assert await discovery.verify("http://127.0.0.1:11434") is None


def test_journal_log_is_private(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(journal, "data_dir", lambda: tmp_path)
    item = journal.Journal.open("privacy-test")
    assert item.path is not None
    if os.name != "nt":
        assert item.path.stat().st_mode & 0o777 == 0o600
