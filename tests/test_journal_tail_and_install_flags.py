from __future__ import annotations

import json
from pathlib import Path

from wynxo.journal import Journal


def test_journal_tail_zero_is_empty(tmp_path: Path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"kind": "one"}) + "\n" + json.dumps({"kind": "two"}) + "\n", encoding="utf-8")
    journal = Journal("x", path=path)

    assert journal.tail(0) == []
