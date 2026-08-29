"""The session journal: append-only records, per-field trimming, pruning
old sessions and credential redaction."""

from __future__ import annotations

import json
import os
import time

from wynxo.journal import Journal, MAX_FIELD, prune
from wynxo.secrets import redact


class TestJournalRecords:
    def _journal(self, tmp_path):
        path = tmp_path / "j.jsonl"
        return Journal(session_id="sess-1", path=path, enabled=True), path

    def test_write_appends_one_json_object_per_line(self, tmp_path):
        journal, path = self._journal(tmp_path)
        journal.write("question", content="hello")
        journal.write("answer", content="hi back")
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["kind"] == "question"
        assert first["content"] == "hello"
        assert "t" in first

    def test_disabled_journal_writes_nothing(self, tmp_path):
        path = tmp_path / "off.jsonl"
        journal = Journal(session_id="s", path=path, enabled=False)
        journal.write("x", content="y")
        assert not path.exists()

    def test_long_field_is_trimmed_not_lost(self, tmp_path):
        journal, path = self._journal(tmp_path)
        big = "z" * (MAX_FIELD + 500)
        journal.write("result", content=big)
        record = json.loads(path.read_text(encoding="utf-8"))
        assert len(record["content"]) <= MAX_FIELD + 60
        assert "more characters" in record["content"]

    def test_credentials_are_scrubbed_on_the_way_in(self, tmp_path):
        journal, path = self._journal(tmp_path)
        secret = "sk-ant-abcdef1234567890"
        journal.write("result", config={"api_key": secret})
        record = json.loads(path.read_text(encoding="utf-8"))
        assert secret not in record["config"]["api_key"]
        assert record["config"]["api_key"] == redact(secret)[0]


class TestPrune:
    def _make_logs(self, directory, n):
        paths = []
        base = time.time() - n
        for i in range(n):
            p = directory / f"{i:02d}-sess.jsonl"
            p.write_text("{}\n", encoding="utf-8")
            os.utime(p, (base + i, base + i))
            paths.append(p)
        return paths

    def test_keeps_only_the_most_recent(self, tmp_path):
        directory = tmp_path / "logs"
        directory.mkdir()
        self._make_logs(directory, 25)
        prune(directory, keep=20)
        remaining = sorted(directory.glob("*.jsonl"))
        assert len(remaining) == 20

    def test_nothing_when_directory_is_absent(self, tmp_path):
        prune(tmp_path / "nope")   # must not raise