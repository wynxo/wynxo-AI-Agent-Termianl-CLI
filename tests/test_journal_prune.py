from __future__ import annotations

from pathlib import Path

from wynxo.journal import prune


def test_prune_keep_zero_removes_all_logs(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"{index}.jsonl").write_text("{}\n", encoding="utf-8")

    prune(tmp_path, keep=0)

    assert list(tmp_path.glob("*.jsonl")) == []
