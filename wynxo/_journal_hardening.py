from __future__ import annotations


def install() -> None:
    from . import journal

    original_prune = journal.prune
    if not getattr(original_prune, "_wynxo_keep_zero", False):
        def prune(directory, keep=journal.KEEP_SESSIONS):
            try:
                logs = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
            except OSError:
                return
            old_logs = logs if keep <= 0 else logs[:-keep]
            for old in old_logs:
                try:
                    old.unlink()
                except OSError:
                    pass
        prune._wynxo_keep_zero = True
        journal.prune = prune

    original_tail = journal.Journal.tail
    if getattr(original_tail, "_wynxo_zero_safe", False):
        return

    def tail(self, count=40):
        if count <= 0:
            return []
        return original_tail(self, count)

    tail._wynxo_zero_safe = True
    journal.Journal.tail = tail


install()
