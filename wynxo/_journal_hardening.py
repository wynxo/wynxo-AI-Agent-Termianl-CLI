from __future__ import annotations


def install() -> None:
    from . import journal

    original = journal.prune
    if getattr(original, "_wynxo_keep_zero", False):
        return

    def prune(directory, keep=journal.KEEP_SESSIONS):
        try:
            logs = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        if keep <= 0:
            old_logs = logs
        else:
            old_logs = logs[:-keep]
        for old in old_logs:
            try:
                old.unlink()
            except OSError:
                pass

    prune._wynxo_keep_zero = True
    journal.prune = prune


install()
