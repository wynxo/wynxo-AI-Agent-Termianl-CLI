from __future__ import annotations


def install() -> None:
    from . import journal

    original_open = journal.Journal.open.__func__
    if getattr(original_open, "_wynxo_mode_600", False):
        return

    def open_(cls, session_id: str, enabled: bool = True):
        result = original_open(cls, session_id, enabled)
        if result.path is not None:
            try:
                result.path.chmod(0o600)
            except OSError:
                pass
        return result

    open_._wynxo_mode_600 = True
    journal.Journal.open = classmethod(open_)


install()
