from __future__ import annotations


def install() -> None:
    from .memory import MemoryFile

    original = MemoryFile.forget
    if getattr(original, "_wynxo_empty_safe", False):
        return

    def forget(self, pattern: str):
        if not pattern or not pattern.strip():
            return 0, "A non-empty pattern is required; nothing was forgotten."
        return original(self, pattern)

    forget._wynxo_empty_safe = True
    MemoryFile.forget = forget


install()
