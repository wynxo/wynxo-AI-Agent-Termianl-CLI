from __future__ import annotations

import json


def install() -> None:
    from .config import atomic_write
    from .session import Session, Usage

    original_save = Session.save
    if not getattr(original_save, "_wynxo_generation_stats", False):
        def save(self):
            path = original_save(self)
            if path is None:
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                usage = data.setdefault("usage", {})
                usage["generation_seconds"] = self.usage.generation_seconds
                atomic_write(path, json.dumps(data, indent=2, default=str) + "\n")
            except (OSError, ValueError, TypeError):
                # The original save already succeeded. Do not make a stats
                # field failure turn a valid session into an unusable one.
                pass
            return path
        save._wynxo_generation_stats = True
        Session.save = save

    original_load = Session.load
    if getattr(original_load, "_wynxo_generation_stats", False):
        return

    @classmethod
    def load(cls, session_id, workspace):
        session = original_load(session_id, workspace)
        if session is None:
            return None
        path = session.path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(usage, dict):
                raw = usage.get("generation_seconds", 0.0)
                if isinstance(raw, (int, float)) and raw >= 0:
                    session.usage.generation_seconds = float(raw)
        except (OSError, ValueError, TypeError):
            pass
        return session
    load._wynxo_generation_stats = True
    Session.load = load


install()
