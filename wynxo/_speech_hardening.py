from __future__ import annotations


def install() -> None:
    from . import speech

    original_available = speech.available
    if getattr(original_available, "_wynxo_piper_audio", False):
        return

    def available():
        engines = original_available()
        # Piper's raw PCM is not audible when stdout is redirected to DEVNULL.
        # Only expose it automatically when the user supplied an explicit
        # playback command via WYNXO_PIPER_PLAYER; otherwise fall through to
        # an engine that actually produces sound by itself.
        if any(e.name == "piper" for e in engines):
            import os
            if not os.environ.get("WYNXO_PIPER_PLAYER"):
                engines = [e for e in engines if e.name != "piper"]
        return engines

    available._wynxo_piper_audio = True
    speech.available = available


install()
