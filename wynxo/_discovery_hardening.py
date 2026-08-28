from __future__ import annotations


def install() -> None:
    from . import discovery

    original = discovery.verify
    if getattr(original, "_wynxo_json_safe", False):
        return

    async def verify(url, timeout=discovery.VERIFY_TIMEOUT):
        try:
            return await original(url, timeout=timeout)
        except (AttributeError, TypeError, ValueError):
            # Discovery is a probe. A live HTTP service with malformed JSON is
            # not a reason to crash the setup wizard; treat it as "not Ollama".
            return None

    verify._wynxo_json_safe = True
    discovery.verify = verify


install()
