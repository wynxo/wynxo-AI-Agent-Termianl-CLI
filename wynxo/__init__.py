"""wynxo -- a terminal coding agent for local models served by Ollama."""

__version__ = "0.1.0"

# Install cross-cutting safety/reliability hardening before callers import the
# tool registry. The module is idempotent, so repeated imports are harmless.
from . import hardening as _hardening  # noqa: F401,E402
# Streaming-specific hardening patches the UI renderer without changing the
# stable public UI API.
from . import _streaming_hardening as _streaming_hardening  # noqa: F401,E402
# Agent observability hardening fixes per-turn state and streams automatic
# verification output through the same callbacks as user shell commands.
from . import _agent_hardening as _agent_hardening  # noqa: F401,E402
# UI streaming polish keeps live code progress visible in the pinned bar.
from . import _ui_hardening as _ui_hardening  # noqa: F401,E402

__all__ = ["__version__"]
