"""wynxo -- a terminal coding agent for local models served by Ollama."""

__version__ = "0.1.0"

# Install cross-cutting safety/reliability hardening before callers import the
# tool registry. The module is idempotent, so repeated imports are harmless.
from . import hardening as _hardening  # noqa: F401,E402
# Streaming-specific hardening is kept separate because it patches the UI
# renderer, not the filesystem/shell tools.
from . import _streaming_hardening as _streaming_hardening  # noqa: F401,E402

__all__ = ["__version__"]
