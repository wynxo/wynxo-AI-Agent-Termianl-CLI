"""wynxo -- a terminal coding agent for local models served by Ollama."""

__version__ = "0.1.0"

# Compatibility and safety hooks are intentionally loaded with the package so
# they affect the real CLI, not only direct unit-test imports. Each hook is
# idempotent and guards its target API, so importing wynxo remains safe across
# refactors.
from . import _agent_hardening as _agent_hardening  # noqa: F401,E402

__all__ = ["__version__"]
