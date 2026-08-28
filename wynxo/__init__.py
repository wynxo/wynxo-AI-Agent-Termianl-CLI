"""wynxo -- a terminal coding agent for local models served by Ollama."""

__version__ = "0.1.0"

# Compatibility/safety hooks load with the package so the behavior exercised
# by the tests is the behavior used by the installed CLI. Each hook is
# idempotent and guards its target API.
from . import _agent_hardening as _agent_hardening  # noqa: F401,E402
from . import _provider_hardening as _provider_hardening  # noqa: F401,E402
from . import _shell_hardening as _shell_hardening  # noqa: F401,E402
from . import _ui_hardening as _ui_hardening  # noqa: F401,E402
from . import _tool_hardening as _tool_hardening  # noqa: F401,E402

__all__ = ["__version__"]
