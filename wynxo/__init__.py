"""wynxo -- a terminal coding agent for local models served by Ollama."""

__version__ = "0.1.0"

from . import hardening as _hardening  # noqa: F401,E402
from . import _streaming_hardening as _streaming_hardening  # noqa: F401,E402
from . import _agent_hardening as _agent_hardening  # noqa: F401,E402
from . import _ui_hardening as _ui_hardening  # noqa: F401,E402
from . import _provider_hardening as _provider_hardening  # noqa: F401,E402
from . import _shell_hardening as _shell_hardening  # noqa: F401,E402

__all__ = ["__version__"]
