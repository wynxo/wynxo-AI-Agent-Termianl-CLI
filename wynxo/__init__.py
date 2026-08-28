"""wynxo -- a terminal coding agent for local models served by Ollama."""

__version__ = "0.1.0"

# Runtime hardening hooks are loaded with the package so tests and the
# installed CLI exercise the same safety behavior. Each hook is idempotent.
from . import _agent_hardening as _agent_hardening  # noqa: F401,E402
from . import _provider_hardening as _provider_hardening  # noqa: F401,E402
from . import _shell_hardening as _shell_hardening  # noqa: F401,E402
from . import _shell_secret_hardening as _shell_secret_hardening  # noqa: F401,E402
from . import _ui_hardening as _ui_hardening  # noqa: F401,E402
from . import _tool_hardening as _tool_hardening  # noqa: F401,E402
from . import _search_hardening as _search_hardening  # noqa: F401,E402
from . import _session_hardening as _session_hardening  # noqa: F401,E402
from . import _memory_hardening as _memory_hardening  # noqa: F401,E402
from . import _testing_hardening as _testing_hardening  # noqa: F401,E402
from . import _symlink_hardening as _symlink_hardening  # noqa: F401,E402
from . import _projectmap_hardening as _projectmap_hardening  # noqa: F401,E402
from . import _speech_hardening as _speech_hardening  # noqa: F401,E402

__all__ = ["__version__"]
