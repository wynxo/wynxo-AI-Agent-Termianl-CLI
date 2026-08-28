"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations

import sys


def main():
    """Start the CLI with a platform-safe terminal mode.

    Windows terminals vary in their full-screen prompt-toolkit support. The
    classic scrolling renderer is the safe default there; ``--chat`` opts into
    the full-screen interface explicitly. Other platforms keep their normal
    chat-layout behavior.
    """
    if sys.platform == "win32" and "--chat" not in sys.argv[1:]:
        from . import tui

        original = tui.usable
        tui.usable = lambda: False
        try:
            from .cli import main as cli_main
            return cli_main()
        finally:
            tui.usable = original

    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
