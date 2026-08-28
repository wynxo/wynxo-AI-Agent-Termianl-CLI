"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations

import sys


def main():
    """Start the CLI with a safe terminal-mode choice.

    Windows consoles differ more than POSIX terminals, especially around
    full-screen prompt-toolkit output. The scrolling renderer is the robust
    default there; ``--chat`` is an explicit opt-in for the full-screen UI.
    This keeps the installed command usable even when a terminal advertises
    ANSI support incompletely.
    """
    if sys.platform == "win32" and "--chat" not in sys.argv[1:]:
        from . import tui

        original_usable = tui.usable
        tui.usable = lambda: False
        try:
            from .cli import main as cli_main
            return cli_main()
        finally:
            tui.usable = original_usable

    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
