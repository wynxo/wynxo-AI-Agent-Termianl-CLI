"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations


def main():
    """Start the CLI.

    wynxo always runs the scrolling prompt: output goes to the terminal's
    own scrollback and the mouse is never captured, so scrolling, selecting
    and copying behave exactly like any other command-line program. The
    layout decision lives entirely in cli; bootstrap only delegates.
    """
    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
