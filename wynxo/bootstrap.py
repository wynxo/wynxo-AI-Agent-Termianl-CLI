"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations

import sys


def main():
    """Start the CLI.

    The chat layout -- composer pinned to the bottom, conversation flowing
    above it -- is the product's default wherever the terminal can host it,
    Windows included. ``--classic`` opts out to the scrolling prompt; that
    decision belongs to cli.apply_flags with the rest of the run flags, so
    bootstrap has no business overriding it for one platform.
    """
    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
