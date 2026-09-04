"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations


def main():
    """Start the CLI with the optional visual dashboard installed first."""
    # Install the visual shell before cli imports UI. The dashboard only
    # replaces UI.banner; all existing streaming, tools, prompt handling,
    # scrolling, and tests keep using the original UI implementation.
    from .dashboard import install
    install()

    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
