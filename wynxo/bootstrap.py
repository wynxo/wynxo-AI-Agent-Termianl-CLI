"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations


def main():
    """Start the CLI with the visual shell and runtime compatibility installed first."""
    # Install compatibility shims before cli imports the provider/client.
    # This keeps startup resilient when the CLI and provider signatures are
    # briefly out of sync during development.
    from . import runtime_compat  # noqa: F401

    # Install the visual shell before cli imports UI. The dashboard replaces
    # UI.banner; all existing streaming, tools, prompt handling, scrolling,
    # and tests keep using the original UI implementation.
    from .dashboard import install
    install()

    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
