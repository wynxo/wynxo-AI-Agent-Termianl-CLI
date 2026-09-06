"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations


def main():
    """Start the CLI with compatibility and the product shell installed.

    The agent and REPL stay in ``cli.py``.  The visual layer is installed
    separately so the terminal chrome can evolve without coupling layout
    decisions to provider/tool behaviour.
    """
    # Install compatibility shims before cli imports the provider/client.
    # This keeps startup resilient when the CLI and provider signatures are
    # briefly out of sync during development.
    from . import runtime_compat  # noqa: F401

    from . import cli
    from . import product_ui

    product_ui.install()
    return cli.main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
