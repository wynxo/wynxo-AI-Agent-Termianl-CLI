"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations


def main():
    """Start the CLI with compatibility and the product shell installed.

    The agent and REPL stay in ``cli.py``. The visual layer is installed
    separately so terminal chrome can evolve without coupling layout
    decisions to provider/tool behaviour.
    """
    # Install compatibility shims before cli imports the provider/client.
    from . import runtime_compat  # noqa: F401

    from . import clean_ui
    from . import cli
    from . import product_ui

    # Tests and embedders sometimes replace cli.main with their own callable.
    # In that case this function is only a dispatcher; do not globally restyle
    # classes in the host process. The installed command always reaches the
    # real wynxo.cli.main and therefore gets the product shell.
    if getattr(cli.main, "__module__", "") == cli.__name__:
        product_ui.install()
        clean_ui.install()
    return cli.main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
