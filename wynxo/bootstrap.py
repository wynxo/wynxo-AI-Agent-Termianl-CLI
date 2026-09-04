"""Stable process bootstrap for the installed ``wynxo`` command."""

from __future__ import annotations


def main():
    """Start the CLI with runtime compatibility installed first.

    No layout decisions here. The visual shell used to be installed from
    this function, by replacing ``UI.banner`` on the class -- which meant
    the look of the application depended on which entry point had been
    called, and a single import of ``bootstrap`` in a test suite silently
    changed the header for everything that ran after it. The shell is part
    of ``UI`` now, so there is nothing to install.
    """
    # Install compatibility shims before cli imports the provider/client.
    # This keeps startup resilient when the CLI and provider signatures are
    # briefly out of sync during development.
    from . import runtime_compat  # noqa: F401

    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
