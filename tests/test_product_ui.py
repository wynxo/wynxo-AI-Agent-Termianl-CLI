"""Regression checks for the polished interactive terminal shell."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(code: str):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_product_shell_installs_on_the_real_runtime_classes():
    """The installed command must actually opt into the new presentation."""
    result = _run(
        """
        from wynxo import cli, product_ui
        from wynxo.ui import UI

        product_ui.install()

        assert UI.home.__module__ == "wynxo.product_ui"
        assert UI.user_line.__module__ == "wynxo.product_ui"
        assert UI.tool_call.__module__ == "wynxo.product_ui"
        assert cli.Repl._prompt_message.__module__ == "wynxo.product_ui"
        assert cli.Repl._bottom_toolbar.__module__ == "wynxo.product_ui"
        assert cli.TerminalCallbacks.on_content.__module__ == "wynxo.product_ui"
        """
    )
    assert result.returncode == 0, result.stderr


def test_product_shell_install_is_idempotent():
    """Reloading/bootstrap reuse must not wrap the renderer repeatedly."""
    result = _run(
        """
        from wynxo import cli, product_ui

        product_ui.install()
        prompt = cli.Repl._prompt_message
        toolbar = cli.Repl._bottom_toolbar
        product_ui.install()

        assert cli.Repl._prompt_message is prompt
        assert cli.Repl._bottom_toolbar is toolbar
        """
    )
    assert result.returncode == 0, result.stderr
