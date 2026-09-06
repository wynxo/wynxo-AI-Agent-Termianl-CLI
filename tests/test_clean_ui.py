"""Regression checks for the cleaned product-shell typography."""

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


def test_clean_shell_replaces_font_fragile_renderers():
    result = _run(
        """
        from wynxo import clean_ui, cli, product_ui
        from wynxo.ui import UI

        product_ui.install()
        clean_ui.install()

        assert UI.home.__module__ == "wynxo.clean_ui"
        assert UI.user_line.__module__ == "wynxo.clean_ui"
        assert UI.tool_call.__module__ == "wynxo.clean_ui"
        assert UI.todos.__module__ == "wynxo.clean_ui"
        assert cli.Repl._prompt_message.__module__ == "wynxo.clean_ui"
        assert cli.Repl._bottom_toolbar.__module__ == "wynxo.clean_ui"
        assert product_ui._assistant_heading.__module__ == "wynxo.clean_ui"
        """
    )
    assert result.returncode == 0, result.stderr


def test_clean_shell_install_is_idempotent():
    result = _run(
        """
        from wynxo import clean_ui, cli, product_ui

        product_ui.install()
        clean_ui.install()
        home = clean_ui.ui_mod.UI.home
        toolbar = cli.Repl._bottom_toolbar
        clean_ui.install()

        assert clean_ui.ui_mod.UI.home is home
        assert cli.Repl._bottom_toolbar is toolbar
        """
    )
    assert result.returncode == 0, result.stderr


def test_tool_renderer_does_not_depend_on_cli_verb():
    """Regression for launch_application crashing after a successful tool call."""
    result = _run(
        """
        import io
        from wynxo import clean_ui, product_ui
        from wynxo.ui import Glyphs, UI

        product_ui.install()
        clean_ui.install()

        ui = UI()
        ui.g = Glyphs(True)
        ui.narrow = False
        ui.width = 100
        ui.console.file = io.StringIO()
        ui.tool_call("launch_application", "kcalc", "opened", True)

        rendered = ui.console.file.getvalue()
        assert "kcalc" in rendered
        assert "opened" in rendered
        """
    )
    assert result.returncode == 0, result.stderr
