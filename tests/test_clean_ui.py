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


def test_clean_shell_replaces_the_complete_message_path():
    result = _run(
        """
        from wynxo import clean_ui, cli, product_ui
        from wynxo.ui import UI

        product_ui.install()
        clean_ui.install()

        assert UI.home.__module__ == "wynxo.clean_ui"
        assert UI.user_line.__module__ == "wynxo.clean_ui"
        assert UI.assistant_markdown.__module__ == "wynxo.clean_ui"
        assert UI.tool_call.__module__ == "wynxo.clean_ui"
        assert UI.todos.__module__ == "wynxo.clean_ui"
        assert cli.Repl._prompt_message.__module__ == "wynxo.clean_ui"
        assert cli.Repl._bottom_toolbar.__module__ == "wynxo.clean_ui"
        assert cli.TerminalCallbacks.on_content.__module__ == "wynxo.clean_ui"
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
        content = cli.TerminalCallbacks.on_content
        clean_ui.install()

        assert clean_ui.ui_mod.UI.home is home
        assert cli.Repl._bottom_toolbar is toolbar
        assert cli.TerminalCallbacks.on_content is content
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
        ui.tool_call("list_applications", "terminal", "2 matches", True)

        rendered = ui.console.file.getvalue()
        assert "kcalc" in rendered
        assert "opened" in rendered
        assert "list applications" in rendered
        assert "2 matches" in rendered
        """
    )
    assert result.returncode == 0, result.stderr


def test_assistant_heading_and_body_share_one_indent_contract():
    """The two-cell clean rail must not leave the old six-cell body indent."""
    result = _run(
        """
        import io
        from wynxo import clean_ui, product_ui
        from wynxo.ui import Glyphs, UI

        product_ui.install()
        clean_ui.install()

        seen = []

        class Streamer:
            def __init__(self, ui, indent="", **_kwargs):
                seen.append(indent)
            def feed(self, _text):
                pass
            def finish(self):
                pass

        clean_ui.ui_mod.CodeStreamer = Streamer

        ui = UI()
        ui.g = Glyphs(True)
        ui.narrow = False
        ui.width = 100
        ui.console.file = io.StringIO()
        ui.assistant_markdown("aligned answer")

        assert clean_ui.MESSAGE_INDENT == "   "
        assert seen == [clean_ui.MESSAGE_INDENT]
        """
    )
    assert result.returncode == 0, result.stderr
