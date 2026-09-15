"""Regression checks for the calm interactive terminal shell."""

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


def test_high_frequency_transcript_events_are_borderless():
    """Messages and tools should read like a transcript, not stacked cards."""
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

        ui.user_line("hello")
        ui.tool_call("read_file", "src/main.py", "24 lines", True)

        rendered = ui.console.file.getvalue()
        assert "hello" in rendered
        assert "src/main.py" in rendered
        for edge in "╭╮╰╯┌┐└┘":
            assert edge not in rendered
        """
    )
    assert result.returncode == 0, result.stderr


def test_prompt_and_toolbar_do_not_draw_frames():
    result = _run(
        """
        from pathlib import Path
        from types import SimpleNamespace
        from prompt_toolkit.formatted_text import to_formatted_text
        from wynxo import clean_ui, product_ui
        from wynxo.ui import Glyphs, UI

        product_ui.install()
        clean_ui.install()
        clean_ui.is_dumb_terminal = lambda: False
        product_ui._prompt_hint = lambda repl: "Ask anything  ·  / for commands"

        ui = UI()
        ui.g = Glyphs(True)
        ui.narrow = False
        ui.width = 100

        prompt_repl = SimpleNamespace(ui=ui)
        prompt = clean_ui._prompt_message(prompt_repl)
        prompt_text = "".join(
            fragment[1] for fragment in to_formatted_text(prompt)
        )
        assert "Ask anything" in prompt_text
        assert "╭" not in prompt_text and "╰" not in prompt_text

        toolbar_repl = SimpleNamespace(
            ui=ui,
            workspace=Path("."),
            _status_line=lambda: "qwen3-coder  ·  code  ·  ctx 12%",
        )
        toolbar = clean_ui._bottom_toolbar(toolbar_repl)
        toolbar_text = "".join(
            fragment[1] for fragment in to_formatted_text(toolbar)
        )
        assert "\n" not in toolbar_text
        assert "ready" in toolbar_text
        assert "╭" not in toolbar_text and "╰" not in toolbar_text
        """
    )
    assert result.returncode == 0, result.stderr
