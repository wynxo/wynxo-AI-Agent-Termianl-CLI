"""Runtime regressions for the product UI before clean_ui layers over it."""

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


def test_product_ui_tool_renderer_works_without_clean_ui():
    """The lower presentation layer must not depend on a cli.verb alias."""
    result = _run(
        """
        import io
        from wynxo import product_ui
        from wynxo.ui import Glyphs, UI

        product_ui.install()

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


def test_product_ui_tool_detail_is_sanitised():
    result = _run(
        """
        import io
        from wynxo import product_ui
        from wynxo.ui import Glyphs, UI

        product_ui.install()

        ui = UI()
        ui.g = Glyphs(True)
        ui.narrow = False
        ui.width = 100
        ui.console.file = io.StringIO()
        ui.tool_call("list_applications", "terminal", "safe\\x1b[2Jdetail", True)

        rendered = ui.console.file.getvalue()
        assert "safe" in rendered and "detail" in rendered
        assert "\\x1b[2J" not in rendered
        """
    )
    assert result.returncode == 0, result.stderr
