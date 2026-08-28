from __future__ import annotations

from wynxo.ui import ActivityBar, UI


def test_partial_code_line_updates_pinned_activity() -> None:
    ui = UI()
    ui.width = 80
    bar = ActivityBar(ui, "high")
    ui.bar = bar

    line = ui.highlight("return 42", "python")
    bar.set_lead(line)

    assert bar.activity == "writing code"
    assert "9 chars" in bar.detail
    assert bar.lead is line


def test_clearing_partial_code_leaves_activity_usable() -> None:
    ui = UI()
    bar = ActivityBar(ui, "medium")
    ui.bar = bar

    bar.set_lead(ui.highlight("x = 1", "python"))
    bar.set_lead(None)

    assert bar.lead is None
    assert bar._render().plain
