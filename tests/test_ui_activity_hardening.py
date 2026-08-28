from __future__ import annotations

from wynxo.ui import ActivityBar, UI


def test_streamed_code_updates_activity_bar() -> None:
    ui = UI()
    bar = ActivityBar(ui, "high")
    ui.bar = bar

    bar.set_lead(ui.highlight("return 42", "python"))

    assert bar.activity == "writing code"
    assert "9 chars" in bar.detail
    assert bar.lead is not None
