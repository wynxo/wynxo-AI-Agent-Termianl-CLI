from __future__ import annotations


def test_windows_uses_the_chat_layout_by_default(monkeypatch):
    """The pinned-composer chat layout is the product's default on Windows;
    bootstrap must not silently downgrade it to the scrolling prompt."""
    import sys

    from wynxo import bootstrap

    seen = []

    def fake_main():
        import wynxo.tui as tui
        seen.append(tui.usable())
        return 17

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["wynxo"])
    monkeypatch.setattr("wynxo.tui.usable", lambda: True)
    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 17
    assert seen == [True], "the chat layout must not be disabled on Windows"


def test_windows_classic_is_an_explicit_opt_out(monkeypatch):
    """--classic is honoured through cli.apply_flags; bootstrap leaves the
    decision alone rather than overriding it."""
    import sys

    from wynxo import bootstrap

    seen = []

    def fake_main():
        import wynxo.tui as tui
        seen.append(tui.usable())
        return 0

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["wynxo", "--classic"])
    monkeypatch.setattr("wynxo.tui.usable", lambda: True)
    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 0
    assert seen == [True]


def test_non_windows_keeps_normal_ui(monkeypatch):
    import sys

    from wynxo import bootstrap

    called = []

    def fake_main():
        called.append(True)
        return 0

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["wynxo"])
    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 0
    assert called == [True]
