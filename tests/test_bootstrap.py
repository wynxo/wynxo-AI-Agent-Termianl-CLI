from __future__ import annotations


def test_windows_default_uses_classic_renderer(monkeypatch):
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
    assert seen == [False]


def test_windows_explicit_chat_keeps_chat_renderer(monkeypatch):
    import sys
    from wynxo import bootstrap

    seen = []

    def fake_main():
        import wynxo.tui as tui
        seen.append(tui.usable())
        return 0

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["wynxo", "--chat"])
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
