from __future__ import annotations


def test_windows_default_uses_classic_renderer(monkeypatch):
    import sys
    from wynxo import bootstrap

    calls = []

    class DummyTui:
        usable = staticmethod(lambda: True)

    def fake_main():
        import wynxo.tui as real_tui
        calls.append(real_tui.usable())
        return 17

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("wynxo.tui.usable", lambda: True)
    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 17
    assert calls == [False]


def test_windows_explicit_chat_keeps_chat_renderer(monkeypatch):
    import sys
    from wynxo import bootstrap

    calls = []

    def fake_main():
        import wynxo.tui as real_tui
        calls.append(real_tui.usable())
        return 0

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["wynxo", "--chat"])
    monkeypatch.setattr("wynxo.tui.usable", lambda: True)
    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 0
    assert calls == [True]


def test_linux_keeps_default_ui(monkeypatch):
    import sys
    from wynxo import bootstrap

    calls = []

    def fake_main():
        calls.append(True)
        return 0

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 0
    assert calls == [True]
