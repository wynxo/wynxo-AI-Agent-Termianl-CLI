"""The bootstrap shim exists to wrap ``cli.main`` for the installed command.

There are no layout decisions here -- wynxo always runs the scrolling
prompt, and bootstrap must not override or even inspect the terminal state.
"""

from __future__ import annotations


def test_bootstrap_delegates_to_cli_main(monkeypatch):
    import sys

    from wynxo import bootstrap

    called = []

    def fake_main():
        called.append(True)
        return 17

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["wynxo"])
    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 17
    assert called == [True], "bootstrap must only forward to cli.main"


def test_bootstrap_does_not_inspect_the_terminal(monkeypatch):
    """The scrolling prompt is the only layout, on every platform.

    A regression guard: bootstrap used to call the old full-screen
    tui.usable() and could silently downgrade or upgrade the layout. The
    tui is gone, so the sharper guard is that bootstrap is a bare
    delegation: no arguments are passed or inspected, on any platform.
    """
    import sys

    from wynxo import bootstrap

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["wynxo"])
    calls = []

    def fake_main(*args, **kwargs):
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr("wynxo.cli.main", fake_main)

    assert bootstrap.main() == 0
    # A bare delegation: nothing was inspected or passed along.
    assert calls == [((), {})]
