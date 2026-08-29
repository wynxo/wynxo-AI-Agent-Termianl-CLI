"""The shell output cleaner: line endings and progress-bar redraws."""

from __future__ import annotations

from wynxo.tools.shell import _clean


def test_plain_line():
    assert _clean(b"hello\n") == "hello"


def test_crlf_becomes_the_line():
    # Windows CRLF must not collapse to empty.
    assert _clean(b"slow work done\r\n") == "slow work done"


def test_double_carriage_return_still_keeps_the_text():
    # PowerShell can add a second \r to an already-CRLF line.
    assert _clean(b"slow work done\r\r\n") == "slow work done"


def test_progress_bar_keeps_the_last_frame():
    assert _clean(b"10%\r20%\r30%\r100%") == "100%"


def test_multiline_keeps_every_line():
    text = _clean(b"one\n2\r\ntwo\r\nthree\n")
    assert text == "one\n2\ntwo\nthree"