"""Editing a file without rewriting how it is stored.

read_file numbers lines, so what the model sees is LF whatever the file
actually uses. It can only send back what it saw. Without knowing the
file's own line endings, a multi-line old_text never matched a CRLF file
-- the agent simply could not edit it -- and write_file quietly converted
every line ending in the file, which reads as a whole-file diff to
everyone else on the project.
"""

import asyncio

import pytest

from wynxo.tools.files import EditFile, MultiEdit, ReadFile, WriteFile, _decode

CRLF = "def f():\r\n    value = 1\r\n    return value\r\n"
LF = CRLF.replace("\r\n", "\n")


def run(tool, **arguments):
    return asyncio.run(tool.invoke(arguments))


@pytest.fixture
def crlf(tmp_path):
    (tmp_path / "win.py").write_bytes(CRLF.encode())
    return tmp_path


@pytest.fixture
def lf(tmp_path):
    (tmp_path / "nix.py").write_bytes(LF.encode())
    return tmp_path


# -- what the model is given --------------------------------------------------


def test_decoded_text_is_normalised_so_one_representation_is_compared(crlf):
    decoded = _decode(crlf / "win.py")
    assert "\r" not in decoded.text
    assert decoded.newline == "\r\n"
    assert decoded.mixed is False


def test_an_lf_file_reports_lf(lf):
    assert _decode(lf / "nix.py").newline == "\n"


def test_the_model_never_sees_a_carriage_return(crlf):
    shown = run(ReadFile(crlf), path="win.py")
    assert "\r" not in shown.output


# -- editing ------------------------------------------------------------------


def test_a_multi_line_edit_works_on_a_crlf_file(crlf):
    # The whole defect: the model can only send the LF it was shown.
    result = run(EditFile(crlf), path="win.py",
                 old_text="    value = 1\n    return value",
                 new_text="    return 2")
    assert result.ok, result.error
    assert (crlf / "win.py").read_bytes() == b"def f():\r\n    return 2\r\n"


def test_editing_a_crlf_file_leaves_every_other_line_ending_alone(crlf):
    run(EditFile(crlf), path="win.py", old_text="    value = 1",
        new_text="    value = 99")
    raw = (crlf / "win.py").read_bytes()
    assert raw.count(b"\r\n") == 3
    assert b"\n\n" not in raw.replace(b"\r\n", b"")


def test_a_multi_line_edit_still_works_on_an_lf_file(lf):
    result = run(EditFile(lf), path="nix.py",
                 old_text="    value = 1\n    return value",
                 new_text="    return 2")
    assert result.ok
    assert (lf / "nix.py").read_bytes() == b"def f():\n    return 2\n"


def test_an_lf_file_does_not_gain_carriage_returns(lf):
    run(EditFile(lf), path="nix.py", old_text="    value = 1", new_text="    value = 9")
    assert b"\r" not in (lf / "nix.py").read_bytes()


def test_multi_edit_works_on_a_crlf_file(crlf):
    result = run(MultiEdit(crlf), path="win.py", edits=[
        {"old_text": "def f():\n    value = 1", "new_text": "def f():\n    value = 7"},
        {"old_text": "    return value", "new_text": "    return value * 2"},
    ])
    assert result.ok, result.error
    assert (crlf / "win.py").read_bytes() == \
        b"def f():\r\n    value = 7\r\n    return value * 2\r\n"


# -- writing ------------------------------------------------------------------


def test_write_file_keeps_the_line_endings_the_file_had(crlf):
    run(WriteFile(crlf), path="win.py", content="def g():\n    return 9\n")
    assert (crlf / "win.py").read_bytes() == b"def g():\r\n    return 9\r\n"


def test_write_file_does_not_give_an_lf_file_carriage_returns(lf):
    run(WriteFile(lf), path="nix.py", content="def g():\n    return 9\n")
    assert (lf / "nix.py").read_bytes() == b"def g():\n    return 9\n"


def test_a_brand_new_file_is_written_with_lf(tmp_path):
    run(WriteFile(tmp_path), path="fresh.py", content="a = 1\n")
    assert (tmp_path / "fresh.py").read_bytes() == b"a = 1\n"


# -- the cases where lossless is impossible -----------------------------------


def test_a_mixed_file_is_detected(tmp_path):
    (tmp_path / "m.py").write_bytes(b"a = 1\r\nb = 2\nc = 3\r\n")
    decoded = _decode(tmp_path / "m.py")
    assert decoded.mixed is True
    assert decoded.newline == "\r\n"      # the majority


def test_normalising_a_mixed_file_is_said_out_loud(tmp_path):
    # It is a change the user did not ask for, so it does not happen
    # quietly. A mixed file cannot survive a whole-file write intact.
    (tmp_path / "m.py").write_bytes(b"a = 1\r\nb = 2\nc = 3\r\n")
    result = run(EditFile(tmp_path), path="m.py", old_text="b = 2", new_text="b = 4")
    assert result.ok
    assert "mixed" in result.output
    assert (tmp_path / "m.py").read_bytes() == b"a = 1\r\nb = 4\r\nc = 3\r\n"


def test_encoding_and_bom_are_still_preserved(tmp_path):
    # The line-ending work must not cost the encoding work that was
    # already there.
    (tmp_path / "u.py").write_bytes(b"\xff\xfe" + "a = 1\r\n".encode("utf-16-le"))
    result = run(EditFile(tmp_path), path="u.py", old_text="a = 1", new_text="a = 2")
    assert result.ok
    raw = (tmp_path / "u.py").read_bytes()
    assert raw.startswith(b"\xff\xfe")
    assert raw[2:].decode("utf-16-le") == "a = 2\r\n"


def test_a_crlf_file_with_no_trailing_newline_is_unchanged_apart_from_the_edit(tmp_path):
    (tmp_path / "t.py").write_bytes(b"a = 1\r\nb = 2")
    run(EditFile(tmp_path), path="t.py", old_text="b = 2", new_text="b = 3")
    assert (tmp_path / "t.py").read_bytes() == b"a = 1\r\nb = 3"


def test_grep_matches_a_crlf_file_without_the_carriage_return_leaking(crlf):
    from wynxo.tools.search import Grep

    result = run(Grep(crlf), pattern="value = 1")
    assert result.ok
    assert "\r" not in result.output
