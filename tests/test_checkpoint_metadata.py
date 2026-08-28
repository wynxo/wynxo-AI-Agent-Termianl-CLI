from __future__ import annotations

from wynxo.checkpoints import Checkpoints


def test_undo_preserves_file_mode(tmp_path):
    path = tmp_path / "executable.sh"
    path.write_bytes(b"#!/bin/sh\necho old\n")
    path.chmod(0o750)
    before_mode = path.stat().st_mode & 0o777

    points = Checkpoints()
    points.capture(path, "edit_file")
    path.write_bytes(b"#!/bin/sh\necho changed\n")
    path.chmod(0o600)

    ok, _ = points.undo()

    assert ok
    assert path.read_bytes() == b"#!/bin/sh\necho old\n"
    assert path.stat().st_mode & 0o777 == before_mode
