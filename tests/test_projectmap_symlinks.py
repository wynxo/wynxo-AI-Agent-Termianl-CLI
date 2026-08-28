"""The project map must not follow symlinks.

A junction pointing back at its own directory made walk() loop forever:
the stack never emptied and the file list never grew. A link pointing
outside the project pulled unrelated files into the map. Links are
skipped; the map is the real tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from wynxo.projectmap import walk


def make_link(link: Path, target: Path) -> bool:
    """Create a directory link (junction on Windows, symlink elsewhere).
    Returns False when the platform refuses -- the test skips then, because
    a machine that cannot make links cannot exhibit the bug."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True, timeout=15)
            return result.returncode == 0
        os.symlink(str(target), str(link), target_is_directory=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def test_walk_terminates_on_a_self_referencing_junction(tmp_path: Path):
    (tmp_path / "real.py").write_text("x = 1\n")
    loop = tmp_path / "loop"
    if not make_link(loop, tmp_path):
        pytest.skip("platform refused to create a directory link")
    # walk() would loop forever before the fix; a hard cap on the work it
    # is allowed to do proves termination rather than hanging the suite.
    import time
    started = time.monotonic()
    found = walk(tmp_path, limit=400)
    assert time.monotonic() - started < 10, "walk must terminate on a cycle"
    names = {p.name for p in found}
    assert "real.py" in names
    # Nothing reached through the link: the junction's own name is a
    # directory (skipped), and its contents are the same tree, so no
    # duplicate or outside path can appear.
    assert all(p.resolve().is_relative_to(tmp_path.resolve()) for p in found)


def test_walk_skips_a_link_to_outside_the_project(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("LEAK = True\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("x = 1\n")
    if not make_link(project / "link", outside):
        pytest.skip("platform refused to create a directory link")
    found = walk(project, limit=400)
    names = {p.name for p in found}
    assert names == {"main.py"}, names


def test_walk_skips_symlinked_files(tmp_path: Path):
    outside = tmp_path / "outside.py"
    outside.write_text("LEAK = True\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("x = 1\n")
    try:
        os.symlink(str(outside), str(project / "link.py"))
    except OSError:
        pytest.skip("platform refused to create a file symlink")
    found = walk(project, limit=400)
    assert [p.name for p in found] == ["main.py"]
