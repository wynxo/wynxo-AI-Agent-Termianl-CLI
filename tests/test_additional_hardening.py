from __future__ import annotations

from pathlib import Path

import pytest

from wynxo._shell_secret_hardening import redact_shell_text
from wynxo.tools.search import Glob, Grep
from wynxo import testing


def test_shell_output_redaction_masks_named_secret() -> None:
    text = "API_KEY=supersecretvalue123"
    assert "supersecretvalue123" not in redact_shell_text(text)
    assert "[redacted by wynxo]" in redact_shell_text(text)


def test_bun_lockfile_selects_bun() -> None:
    root = Path("/tmp/wynxo-bun-test")
    # The shim only inspects existence; avoid touching the actual filesystem.
    class FakeRoot:
        def __truediv__(self, value):
            class FakePath:
                def is_file(self):
                    return value == "bun.lock"
            return FakePath()
    assert testing._node_agent(FakeRoot()) == "bun"


@pytest.mark.asyncio
async def test_search_finds_github_config(tmp_path: Path) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "ci.yml").write_text("name: test\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("internal\n", encoding="utf-8")

    glob = await Glob(tmp_path).invoke({"pattern": "**/*"})
    grep = await Grep(tmp_path).invoke({"pattern": "name:", "path": "."})

    assert ".github/ci.yml" in glob.output.replace("\\", "/")
    assert ".git/config" not in glob.output.replace("\\", "/")
    assert ".github/ci.yml" in grep.output.replace("\\", "/")
    assert ".git/config" not in grep.output.replace("\\", "/")
