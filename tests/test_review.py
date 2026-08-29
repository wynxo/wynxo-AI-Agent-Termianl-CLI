"""The /review command: it must hand the model the working diff and print
the review, and treat a clean tree as a no-op."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from rich.console import Console

from wynxo import cli, prompts, repo

UNSTAGED = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-print(1)\n+return 2\n"
STAGED = "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-a\n+b\n"


class _UI:
    def __init__(self):
        self.buffer = io.StringIO()
        self.console = Console(file=self.buffer, force_terminal=False, width=100)
        self.notes = []

    def warn(self, message):
        self.notes.append(("warn", message))

    def info(self, message):
        self.notes.append(("info", message))

    def status(self, _message):
        return _Status()


class _Status:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Turn:
    content = "Looks fine, but `return 2` changes behaviour without a test."


class _Agent:
    def __init__(self):
        self.seen = None

    async def _call_model(self, messages, use_tools=False, stream_content=False):
        self.seen = messages
        return _Turn()


class TestReview:
    def _run(self, repl, args):
        return cli.Repl.cmd_review(repl, args)

    def test_clean_tree_is_a_noop(self, monkeypatch):
        def fake_git(cmd, cwd=None, timeout=None):
            if cmd == ["rev-parse", "--git-dir"]:
                return (True, ".git")
            if cmd in (["diff"], ["diff", "--staged"]):
                return (True, "")    # empty patch = nothing to review
            return (False, "unexpected")
        monkeypatch.setattr(repo, "run_git", fake_git)
        ui = _UI()
        repl = SimpleNamespace(workspace="/tmp/x", ui=ui, agent=_Agent())
        asyncio.run(self._run(repl, ["all"]))
        assert any(m == "no changes to review" for _, m in ui.notes)

    def test_review_asks_the_model_and_prints(self, monkeypatch):
        def fake_git(cmd, cwd=None, timeout=None):
            if cmd == ["rev-parse", "--git-dir"]:
                return (True, ".git")
            if cmd == ["diff"]:
                return (True, UNSTAGED)
            if cmd == ["diff", "--staged"]:
                return (True, STAGED)
            return (False, "unexpected")
        monkeypatch.setattr(repo, "run_git", fake_git)
        agent = _Agent()
        ui = _UI()
        repl = SimpleNamespace(workspace="/tmp/x", ui=ui, agent=agent)
        assert asyncio.run(self._run(repl, ["all"])) is True
        # the model was asked with the review prompt over the diff
        assert agent.seen is not None
        sent = agent.seen[0]["content"]
        assert prompts.REVIEW_PROMPT in sent
        assert "return 2" in sent
        # and the review was printed
        assert "Looks fine" in ui.buffer.getvalue()