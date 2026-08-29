"""The cloud GitHub layer: GitHubClient's gh-CLI plumbing, and the /gh
command group that drives it from the REPL.

Everything is tested against a fake `subprocess.run`, so no network and no
real gh install is involved.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from wynxo.gh import GitHubClient, GitHubError


def _gh(repl, args, reply="y"):
    """Drive the async cmd_gh from a sync test, confirming any prompt."""
    repl.prompt_session = _FakePrompt(reply)
    coro = repl.cmd_gh(args)
    return asyncio.run(coro)


class Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeGh:
    """Records gh invocations and answers from canned scripts."""

    def __init__(self, monkeypatch, script: dict | None = None):
        self.calls: list[tuple[list, dict]] = []
        self.script = script or {}
        monkeypatch.setattr("subprocess.run", self._run)

    def _run(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        key = tuple(args)
        if key in self.script:
            answer = self.script[key]
            if isinstance(answer, Proc):
                return answer
            return Proc(stdout=answer)
        # Default: no answer found -> a generic success.
        return Proc(stdout="{}")

    def called(self, *want: str) -> bool:
        return any(a[: len(want)] == list(want) for a, _ in self.calls)

    def input_for(self, *want: str) -> dict:
        for args, kwargs in self.calls:
            if args[: len(want)] == list(want):
                return json.loads(kwargs.get("input") or "{}")
        raise AssertionError(f"no call matching {want!r}")


@pytest.fixture
def client(monkeypatch):
    gh = FakeGh(monkeypatch)
    return GitHubClient(), gh


class TestIdentity:
    def test_available_reports_the_cli(self, monkeypatch):
        import shutil

        monkeypatch.setattr(shutil, "which", lambda tool: tool == "gh")
        assert GitHubClient().available() is True
        monkeypatch.setattr(shutil, "which", lambda tool: None)
        assert GitHubClient().available() is False

    def test_auth_user_returns_the_account(self, client):
        c, gh = client
        gh.script[("gh", "api", "user", "--jq", ".login")] = "wynxo\n"
        assert c.auth_user() == "wynxo"

    def test_not_logged_in_gets_a_useful_message(self, client):
        c, gh = client
        gh.script[("gh", "api", "user", "--jq", ".login")] = Proc(
            stderr="To get started with GitHub CLI, please run:  gh auth login",
            returncode=1)
        with pytest.raises(GitHubError) as exc:
            c.auth_user()
        assert "gh auth login" in str(exc.value)

    def test_missing_gh_cli_mentions_installation(self, client, monkeypatch):
        c, _ = client

        def boom(*a, **k):
            raise FileNotFoundError

        monkeypatch.setattr("subprocess.run", boom)
        with pytest.raises(GitHubError) as exc:
            c.auth_user()
        assert "cli.github.com" in str(exc.value)


class TestReading:
    def test_default_branch(self, client):
        c, gh = client
        gh.script[("gh", "api", "repos/o/r", "--jq", ".default_branch")] = "main\n"
        assert c.repo_default_branch("o", "r") == "main"

    def test_tree_parses_recursive_entries(self, client):
        c, gh = client
        gh.script[("gh", "api", "repos/o/r/git/trees/main?recursive=1",
                   "--jq", ".tree[]")] = (
            '{"path":"src","type":"tree","size":0}\n'
            '{"path":"src/a.py","type":"blob","size":42}\n')
        entries = c.tree("o", "r", "main")
        assert [e["path"] for e in entries] == ["src", "src/a.py"]

    def test_read_decodes_base64_and_returns_sha(self, client):
        c, gh = client
        payload = {"content": base64.b64encode(b"hello\n").decode(),
                   "sha": "abc123", "size": 6}
        gh.script[("gh", "api", "repos/o/r/contents/src/a.py?ref=main",
                   "--jq", "{content, sha, size}")] = json.dumps(payload)
        text, sha = c.read("o", "r", "src/a.py", "main")
        assert text == "hello\n"
        assert sha == "abc123"

    def test_oversized_files_are_refused(self, client):
        c, gh = client
        payload = {"content": base64.b64encode(b"x").decode(),
                   "sha": "s", "size": 2_000_000}
        gh.script[("gh", "api", "repos/o/r/contents/big.bin?ref=main",
                   "--jq", "{content, sha, size}")] = json.dumps(payload)
        with pytest.raises(GitHubError) as exc:
            c.read("o", "r", "big.bin", "main")
        assert "1000000" in str(exc.value)


class TestWriting:
    def test_write_sends_commit_fields_via_stdin(self, client):
        c, gh = client
        url = "repos/o/r/contents/src/a.py"
        gh.script[("gh", "api", "--method", "PUT", url, "--jq",
                   ".commit.sha", "--input", "-")] = "newcommit\n"
        commit = c.write("o", "r", "src/a.py", "new content",
                         "add retry", "feature", sha="oldsha")
        assert commit == "newcommit"
        payload = gh.input_for("gh", "api", "--method", "PUT", url)
        assert payload["message"] == "add retry"
        assert payload["branch"] == "feature"
        assert payload["sha"] == "oldsha"
        assert payload["content"] == base64.b64encode(b"new content").decode()

    def test_write_without_sha_creates_a_new_file(self, client):
        c, gh = client
        gh.script[("gh", "api", "--method", "PUT",
                   "repos/o/r/contents/new.md", "--jq", ".commit.sha",
                   "--input", "-")] = "c1\n"
        commit = c.write("o", "r", "new.md", "hi", "add new.md", "main")
        assert commit == "c1"
        assert "sha" not in gh.input_for("gh", "api", "--method", "PUT",
                                         "repos/o/r/contents/new.md")

    def test_create_branch_posts_a_ref(self, client):
        c, gh = client
        c.create_branch("o", "r", "feature", "basehead")
        payload = gh.input_for("gh", "api", "--method", "POST",
                               "repos/o/r/git/refs")
        assert payload == {"ref": "refs/heads/feature", "sha": "basehead"}

    def test_commits_parses_quoted_messages(self, client):
        c, gh = client
        gh.script[("gh", "api", "repos/o/r/commits?sha=feature", "--jq",
                   ".[0:15] | .[].commit.message")] = (
            '"first line\\n\\nbody"\n"second"\n')
        assert c.commits("o", "r", "feature") == ["first line\n\nbody", "second"]

    def test_open_pr_uses_gh_pr_create(self, client):
        c, gh = client
        gh.script[("gh", "pr", "create", "--repo", "o/r", "--base", "main",
                   "--head", "feature", "--title", "T", "--body", "B")] = (
            "https://github.com/o/r/pull/7\n")
        url = c.open_pr("o", "r", "main", "feature", "T", "B")
        assert url == "https://github.com/o/r/pull/7"


class _FakePrompt:
    """Answers the diff-review question so /gh edit can commit in a test."""

    def __init__(self, reply="y"):
        self.reply = reply
        self.answers = []

    async def prompt_async(self, *args, **kwargs):
        self.answers.append(kwargs.get("default"))
        return self.reply


class _UI:
    def __init__(self):
        self.msgs: list[tuple[str, str]] = []
        self.printed: list[str] = []
        self.diffs: list[str] = []

    def info(self, m): self.msgs.append(("info", m))
    def success(self, m): self.msgs.append(("success", m))
    def error(self, m): self.msgs.append(("error", m))
    def warn(self, m): self.msgs.append(("warn", m))
    def diff(self, text): self.diffs.append(str(text))

    class Console:
        def __init__(self, ui):
            self._ui = ui

        def print(self, text="", **kwargs):
            self._ui.printed.append(str(text))

    @property
    def console(self):
        return self.Console(self)


class StubClient:
    """The GitHubClient surface cmd_gh touches, with canned answers."""

    def __init__(self):
        self.writes = []
        self.prs = []
        self.branches = []
        self.user = "wynxo"

    def auth_user(self):
        return self.user

    def repo_default_branch(self, owner, repo):
        return "main"

    def tree(self, owner, repo, branch):
        return [
            {"path": "README.md", "type": "blob", "size": 5},
            {"path": "src", "type": "tree", "size": 0},
            {"path": "src/a.py", "type": "blob", "size": 42},
            {"path": "src/deep/b.py", "type": "blob", "size": 7},
        ]

    def read(self, owner, repo, path, branch):
        return "hello\n", "sha1"

    def ref_sha(self, owner, repo, branch):
        return "headsha"

    def create_branch(self, owner, repo, name, sha):
        self.branches.append((name, sha))

    def commits(self, owner, repo, branch):
        return ["feat: retry upload", "fix: timeout"]

    def open_pr(self, owner, repo, base, head, title, body):
        self.prs.append((base, head, title, body))
        return "https://github.com/o/r/pull/9"

    def write(self, owner, repo, path, content, message, branch, sha=None):
        self.writes.append((path, content, message, branch, sha))
        return "commit123"


def _repl(client: StubClient) -> tuple:
    from wynxo.cli import Repl

    repl = Repl.__new__(Repl)
    repl.gh = client
    repl.gh_ws = None
    ui = _UI()
    repl.ui = ui
    repl.prompt_session = _FakePrompt("y")
    return repl, ui


class TestGhCommand:
    def test_status_without_a_workspace(self):
        repl, ui = _repl(StubClient())
        _gh(repl, [])
        assert any("logged in as wynxo" in m for _, m in ui.msgs)

    def test_open_sets_the_workspace(self):
        repl, ui = _repl(StubClient())
        assert _gh(repl, ["open", "o/r"]) is True
        assert repl.gh_ws["owner"] == "o"
        assert repl.gh_ws["branch"] == "main"
        assert any("opened o/r @ main" in m for _, m in ui.msgs)

    def test_ls_lists_direct_children_with_sizes(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        _gh(repl, ["ls"])
        printed = " ".join(ui.printed)
        assert "README.md  (5 B)" in printed
        assert "src/" in printed
        assert "src/deep" not in printed

    def test_ls_with_a_prefix_stays_at_that_level(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        _gh(repl, ["ls", "src"])
        printed = " ".join(ui.printed)
        assert "a.py  (42 B)" in printed
        assert "deep" not in printed

    def test_cat_prints_the_file(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        _gh(repl, ["cat", "README.md"])
        assert "hello" in ui.printed[0]

    def test_commands_need_an_open_workspace(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["ls"])
        assert any("no repository open" in m for _, m in ui.msgs)

    def test_branch_creates_and_switches(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        _gh(repl, ["branch", "feature"])
        assert repl.gh.branches == [("feature", "headsha")]
        assert repl.gh_ws["branch"] == "feature"

    def test_pr_opens_from_the_workspace_branch(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        _gh(repl, ["branch", "feature"])
        _gh(repl, ["pr"])
        assert repl.gh.prs[0][:2] == ("main", "feature")
        assert any("pull/9" in m for _, m in ui.msgs)

    def test_pr_on_the_default_branch_refuses(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        _gh(repl, ["pr"])
        assert repl.gh.prs == []
        assert any("default branch" in m for _, m in ui.msgs)

    def test_edit_round_trips_through_the_editor(self, monkeypatch):
        import subprocess

        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        monkeypatch.setenv("EDITOR", "nano")

        def fake_run(args, **kwargs):
            if args and args[0] == "nano":
                with open(args[1], "w", encoding="utf-8") as handle:
                    handle.write("edited!")
            return Proc()

        monkeypatch.setattr(subprocess, "run", fake_run)
        _gh(repl, ["edit", "README.md", "improve it"])
        assert repl.gh.writes[0][0] == "README.md"
        assert repl.gh.writes[0][1] == "edited!"
        assert repl.gh.writes[0][2] == "improve it"

    def test_edit_shows_the_diff_and_refuses_on_no(self, monkeypatch):
        import subprocess

        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        monkeypatch.setenv("EDITOR", "nano")

        def fake_run(args, **kwargs):
            if args and args[0] == "nano":
                with open(args[1], "w", encoding="utf-8") as handle:
                    handle.write("edited!")
            return Proc()

        monkeypatch.setattr(subprocess, "run", fake_run)
        # "no" aborts the commit, but the diff must already have been shown.
        _gh(repl, ["edit", "README.md", "improve it"], reply="n")
        assert any("nothing committed" in m for _, m in ui.msgs)
        assert repl.gh.writes == [], "the 'no' answer must not commit"
        assert ui.diffs and "+edited!" in ui.diffs[0]

        # "yes" goes through.
        _gh(repl, ["edit", "README.md", "improve it"], reply="y")
        assert repl.gh.writes[0][0] == "README.md"
        assert repl.gh.writes[0][1] == "edited!"

    def test_close_drops_the_workspace(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["open", "o/r"])
        _gh(repl, ["close"])
        assert repl.gh_ws is None

    def test_unknown_action_gets_the_menu(self):
        repl, ui = _repl(StubClient())
        _gh(repl, ["frobnicate"])
        assert any("unknown /gh action" in m for _, m in ui.msgs)

    def test_a_failing_gh_call_is_reported(self):
        from wynxo.gh import GitHubError

        class Flaky(StubClient):
            def auth_user(self):
                raise GitHubError("something went wrong")

        repl, ui = _repl(Flaky())
        _gh(repl, ["status"])
        assert any("something went wrong" in m for _, m in ui.msgs)
