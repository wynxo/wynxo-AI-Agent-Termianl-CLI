"""Committing to GitHub without a checkout.

The contents API can only touch one file per commit, so a change spanning
four files landed as four commits -- three of them states nobody wrote,
where half the change exists, all of them in the history and in the pull
request. This drives the Git Data API instead: blobs, one tree, one commit,
one ref update. Every file lands together or none does.

Everything here runs against a fake `gh`, so no network and no real
repository is involved.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from wynxo.gh import Blob, Change, GitHubClient, GitHubError
from wynxo.scope import Boundary, Scope
from wynxo.tools.github_tool import GitHubWrite, GitHubWriteInput


class Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class FakeApi:
    """Enough of GitHub's Git Data API to build a commit against."""

    def __init__(self, modes=None, truncated=False):
        self.modes = modes or {"a.py": "100644"}
        self.truncated = truncated
        self.blobs: dict[str, str] = {}
        self.tree_request = None
        self.commit_request = None
        self.ref_request = None
        self.not_fast_forward = False
        self.n = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", self.run)
        return self

    def run(self, argv, **kwargs):
        body = json.loads(kwargs["input"]) if kwargs.get("input") else None
        path = self._path(argv)
        if path.endswith("/git/blobs"):
            self.n += 1
            self.blobs[f"blob{self.n}"] = body["content"]
            return Proc(f"blob{self.n}\n")
        if "/git/commits/" in path:
            return Proc("basetree\n")
        if path.endswith("/git/trees"):
            self.tree_request = body
            return Proc("newtree\n")
        if path.endswith("/git/commits"):
            self.commit_request = body
            return Proc("newcommit\n")
        if "/git/refs/heads/" in path:
            self.ref_request = body
            if self.not_fast_forward:
                return Proc(stderr="gh: Update is not a fast forward (HTTP 422)",
                            returncode=1)
            return Proc("newcommit\n")
        if "recursive=1" in path:
            return Proc(json.dumps({
                "sha": "t", "truncated": self.truncated,
                "tree": [{"path": p, "type": "blob", "mode": m}
                         for p, m in self.modes.items()]}))
        return Proc("{}\n")

    TAKES_A_VALUE = {"--method", "--jq", "--input", "-X", "-f", "-F", "-H"}

    @classmethod
    def _path(cls, argv):
        """The endpoint out of a `gh api ...` argv.

        Flags and their values are skipped rather than just flags: the
        first attempt took the first token without a dash, which for
        `--method POST repos/...` is "POST".
        """
        argv = list(argv)
        if "api" not in argv:
            return ""
        rest = argv[argv.index("api") + 1:]
        i = 0
        while i < len(rest):
            token = rest[i]
            if token in cls.TAKES_A_VALUE:
                i += 2
                continue
            if token.startswith("-"):
                i += 1
                continue
            return token
        return ""

    @property
    def entries(self):
        return {e["path"]: e for e in (self.tree_request or {}).get("tree", [])}


@pytest.fixture
def api(monkeypatch):
    return FakeApi().install(monkeypatch)


class TestOneCommitManyFiles:
    def test_every_file_is_in_the_same_commit(self, api):
        sha = GitHubClient().commit(
            "o", "r", "main",
            [Change("a.py", "one\n"), Change("b.py", "two\n")],
            "both at once", "parent1")
        assert sha == "newcommit"
        assert set(api.entries) == {"a.py", "b.py"}
        assert api.commit_request["parents"] == ["parent1"]

    def test_it_builds_on_the_parent_tree(self, api):
        """Otherwise the commit holds only the files it touched, and every
        other file in the repository is deleted."""
        GitHubClient().commit("o", "r", "main", [Change("a.py", "x\n")],
                              "m", "parent1")
        assert api.tree_request["base_tree"] == "basetree"

    def test_a_deletion_is_a_null_sha(self, api):
        GitHubClient().commit("o", "r", "main", [Change("gone.txt", None)],
                              "remove it", "parent1")
        assert api.entries["gone.txt"]["sha"] is None

    def test_an_executable_keeps_its_bit(self, monkeypatch):
        """Rebuilding every entry as a plain 100644 turns a script into a
        file that no longer runs, and nothing about the diff says so."""
        api = FakeApi(modes={"run.sh": "100755"}).install(monkeypatch)
        GitHubClient().commit("o", "r", "main",
                              [Change("run.sh", "#!/bin/sh\necho hi\n")],
                              "m", "parent1")
        assert api.entries["run.sh"]["mode"] == "100755"

    def test_a_new_file_gets_the_ordinary_mode(self, api):
        GitHubClient().commit("o", "r", "main", [Change("new.py", "x\n")],
                              "m", "parent1")
        assert api.entries["new.py"]["mode"] == "100644"

    def test_a_symlink_is_refused(self, monkeypatch):
        """Writing text over a 120000 entry replaces the link with a file
        of that name -- a change nobody asked for, in a commit about
        something else."""
        FakeApi(modes={"link": "120000"}).install(monkeypatch)
        with pytest.raises(GitHubError, match="symlink"):
            GitHubClient().commit("o", "r", "main", [Change("link", "x\n")],
                                  "m", "parent1")

    def test_a_submodule_is_refused(self, monkeypatch):
        FakeApi(modes={"vendor": "160000"}).install(monkeypatch)
        with pytest.raises(GitHubError, match="submodule"):
            GitHubClient().commit("o", "r", "main", [Change("vendor", "x\n")],
                                  "m", "parent1")

    def test_content_goes_up_as_base64(self, api):
        """A file holding a BOM, CRLF or a lone surrogate survives base64
        unchanged; JSON has nowhere to put the last of those at all."""
        GitHubClient().commit("o", "r", "main",
                              [Change("w.txt", "line\r\n﻿")], "m", "p")
        import base64
        raw, = api.blobs.values()
        assert base64.b64decode(raw).decode("utf-8") == "line\r\n﻿"

    def test_a_repository_too_large_to_list_still_commits(self, monkeypatch):
        """Modes are best-effort: a truncated tree means a path may be
        missing from the listing, and a new file's default mode is what it
        would have got anyway."""
        api = FakeApi(modes={}, truncated=True).install(monkeypatch)
        GitHubClient().commit("o", "r", "main", [Change("a.py", "x\n")],
                              "m", "p")
        assert api.entries["a.py"]["mode"] == "100644"


class TestSomebodyElsesPushIsNotErased:
    def test_the_ref_is_updated_without_force(self, api):
        GitHubClient().commit("o", "r", "main", [Change("a.py", "x\n")],
                              "m", "parent1")
        assert api.ref_request["force"] is False

    def test_a_branch_that_moved_is_refused(self, api):
        api.not_fast_forward = True
        with pytest.raises(GitHubError) as caught:
            GitHubClient().commit("o", "r", "main", [Change("a.py", "x\n")],
                                  "m", "parent1")
        assert caught.value.kind == "conflict"
        assert "moved on GitHub" in str(caught.value)

    def test_nothing_reaches_a_branch_before_the_ref_moves(self, api):
        """The blobs, the tree and the commit are all unreachable until the
        ref update, so a refusal leaves the repository exactly as it was
        rather than half-changed."""
        api.not_fast_forward = True
        with pytest.raises(GitHubError):
            GitHubClient().commit("o", "r", "main", [Change("a.py", "x\n")],
                                  "m", "parent1")
        assert api.ref_request is not None, "it did try"
        # And the attempt was the last thing it did.
        assert api.commit_request["parents"] == ["parent1"]

    def test_an_empty_commit_is_refused(self, api):
        with pytest.raises(GitHubError, match="nothing to commit"):
            GitHubClient().commit("o", "r", "main", [], "m", "p")

    def test_a_commit_needs_a_message(self, api):
        with pytest.raises(GitHubError, match="needs a message"):
            GitHubClient().commit("o", "r", "main", [Change("a.py", "x\n")],
                                  "   ", "p")


# -- the tool on top of it ---------------------------------------------------

FILE = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"


class Client(GitHubClient):
    """The client surface github_write touches, recording what it is told."""

    def __init__(self, text=FILE):
        self.text = text
        self.commits_made: list[tuple] = []
        self.writes: list[tuple] = []

    def available(self):
        return True

    def repo_default_branch(self, owner, repo):
        return "main"

    def ref_sha(self, owner, repo, branch):
        return "head999"

    def read(self, owner, repo, path, branch, start=0, end=0):
        return Blob(text=self.text, sha="blob1", size=len(self.text),
                    path=path, total_lines=self.text.count("\n"))

    def commit(self, owner, repo, branch, changes, message, parent):
        self.commits_made.append((branch, [(c.path, c.deletes) for c in changes],
                                  message, parent))
        return "commitsha1"

    def write(self, owner, repo, path, content, message, branch, sha=None):
        self.writes.append((path, content, message, branch, sha))
        self.text = content
        return "commitsha2"


def _run(client, **kw):
    ws = Path("/tmp")
    tool = GitHubWrite(ws, Boundary(Scope.REPO, ws), client=client)
    return asyncio.run(tool.run(GitHubWriteInput(repo="o/r", **kw)))


class TestTheCommitOperation:
    def test_several_files_reach_the_client_as_one_commit(self):
        client = Client()
        out = _run(client, operation="commit", parent="head999", message="m",
                   files=[{"path": "a.py", "content": "x\n"},
                          {"path": "b.py", "content": "y\n"}])
        assert out.ok
        branch, paths, message, parent = client.commits_made[0]
        assert paths == [("a.py", False), ("b.py", False)]
        assert parent == "head999"

    def test_a_delete_travels_with_the_writes(self):
        client = Client()
        _run(client, operation="commit", parent="h", message="m",
             files=[{"path": "a.py", "content": "x\n"},
                    {"path": "gone.txt", "delete": True}])
        _branch, paths, _m, _p = client.commits_made[0]
        assert ("gone.txt", True) in paths

    def test_it_will_not_commit_without_a_parent(self):
        """The parent is the whole protection. Fetching one here would
        satisfy the check by construction and protect nothing, so it
        refuses and reports the head instead."""
        client = Client()
        out = _run(client, operation="commit", message="m",
                   files=[{"path": "a.py", "content": "x\n"}])
        assert not out.ok and not client.commits_made
        assert out.metadata["sha"] == "head999"
        assert "read them again rather than passing it blind" in out.error

    def test_the_same_path_twice_is_refused(self):
        """A tree with two answers for one question. GitHub takes the last
        one silently, which is not a thing to guess at."""
        client = Client()
        out = _run(client, operation="commit", parent="h", message="m",
                   files=[{"path": "a.py", "content": "x\n"},
                          {"path": "a.py", "content": "y\n"}])
        assert not out.ok and not client.commits_made
        assert "appears twice" in out.error

    def test_an_empty_file_is_refused_rather_than_guessed_at(self):
        client = Client()
        out = _run(client, operation="commit", parent="h", message="m",
                   files=[{"path": "a.py", "content": ""}])
        assert not out.ok and "delete=true" in out.error

    def test_a_missing_message_names_the_files(self):
        """"wynxo: update" is a commit nobody can find again."""
        client = Client()
        _run(client, operation="commit", parent="h",
             files=[{"path": "a.py", "content": "x\n"},
                    {"path": "b.py", "content": "y\n"}])
        _b, _p, message, _parent = client.commits_made[0]
        assert "a.py" in message and "b.py" in message


class TestTheEditOperation:
    def test_it_replaces_exact_text_and_commits(self):
        client = Client()
        out = _run(client, operation="edit", path="m.py",
                   old_text="return 1", new_text="return 42")
        assert out.ok
        path, content, _m, _b, sha = client.writes[0]
        assert content == FILE.replace("return 1", "return 42")
        assert sha == "blob1", "committed against the blob it read"

    def test_the_rest_of_the_file_is_untouched(self):
        """The point of editing rather than writing: nothing outside
        old_text can be dropped, because nothing else was resent."""
        client = Client()
        _run(client, operation="edit", path="m.py",
             old_text="return 1", new_text="return 42")
        _p, content, _m, _b, _s = client.writes[0]
        assert "def g():" in content and "return 2" in content

    def test_ambiguous_text_is_refused(self):
        client = Client()
        out = _run(client, operation="edit", path="m.py",
                   old_text="    return", new_text="    yield")
        assert not out.ok and not client.writes
        assert out.metadata["occurrences"] == 2

    def test_text_that_is_not_there_is_refused(self):
        """This is also the drift check. If somebody's commit moved or
        removed that text, the match fails and nothing is written -- the
        anchor is the content, which is a statement about the version being
        edited in a way a re-fetched sha is not."""
        client = Client()
        out = _run(client, operation="edit", path="m.py",
                   old_text="return 999", new_text="x")
        assert not out.ok and not client.writes
        assert out.metadata["kind"] == "no_match"

    def test_an_identical_replacement_is_refused(self):
        client = Client()
        out = _run(client, operation="edit", path="m.py",
                   old_text="return 1", new_text="return 1")
        assert not out.ok and not client.writes

    def test_editing_a_file_that_does_not_exist_says_so(self):
        class Missing(Client):
            def read(self, *a, **k):
                raise GitHubError("nope", "not_found")

        out = _run(Missing(), operation="edit", path="new.py",
                   old_text="a", new_text="b")
        assert not out.ok and "Use 'write' to create it" in out.error


class TestASecretIsNotPublished:
    """Reading a committed secret is the smaller half. This is the tool
    that puts one *into* a repository, from a machine with no checkout to
    notice, and it had no check at all."""

    @pytest.mark.parametrize("path", [".env", "app/.env.local", "deploy/id_rsa"])
    @pytest.mark.parametrize("operation", ["commit", "edit", "write"])
    def test_it_is_refused_on_every_path_that_commits(self, path, operation):
        client = Client(text="KEY=1\n")
        kw = {
            "commit": dict(operation="commit", parent="h", message="m",
                           files=[{"path": path, "content": "K=1\n"}]),
            "edit": dict(operation="edit", path=path, old_text="KEY=1",
                         new_text="KEY=2"),
            "write": dict(operation="write", path=path, branch="main",
                          content="K=1\n", message="m"),
        }[operation]
        out = _run(client, **kw)
        assert not out.ok, f"{operation} committed {path}"
        assert not client.commits_made and not client.writes

    def test_the_reason_describes_what_actually_happened(self):
        """The reader's wording says the file was not read, which on this
        side would be describing something that did not happen -- and its
        advice is for somebody trying to read a value, not for somebody
        about to publish one."""
        out = _run(Client(), operation="commit", parent="h", message="m",
                   files=[{"path": ".env", "content": "K=1\n"}])
        assert "did not commit it" in out.error
        assert "publishes it" in out.error

    def test_an_ordinary_path_still_goes_through(self):
        client = Client()
        out = _run(client, operation="commit", parent="h", message="m",
                   files=[{"path": "src/env_loader.py", "content": "x\n"}])
        assert out.ok and client.commits_made
