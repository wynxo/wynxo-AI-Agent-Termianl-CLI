"""The remote-repository system, against GitHub's real response shapes.

`gh` is replaced with something that answers what github.com answers --
including the fields the old code discarded. Nothing here reaches the
network, but every payload is the shape GitHub documents, because the bugs
this file exists for were all cases of the code looking at the wrong part
of a correct response.
"""

from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import subprocess
import tempfile

import pytest

from wynxo.gh import GitHubClient, GitHubError
from wynxo.tools.github_tool import GitHubRead, GitHubWrite


class Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class FakeGitHub:
    """A repository, and gh's view of it.

    Holds real state so a write can be observed rather than asserted about:
    the tests that matter here are about what happens to the file.
    """

    def __init__(self, files=None, truncated=False, malformed=0):
        self.files = dict(files or {"a.py": ("original line\n", "sha-v1")})
        self.truncated = truncated
        self.malformed = malformed
        self.search_hits: list[str] = []
        self.calls: list[str] = []
        self.commits = 0

    def run(self, args, **kwargs):
        line = " ".join(args)
        self.calls.append(line)
        if "--jq .default_branch" in line:
            return Proc("main\n")
        if "git/trees" in line:
            tree = [{"path": p, "type": "blob", "size": len(c), "sha": s}
                    for p, (c, s) in self.files.items()]
            tree += [{"type": "blob"}] * self.malformed      # no path
            return Proc(json.dumps({"sha": "t", "truncated": self.truncated,
                                    "tree": tree}))
        if "search/code" in line:
            items = [{"path": p, "sha": "s", "html_url": f"http://x/{p}"}
                     for p in self.search_hits]
            return Proc("\n".join(json.dumps(i) for i in items))
        if "contents/" in line and "--method" in args:
            body = json.loads(kwargs.get("input") or "{}")
            path = line.split("contents/")[1].split()[0]
            current = self.files.get(path)
            if current and body.get("sha") != current[1]:
                return Proc(stderr=f"HTTP 409: is at {current[1]}",
                            returncode=1)
            text = base64.b64decode(body["content"]).decode()
            self.commits += 1
            self.files[path] = (text, f"sha-after-{self.commits}")
            return Proc(f"commit-{self.commits}\n")
        if "contents/" in line:
            path = line.split("contents/")[1].split("?")[0]
            if path not in self.files:
                return Proc(stderr="gh: Not Found (HTTP 404)", returncode=1)
            text, sha = self.files[path]
            return Proc(json.dumps({
                "content": base64.b64encode(text.encode()).decode(),
                "sha": sha, "size": len(text), "encoding": "base64",
                "type": "file"}))
        return Proc("{}")


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(subprocess, "run", fake.run)
    return fake


def _tools(client=None):
    workspace = pathlib.Path(tempfile.mkdtemp())
    client = client or GitHubClient()
    return (GitHubRead(workspace, client=client),
            GitHubWrite(workspace, client=client), client)


def _run(tool, **kwargs):
    return asyncio.run(tool.run(tool.Input(**kwargs)))


class TestATruncatedTreeIsNotAWholeRepository:
    """GitHub caps a recursive tree at 100,000 entries or 7MB and then sets
    `truncated: true` beside the entries. The client selected `.tree[]` with
    jq, which threw that flag away -- so on any large repository the agent
    was handed a partial map and told it was the repository. It would then
    conclude a file did not exist because it was not in a list that was
    never complete.
    """

    def test_the_client_reports_truncation(self, github):
        github.truncated = True
        assert GitHubClient().tree("o", "r", "main").truncated is True

    def test_the_model_is_told_in_words(self, github):
        github.truncated = True
        read, _write, _c = _tools()
        result = _run(read, operation="tree", repo="o/r")
        assert "INCOMPLETE" in result.output
        assert "not conclude a file is absent" in result.output

    def test_and_in_the_metadata(self, github):
        github.truncated = True
        read, _write, _c = _tools()
        result = _run(read, operation="tree", repo="o/r")
        assert result.metadata["truncated"] is True
        assert result.metadata["complete"] is False

    def test_a_complete_tree_says_nothing_alarming(self, github):
        read, _write, _c = _tools()
        result = _run(read, operation="tree", repo="o/r")
        assert "INCOMPLETE" not in result.output
        assert result.metadata["complete"] is True

    def test_unusable_entries_are_counted_not_dropped_silently(self, github):
        github.malformed = 3
        read, _write, _c = _tools()
        result = _run(read, operation="tree", repo="o/r")
        assert "3 entr" in result.output
        assert result.metadata["malformed"] == 3


class TestAWriteCannotEraseSomebodyElsesCommit:
    """This is the one that lost work.

    github_write fetched the file's current sha immediately before writing
    and passed that to GitHub. GitHub compares the sha it is given with the
    sha that is there -- so a freshly fetched one always matches, the check
    always passes, and a change committed between the model's read and its
    write was overwritten with no error and a "committed" message. The sha
    is only protection if it is the version the edit was *based on*.
    """

    def _read_then_someone_else_commits(self, github):
        read, write, client = _tools()
        blob = client.read("o", "r", "a.py", "main")
        github.files["a.py"] = ("SOMEBODY ELSE'S FIX\noriginal line\n", "sha-v2")
        return read, write, blob

    def test_the_write_is_refused(self, github):
        _read, write, blob = self._read_then_someone_else_commits(github)
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="a.py", content="edited line\n", message="m",
                      base_sha=blob.sha)
        assert not result.ok
        assert result.metadata["kind"] == "conflict"

    def test_and_the_other_change_survives(self, github):
        _read, write, blob = self._read_then_someone_else_commits(github)
        _run(write, operation="write", repo="o/r", branch="main", path="a.py",
             content="edited line\n", message="m", base_sha=blob.sha)
        assert "SOMEBODY ELSE'S FIX" in github.files["a.py"][0]
        assert github.commits == 0, "nothing should have been committed"

    def test_the_model_is_told_what_to_do_about_it(self, github):
        _read, write, blob = self._read_then_someone_else_commits(github)
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="a.py", content="edited line\n", message="m",
                      base_sha=blob.sha)
        assert "Read it again" in result.output
        assert "Nothing was written" in result.output

    def test_updating_without_a_base_sha_is_refused(self, github):
        """Not defaulted, not fetched: asked for. Filling it in here is
        exactly the bug, and a tool that quietly does the unsafe thing when
        an argument is missing has no protection at all."""
        _read, write, _client = _tools()
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="a.py", content="edited\n", message="m")
        assert not result.ok
        assert result.metadata["kind"] == "needs_base_sha"
        assert github.commits == 0

    def test_a_current_base_sha_commits(self, github):
        read, write, client = _tools()
        blob = client.read("o", "r", "a.py", "main")
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="a.py", content="edited line\n", message="m",
                      base_sha=blob.sha)
        assert result.ok, result.output
        assert github.files["a.py"][0] == "edited line\n"
        assert github.commits == 1

    def test_a_new_file_needs_no_base_sha(self, github):
        _read, write, _client = _tools()
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="new.py", content="x = 1\n", message="add")
        assert result.ok, result.output
        assert github.files["new.py"][0] == "x = 1\n"

    def test_a_base_sha_for_a_file_that_is_gone_is_refused(self, github):
        _read, write, _client = _tools()
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="gone.py", content="x\n", message="m",
                      base_sha="sha-v1")
        assert not result.ok
        assert result.metadata["kind"] == "not_found"

    def test_an_identical_write_is_not_a_commit(self, github):
        read, write, client = _tools()
        blob = client.read("o", "r", "a.py", "main")
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="a.py", content=blob.text, message="m",
                      base_sha=blob.sha)
        assert result.ok
        assert result.metadata["changed"] is False
        assert github.commits == 0


class TestSuccessIsCheckedRatherThanAssumed:
    def test_a_put_with_no_commit_is_not_a_commit(self, github):
        """It reported "committed 'm' on main ()" -- a success message with
        an empty commit in it. The model then moves on believing the work is
        saved."""
        read, write, _client = _tools()

        def no_commit(args, **kwargs):
            if "contents/" in " ".join(args) and "--method" in args:
                return Proc("\n")
            return github.run(args, **kwargs)

        blob = GitHubClient().read("o", "r", "a.py", "main")
        with pytest.MonkeyPatch().context() as patcher:
            patcher.setattr(subprocess, "run", no_commit)
            result = _run(write, operation="write", repo="o/r", branch="main",
                          path="a.py", content="edited\n", message="m",
                          base_sha=blob.sha)
        assert not result.ok
        assert "cannot be confirmed" in result.output

    def test_the_file_is_read_back_after_writing(self, github):
        read, write, client = _tools()
        blob = client.read("o", "r", "a.py", "main")
        before = len([c for c in github.calls if "contents/" in c])
        _run(write, operation="write", repo="o/r", branch="main", path="a.py",
             content="edited line\n", message="m", base_sha=blob.sha)
        after = len([c for c in github.calls if "contents/" in c])
        assert after - before >= 3, "read, write, and read back"

    def test_a_write_that_did_not_take_is_reported_as_failure(self, github):
        read, write, client = _tools()
        blob = client.read("o", "r", "a.py", "main")

        real_run = github.run
        state = {"n": 0}

        def drifting(args, **kwargs):
            result = real_run(args, **kwargs)
            line = " ".join(args)
            if "contents/" in line and "--method" in args:
                # The commit lands, and something else changes it at once.
                github.files["a.py"] = ("A THIRD PARTY\n", "sha-v9")
                state["n"] += 1
            return result

        with pytest.MonkeyPatch().context() as patcher:
            patcher.setattr(subprocess, "run", drifting)
            result = _run(write, operation="write", repo="o/r", branch="main",
                          path="a.py", content="edited line\n", message="m",
                          base_sha=blob.sha)
        assert state["n"] == 1
        assert not result.ok
        assert "does not match what was sent" in result.output

    def test_the_summary_carries_a_real_line_count(self, github):
        read, write, client = _tools()
        blob = client.read("o", "r", "a.py", "main")
        result = _run(write, operation="write", repo="o/r", branch="main",
                      path="a.py", content="one\ntwo\nthree\n", message="m",
                      base_sha=blob.sha)
        assert result.metadata["added"] == 3
        assert result.metadata["removed"] == 1
        assert "+3 -1" in result.output


class TestFindingBeforeReading:
    """The old tool offered two operations: list every file, or download a
    whole one. Answering "where is authentication implemented" meant walking
    the tree and reading files until you found it -- which for a large
    repository does not fit in a context window at all."""

    def test_search_reports_where_a_thing_is(self, github):
        github.search_hits = ["src/auth.py", "src/middleware/session.py"]
        read, _write, _c = _tools()
        result = _run(read, operation="search", repo="o/r", query="authenticate")
        assert "src/auth.py" in result.output
        assert result.metadata["found"] == 2
        assert result.metadata["paths"] == github.search_hits

    def test_search_needs_something_to_search_for(self, github):
        read, _write, _c = _tools()
        assert not _run(read, operation="search", repo="o/r").ok

    def test_no_match_says_what_that_does_and_does_not_prove(self, github):
        """GitHub's code search indexes the default branch and skips very
        large files. Reporting "not found" without that is a wrong answer
        dressed as a fact."""
        read, _write, _c = _tools()
        result = _run(read, operation="search", repo="o/r", query="nothing")
        assert result.ok
        assert "default branch" in result.output
        assert result.metadata["found"] == 0

    def test_a_line_range_returns_only_those_lines(self, github):
        github.files["big.py"] = ("\n".join(f"line {i}" for i in range(1, 501)),
                                  "sha-big")
        read, _write, _c = _tools()
        result = _run(read, operation="read", repo="o/r", path="big.py",
                      start_line=100, end_line=104)
        body = result.output.split("---")[2]
        assert "line 100" in body and "line 104" in body
        assert "line 99" not in body and "line 105" not in body
        assert result.metadata["ranged"] is True

    def test_a_range_says_how_big_the_file_really_is(self, github):
        github.files["big.py"] = ("\n".join(f"line {i}" for i in range(1, 501)),
                                  "sha-big")
        read, _write, _c = _tools()
        result = _run(read, operation="read", repo="o/r", path="big.py",
                      start_line=10, end_line=12)
        assert "of 500" in result.output
        assert result.metadata["lines"] == 500

    def test_a_range_past_the_end_is_an_error_not_an_empty_answer(self, github):
        read, _write, _c = _tools()
        result = _run(read, operation="read", repo="o/r", path="a.py",
                      start_line=9000)
        assert not result.ok
        assert "past the end" in result.output

    def test_reading_reports_the_sha_the_edit_must_be_based_on(self, github):
        read, _write, _c = _tools()
        result = _run(read, operation="read", repo="o/r", path="a.py")
        assert "sha-v1" in result.output
        assert result.metadata["sha"] == "sha-v1"
        assert "github_write" in result.output

    def test_stat_reports_size_without_downloading(self, github):
        github.files["big.py"] = ("x" * 5000, "sha-big")
        read, _write, _c = _tools()
        result = _run(read, operation="stat", repo="o/r", path="big.py")
        assert result.metadata["size"] == 5000
        assert "x" * 100 not in result.output, "stat must not carry content"


class TestKnowingWhatWentWrong:
    """gh reports HTTP failures by echoing the status and GitHub's message,
    which is accurate and says nothing about what to do next. Each of these
    is a different next step."""

    CASES = [
        ("HTTP 403: API rate limit exceeded for user", "rate_limit",
         "rate-limited"),
        ("gh: Not Found (HTTP 404)", "not_found", "could not find"),
        ("HTTP 401: Bad credentials", "auth", "gh auth login"),
        ("HTTP 403: Resource not accessible by integration", "denied",
         "refused this operation"),
        ("HTTP 409: a.py does not match", "conflict", "changed on GitHub"),
        ("HTTP 422: Reference already exists", "malformed", "already exists"),
    ]

    @pytest.mark.parametrize("stderr,kind,phrase", CASES)
    def test_each_failure_is_named_and_explained(self, monkeypatch, stderr,
                                                 kind, phrase):
        monkeypatch.setattr(
            subprocess, "run",
            lambda a, **k: Proc(stderr=stderr, returncode=1))
        with pytest.raises(GitHubError) as caught:
            GitHubClient().repo_default_branch("o", "r")
        assert caught.value.kind == kind
        assert phrase in str(caught.value)

    def test_the_original_message_is_kept_as_well(self, monkeypatch):
        """When the guess above is wrong, the raw line is the only thing
        that says what actually happened."""
        monkeypatch.setattr(
            subprocess, "run",
            lambda a, **k: Proc(stderr="gh: Not Found (HTTP 404)",
                                returncode=1))
        with pytest.raises(GitHubError) as caught:
            GitHubClient().repo_default_branch("o", "r")
        assert "HTTP 404" in str(caught.value)

    def test_a_missing_gh_says_how_to_get_it(self, monkeypatch):
        def missing(*a, **k):
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", missing)
        with pytest.raises(GitHubError) as caught:
            GitHubClient().repo_default_branch("o", "r")
        assert caught.value.kind == "missing_cli"
        assert "cli.github.com" in str(caught.value)

    def test_a_failure_reaches_the_model_with_its_kind(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda a, **k: Proc(stderr="HTTP 403: API rate limit exceeded",
                                returncode=1))
        read, _write, _c = _tools()
        result = _run(read, operation="tree", repo="o/r")
        assert not result.ok
        assert result.metadata["kind"] == "rate_limit"

    def test_a_tree_that_is_not_json_is_an_error_not_an_empty_repository(
            self, monkeypatch):
        """Answering "0 files" for a broken response is how an agent decides
        a repository is empty and starts creating things."""
        def html(args, **kwargs):
            if "--jq .default_branch" in " ".join(args):
                return Proc("main\n")
            return Proc("<html>502 Bad Gateway</html>")

        monkeypatch.setattr(subprocess, "run", html)
        with pytest.raises(GitHubError) as caught:
            GitHubClient().tree("o", "r", "main")
        assert caught.value.kind == "malformed"


class TestNotAskingTwiceForTheSameThing:
    def test_the_default_branch_is_looked_up_once(self, github):
        client = GitHubClient()
        for _ in range(4):
            client.repo_default_branch("o", "r")
        assert len([c for c in github.calls if "default_branch" in c]) == 1

    def test_the_tree_is_fetched_once(self, github):
        client = GitHubClient()
        for _ in range(3):
            client.tree("o", "r", "main")
        assert len([c for c in github.calls if "git/trees" in c]) == 1

    def test_a_write_drops_what_it_invalidated(self, github):
        """A file listing from before the commit describes a repository that
        no longer exists."""
        client = GitHubClient()
        client.tree("o", "r", "main")
        blob = client.read("o", "r", "a.py", "main")
        _read, write, _c = _tools(client)
        _run(write, operation="write", repo="o/r", branch="main", path="a.py",
             content="edited\n", message="m", base_sha=blob.sha)
        client.tree("o", "r", "main")
        assert len([c for c in github.calls if "git/trees" in c]) == 2

    def test_refresh_forces_a_re_fetch(self, github):
        client = GitHubClient()
        read, _write, _c = _tools(client)
        _run(read, operation="tree", repo="o/r")
        _run(read, operation="tree", repo="o/r", refresh=True)
        assert len([c for c in github.calls if "git/trees" in c]) == 2

    def test_forgetting_one_repository_leaves_the_others(self, github):
        client = GitHubClient()
        client.repo_default_branch("o", "r")
        client.repo_default_branch("o", "other")
        client.forget("o", "r")
        client.repo_default_branch("o", "other")
        assert len([c for c in github.calls if "default_branch" in c]) == 2


class TestARemoteFileHitsTheSameWallAsALocalOne:
    """A committed .env is a committed .env whether it is on this disk or on
    GitHub. The local read refused one and the remote read handed it over,
    which put the credential in the model's context and in the transcript by
    the longer route."""

    SECRET = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY\n"
    IN_CODE = "TOKEN = 'ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789'\nx = 1\n"

    def _read(self, github, path, body):
        github.files[path] = (body, "sha-x")
        read, _write, _c = _tools()
        return _run(read, operation="read", repo="o/r", path=path)

    @pytest.mark.parametrize("path", [".env", ".env.local", "id_rsa",
                                      "config/.env.production"])
    def test_a_credentials_file_is_refused(self, github, path):
        result = self._read(github, path, self.SECRET)
        assert not result.ok, f"{path} was handed over"
        assert "wJalrXUtnFEMI" not in result.output
        assert result.metadata["kind"] == "secret"

    def test_the_local_tool_refuses_the_same_name(self, github):
        """The premise: these two must not disagree."""
        import pathlib
        import tempfile

        from wynxo.tools.files import ReadFile

        workspace = pathlib.Path(tempfile.mkdtemp())
        (workspace / ".env").write_text(self.SECRET)
        local = ReadFile(workspace)
        assert not asyncio.run(local.run(local.Input(path=".env"))).ok

    def test_a_credential_inside_an_ordinary_file_is_masked(self, github):
        result = self._read(github, "src/app.py", self.IN_CODE)
        assert result.ok
        assert "ghp_aBcDeFgHiJkLmNoP" not in result.output
        assert result.metadata["masked"] == 1
        assert "x = 1" in result.output, "the rest of the file still arrives"

    def test_an_ordinary_file_is_untouched(self, github):
        result = self._read(github, "README.md", "# hello\nworld\n")
        assert result.ok
        assert "world" in result.output
        assert result.metadata["masked"] == 0
