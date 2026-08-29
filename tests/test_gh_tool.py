"""The agent-side cloud GitHub tools: github_read and github_write.

The tools are exercised against a stub GitHubClient so no gh CLI or network
is involved; what is being tested is the tool contract -- schema validation,
operation dispatch, and how results and failures reach the agent.
"""

from __future__ import annotations

import pytest

from wynxo.gh import GitHubError
from wynxo.tools.github_tool import GitHubRead, GitHubReadInput, \
    GitHubWrite, GitHubWriteInput


class StubClient:
    def __init__(self):
        self.writes = []
        self.branches = []
        self.prs = []

    def repo_default_branch(self, owner, repo):
        return "main"

    def tree(self, owner, repo, branch):
        return [
            {"path": "README.md", "type": "blob", "size": 5},
            {"path": "src", "type": "tree", "size": 0},
            {"path": "src/a.py", "type": "blob", "size": 42},
        ]

    def read(self, owner, repo, path, branch):
        if path == "missing.txt":
            raise GitHubError("path not found")
        return "def hello():\n    return 1\n", "sha1"

    def ref_sha(self, owner, repo, branch):
        return "headsha"

    def create_branch(self, owner, repo, name, sha):
        self.branches.append((name, sha))

    def commits(self, owner, repo, branch):
        return ["feat: x"]

    def open_pr(self, owner, repo, base, head, title, body):
        self.prs.append((base, head, title, body))
        return "https://github.com/o/r/pull/3"

    def write(self, owner, repo, path, content, message, branch, sha=None):
        self.writes.append((path, content, message, branch, sha))
        return "commit123"


def _read(client=None, workspace=None):
    return GitHubRead(workspace, client=client or StubClient())


def _write(client=None, workspace=None):
    return GitHubWrite(workspace, client=client or StubClient())


async def _run(tool, input):
    return await tool.run(input)


class TestGitHubRead:
    @pytest.fixture
    def ws(self, tmp_path):
        return tmp_path

    async def test_tree_lists_files(self, ws):
        result = await _run(_read(workspace=ws),
                            GitHubReadInput(repo="o/r", operation="tree"))
        assert result.ok
        assert "README.md" in result.output
        assert "src/a.py" in result.output
        assert "2 files" in result.output

    async def test_read_returns_decoded_content(self, ws):
        result = await _run(_read(workspace=ws), GitHubReadInput(
            repo="o/r", operation="read", path="src/a.py"))
        assert result.ok
        assert "def hello():" in result.output

    async def test_read_failure_becomes_a_result_not_a_crash(self, ws):
        result = await _run(_read(workspace=ws), GitHubReadInput(
            repo="o/r", operation="read", path="missing.txt"))
        assert not result.ok
        assert "path not found" in result.output

    async def test_repo_is_required(self, ws):
        result = await _run(_read(workspace=ws),
                            GitHubReadInput(repo="not-a-slash"))
        assert not result.ok
        assert "owner/name" in result.output

    def test_never_prompts(self):
        assert GitHubRead.mutating is False


class TestGitHubWrite:
    @pytest.fixture
    def ws(self, tmp_path):
        return tmp_path

    async def test_branch_creates_off_the_default(self, ws):
        client = StubClient()
        result = await _run(_write(client, ws), GitHubWriteInput(
            repo="o/r", operation="branch", branch="feature"))
        assert result.ok
        assert client.branches == [("feature", "headsha")]

    async def test_write_commits_new_content(self, ws):
        client = StubClient()
        result = await _run(_write(client, ws), GitHubWriteInput(
            repo="o/r", operation="write", branch="feature",
            path="src/a.py", content="new body", message="rewrite a.py"))
        assert result.ok
        path, content, message, branch, sha = client.writes[0]
        assert (path, content, message, branch) == \
            ("src/a.py", "new body", "rewrite a.py", "feature")
        assert sha == "sha1"          # existing file -> update, not create
        assert "commit123" in result.output

    async def test_write_a_new_file_has_no_sha(self, ws):
        class NewFileClient(StubClient):
            def read(self, owner, repo, path, branch):
                raise GitHubError("not found")

        client = NewFileClient()
        result = await _run(_write(client, ws), GitHubWriteInput(
            repo="o/r", operation="write", branch="feature",
            path="NEW.txt", content="hi", message="add NEW.txt"))
        assert result.ok
        assert client.writes[0][4] is None

    async def test_write_requires_content(self, ws):
        result = await _run(_write(workspace=ws), GitHubWriteInput(
            repo="o/r", operation="write", branch="feature",
            path="a.txt", content="   ", message="m"))
        assert not result.ok
        assert "empty file" in result.output

    async def test_pr_opens_and_reports_the_url(self, ws):
        client = StubClient()
        result = await _run(_write(client, ws), GitHubWriteInput(
            repo="o/r", operation="pr", branch="feature"))
        assert result.ok
        assert client.prs[0][:2] == ("main", "feature")
        assert "pull/3" in result.output

    def test_writes_prompt(self):
        assert GitHubWrite.mutating is True
