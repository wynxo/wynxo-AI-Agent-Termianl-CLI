"""Cloud GitHub operations, as agent tools.

The user's repository lives on GitHub, not on this machine. ``github_read``
browses a repo's tree and returns file contents through the API, and
``github_write`` creates or updates files, branches and pull requests --
with the same permission prompt as a local write. Nothing is cloned; the
token comes from ``gh auth login`` via the ``gh`` CLI. This is what makes
\"edit the repo in the cloud\" possible for the model, on top of the /gh
commands a person can drive by hand.
"""

from __future__ import annotations

from ..gh import GitHubClient, GitHubError
from ..schema import Field, Schema
from .base import Tool, ToolResult


class GitHubReadInput(Schema):
    operation = Field(
        str,
        "What to do: 'tree' lists the repository's files at a branch; "
        "'read' returns one file's decoded text.",
        choices=("tree", "read"), default="tree")
    repo = Field(
        str,
        "The repository as owner/name, e.g. 'wynxo/wynxo-AI-Agent-Termianl-CLI'.")
    branch = Field(
        str, "Branch or ref to look at; leave empty for the default branch.",
        default="")
    path = Field(
        str, "Path within the repository. Required for 'read'.", default="")


class GitHubRead(Tool):
    name = "github_read"
    description = (
        "Browse a GitHub repository without a local checkout, using the "
        "account that ran `gh auth login`. With operation 'tree' it lists "
        "the files at a branch (the first argument to give the model a map "
        "of an unfamiliar repo). With operation 'read' it returns the "
        "decoded text of one file, which is the only way the model can see "
        "cloud code before editing it. Provide repo as owner/name and an "
        "optional branch. This tool never writes anything."
    )
    Input = GitHubReadInput

    def __init__(self, workspace, boundary=None, shield=None,
                 client: GitHubClient | None = None):
        super().__init__(workspace, boundary, shield)
        self._client = client or GitHubClient()

    async def run(self, args: GitHubReadInput) -> ToolResult:
        if not args.repo or "/" not in args.repo:
            return ToolResult.failure(
                "github_read needs a repository as owner/name, e.g. "
                "'wynxo/wynxo-AI-Agent-Termianl-CLI'.")
        owner, repo = args.repo.strip().split("/", 1)
        try:
            branch = args.branch or self._client.repo_default_branch(owner, repo)
            if args.operation == "tree":
                entries = self._client.tree(owner, repo, branch)
                if not entries:
                    return ToolResult.success(
                        f"{args.repo}@{branch}: no files found.")
                dirs = [e["path"] + "/" for e in entries
                        if e.get("type") == "tree"]
                files = [e["path"] for e in entries
                         if e.get("type") == "blob"]
                lines = dirs + files
                return ToolResult.success(
                    f"{args.repo}@{branch} ({len(files)} files):\n"
                    + "\n".join(lines))
            if not args.path:
                return ToolResult.failure(
                    "github_read operation 'read' needs a path.")
            content, _sha = self._client.read(owner, repo, args.path, branch)
            return ToolResult.success(
                f"--- {args.repo}:{args.path} ({branch}) ---\n{content}")
        except GitHubError as exc:
            return ToolResult.failure(str(exc))


class GitHubWriteInput(Schema):
    operation = Field(
        str,
        "What to do: 'write' creates or updates one file (with content and "
        "a commit message); 'branch' creates a feature branch; 'pr' opens a "
        "pull request from the branch to the base.",
        choices=("write", "branch", "pr"), default="write")
    repo = Field(
        str,
        "The repository as owner/name, e.g. 'wynxo/wynxo-AI-Agent-Termianl-CLI'.")
    branch = Field(
        str,
        "The branch to act on: write commits to this branch, 'branch' names "
        "the new branch to create, 'pr' is the head branch. For 'write' and "
        "'pr' this is required and should be a feature branch, not the "
        "default.",
        default="")
    base = Field(
        str, "The branch a pull request targets; empty means the default.",
        default="")
    path = Field(
        str, "File path to write (relative to the repo root), for 'write'.",
        default="")
    content = Field(
        str, "The complete new file content, for 'write'.", default="")
    message = Field(
        str, "Commit message for 'write'.", default="")
    title = Field(
        str, "Pull request title for 'pr'; a sensible default is used if empty.",
        default="")
    body = Field(
        str, "Pull request body for 'pr'; empty means the commit messages.",
        default="")


class GitHubWrite(Tool):
    name = "github_write"
    description = (
        "Edit a GitHub repository in the cloud, committing through the "
        "API, using the account that ran `gh auth login`. Nothing is cloned "
        "and no local files are touched. Workflow: first operation 'branch' "
        "to create a feature branch off the default, then one or more "
        "operation 'write' calls (each needs path, the complete new content, "
        "a commit message, and the branch), then operation 'pr' to open a "
        "pull request from the branch to the default. Each write is its own "
        "commit. Writes prompt for approval like any other file change."
    )
    Input = GitHubWriteInput
    mutating = True
    concurrency_safe = False

    def __init__(self, workspace, boundary=None, shield=None,
                 client: GitHubClient | None = None):
        super().__init__(workspace, boundary, shield)
        self._client = client or GitHubClient()

    async def run(self, args: GitHubWriteInput) -> ToolResult:
        if not args.repo or "/" not in args.repo:
            return ToolResult.failure(
                "github_write needs a repository as owner/name.")
        owner, repo = args.repo.strip().split("/", 1)
        try:
            default = self._client.repo_default_branch(owner, repo)
            if args.operation == "branch":
                if not args.branch:
                    return ToolResult.failure(
                        "operation 'branch' needs a branch name to create.")
                head = self._client.ref_sha(owner, repo, default)
                self._client.create_branch(owner, repo, args.branch, head)
                return ToolResult.success(
                    f"created branch {args.branch} (from {default}) "
                    f"in {args.repo}.")
            if args.operation == "pr":
                if not args.branch:
                    return ToolResult.failure(
                        "operation 'pr' needs the head branch to open from.")
                base = args.base or default
                title = args.title or f"wynxo: changes on {args.branch}"
                body = args.body or self._pr_body(
                    owner, repo, args.branch, base)
                url = self._client.open_pr(owner, repo, base, args.branch,
                                           title, body)
                return ToolResult.success(
                    f"opened {url}", display=f"PR: {url}")
            # write
            if not args.path or not args.branch:
                return ToolResult.failure(
                    "operation 'write' needs a path, the new content, a "
                    "commit message and a branch.")
            if not args.content.strip():
                return ToolResult.failure("refusing to write an empty file.")
            message = args.message or f"wynxo: update {args.path}"
            try:
                _current, sha = self._client.read(
                    owner, repo, args.path, args.branch)
            except GitHubError:
                sha = None  # new file
            commit = self._client.write(
                owner, repo, args.path, args.content, message,
                args.branch, sha=sha)
            return ToolResult.success(
                f"committed {message!r} on {args.branch} ({commit[:10]}) "
                f"in {args.repo}:{args.path}")
        except GitHubError as exc:
            return ToolResult.failure(str(exc))

    def _pr_body(self, owner: str, repo: str, branch: str,
                 base: str) -> str:
        try:
            messages = self._client.commits(owner, repo, branch)
        except GitHubError:
            messages = []
        lines = [f"Changes on `{branch}` → `{base}`:"]
        lines += [f"- {m.splitlines()[0]}" for m in messages
                  if m.splitlines()]
        return "\n".join(lines) or f"Changes on `{branch}`."
