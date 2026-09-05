"""Cloud GitHub operations, as agent tools.

The user's repository lives on GitHub, not on this machine. Nothing is
cloned; the token comes from ``gh auth login`` via the ``gh`` CLI.
"""

from __future__ import annotations

from ..gh import GitHubClient, GitHubError
from ..schema import Field, Schema
from .base import Tool, ToolResult
from pathlib import PurePosixPath


MAX_TREE_LINES = 600


def _split_repo(value: str) -> tuple[str, str]:
    owner, _, repo = (value or "").strip().strip("/").partition("/")
    return owner, repo


def _secret_path(path: str) -> bool:
    from ..secrets import is_secret_file
    return is_secret_file(PurePosixPath((path or "").strip().lstrip("/")))


class GitHubReadInput(Schema):
    operation = Field(str, "What to do. 'search', 'tree', 'stat', or 'read'.",
                      choices=("search", "tree", "stat", "read"), default="search")
    repo = Field(str, "The repository as owner/name.")
    branch = Field(str, "Branch or ref; empty means default.", default="")
    path = Field(str, "Path within the repository.", default="")
    query = Field(str, "Search text for operation 'search'.", default="")
    start_line = Field(int, "First line for 'read' (1-based).", default=0)
    end_line = Field(int, "Last line for 'read' (1-based, inclusive).", default=0)
    refresh = Field(bool, "Re-fetch instead of session cache.", default=False)


class GitHubRead(Tool):
    name = "github_read"
    description = (
        "Look inside a GitHub repository without cloning it, using the account "
        "that ran `gh auth login`. Search first, then read relevant files. "
        "Never writes anything."
    )
    Input = GitHubReadInput

    def __init__(self, workspace, boundary=None, shield=None, client: GitHubClient | None = None):
        super().__init__(workspace, boundary, shield)
        self._client = client or GitHubClient()

    def unavailable(self) -> str:
        if not self._client.available():
            return ("the GitHub CLI `gh` is not installed -- install it "
                    "from https://cli.github.com, then `gh auth login`")
        return ""

    async def run(self, args: GitHubReadInput) -> ToolResult:
        owner, repo = _split_repo(args.repo)
        if not owner or not repo:
            return ToolResult.failure("github_read needs a repository as owner/name.")
        try:
            if args.refresh:
                self._client.forget(owner, repo)
            branch = args.branch or self._client.repo_default_branch(owner, repo)
            if args.operation == "search":
                return self._search(owner, repo, branch, args)
            if args.operation == "tree":
                return self._tree(owner, repo, branch, args)
            if args.operation == "stat":
                return self._stat(owner, repo, branch, args)
            return self._read(owner, repo, branch, args)
        except GitHubError as exc:
            return ToolResult.failure(str(exc), kind=exc.kind, repo=args.repo, operation=args.operation)

    def _search(self, owner, repo, branch, args) -> ToolResult:
        if not args.query.strip():
            return ToolResult.failure("operation 'search' needs a query.")
        hits = self._client.search_code(owner, repo, args.query.strip())
        if not hits:
            return ToolResult.success(
                f"No match for {args.query!r} in {args.repo}.\n"
                "GitHub code search indexes the default branch only and may skip very large files.",
                found=0, query=args.query)
        lines = [f"{len(hits)} file(s) matching {args.query!r} in {args.repo}:"]
        lines += [f"  {hit.get('path')}" for hit in hits]
        lines.append("")
        lines.append("Read the relevant ones with operation 'read'.")
        return ToolResult.success("\n".join(lines), found=len(hits),
                                  paths=[h.get("path") for h in hits], query=args.query)

    def _tree(self, owner, repo, branch, args) -> ToolResult:
        tree = self._client.tree(owner, repo, branch)
        prefix = args.path.strip().strip("/")
        files = [e["path"] for e in tree.files if not prefix or e["path"].startswith(prefix + "/") or e["path"] == prefix]
        dirs = [e["path"] + "/" for e in tree.dirs if not prefix or e["path"].startswith(prefix + "/")]
        listing = dirs + files
        notes: list[str] = []
        if tree.truncated:
            notes.append("INCOMPLETE: GitHub truncated this tree. Do not conclude a file is absent from this listing. Use search or stat.")
        if tree.malformed:
            notes.append(f"{tree.malformed} entries were unusable and omitted; this listing is incomplete.")
        shown = listing[:MAX_TREE_LINES]
        if len(listing) > MAX_TREE_LINES:
            notes.append(f"Showing {MAX_TREE_LINES} of {len(listing)} paths. Narrow with 'path' or use search.")
        body = [f"{args.repo}@{branch}" + (f" under {prefix}/" if prefix else "")]
        if notes:
            body += ["", *notes, ""]
        body += [f"{len(files)} files", *shown]
        complete = not (tree.truncated or tree.malformed or len(listing) > MAX_TREE_LINES)
        return ToolResult.success("\n".join(body), truncated=tree.truncated, complete=complete,
                                  files=len(files), shown=len(shown), malformed=tree.malformed)

    def _stat(self, owner, repo, branch, args) -> ToolResult:
        if not args.path:
            return ToolResult.failure("operation 'stat' needs a path.")
        info = self._client.stat(owner, repo, args.path, branch)
        if info.get("type") == "dir":
            entries = info.get("entries") or []
            lines = [f"{args.path}/ is a directory ({len(entries)} entries):"]
            lines += [f"  {e['path']}{'/' if e['type'] == 'dir' else ''} {e.get('size') or ''}" for e in entries]
            return ToolResult.success("\n".join(lines), kind="dir", entries=len(entries))
        return ToolResult.success(
            f"{info['path']}  {info.get('size', 0)} bytes  blob {info.get('sha', '')[:12]}",
            kind="file", size=info.get("size", 0), sha=info.get("sha", ""))

    def _read(self, owner, repo, branch, args) -> ToolResult:
        if not args.path:
            return ToolResult.failure("operation 'read' needs a path.")
        if _secret_path(args.path):
            from ..secrets import refusal
            return ToolResult.failure(refusal(args.path), kind="secret")
        blob = self._client.read(owner, repo, args.path, branch,
                                 start=max(0, args.start_line), end=max(0, args.end_line))
        text, masked = self.shield.clean(blob.text)
        where = f"{args.repo}:{args.path} ({branch})"
        if blob.ranged:
            where += f" lines {blob.start}-{blob.end} of {blob.total_lines}"
        note = f"\n[{masked} credential(s) masked]" if masked else ""
        return ToolResult.success(
            f"--- {where} ---\n{text}\n--- blob sha: {blob.sha} (pass this to github_write) ---{note}",
            sha=blob.sha, path=args.path, branch=branch, lines=blob.total_lines,
            ranged=blob.ranged, masked=masked)


class GitHubWriteInput(Schema):
    operation = Field(str, "What to do: write, branch, or pr.",
                      choices=("write", "branch", "pr"), default="write")
    repo = Field(str, "The repository as owner/name.")
    branch = Field(str, "Branch to act on.", default="")
    base = Field(str, "Base branch for branch/pr.", default="")
    path = Field(str, "File path to write.", default="")
    content = Field(str, "Complete new file content.", default="")
    base_sha = Field(str, "Blob SHA from github_read for an existing file.", default="")
    message = Field(str, "Commit message.", default="")
    title = Field(str, "Pull request title.", default="")
    body = Field(str, "Pull request body.", default="")


class GitHubWrite(Tool):
    name = "github_write"
    description = (
        "Change a GitHub repository through the API. For file writes, read the "
        "file first and pass the blob sha as base_sha. Branch creation and pull "
        "requests are also supported. This is a remote mutation and requires "
        "explicit permission even in AUTO/REVIEW mode."
    )
    Input = GitHubWriteInput
    mutating = True
    concurrency_safe = False

    def __init__(self, workspace, boundary=None, shield=None, client: GitHubClient | None = None):
        super().__init__(workspace, boundary, shield)
        self._client = client or GitHubClient()

    def unavailable(self) -> str:
        if not self._client.available():
            return ("the GitHub CLI `gh` is not installed -- install it "
                    "from https://cli.github.com, then `gh auth login`")
        return ""

    async def run(self, args: GitHubWriteInput) -> ToolResult:
        owner, repo = _split_repo(args.repo)
        if not owner or not repo:
            return ToolResult.failure("github_write needs a repository as owner/name.")
        try:
            default = self._client.repo_default_branch(owner, repo)
            if args.operation == "branch":
                return self._branch(owner, repo, default, args)
            if args.operation == "pr":
                return self._pr(owner, repo, default, args)
            return self._write(owner, repo, args)
        except GitHubError as exc:
            return ToolResult.failure(str(exc), kind=exc.kind, repo=args.repo, operation=args.operation)

    def _branch(self, owner, repo, default, args) -> ToolResult:
        if not args.branch:
            return ToolResult.failure("operation 'branch' needs a name for the new branch.")
        from_branch = args.base or default
        head = self._client.ref_sha(owner, repo, from_branch)
        self._client.create_branch(owner, repo, args.branch, head)
        return ToolResult.success(f"created branch {args.branch} in {args.repo}, from {from_branch} at {head[:10]}.",
                                  branch=args.branch, base=from_branch, sha=head)

    def _pr(self, owner, repo, default, args) -> ToolResult:
        if not args.branch:
            return ToolResult.failure("operation 'pr' needs the head branch to open from.")
        base = args.base or default
        title = args.title or f"wynxo: changes on {args.branch}"
        body = args.body or self._pr_body(owner, repo, args.branch, base)
        url = self._client.open_pr(owner, repo, base, args.branch, title, body)
        if not url.strip():
            return ToolResult.failure("GitHub did not return a pull request URL, so it is not confirmed that one was opened.")
        return ToolResult.success(f"opened {url}", display=f"PR: {url}", url=url.strip(), head=args.branch, base=base)

    def _write(self, owner, repo, args) -> ToolResult:
        if not args.path or not args.branch:
            return ToolResult.failure("operation 'write' needs a path, content, commit message and branch.")
        if not args.content.strip():
            return ToolResult.failure("refusing to write an empty file.")
        if _secret_path(args.path):
            return ToolResult.failure(
                f"{args.path} is treated as a credential-bearing path and may not be changed through github_write. "
                "Use a normal non-secret file and keep credentials out of the repository.",
                kind="secret_path", path=args.path)
        try:
            current = self._client.read(owner, repo, args.path, args.branch)
        except GitHubError as exc:
            if exc.kind != "not_found":
                raise
            current = None

        if current is None:
            if args.base_sha:
                return ToolResult.failure(
                    f"{args.path} does not exist on {args.branch}, but a base_sha was given. Read the path again.",
                    kind="not_found")
        else:
            if not args.base_sha:
                return ToolResult.failure(
                    f"{args.path} already exists on {args.branch}. Updating it needs base_sha from github_read; read it again rather than passing the current sha blindly.",
                    kind="needs_base_sha", sha=current.sha)
            if args.base_sha != current.sha:
                return ToolResult.failure(
                    f"{args.path} changed on {args.branch} since it was read: expected {args.base_sha[:12]}, found {current.sha[:12]}. Nothing was written. Read it again and reapply your changes to the current content.",
                    kind="conflict", expected=args.base_sha, actual=current.sha, path=args.path)
            if current.text == args.content:
                return ToolResult.success(f"{args.path} on {args.branch} already has exactly this content; nothing to commit.",
                                          changed=False, path=args.path, sha=current.sha)

        message = args.message or f"wynxo: update {args.path}"
        commit = self._client.write(owner, repo, args.path, args.content, message, args.branch, sha=args.base_sha or None)
        verified, note = self._verify(owner, repo, args)
        added, removed = _line_delta(current.text if current is not None else "", args.content)
        summary = f"committed {message!r} to {args.repo}:{args.path} on {args.branch} ({commit[:10]}) +{added} -{removed}"
        if not verified:
            return ToolResult.failure(f"{summary}\n\nBut the result could not be confirmed: {note} Check the repository before relying on this.",
                                      kind="unverified", commit=commit, path=args.path)
        return ToolResult.success(summary, display=f"{args.repo}:{args.path} +{added} -{removed}",
                                  commit=commit, path=args.path, branch=args.branch,
                                  added=added, removed=removed, sha=note)

    def _verify(self, owner, repo, args) -> tuple[bool, str]:
        try:
            after = self._client.read(owner, repo, args.path, args.branch)
        except GitHubError as exc:
            return False, f"reading it back failed: {exc}"
        if after.text != args.content:
            return False, "the file on GitHub does not match what was sent."
        return True, after.sha

    def _pr_body(self, owner: str, repo: str, branch: str, base: str) -> str:
        try:
            messages = self._client.commits(owner, repo, branch)
        except GitHubError:
            messages = []
        lines = [f"Changes on `{branch}` → `{base}`:"]
        lines += [f"- {m.splitlines()[0]}" for m in messages if m.splitlines()]
        return "\n".join(lines) or f"Changes on `{branch}`."


def _line_delta(before: str, after: str) -> tuple[int, int]:
    import difflib
    added = removed = 0
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed
