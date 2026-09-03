"""Cloud GitHub operations, as agent tools.

The user's repository lives on GitHub, not on this machine. Nothing is
cloned; the token comes from ``gh auth login`` via the ``gh`` CLI.

The shape of these two tools is the shape of the workflow they are for:

    search  ->  where is this thing?
    tree    ->  what is in this repository?
    stat    ->  what is at this path, and how big?
    read    ->  the part of that file that matters
    write   ->  change it, based on what was read

The point of the first four is that the last one should never need a
whole repository in context to happen.
"""

from __future__ import annotations

from ..gh import GitHubClient, GitHubError
from ..schema import Field, Schema
from .base import Tool, ToolResult

MAX_TREE_LINES = 600
"""How much of a file listing to put in one answer. Past this the model is
reading a phone book, and the paths it needs are better found by search."""


def _split_repo(value: str) -> tuple[str, str]:
    owner, _, repo = (value or "").strip().strip("/").partition("/")
    return owner, repo


class GitHubReadInput(Schema):
    operation = Field(
        str,
        "What to do. 'search' finds where text appears in the repository and "
        "is the right first step in an unfamiliar one. 'tree' lists files. "
        "'stat' reports one path's type and size without downloading it. "
        "'read' returns a file's text, optionally just a line range.",
        choices=("search", "tree", "stat", "read"), default="search")
    repo = Field(
        str,
        "The repository as owner/name, e.g. 'wynxo/wynxo-AI-Agent-Termianl-CLI'.")
    branch = Field(
        str, "Branch or ref; empty means the repository's default branch.",
        default="")
    path = Field(
        str, "Path within the repository. Required for 'read' and 'stat'; "
        "for 'tree' it limits the listing to that subtree.", default="")
    query = Field(
        str, "What to search for, for 'search'. Plain text or GitHub code "
        "search syntax.", default="")
    start_line = Field(
        int, "First line to return for 'read' (1-based). 0 means the "
        "beginning.", default=0)
    end_line = Field(
        int, "Last line to return for 'read' (1-based, inclusive). 0 means "
        "the end of the file.", default=0)
    refresh = Field(
        bool, "Re-fetch instead of using what was already looked up this "
        "session.", default=False)


class GitHubRead(Tool):
    name = "github_read"
    description = (
        "Look inside a GitHub repository without cloning it, using the "
        "account that ran `gh auth login`. Start with operation 'search' to "
        "find which files matter, then 'read' those files -- with "
        "start_line/end_line when only part of one is relevant. 'tree' lists "
        "the files and 'stat' reports a path's size and type without "
        "downloading it. Never writes anything."
    )
    Input = GitHubReadInput

    def __init__(self, workspace, boundary=None, shield=None,
                 client: GitHubClient | None = None):
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
            return ToolResult.failure(
                "github_read needs a repository as owner/name, e.g. "
                "'wynxo/wynxo-AI-Agent-Termianl-CLI'.")
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
            return ToolResult.failure(str(exc), kind=exc.kind,
                                      repo=args.repo, operation=args.operation)

    def _refuse_secret(self, path: str) -> str:
        """Why this remote path may not be read, or "" if it may.

        Asked of the path alone. There is no local file to consult, and the
        name is what the local shield keys on anyway -- .env is .env
        wherever it lives.
        """
        from pathlib import PurePosixPath

        from ..secrets import is_secret_file, refusal

        clean = (path or "").strip().lstrip("/")
        return refusal(clean) if is_secret_file(PurePosixPath(clean)) else ""

    # -- the operations ----------------------------------------------------

    def _search(self, owner, repo, branch, args) -> ToolResult:
        if not args.query.strip():
            return ToolResult.failure(
                "operation 'search' needs a query: the text, symbol or "
                "phrase to look for.")
        hits = self._client.search_code(owner, repo, args.query.strip())
        if not hits:
            return ToolResult.success(
                f"No match for {args.query!r} in {args.repo}.\n"
                "GitHub's code search indexes the default branch only, and "
                "skips very large files -- so this is evidence of absence "
                "there, not everywhere.",
                found=0, query=args.query)
        lines = [f"{len(hits)} file(s) matching {args.query!r} in {args.repo}:"]
        lines += [f"  {hit.get('path')}" for hit in hits]
        lines.append("")
        lines.append("Read the relevant ones with operation 'read' -- with a "
                     "line range where the file is large.")
        return ToolResult.success("\n".join(lines), found=len(hits),
                                  paths=[h.get("path") for h in hits],
                                  query=args.query)

    def _tree(self, owner, repo, branch, args) -> ToolResult:
        tree = self._client.tree(owner, repo, branch)
        prefix = args.path.strip().strip("/")
        files = [e["path"] for e in tree.files
                 if not prefix or e["path"].startswith(prefix + "/")
                 or e["path"] == prefix]
        dirs = [e["path"] + "/" for e in tree.dirs
                if not prefix or e["path"].startswith(prefix + "/")]
        listing = dirs + files
        header = f"{args.repo}@{branch}"
        if prefix:
            header += f" under {prefix}/"
        notes: list[str] = []
        if tree.truncated:
            # The one thing that must never be silent. GitHub gives up on a
            # recursive tree past 100,000 entries or 7MB, and an agent that
            # believes a partial listing is the whole repository concludes
            # that files it cannot see do not exist.
            notes.append(
                "INCOMPLETE: GitHub truncated this tree, so it is only part "
                "of the repository. Do not conclude a file is absent because "
                "it is not listed here -- use operation 'search', or 'stat' "
                "a specific path.")
        if tree.malformed:
            notes.append(f"{tree.malformed} entr(y/ies) came back unusable "
                         "and were left out.")
        shown = listing[:MAX_TREE_LINES]
        if len(listing) > MAX_TREE_LINES:
            notes.append(
                f"Showing {MAX_TREE_LINES} of {len(listing)} paths. Use "
                "'search', or 'tree' with a path, to narrow this.")
        body = [f"{header} ({len(files)} files)"]
        if notes:
            body += ["", *notes, ""]
        body += shown
        return ToolResult.success(
            "\n".join(body), truncated=tree.truncated, complete=not tree.truncated,
            files=len(files), shown=len(shown), malformed=tree.malformed)

    def _stat(self, owner, repo, branch, args) -> ToolResult:
        if not args.path:
            return ToolResult.failure("operation 'stat' needs a path.")
        info = self._client.stat(owner, repo, args.path, branch)
        if info.get("type") == "dir":
            entries = info.get("entries") or []
            lines = [f"{args.path}/ is a directory ({len(entries)} entries):"]
            lines += [f"  {e['path']}{'/' if e['type'] == 'dir' else ''}"
                      f"  {e.get('size') or ''}" for e in entries]
            return ToolResult.success("\n".join(lines), kind="dir",
                                      entries=len(entries))
        return ToolResult.success(
            f"{info['path']}  {info.get('size', 0)} bytes  "
            f"blob {info.get('sha', '')[:12]}",
            kind="file", size=info.get("size", 0), sha=info.get("sha", ""))

    def _read(self, owner, repo, branch, args) -> ToolResult:
        if not args.path:
            return ToolResult.failure("operation 'read' needs a path.")
        # The same wall a local read hits. A committed .env is a committed
        # .env whether it is on this disk or on GitHub, and refusing one
        # while handing over the other put the credential in the model's
        # context and in the transcript by the longer route.
        if refused := self._refuse_secret(args.path):
            return ToolResult.failure(refused, kind="secret")
        blob = self._client.read(owner, repo, args.path, branch,
                                 start=max(0, args.start_line),
                                 end=max(0, args.end_line))
        # And credentials inside a file that is otherwise fine to read are
        # masked, exactly as they are locally.
        text, masked = self.shield.clean(blob.text)
        where = f"{args.repo}:{args.path} ({branch})"
        if blob.ranged:
            where += f" lines {blob.start}-{blob.end} of {blob.total_lines}"
        # The sha goes back with the content because it is what the *edit*
        # has to be based on: github_write needs the version this was read
        # at, not whatever is there when the write happens.
        note = (f"\n[{masked} credential(s) masked]" if masked else "")
        return ToolResult.success(
            f"--- {where} ---\n{text}\n"
            f"--- blob sha: {blob.sha} (pass this to github_write) ---{note}",
            sha=blob.sha, path=args.path, branch=branch,
            lines=blob.total_lines, ranged=blob.ranged, masked=masked)


class GitHubWriteInput(Schema):
    operation = Field(
        str,
        "What to do: 'write' creates or updates one file; 'branch' creates a "
        "branch; 'pr' opens a pull request from a branch to a base.",
        choices=("write", "branch", "pr"), default="write")
    repo = Field(
        str,
        "The repository as owner/name, e.g. 'wynxo/wynxo-AI-Agent-Termianl-CLI'.")
    branch = Field(
        str,
        "The branch to act on: 'write' commits here, 'branch' names the new "
        "branch, 'pr' is the head branch.",
        default="")
    base = Field(
        str, "The branch a pull request targets; empty means the default.",
        default="")
    path = Field(
        str, "File path to write, relative to the repository root.",
        default="")
    content = Field(
        str, "The complete new file content, for 'write'.", default="")
    base_sha = Field(
        str,
        "The blob sha github_read reported for this file -- the version this "
        "change was written against. Required when updating an existing "
        "file: GitHub compares it with what is there now and refuses the "
        "write if somebody else has committed in the meantime. Leave empty "
        "only when creating a file that does not exist yet.",
        default="")
    message = Field(str, "Commit message for 'write'.", default="")
    title = Field(
        str, "Pull request title; a sensible default is used if empty.",
        default="")
    body = Field(
        str, "Pull request body; empty means the commit messages.", default="")


class GitHubWrite(Tool):
    name = "github_write"
    description = (
        "Change a GitHub repository in the cloud, committing through the API "
        "with the account that ran `gh auth login`. Nothing is cloned. "
        "Workflow: github_read the file first (it reports a blob sha), edit "
        "the content, then 'write' passing that same base_sha -- GitHub "
        "refuses the commit if the file moved underneath you, which is what "
        "stops one change from erasing another. 'branch' creates a branch "
        "and 'pr' opens a pull request. Writes prompt for approval like any "
        "other file change."
    )
    Input = GitHubWriteInput
    mutating = True
    concurrency_safe = False

    def __init__(self, workspace, boundary=None, shield=None,
                 client: GitHubClient | None = None):
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
            return ToolResult.failure(
                "github_write needs a repository as owner/name.")
        try:
            default = self._client.repo_default_branch(owner, repo)
            if args.operation == "branch":
                return self._branch(owner, repo, default, args)
            if args.operation == "pr":
                return self._pr(owner, repo, default, args)
            return self._write(owner, repo, args)
        except GitHubError as exc:
            return ToolResult.failure(str(exc), kind=exc.kind,
                                      repo=args.repo, operation=args.operation)

    # -- the operations ----------------------------------------------------

    def _branch(self, owner, repo, default, args) -> ToolResult:
        if not args.branch:
            return ToolResult.failure(
                "operation 'branch' needs a name for the new branch.")
        from_branch = args.base or default
        head = self._client.ref_sha(owner, repo, from_branch)
        self._client.create_branch(owner, repo, args.branch, head)
        return ToolResult.success(
            f"created branch {args.branch} in {args.repo}, from "
            f"{from_branch} at {head[:10]}.",
            branch=args.branch, base=from_branch, sha=head)

    def _pr(self, owner, repo, default, args) -> ToolResult:
        if not args.branch:
            return ToolResult.failure(
                "operation 'pr' needs the head branch to open from.")
        base = args.base or default
        title = args.title or f"wynxo: changes on {args.branch}"
        body = args.body or self._pr_body(owner, repo, args.branch, base)
        url = self._client.open_pr(owner, repo, base, args.branch, title, body)
        if not url.strip():
            return ToolResult.failure(
                "GitHub did not return a pull request URL, so it is not "
                "confirmed that one was opened.")
        return ToolResult.success(f"opened {url}", display=f"PR: {url}",
                                  url=url.strip(), head=args.branch, base=base)

    def _write(self, owner, repo, args) -> ToolResult:
        if not args.path or not args.branch:
            return ToolResult.failure(
                "operation 'write' needs a path, the complete new content, a "
                "commit message and a branch.")
        if not args.content.strip():
            return ToolResult.failure("refusing to write an empty file.")

        # What is on the branch right now, and whether this change was
        # written against it.
        try:
            current = self._client.read(owner, repo, args.path, args.branch)
        except GitHubError as exc:
            if exc.kind != "not_found":
                raise
            current = None

        if current is None:
            if args.base_sha:
                return ToolResult.failure(
                    f"{args.path} does not exist on {args.branch}, but a "
                    "base_sha was given for it. Read the path again: either "
                    "the branch is wrong or the file was deleted.",
                    kind="not_found")
        else:
            if not args.base_sha:
                # The whole protection lives on this argument. Fetching the
                # current sha here instead -- which is what this tool used to
                # do -- satisfies GitHub's check while defeating it, because
                # the sha then always matches by construction and the write
                # always succeeds. A change made between the read and the
                # write was destroyed with no error and a "committed"
                # message.
                return ToolResult.failure(
                    f"{args.path} already exists on {args.branch}. Updating "
                    "it needs base_sha: the blob sha github_read reported "
                    "for the version this change was written against. "
                    f"It is {current.sha} right now -- but read the file "
                    "again rather than passing that blind, or an edit based "
                    "on stale content will be committed as if it were "
                    "current.",
                    kind="needs_base_sha", sha=current.sha)
            if args.base_sha != current.sha:
                return ToolResult.failure(
                    f"{args.path} changed on {args.branch} since it was "
                    f"read: this change is based on {args.base_sha[:12]} and "
                    f"the file is now {current.sha[:12]}. Nothing was "
                    "written. Read it again and rebuild the change on the "
                    "new content -- committing this one would erase whatever "
                    "landed in between.",
                    kind="conflict", expected=args.base_sha,
                    actual=current.sha, path=args.path)
            if current.text == args.content:
                return ToolResult.success(
                    f"{args.path} on {args.branch} already has exactly this "
                    "content; nothing to commit.",
                    changed=False, path=args.path, sha=current.sha)

        message = args.message or f"wynxo: update {args.path}"
        commit = self._client.write(
            owner, repo, args.path, args.content, message, args.branch,
            sha=args.base_sha or None)

        # Say it landed only after looking. A PUT that returns a commit has
        # usually worked, but "usually" is not what a report of success
        # should mean, and the read costs one request against a change that
        # is already made.
        verified, note = self._verify(owner, repo, args, commit)
        added, removed = _line_delta(
            current.text if current is not None else "", args.content)
        summary = (f"committed {message!r} to {args.repo}:{args.path} on "
                   f"{args.branch} ({commit[:10]}) +{added} -{removed}")
        if not verified:
            return ToolResult.failure(
                f"{summary}\n\nBut the result could not be confirmed: {note} "
                "Check the repository before relying on this.",
                kind="unverified", commit=commit, path=args.path)
        return ToolResult.success(
            summary, display=f"{args.repo}:{args.path} +{added} -{removed}",
            commit=commit, path=args.path, branch=args.branch,
            added=added, removed=removed, sha=note)

    def _verify(self, owner, repo, args, commit: str) -> tuple[bool, str]:
        """Read the file back. Returns (ok, new sha or why not)."""
        try:
            after = self._client.read(owner, repo, args.path, args.branch)
        except GitHubError as exc:
            return False, f"reading it back failed: {exc}"
        if after.text != args.content:
            return False, ("the file on GitHub does not match what was sent, "
                           "so something else changed it at the same time.")
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
    """(added, removed) between two versions, for an honest summary."""
    import difflib

    added = removed = 0
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                     lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
