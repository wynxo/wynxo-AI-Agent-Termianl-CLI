"""Cloud access to GitHub, through the ``gh`` command-line tool.

The whole point of this module is that the repository never has to be on
disk. GitHub's REST API lets an authenticated account create, edit and
delete files, branches and pull requests in any repository it can see --
which is exactly "code in the cloud". ``gh auth login`` stores the token,
keyring and host configuration, so wynxo never touches credentials: every
operation here shells out to ``gh api`` (or ``gh pr create``) and parses the
JSON that comes back.

Why the CLI instead of the REST API directly? Because ``gh`` already solved
authentication, token refresh and rate limits, and because the user's own
``gh auth status`` is the single source of truth for what the account can
reach. No API key is stored anywhere in wynxo.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from urllib.parse import quote

GITHUB = "https://github.com"


@dataclass
class Tree:
    """A repository's file listing, and whether it is the whole of one."""

    entries: list[dict] = field(default_factory=list)
    truncated: bool = False
    """GitHub gave up before listing everything. Anything derived from this
    tree is a partial view and must say so."""
    malformed: int = 0
    """Entries that came back without a path or a type."""
    sha: str = ""

    @property
    def files(self) -> list[dict]:
        return [e for e in self.entries if e.get("type") == "blob"]

    @property
    def dirs(self) -> list[dict]:
        return [e for e in self.entries if e.get("type") == "tree"]


@dataclass
class Blob:
    """One file's text, with the identity an edit has to be based on."""

    text: str
    sha: str
    size: int = 0
    path: str = ""
    total_lines: int = 0
    start: int = 0
    """1-based first line, when a range was asked for."""
    end: int = 0

    @property
    def ranged(self) -> bool:
        return bool(self.start or self.end)

# The contents endpoint refuses files above this size; the git blobs
# endpoint handles more, but a 1MB wall keeps the model's context sane.
MAX_FILE_BYTES = 1_000_000

DEFAULT_TIMEOUT = 30.0


class GitHubError(Exception):
    """A gh call failed; the message says what to do about it.

    ``kind`` is the machine-readable half. The model gets a sentence it can
    act on, but a caller that wants to *decide* -- retry, re-read, give up --
    should not be parsing English, and the tool layer turns a stale-sha
    conflict into a re-read rather than a failure.
    """

    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


AUTH = "auth"
NOT_FOUND = "not_found"
CONFLICT = "conflict"
RATE_LIMIT = "rate_limit"
DENIED = "denied"
TOO_LARGE = "too_large"
BINARY = "binary"
TIMEOUT = "timeout"
MALFORMED = "malformed"
MISSING_CLI = "missing_cli"


def _classify(stderr: str) -> tuple[str, str]:
    """(kind, an explanation worth reading) for one gh failure.

    gh reports HTTP failures by echoing the status and GitHub's own message,
    which is accurate and says nothing about what to do next. "gh: Not Found
    (HTTP 404)" does not tell anybody whether the repository, the branch or
    the path was the thing that was missing, and a rate-limit body is a
    paragraph of URL. Each of these is a different next step, so each gets
    named.
    """
    low = stderr.lower()
    if "rate limit" in low or "secondary rate" in low:
        return RATE_LIMIT, (
            "GitHub rate-limited this account. Wait before retrying; "
            "unauthenticated requests are limited far more sharply, so check "
            "`gh auth status` if this was unexpected.")
    if "401" in stderr or "not logged in" in low or "bad credentials" in low:
        return AUTH, ("GitHub rejected the credentials. Run `gh auth login` "
                      "in a terminal to connect the account.")
    if "403" in stderr:
        return DENIED, (
            "GitHub refused this operation for the authenticated account. "
            "That is usually a missing scope or write access to the "
            "repository: check `gh auth status` for the token's scopes.")
    if "404" in stderr or "not found" in low:
        return NOT_FOUND, (
            "GitHub could not find that. A 404 also covers a private "
            "repository the account cannot see, so check the owner/name, the "
            "branch, and the path -- and that the account has access.")
    if "409" in stderr or "does not match" in low or "is at" in low:
        return CONFLICT, (
            "The file changed on GitHub since it was read, so the write was "
            "refused rather than overwriting somebody else's commit. Read it "
            "again and rebuild the change on the new content.")
    if "422" in stderr:
        return MALFORMED, (
            "GitHub rejected the request as invalid. For a branch this "
            "usually means it already exists; for a write, that the sha does "
            "not belong to that path.")
    if "timeout" in low or "timed out" in low:
        return TIMEOUT, "The GitHub request timed out."
    return "error", stderr or "the GitHub request failed."


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


class GitHubClient:
    """A thin wrapper over ``gh api`` for the operations a coding agent
    needs: browse a repo, read and write files, create branches, open PRs.

    Every call is a subprocess, so the things that do not change within a
    turn -- which branch is the default, what the tree looks like -- are
    remembered for the life of the client rather than fetched again for
    every question about the same repository. Anything that writes drops
    what it could have invalidated, because a cache that outlives the truth
    is worse than no cache: it would hand the model a file listing that no
    longer matches the repository it just changed.
    """

    def __init__(self) -> None:
        self._branch_cache: dict[tuple[str, str], str] = {}
        self._tree_cache: dict[tuple[str, str, str], Tree] = {}

    def forget(self, owner: str = "", repo: str = "") -> None:
        """Drop cached knowledge, of one repository or of everything.

        Called after any write, and by an explicit refresh.
        """
        if not owner:
            self._branch_cache.clear()
            self._tree_cache.clear()
            return
        self._branch_cache.pop((owner, repo), None)
        for key in [k for k in self._tree_cache if k[:2] == (owner, repo)]:
            del self._tree_cache[key]

    def available(self) -> bool:
        """Whether the gh CLI is installed at all."""
        return shutil.which("gh") is not None

    # -- plumbing ----------------------------------------------------------

    def _run(self, args: list[str], *, check: bool = True,
             timeout: float = DEFAULT_TIMEOUT,
             input: str | None = None) -> subprocess.CompletedProcess:
        """Run one gh command, translating failure into a useful message."""
        try:
            proc = subprocess.run(
                ["gh"] + args, capture_output=True, text=True, timeout=timeout,
                input=input)
        except FileNotFoundError:
            raise GitHubError(
                "the GitHub CLI `gh` is not installed. "
                "Install it from https://cli.github.com, then run "
                "`gh auth login`.", MISSING_CLI) from None
        except subprocess.TimeoutExpired:
            raise GitHubError(
                f"the GitHub request timed out after {timeout:.0f}s.",
                TIMEOUT) from None
        if check and proc.returncode != 0:
            raise self._describe(args, proc)
        return proc

    def _describe(self, args: list[str], proc: subprocess.CompletedProcess) -> GitHubError:
        stderr = (proc.stderr or "").strip()
        if not stderr:
            return GitHubError(f"gh {' '.join(args[:2])} failed with no output.")
        kind, explanation = _classify(stderr)
        # The raw line is kept on the end, once, because when the guess above
        # is wrong it is the only thing that says what actually happened --
        # but it goes after the sentence that is useful, not instead of it.
        detail = stderr.splitlines()[0][:200]
        if explanation != stderr:
            return GitHubError(f"{explanation} (GitHub said: {detail})", kind)
        return GitHubError(detail, kind)

    def _api(self, path: str, *, method: str = "GET", fields: dict | None = None,
             jq: str | None = None, silent: bool = False,
             check: bool = True) -> str:
        args = ["api"]
        if method != "GET":
            args += ["--method", method]
        if silent:
            args.append("--silent")
        args.append(path)
        if jq:
            args += ["--jq", jq]
        if fields is not None:
            args += ["--input", "-"]
            return self._run(args, check=check,
                             input=json.dumps(fields)).stdout
        return self._run(args, check=check).stdout

    # -- identity ----------------------------------------------------------

    def auth_user(self) -> str:
        """The logged-in account, e.g. ``wynxo``. Raises GitHubError when
        not authenticated, with instructions to run ``gh auth login``."""
        out = self._run(["api", "user", "--jq", ".login"]).stdout.strip()
        return out

    # -- reading a repository ----------------------------------------------

    def repo_default_branch(self, owner: str, repo: str) -> str:
        key = (owner, repo)
        if key not in self._branch_cache:
            branch = self._api(f"repos/{owner}/{repo}",
                               jq=".default_branch").strip()
            if not branch:
                raise GitHubError(
                    f"{owner}/{repo} did not report a default branch. Check "
                    "the owner and name, and that the account can see it.",
                    NOT_FOUND)
            self._branch_cache[key] = branch
        return self._branch_cache[key]

    def tree(self, owner: str, repo: str, branch: str) -> "Tree":
        """The repository's file tree at a branch.

        The whole response is parsed rather than ``.tree[]``, because the
        field that says whether this is the *whole* tree lives beside it.
        GitHub caps a recursive tree at 100,000 entries or 7MB and then sets
        ``truncated: true`` -- and selecting ``.tree[]`` with jq threw that
        away, so on any large repository the agent was handed a partial map
        and told nothing. It would then conclude a file did not exist
        because it was not in a list that was never complete.
        """
        key = (owner, repo, branch)
        if key in self._tree_cache:
            return self._tree_cache[key]
        raw = self._api(f"repos/{owner}/{repo}/git/trees/{quote(branch, safe='')}"
                        "?recursive=1")
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            raise GitHubError(
                f"could not read the file tree of {owner}/{repo}@{branch}: "
                "GitHub's response was not JSON.", MALFORMED) from None
        if not isinstance(data, dict):
            raise GitHubError(
                f"could not read the file tree of {owner}/{repo}@{branch}: "
                "unexpected response shape.", MALFORMED)
        entries: list[dict] = []
        malformed = 0
        for entry in data.get("tree") or []:
            # Every entry needs a path and a type to be usable. A shape that
            # does not have them is counted rather than dropped in silence:
            # a map with holes in it is worth knowing about.
            if isinstance(entry, dict) and entry.get("path") and entry.get("type"):
                entries.append(entry)
            else:
                malformed += 1
        tree = Tree(entries=entries,
                    truncated=bool(data.get("truncated")),
                    malformed=malformed,
                    sha=str(data.get("sha") or ""))
        self._tree_cache[key] = tree
        return tree

    def stat(self, owner: str, repo: str, path: str, branch: str) -> dict:
        """What is at a path, without downloading it.

        A directory answers with a list, a file with an object. Either way
        this costs one request and no content, which is what makes it worth
        having: deciding whether a file is worth reading should not require
        reading it.
        """
        out = self._api(f"repos/{owner}/{repo}/contents/{quote(path, safe='/')}"
                        f"?ref={quote(branch, safe='')}")
        try:
            data = json.loads(out or "null")
        except json.JSONDecodeError:
            raise GitHubError(f"could not stat {path}: unexpected response.",
                              MALFORMED) from None
        if isinstance(data, list):
            return {"type": "dir", "path": path,
                    "entries": [{"path": e.get("path"), "type": e.get("type"),
                                 "size": e.get("size", 0), "sha": e.get("sha")}
                                for e in data if isinstance(e, dict)]}
        if not isinstance(data, dict):
            raise GitHubError(f"could not stat {path}: unexpected response.",
                              MALFORMED)
        return {"type": data.get("type", "file"), "path": data.get("path", path),
                "size": int(data.get("size") or 0),
                "sha": str(data.get("sha") or "")}

    def read(self, owner: str, repo: str, path: str, branch: str,
             start: int = 0, end: int = 0) -> "Blob":
        """One file's text and the blob sha the edit will be based on.

        ``start``/``end`` are 1-based inclusive line numbers. GitHub has no
        ranged-content endpoint, so the fetch is whole either way -- but what
        goes back to the model is only the region asked for, which is the
        expensive half. Reading one function out of a 10,000-line file should
        not put 10,000 lines in the context window.
        """
        out = self._api(f"repos/{owner}/{repo}/contents/{quote(path, safe='/')}"
                        f"?ref={quote(branch, safe='')}",
                        jq="{content, sha, size, encoding, type}")
        try:
            data = json.loads(out or "null")
        except json.JSONDecodeError:
            raise GitHubError(f"could not read {path}: unexpected response.",
                              MALFORMED) from None
        if not isinstance(data, dict):
            raise GitHubError(
                f"{path} is not a file (a directory answers with a list; use "
                "operation 'stat' to look inside one).", MALFORMED)
        size = int(data.get("size") or 0)
        if size > MAX_FILE_BYTES:
            raise GitHubError(
                f"{path} is {size} bytes. The contents API caps at "
                f"{MAX_FILE_BYTES}, and reading that into context would be "
                "unwise anyway -- use a line range, or search for the part "
                "that matters.", TOO_LARGE)
        encoding = data.get("encoding") or "base64"
        if encoding != "base64":
            raise GitHubError(
                f"{path} came back {encoding}-encoded, which wynxo cannot "
                "decode as text.", BINARY)
        raw = base64.b64decode(data.get("content", "") or "")
        if b"\x00" in raw[:8192]:
            raise GitHubError(
                f"{path} is a binary file; there is no text to read.", BINARY)
        text = raw.decode("utf-8", "replace")
        sha = str(data.get("sha", ""))
        total = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        if not (start or end):
            return Blob(text=text, sha=sha, size=size, path=path,
                        total_lines=total)
        lines = text.splitlines()
        first = max(1, start or 1)
        last = min(len(lines), end or len(lines))
        if first > len(lines):
            raise GitHubError(
                f"{path} has {len(lines)} lines; line {first} is past the end.",
                NOT_FOUND)
        return Blob(text="\n".join(lines[first - 1:last]), sha=sha, size=size,
                    path=path, total_lines=len(lines),
                    start=first, end=last)

    def search_code(self, owner: str, repo: str, query: str,
                    limit: int = 20) -> list[dict]:
        """Where a thing appears in a repository, without cloning it.

        This is what makes "find where authentication is implemented"
        answerable in one request instead of a tree walk and a hundred reads.
        GitHub's code search indexes the default branch only, which is a real
        limitation and is reported rather than hidden.
        """
        scoped = f"{query} repo:{owner}/{repo}"
        out = self._api(
            "search/code?q=" + quote(scoped, safe="") + f"&per_page={min(limit, 100)}",
            jq=".items[] | {path, sha, url: .html_url}")
        hits: list[dict] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                hits.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return hits

    # -- writing a repository ----------------------------------------------

    def write(self, owner: str, repo: str, path: str, content: str,
              message: str, branch: str, sha: str | None = None) -> str:
        """Create or update one file, returning the new commit sha.

        ``sha`` is the blob the change was *based on*, not the blob that is
        there now. That distinction is the entire value of it: GitHub
        compares the two and refuses the write if they differ, which is what
        stops one edit from silently erasing another. Passing a freshly
        fetched sha satisfies the check while defeating the protection, so
        callers must hand in the one they read.

        Omit it only to create a file that does not exist yet.
        """
        fields: dict = {"message": message, "content": _b64(content),
                        "branch": branch}
        if sha:
            fields["sha"] = sha
        out = self._api(f"repos/{owner}/{repo}/contents/{quote(path, safe='/')}",
                        method="PUT", fields=fields, jq=".commit.sha")
        # Whatever was known about this repository describes the version
        # before this commit.
        self.forget(owner, repo)
        commit = out.strip()
        if not commit:
            # A PUT that returns no commit is not a commit. Reporting one
            # anyway is the worst possible outcome: the model believes the
            # work is saved and moves on.
            raise GitHubError(
                f"GitHub accepted the request for {path} but returned no "
                "commit, so the write cannot be confirmed. Read the file "
                "again before assuming anything changed.", MALFORMED)
        return commit

    def ref_sha(self, owner: str, repo: str, branch: str) -> str:
        out = self._api(f"repos/{owner}/{repo}/git/ref/heads/{quote(branch, safe='')}",
                        jq=".object.sha")
        return out.strip()

    def create_branch(self, owner: str, repo: str, name: str,
                      from_sha: str) -> None:
        """Create a branch at a commit sha (use ref_sha for the head)."""
        self._api(f"repos/{owner}/{repo}/git/refs", method="POST",
                  fields={"ref": f"refs/heads/{name}", "sha": from_sha})
        self.forget(owner, repo)

    def commits(self, owner: str, repo: str, branch: str,
                limit: int = 15) -> list[str]:
        """Commit messages on a branch, newest first, for a PR body."""
        out = self._api(f"repos/{owner}/{repo}/commits?sha={quote(branch, safe='')}",
                        jq=f".[0:{limit}] | .[].commit.message")
        messages: list[str] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                messages.append(line)
        return messages

    def open_pr(self, owner: str, repo: str, base: str, head: str,
                title: str, body: str) -> str:
        """Open a pull request and return its URL."""
        proc = self._run([
            "pr", "create", "--repo", f"{owner}/{repo}",
            "--base", base, "--head", head,
            "--title", title, "--body", body,
        ])
        return proc.stdout.strip()
