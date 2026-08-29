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
from urllib.parse import quote

GITHUB = "https://github.com"

# The contents endpoint refuses files above this size; the git blobs
# endpoint handles more, but a 1MB wall keeps the model's context sane.
MAX_FILE_BYTES = 1_000_000

DEFAULT_TIMEOUT = 30.0


class GitHubError(Exception):
    """A gh call failed; the message says what to do about it."""


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _unb64(text: str) -> str:
    return base64.b64decode(text).decode("utf-8", "replace")


class GitHubClient:
    """A thin wrapper over ``gh api`` for the operations a coding agent
    needs: browse a repo, read and write files, create branches, open PRs.
    """

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
                "Install it from https://cli.github.com, then run `gh auth login`.")
        except subprocess.TimeoutExpired:
            raise GitHubError("the GitHub request timed out.")
        if check and proc.returncode != 0:
            raise GitHubError(self._describe(args, proc))
        return proc

    def _describe(self, args: list[str], proc: subprocess.CompletedProcess) -> str:
        stderr = (proc.stderr or "").strip()
        message = stderr or f"gh {' '.join(args)} failed."
        lowered = stderr.lower()
        if "auth" in lowered or "not logged in" in lowered or "401" in stderr:
            message += (" Run `gh auth login` in a terminal to connect "
                        "your GitHub account.")
        return message

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
        out = self._api(f"repos/{owner}/{repo}", jq=".default_branch").strip()
        return out

    def tree(self, owner: str, repo: str, branch: str) -> list[dict]:
        """The repository's file tree at a branch, as the recursive git
        tree's entries: ``[{"path", "type", "size", "sha"}, ...]``."""
        out = self._api(f"repos/{owner}/{repo}/git/trees/{quote(branch, safe='')}"
                        "?recursive=1", jq=".tree[]")
        entries: list[dict] = []
        for line in out.splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def read(self, owner: str, repo: str, path: str, branch: str) -> tuple[str, str]:
        """The decoded text of one file, and its blob sha (needed to edit it).
        Raises GitHubError for binary or oversized content."""
        out = self._api(f"repos/{owner}/{repo}/contents/{quote(path, safe='/')}"
                        f"?ref={quote(branch, safe='')}",
                        jq="{content, sha, size}")
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            raise GitHubError(f"could not read {path}: unexpected response.")
        if data.get("size", 0) > MAX_FILE_BYTES:
            raise GitHubError(
                f"{path} is {data.get('size')} bytes; the contents API caps at "
                f"{MAX_FILE_BYTES} and reading it into context would be unwise.")
        return _unb64(data.get("content", "")), str(data.get("sha", ""))

    # -- writing a repository ----------------------------------------------

    def write(self, owner: str, repo: str, path: str, content: str,
              message: str, branch: str, sha: str | None = None) -> str:
        """Create or update one file with a commit message, returning the new
        commit sha. ``sha`` must be the current blob sha for updates."""
        fields: dict = {"message": message, "content": _b64(content),
                        "branch": branch}
        if sha:
            fields["sha"] = sha
        out = self._api(f"repos/{owner}/{repo}/contents/{quote(path, safe='/')}",
                        method="PUT", fields=fields, jq=".commit.sha")
        return out.strip()

    def ref_sha(self, owner: str, repo: str, branch: str) -> str:
        out = self._api(f"repos/{owner}/{repo}/git/ref/heads/{quote(branch, safe='')}",
                        jq=".object.sha")
        return out.strip()

    def create_branch(self, owner: str, repo: str, name: str,
                      from_sha: str) -> None:
        """Create a branch at a commit sha (use ref_sha for the head)."""
        self._api(f"repos/{owner}/{repo}/git/refs", method="POST",
                  fields={"ref": f"refs/heads/{name}", "sha": from_sha})

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
