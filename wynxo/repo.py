"""Working on a GitHub repository.

wynxo edits files on disk, so working on a repository means having it on
disk: this clones it into a cache directory and moves the workspace there.
There is no remote-execution mode -- the tools run locally, and pretending
otherwise would be a lie about where your edits are happening.

Pushing is left to you. The agent can run git through the shell tool if you
ask it to, under the same permission prompt as any other command, which is
where that decision belongs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import data_dir

SHORTHAND = re.compile(r"^([\w.-]+)/([\w.-]+?)(?:\.git)?$")

# `..` and `.` are valid GitHub-shaped names as far as the pattern above is
# concerned, and they are directory names as far as the cache is concerned:
# `--repo ../x` put the checkout beside the cache instead of inside it.
# Nothing legitimate is called either.
_NOT_A_NAME = {"", ".", "..", ".git"}


def _usable(owner: str, name: str) -> bool:
    return not ({owner, name} & _NOT_A_NAME)


@dataclass
class Target:
    url: str
    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    def directory(self) -> Path:
        return data_dir() / "repos" / self.owner / self.name


def parse(raw: str) -> Target | None:
    """Accept the shapes people paste.

    ``owner/name``, an https URL, an ssh URL, or a browser URL with extra
    path on the end -- they all name the same repository.
    """
    text = raw.strip().rstrip("/")
    if not text:
        return None

    if match := SHORTHAND.match(text):
        owner, name = match.group(1), match.group(2)
        if not _usable(owner, name):
            return None
        return Target(f"https://github.com/{owner}/{name}.git", owner, name)

    if text.startswith("git@"):
        # git@github.com:owner/name.git
        _, _, path = text.partition(":")
        parts = path.removesuffix(".git").split("/")
        if len(parts) >= 2 and _usable(parts[-2], parts[-1]):
            return Target(text, parts[-2], parts[-1])
        return None

    if text.startswith(("http://", "https://", "ssh://")):
        parts = [p for p in text.split("://", 1)[1].split("/") if p]
        # host, owner, name, then possibly /tree/main/...
        if len(parts) >= 3:
            owner, name = parts[1], parts[2].removesuffix(".git")
            if not _usable(owner, name):
                return None
            base = "/".join(parts[:3]).removesuffix(".git")
            scheme = text.split("://", 1)[0]
            return Target(f"{scheme}://{base}.git", owner, name)
        return None

    return None


def git_available() -> bool:
    return shutil.which("git") is not None


def run_git(args: list[str], cwd: Path | None = None,
            timeout: int = 600) -> tuple[bool, str]:
    """Run git without letting it block on a credential prompt.

    A clone of a private repository would otherwise sit forever waiting for a
    username on a terminal the agent is not reading.
    """
    import os

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"}
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(cwd) if cwd else None, env=env,
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, "git is not installed"
    except subprocess.TimeoutExpired:
        return False, f"git {args[0]} timed out after {timeout}s"
    except OSError as exc:
        return False, str(exc)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def clone_or_update(target: Target) -> tuple[bool, Path, str]:
    """Fetch the repository into the cache. Returns (ok, path, message)."""
    path = target.directory()

    if (path / ".git").is_dir():
        ok, output = run_git(["pull", "--ff-only"], cwd=path)
        if ok:
            return True, path, f"updated {target.slug}"
        # A dirty checkout is not a failure worth throwing away work over.
        return True, path, f"using the existing copy ({_first_line(output)})"

    if path.exists() and any(path.iterdir()):
        return False, path, f"{path} already exists and is not a git checkout"

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, output = run_git(["clone", "--depth", "50", target.url, str(path)])
    if not ok:
        return False, path, _explain_clone_failure(target, output)
    return True, path, f"cloned {target.slug}"


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _explain_clone_failure(target: Target, output: str) -> str:
    low = output.lower()
    if "could not read username" in low or "authentication failed" in low:
        return (
            f"{target.slug} needs credentials and none were available.\n"
            "  For a private repository, either set up a credential helper:\n"
            "    git config --global credential.helper manager   (Windows)\n"
            "    git config --global credential.helper store     (Linux/macOS)\n"
            "  or clone it yourself once, then point wynxo at the folder."
        )
    if "repository not found" in low or "not found" in low:
        return (f"{target.slug} was not found. Check the spelling, and that "
                "you have access if it is private.")
    if "could not resolve host" in low:
        return "No network route to the host. Check your connection."
    return f"clone failed: {_first_line(output)}"


def status(path: Path) -> str:
    """A one-line summary of a checkout, for the status bar."""
    ok, branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path, timeout=10)
    if not ok:
        return ""
    ok, dirty = run_git(["status", "--porcelain"], cwd=path, timeout=10)
    changed = len(dirty.splitlines()) if ok and dirty else 0
    return f"{branch.strip()}{f' +{changed}' if changed else ''}"
