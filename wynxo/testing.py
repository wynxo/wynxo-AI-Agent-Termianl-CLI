"""Finding and running the project's own tests.

The verify pass asks the model to re-read its own work, which for a 7B is
largely self-congratulation -- it wrote the code believing it was right, and
reading it again does not change that belief. A test run does. It is the one
source of truth in the loop that does not come from the model.

So this works out how a project is tested, from the files that are already
there, and hands the failures back. No configuration for the common cases:
if there is a `pytest.ini` there is pytest, and if `package.json` has a test
script there is npm test.

Deliberately narrow. It only recognises a runner when the evidence is
unambiguous, because guessing wrong means running the wrong command in
someone's project -- and a wrong command that happens to succeed is worse
than no test run at all, since it reports confidence nobody earned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Runner:
    name: str
    command: str
    why: str
    """What in the project said so, for when the user asks why this ran."""


# Only the last few lines are worth handing back: a failing suite says what
# failed at the end, and the beginning is setup noise.
TAIL_LINES = 60
DEFAULT_TIMEOUT = 180


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _json(path: Path) -> dict:
    try:
        data = json.loads(_read(path) or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def detect(root: Path) -> Runner | None:
    """The project's test command, or None when it cannot be sure."""
    root = Path(root)

    # -- javascript / typescript ------------------------------------------
    package = root / "package.json"
    if package.is_file():
        scripts = _json(package).get("scripts")
        if isinstance(scripts, dict):
            script = scripts.get("test")
            # `npm init` writes a placeholder test script that exits 1. Running
            # it would report a failure the user did not cause.
            if isinstance(script, str) and script.strip() and \
                    "no test specified" not in script.lower():
                return Runner("npm", f"{_node_agent(root)} test",
                              f"package.json has a test script: {script}")

    # -- rust --------------------------------------------------------------
    if (root / "Cargo.toml").is_file():
        return Runner("cargo", "cargo test", "there is a Cargo.toml")

    # -- go ----------------------------------------------------------------
    if (root / "go.mod").is_file():
        return Runner("go", "go test ./...", "there is a go.mod")

    # -- python ------------------------------------------------------------
    if runner := _python(root):
        return runner

    # -- others where the file itself is the answer ------------------------
    if (root / "Makefile").is_file() and _has_make_target(root / "Makefile", "test"):
        return Runner("make", "make test", "the Makefile has a test target")
    if (root / "mix.exs").is_file():
        return Runner("mix", "mix test", "there is a mix.exs")
    if (root / "Gemfile").is_file() and (root / "spec").is_dir():
        return Runner("rspec", "bundle exec rspec",
                      "there is a Gemfile and a spec/ directory")
    return None


def _has_make_target(makefile: Path, target: str) -> bool:
    """Whether the Makefile actually defines this target.

    A Makefile is not evidence on its own -- plenty have only `build` -- and
    `make test` against a file without the target fails with "no rule to make
    target", which would read as a broken test suite.
    """
    for line in _read(makefile).splitlines():
        if line.startswith(f"{target}:") or line.startswith(f"{target} :"):
            return True
        if line.startswith(".PHONY:") and target in line.split():
            return True
    return False


def _node_agent(root: Path) -> str:
    """Whichever package manager this project is actually using."""
    for lockfile, agent in (("pnpm-lock.yaml", "pnpm"), ("yarn.lock", "yarn"),
                            ("bun.lockb", "bun"), ("package-lock.json", "npm")):
        if (root / lockfile).is_file():
            return agent
    return "npm"


def _python(root: Path) -> Runner | None:
    if (root / "pytest.ini").is_file():
        return Runner("pytest", "python -m pytest", "there is a pytest.ini")
    if (root / "tox.ini").is_file() and "[pytest]" in _read(root / "tox.ini"):
        return Runner("pytest", "python -m pytest",
                      "tox.ini configures pytest")

    pyproject = _read(root / "pyproject.toml")
    if "[tool.pytest" in pyproject:
        return Runner("pytest", "python -m pytest",
                      "pyproject.toml configures pytest")

    if (root / "setup.cfg").is_file() and \
            "[tool:pytest]" in _read(root / "setup.cfg"):
        return Runner("pytest", "python -m pytest",
                      "setup.cfg configures pytest")

    # No configuration, but an unmistakable layout.
    tests = root / "tests"
    if tests.is_dir() and any(tests.glob("test_*.py")):
        return Runner("pytest", "python -m pytest",
                      "there is a tests/ directory of test_*.py files")
    if any(root.glob("test_*.py")):
        return Runner("pytest", "python -m pytest",
                      "there are test_*.py files in the project root")
    return None


def summarise(output: str, limit: int = TAIL_LINES) -> str:
    """The part of a test run worth putting back in the context.

    The tail, because that is where a suite says what failed; the head is
    collection and setup. Local models have little context to spare, and a
    thousand lines of passing dots would push out the code being fixed.
    """
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    omitted = len(lines) - limit
    return f"... [{omitted} earlier lines omitted] ...\n" + \
        "\n".join(lines[-limit:])
