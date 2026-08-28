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
import os
import shlex
import shutil
import sys
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


VENV_DIRS = (".venv", "venv", ".env", "env")
"""Where a Python project keeps its interpreter, in the order people mean."""


def python_command(root: Path) -> str:
    """The interpreter this project's tests should actually run under.

    Two ways "python -m pytest" goes wrong, and both report a failure the
    user did not cause -- which is worse than not running the tests at all,
    because the model then sets about fixing code that was fine.

    A project with a virtualenv keeps its pytest and its dependencies in
    there. Run by whatever `python` happens to be on PATH, the suite fails
    on imports that are installed three directories away.

    And on Debian and Ubuntu `python` is not a command at all unless
    somebody installed python-is-python3, so the whole run fails with "not
    found".
    """
    for directory in VENV_DIRS:
        for relative in ("bin/python", "bin/python3", "Scripts/python.exe"):
            candidate = root / directory / relative
            if candidate.is_file():
                return _runnable(candidate)

    # The interpreter actually running Wynxo is the environment the user
    # chose -- a venv they activated, a conda base -- and its site-packages
    # are where the project's dependencies live. It wins over whatever
    # `python` happens to be on PATH, which on Windows is frequently the
    # Microsoft Store alias: a zero-byte reparse point that opens the Store
    # instead of running anything. A non-empty executable is evidence of a
    # real interpreter; an alias is not.
    current = Path(sys.executable)
    if os.name == "nt" and _real_interpreter(current):
        return _runnable(current)

    if shutil.which("python3") is not None or shutil.which("python") is not None:
        for name in ("python3", "python"):
            if shutil.which(name):
                return name
    if current.is_file():
        return _runnable(current)
    for name in ("python3", "python"):
        if shutil.which(name):
            return name
    return "python3"


def _real_interpreter(path: Path) -> bool:
    """A real executable, not a Windows Store app-execution alias.

    Store aliases under %LOCALAPPDATA%\\Microsoft\\WindowsApps are zero-byte
    reparse points; launching one opens the Microsoft Store. A genuine
    python.exe is always larger than that.
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _runnable(path: Path) -> str:
    """A path the shell will execute, quoted if it has to be."""
    text = str(path)
    if " " not in text:
        return text
    if os.name == "nt":
        # PowerShell treats a quoted string as a string unless the call
        # operator is in front of it.
        return f'& "{text}"'
    return shlex.quote(text)


def _python(root: Path) -> Runner | None:
    run = f"{python_command(root)} -m pytest"
    if (root / "pytest.ini").is_file():
        return Runner("pytest", run, "there is a pytest.ini")
    if (root / "tox.ini").is_file() and "[pytest]" in _read(root / "tox.ini"):
        return Runner("pytest", run, "tox.ini configures pytest")

    pyproject = _read(root / "pyproject.toml")
    if "[tool.pytest" in pyproject:
        return Runner("pytest", run, "pyproject.toml configures pytest")

    if (root / "setup.cfg").is_file() and \
            "[tool:pytest]" in _read(root / "setup.cfg"):
        return Runner("pytest", run, "setup.cfg configures pytest")

    # No configuration, but an unmistakable layout.
    tests = root / "tests"
    if tests.is_dir() and any(tests.glob("test_*.py")):
        return Runner("pytest", run,
                      "there is a tests/ directory of test_*.py files")
    if any(root.glob("test_*.py")):
        return Runner("pytest", run,
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
