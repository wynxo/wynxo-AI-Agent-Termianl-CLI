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
import re
import shlex
import shutil
import subprocess
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

    # -- jvm ---------------------------------------------------------------
    if runner := _jvm(root):
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


def _jvm(root: Path) -> Runner | None:
    """Gradle or Maven, preferring whatever the project pins.

    Until this existed, a Java or Kotlin project got no runner at all, which
    meant wynxo never checked its own work there: the verification step
    skips silently for non-Python changes, and run_tests could only say
    "provide command explicitly". The tool was reachable, but nothing
    reached for it on its own.

    The wrapper wins over the installed tool wherever there is one -- that
    is the whole point of committing a wrapper, and the version it pins is
    usually not the version on PATH.
    """
    for build, wrapper, tool, task in (
        ("build.gradle", "gradlew", "gradle", "test"),
        ("build.gradle.kts", "gradlew", "gradle", "test"),
        ("pom.xml", "mvnw", "mvn", "test"),
    ):
        if not (root / build).is_file():
            continue
        if _is_windows() and (root / f"{wrapper}.bat").is_file():
            return Runner(wrapper, f"{wrapper}.bat {task}",
                          f"there is a {build} and a {wrapper}.bat")
        if not _is_windows() and (root / wrapper).is_file():
            return Runner(wrapper, f"./{wrapper} {task}",
                          f"there is a {build} and a {wrapper}")
        return Runner(tool, f"{tool} {task}", f"there is a {build}")
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
    if _real_interpreter(current) and _belongs_to_project_environment(current, root) \
            and current.parent.name.lower() != "global":
        # Prefer Wynxo's interpreter when it is the active/project environment,
        # but keep the historical PATH fallback for an unrelated system
        # interpreter. This avoids running a test suite with Wynxo's packages
        # merely because the CLI itself was launched globally.
        #
        # The "global" exclusion is not Windows-only: a global install lands
        # in a directory of that name on every platform, and gating it on the
        # OS meant a globally installed wynxo hijacked the interpreter for
        # every project on Linux and Termux.
        return _runnable(current)

    # Prefer PATH when it provides a candidate; the active interpreter is
    # only selected above when it is demonstrably this project's environment.
    for name in ("python3", "python"):
        candidate = shutil.which(name)
        if candidate:
            return name
    # No conventional candidate was found. Return the platform spelling;
    # callers should treat it as a command name, not as proof it exists.
    return "python3"


def _is_windows() -> bool:
    """Whether this is Windows.

    A function rather than a bare ``os.name == "nt"`` at each site so tests
    can exercise the Windows branches by patching this one name. Patching
    ``os.name`` itself would work, but ``os`` is shared with the whole
    interpreter: pathlib picks its flavour from it, so a test that set it
    turned every later ``Path()`` in the process into a ``WindowsPath`` and
    took the rest of the suite down with it.
    """
    return os.name == "nt"


def _belongs_to_project_environment(path: Path, root: Path) -> bool:
    """Whether the active interpreter is plausibly this project's environment."""
    try:
        path = path.resolve()
        root = root.resolve()
    except OSError:
        return False
    parts = {part.lower() for part in path.parts}
    return any(marker in parts for marker in (".venv", "venv", ".env", "env")) \
        or root in path.parents


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
    if _is_windows():
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


# -- structured failure analysis -----------------------------------------


@dataclass(frozen=True)
class Failure:
    kind: str
    """Exception type: AssertionError, ModuleNotFoundError, ..."""
    message: str
    file: str = ""
    line: int = 0
    test: str = ""
    """The pytest node id when known (tests/test_x.py::test_y)."""
    frames: tuple[str, ...] = ()


_FRAME = re.compile(r"^(.+?\.py):(\d+): in (.+)$")
# pytest 8+/9 prints the raise site as ``file.py:line: ExceptionType``
# (no `` in func``), distinct from _FRAME.
_LOC = re.compile(r"^(.+?\.py):(\d+):\s+([A-Za-z_][\w.]*)$")
_E = re.compile(r"^E\s+([^:]+?)(?::\s*(.*))?$")
_SUMMARY = re.compile(r"^(FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$")
_SECTION = re.compile(r"^_{5,}\s*(.*?)\s*_+$")
_PY_FRAME = re.compile(r'^\s*File "(.+)", line (\d+)(?:, in (\S+))?')
_EXC = re.compile(r"^([\w.]+):\s*(.*)$")


def parse_failures(output: str) -> list[Failure]:
    """Structured failure records from pytest or plain Python output.

    Handles both of pytest's shapes -- the full ``E   Type: message``
    traceback sections (with ``file.py:line: in func`` frames) and the
    ``FAILED nodeid - Type: message`` summary lines at the end -- plus a
    plain ``Traceback (most recent call last):`` block. The raise site is
    the frame printed directly above each exception line.
    """
    failures: list[Failure] = []
    seen: set[tuple] = set()
    tests_seen: set[str] = set()
    frames: list[tuple[str, int, str]] = []
    current_test = ""
    traceback_mode = False
    pending_e: list[str] = []
    """Assertion-detail E lines ("assert -1 == 5", "+ where ...") waiting
    for the location line that names their exception type."""

    def add(kind: str, message: str, file: str = "", line: int = 0,
            test: str = "") -> None:
        key = (kind, file, line, test)
        if key in seen:
            return
        seen.add(key)
        if test:
            tests_seen.add(test.rsplit("::", 1)[-1])
        failures.append(Failure(
            kind=kind, message=message.strip(), file=file, line=line,
            test=test, frames=tuple(f"{f}:{l} in {fn}"
                                     for f, l, fn in frames[-4:])))

    for raw in (output or "").splitlines():
        s = raw.rstrip()
        line = s.strip()
        if not line:
            continue

        section = _SECTION.match(line)
        if section:
            head = section.group(1).strip()
            current_test = head.split()[-1] if head else ""
            frames = []
            traceback_mode = False
            pending_e = []
            continue

        frame = _FRAME.match(line)
        if frame:
            frames.append((frame.group(1), int(frame.group(2)), frame.group(3)))
            pending_e = []
            continue

        location = _LOC.match(line)
        if location:
            file = location.group(1)
            lineno = int(location.group(2))
            kind = location.group(3)
            frames.append((file, lineno, ""))
            message = pending_e[0] if pending_e else ""
            add(kind, message, file, lineno, current_test)
            pending_e = []
            continue

        exc = _E.match(line)
        if exc:
            kind = exc.group(1).strip()
            message = exc.group(2) or ""
            if message:
                file, lineno, _ = frames[-1] if frames else ("", 0, "")
                add(kind, message, file, lineno, current_test)
            else:
                # "E   assert -1 == 5" -- assertion detail with no type;
                # its type arrives on the location line below.
                pending_e.append(kind)
            continue

        if line == "Traceback (most recent call last):":
            traceback_mode = True
            frames = []
            continue
        if traceback_mode:
            py_frame = _PY_FRAME.match(line)
            if py_frame:
                frames.append((py_frame.group(1), int(py_frame.group(2)),
                               py_frame.group(3) or ""))
                continue
            if raw.startswith((" ", "\t")):
                continue          # source lines between frames
            plain = _EXC.match(line)
            if plain and frames:
                add(plain.group(1), plain.group(2), frames[-1][0],
                    frames[-1][1], current_test)
                frames = []
                traceback_mode = False
                continue
            traceback_mode = False  # traceback over; other lines may follow

        summary = _SUMMARY.match(line)
        if summary:
            kind = "error" if summary.group(1) == "ERROR" else "failure"
            node = summary.group(2)
            rest = summary.group(3) or ""
            if " " in node:      # "ERROR collecting test_x.py" style
                continue
            if rest and ":" in rest:
                exc_kind, _, msg = rest.partition(":")
                exc_kind = exc_kind.strip() or kind
            else:
                exc_kind, msg = kind, rest
            if node.rsplit("::", 1)[-1] in tests_seen:
                continue          # already captured with file/line above
            add(exc_kind, msg, test=node)
    return failures


ENV_TOOLS = (
    "pytest", "pytest_asyncio", "pytest-asyncio", "pytest_cov",
    "pytest-cov", "ruff", "black", "mypy", "pyright", "flake8",
    "coverage", "tox", "nox", "pre-commit", "setuptools", "wheel",
    "pip", "hypothesis",
)
"""Tools whose absence is an environment problem, not a code problem."""

# Module name as imported -> package name as pip installs it.
_PIP_NAME = {"pytest_asyncio": "pytest-asyncio", "pytest_cov": "pytest-cov"}


def classify_failure(failure: Failure, root: Path) -> tuple[str, str]:
    """(category, reason) for a failure: environment, structure, code, test.

    The category steers the next move: an environment problem must be fixed
    by installing into the active environment, never by editing source;
    a structure problem is about package layout and import paths.
    """
    kind = failure.kind
    message = (failure.message or "").lower()

    if kind in ("ModuleNotFoundError", "ImportError"):
        # Longest first: "pytest_asyncio" must beat "pytest" as a substring.
        for tool in sorted(ENV_TOOLS, key=len, reverse=True):
            if tool.lower() in message:
                package = _PIP_NAME.get(tool, tool)
                return ("environment",
                        f"{package} is not installed in the active environment. "
                        "Install it there -- do not edit application source "
                        f"to paper over it. ({pip_command(root)} install {package})")
        return ("structure",
                "a module cannot be imported. Check the package layout "
                "(src/ vs flat), __init__.py files, and that the package is "
                "importable in the active environment (e.g. an editable "
                "install), before editing code.")

    if kind in ("SyntaxError", "IndentationError", "TabError"):
        return ("code", "the file does not parse -- a syntax error at the "
                "reported line.")

    if kind == "ModuleNotFoundError":
        return ("structure", "the import path does not resolve.")

    if failure.file and _is_test_path(failure.file, root):
        return ("test", "the failure is inside test code -- the test's "
                "expectation or setup is wrong, or the change broke what "
                "the test asserts.")

    if kind == "AssertionError":
        return ("code", "an assertion failed in project code (or the change "
                "broke a behavioral contract).")

    if kind == "TimeoutError" or "timed out" in message:
        return ("code", "the test timed out -- likely an infinite loop or a "
                "hang introduced by the change.")

    if kind in ("NameError", "AttributeError", "TypeError", "KeyError",
                "IndexError", "ValueError", "RecursionError", "ZeroDivisionError",
                "FileNotFoundError", "PermissionError", "OSError", "RuntimeError"):
        return ("code", f"{kind} raised while the code ran; read the failing "
                "frame and the value it was given.")

    return ("unknown", "no strong signal; read the failure context.")


def _is_test_path(path: str, root: Path) -> bool:
    """Whether a failure location is inside test code."""
    name = Path(path).name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    try:
        rel = str(Path(path).resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return False
    return rel.split(os.sep)[0] in ("tests", "test", "spec")


def failure_report(output: str, root: Path) -> str:
    """The structured, classified version of a test run, for the model.

    Appends to the raw tail the things the model can act on directly:
    exception type, file:line, test node id, and what kind of problem it is
    (environment vs code vs test) with the right next move.
    """
    failures = parse_failures(output)
    if not failures:
        return ""
    lines = ["", "[structured failure analysis]"]
    for failure in failures[:6]:
        where = f"{failure.file}:{failure.line}" if failure.file else "?"
        lines.append(f"\u2022 {failure.kind}: {failure.message or '(no message)'}")
        if failure.test:
            lines.append(f"  test: {failure.test}")
        lines.append(f"  at {where}")
        category, reason = classify_failure(failure, root)
        lines.append(f"  \u2192 {category}: {reason}")
    if len(failures) > 6:
        lines.append(f"  \u2026 and {len(failures) - 6} more failures")
    return "\n".join(lines)


# -- the project's Python environment -------------------------------------


_IMPORT_CACHE: dict[str, bool | None] = {}
_VERSION_CACHE: dict[str, str] = {}


def _interpreter_argv(interpreter: str) -> list[str]:
    """The interpreter as an argv, tolerating every quoting style
    python_command() can produce (bare name, quoted path, `& "..."`).

    Never shlex a Windows path: backslashes are shlex escapes on POSIX
    mode, so ``.venv\\Scripts\\python.exe`` would collapse into one token
    with the separators eaten. An unquoted token with no spaces is always
    safe to pass through as-is."""
    if interpreter.startswith("& "):
        return [interpreter[2:].strip().strip('"')]
    if " " not in interpreter:
        return [interpreter]
    try:
        return shlex.split(interpreter)
    except ValueError:
        return [interpreter]


def _run_interpreter(interpreter: str, code: str) -> str:
    """Run a snippet in the resolved interpreter; empty on any failure.

    A non-zero exit is a failure, which the promise above always intended
    and the code did not keep: it returned stderr whenever stdout was empty,
    so an interpreter that did not run had its complaint taken for an
    answer. A half-built .venv -- an interrupted `python -m venv`, a venv
    copied between machines -- reported its error message as the project's
    Python *version*, and /doctor duly displayed it.

    stderr is still read on success, because a working interpreter may warn
    there while answering on stdout.
    """
    argv = _interpreter_argv(interpreter) + ["-c", code]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or proc.stderr or "").strip()


def _module_importable(root: Path, module: str) -> bool | None:
    """Whether the module imports in the project's active interpreter.
    None when it cannot be determined (e.g. the interpreter is missing)."""
    interpreter = python_command(root)
    key = f"{interpreter}\x00{module}"
    if key in _IMPORT_CACHE:
        return _IMPORT_CACHE[key]
    out = _run_interpreter(
        interpreter,
        f"import importlib.util,sys; "
        f"sys.stdout.write(str(importlib.util.find_spec('{module}') is not None))",
    )
    result: bool | None = out.lower() == "true" if out else None
    _IMPORT_CACHE[key] = result
    return result


def pytest_installed(root: Path) -> bool | None:
    """Whether pytest imports in the active interpreter; None if unknown."""
    return _module_importable(root, "pytest")


def pytest_asyncio_installed(root: Path) -> bool | None:
    """Whether pytest-asyncio imports in the active interpreter."""
    return _module_importable(root, "pytest_asyncio")


_ASYNC_TEST = re.compile(r"^\s*async\s+def\s+test", re.MULTILINE)


def async_tests_present(root: Path) -> bool:
    """Whether the project's tests contain async tests (without importing
    anything). Scans a bounded number of files so a huge tree cannot stall."""
    bases = [root / "tests"] if (root / "tests").is_dir() else [root]
    scanned = 0
    for base in bases:
        for path in base.rglob("test_*.py"):
            scanned += 1
            if scanned > 400:
                return False
            try:
                if _ASYNC_TEST.search(
                        path.read_text(encoding="utf-8", errors="replace")):
                    return True
            except OSError:
                continue
    return False


def pytest_asyncio_configured(root: Path) -> bool:
    """Whether the project's config asks for pytest-asyncio behaviour."""
    if "asyncio_mode" in _read(root / "pyproject.toml") or \
            "asyncio" in _read(root / "pyproject.toml"):
        return True
    return any("asyncio_mode" in _read(root / name)
               for name in ("pytest.ini", "setup.cfg", "tox.ini"))


def pip_command(root: Path) -> str:
    """pip through the project's own interpreter, never a stray PATH pip."""
    return f"{python_command(root)} -m pip"


def quote_arg(text: str) -> str:
    """Quote one command argument for the shell this machine runs."""
    if " " not in text:
        return text
    if _is_windows():
        return '"' + text.replace('"', '\\"') + '"'
    return shlex.quote(text)


@dataclass(frozen=True)
class PythonEnvironment:
    interpreter: str
    version: str
    environment: str
    package_manager: str
    config_files: tuple[str, ...]
    pytest_installed: bool | None
    pytest_asyncio_installed: bool | None
    async_tests: bool
    test_runner: str | None


_CONFIG_FILES = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "requirements-dev.txt", "requirements-dev.in", "Pipfile", "poetry.lock",
    "uv.lock", "tox.ini", "noxfile.py", "pytest.ini", ".python-version",
    "MANIFEST.in", "environment.yml", "conda-env.yml",
)


def environment_info(root: Path) -> PythonEnvironment:
    """Everything /doctor and the agent need to know about the project's
    Python environment, from the files that are actually there."""
    root = Path(root)
    interpreter = python_command(root)

    version = _VERSION_CACHE.get(interpreter, "")
    if not version:
        version = _run_interpreter(
            interpreter, "import sys; print('%d.%d.%d' % sys.version_info[:3])")
        _VERSION_CACHE[interpreter] = version

    interp_text = interpreter.lower()
    environment = "system"
    if "windowsapps" in interp_text:
        environment = "windows-store-alias (broken)"
    elif any(marker in interp_text for marker in (
            ".venv\\", "/.venv/", "/.venv\\", "\\venv\\", "/venv/",
            "/venv\\", "\\env\\", "/env/", "\\envs\\", "/envs/")):
        environment = "virtualenv"
    elif "conda" in interp_text or os.environ.get("CONDA_PREFIX"):
        environment = "conda"
    elif (root / "poetry.lock").is_file():
        environment = "poetry"
    elif (root / "uv.lock").is_file():
        environment = "uv"
    elif (root / "Pipfile").is_file():
        environment = "pipenv"

    if (root / "uv.lock").is_file():
        package_manager = "uv"
    elif (root / "poetry.lock").is_file():
        package_manager = "poetry"
    elif (root / "Pipfile").is_file():
        package_manager = "pipenv"
    elif (root / "environment.yml").is_file():
        package_manager = "conda"
    else:
        package_manager = "pip"

    config_files = tuple(name for name in _CONFIG_FILES
                         if (root / name).is_file())

    runner = detect(root)
    return PythonEnvironment(
        interpreter=interpreter,
        version=version,
        environment=environment,
        package_manager=package_manager,
        config_files=config_files,
        pytest_installed=pytest_installed(root),
        pytest_asyncio_installed=pytest_asyncio_installed(root),
        async_tests=async_tests_present(root),
        test_runner=runner.name if runner else None,
    )


def focused_command(root: Path, changed: list[Path]) -> str | None:
    """A pytest command limited to test files plausibly affected by the
    changed files, when the project runs pytest. None when there is nothing
    focused to run -- the caller should fall back to the full suite."""
    runner = detect(root)
    if runner is None or runner.name != "pytest":
        return None
    from .navigation import covering_tests
    # Which tests can this change break? A test that imports the module --
    # directly, or through something that does -- can. Matching test file
    # names against source file names, which is what this used to do,
    # averaged about 30% recall here and selected nothing at all for a
    # module no test is named after.
    files = covering_tests(root, [Path(c) for c in changed])
    if not files:
        return None
    parts = [f"{python_command(root)} -m pytest"]
    parts.extend(quote_arg(path) for path in files)
    return " ".join(parts)
