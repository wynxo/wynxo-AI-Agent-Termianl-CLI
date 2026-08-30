#!/usr/bin/env python3
"""Windows acceptance for wynxo, on a real Windows machine.

Everything wynxo does that is Windows-specific has deterministic coverage
in tests/test_windows_surface.py, which runs anywhere by faking the
platform. Faking it is not the same as being there: nothing on a Linux box
can tell you whether Windows Terminal restores the screen on exit, whether
Shift+drag selects, or whether taskkill really stops a process tree.

So this runs the parts a machine can decide by itself and prints the parts
only a person can, as a checklist. Run it in each host you care about --
Windows Terminal, PowerShell, cmd:

    python scripts\\windows_acceptance.py

It changes nothing and installs nothing. Automatic checks that fail print
what was expected next to what happened, which is the part worth pasting
into a bug report.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
    print(f"{mark} {name}" + (f"\n         {detail}" if detail else ""))


def check(name):
    """Run a check, turning an exception into a failure rather than a crash.

    A check may return SKIP as its verdict when it could not set itself up.
    That is not a pass: a check that reports success without having tested
    anything is worse than no check, because it is the one you stop looking
    at.
    """
    def wrap(fn):
        try:
            ok, detail = fn()
        except Exception as exc:                       # noqa: BLE001
            record(name, FAIL, f"{type(exc).__name__}: {exc}")
        else:
            record(name, ok if ok in (PASS, FAIL, SKIP)
                   else (PASS if ok else FAIL), detail)
        return fn
    return wrap


# ---------------------------------------------------------------- the host --

def host() -> str:
    """Which terminal this is, as far as it will say."""
    if os.environ.get("WT_SESSION"):
        return "Windows Terminal"
    if os.environ.get("TERM_PROGRAM"):
        return os.environ["TERM_PROGRAM"]
    parent = os.environ.get("PSModulePath")
    return "PowerShell" if parent else "cmd or unknown"


def main() -> int:
    if os.name != "nt":
        print("This is for Windows. On anything else, run:")
        print("    python -m pytest tests/test_windows_surface.py")
        return 2

    print(f"wynxo Windows acceptance -- {host()}")
    print(f"python {sys.version.split()[0]}, code page "
          f"{subprocess.run(['chcp'], capture_output=True, text=True, shell=True).stdout.strip()}")
    print()

    workspace = Path(tempfile.mkdtemp(prefix="wynxo-accept-"))

    @check("wynxo imports and reports its version")
    def _():
        out = subprocess.run([sys.executable, "-m", "wynxo", "--version"],
                             capture_output=True, text=True, timeout=120)
        return out.returncode == 0, (out.stdout or out.stderr).strip()[:120]

    @check("the shell tool runs a command and reads its output")
    def _():
        import asyncio

        from wynxo.tools import build_registry

        shell = build_registry(workspace, allow_shell=True).get("shell")
        result = asyncio.run(shell.run(shell.Input(
            command="echo hello-from-wynxo", timeout=60)))
        return "hello-from-wynxo" in result.output, result.output.strip()[:120]

    @check("CRLF from a child arrives as line endings, not blank rows")
    def _():
        import asyncio

        from wynxo.tools import build_registry

        shell = build_registry(workspace, allow_shell=True).get("shell")
        result = asyncio.run(shell.run(shell.Input(
            command="echo one& echo two", timeout=60)))
        lines = [l for l in result.output.splitlines() if l.strip()]
        return lines[:2] == ["one", "two"], repr(result.output[:80])

    @check("a process tree is actually killed (taskkill /T)")
    def _():
        import asyncio

        from wynxo.tools import build_registry
        from wynxo.tools.shell import _BACKGROUND, shutdown_background

        marker = workspace / "ticks.txt"
        shell = build_registry(workspace, allow_shell=True).get("shell")

        async def start():
            return await shell.run(shell.Input(
                command=f'for /L %i in (1,0,2) do @(echo tick >> "{marker}" '
                        f'& timeout /t 1 /nobreak > nul)',
                timeout=600, background=True))

        result = asyncio.run(start())
        if not result.ok:
            return FAIL, result.output[:120]
        time.sleep(3)
        before = marker.stat().st_size if marker.exists() else 0
        shutdown_background()
        _BACKGROUND.clear()
        time.sleep(3)
        after = marker.stat().st_size if marker.exists() else 0
        if before == 0:
            # Nothing was ever written, so killing it proves nothing. Saying
            # PASS here would be the check reporting on itself.
            return SKIP, "the background command never produced output; " \
                         "this check did not test anything"
        return before == after, \
            f"the file grew from {before} to {after} bytes after the kill"

    @check("the interpreter is discovered and answers")
    def _():
        from wynxo import testing

        info = testing.environment_info(workspace)
        return bool(info.version), \
            f"interpreter={info.interpreter} version={info.version!r}"

    @check("a .venv is preferred over the system interpreter")
    def _():
        from wynxo import testing

        project = Path(tempfile.mkdtemp(prefix="wynxo-venv-"))
        made = subprocess.run([sys.executable, "-m", "venv",
                               str(project / ".venv")],
                              capture_output=True, timeout=600)
        if made.returncode != 0:
            return False, "could not create a venv here"
        command = testing.python_command(project)
        return ".venv" in command, command

    @check("pytest is detected in a project that has tests")
    def _():
        from wynxo import testing

        project = Path(tempfile.mkdtemp(prefix="wynxo-tests-"))
        (project / "tests").mkdir()
        (project / "tests" / "test_a.py").write_text(
            "def test_a():\n    assert True\n", encoding="utf-8")
        runner = testing.detect(project)
        return runner is not None and "pytest" in runner.command, \
            runner.command if runner else "nothing detected"

    @check("installed applications are discovered from the real system")
    def _():
        from wynxo.tools.appcatalog import ApplicationCatalog, Sources

        catalog = ApplicationCatalog(Sources.for_platform())
        entries = catalog.entries()
        names = [e.name for e in entries[:6]]
        return len(entries) > 0, f"{len(entries)} found, e.g. {names}"

    @check("a well-known application resolves by name")
    def _():
        from wynxo.tools.appcatalog import ApplicationCatalog, Sources

        catalog = ApplicationCatalog(Sources.for_platform())
        for query in ("notepad", "explorer", "calculator", "terminal"):
            resolution = catalog.resolve(query)
            if resolution.matched:
                return True, f"{query!r} -> {resolution.entry.name}"
        return False, "none of notepad/explorer/calculator/terminal resolved"

    @check("Unicode reaches the terminal without raising")
    def _():
        sys.stdout.write("         日本語  한국어  العربية  🎉  café  ─│╭╮╰╯\n")
        sys.stdout.flush()
        return True, "look at the line above: boxes or question marks mean " \
                     "the code page or font, not wynxo"

    @check("terminal control from tool output is still neutralised")
    def _():
        import io

        from wynxo.ui import UI

        ui = UI()
        sink = io.StringIO()
        ui.console.file = sink
        payload = "\x1b]52;c;cHduZWQ=\x07\x1b[?1049h\x1b[1;5r\x1b[2J\x9b2J"
        ui.tool_result("read_file", True, payload, payload)
        ui.error(payload)
        drawn = sink.getvalue()
        leaked = [n for n, seq in (("clipboard", "\x1b]52;"),
                                   ("alt screen", "\x1b[?1049h"),
                                   ("scroll region", "\x1b[1;5r"),
                                   ("erase", "\x1b[2J"),
                                   ("C1 CSI", "\x9b")) if seq in drawn]
        return not leaked, f"leaked: {leaked}" if leaked else "nothing got out"

    print()
    print("=" * 62)
    print("BY HAND -- start wynxo and work through these in this host.")
    print("=" * 62)
    for i, item in enumerate(MANUAL, 1):
        print(f"  {i:2d}. {item}")

    print()
    failed = [r for r in results if r[1] == FAIL]
    skipped = [r for r in results if r[1] == SKIP]
    passed = len(results) - len(failed) - len(skipped)
    print(f"automatic: {passed} passed, {len(failed)} failed, "
          f"{len(skipped)} could not run")
    for label, rows in (("failed", failed), ("could not run", skipped)):
        if rows:
            print(f"{label}:")
            for name, _status, detail in rows:
                print(f"  - {name}: {detail}")
    print("\nThe checklist above this is the half that matters most, and "
          "no script can do it.")
    return 1 if failed else 0


MANUAL = [
    "It starts, draws a header, a conversation, a composer and a footer, "
    "and the composer sits on the bottom row.",
    "Type a multi-line message with Alt-Enter: the composer grows upward "
    "and the footer does not move.",
    "Send a message: text streams in a word at a time rather than "
    "arriving in one lump.",
    "Mouse wheel scrolls the conversation.",
    "PageUp and PageDown scroll, and the composer keeps the cursor.",
    "Press F2: the footer says [select mode]. Drag to select text, and "
    "copy it (Ctrl-C or right-click, per your host).",
    "Press F2 again: the wheel scrolls again, and typing still works.",
    "Resize the window narrow and wide: nothing is stranded, the composer "
    "stays at the bottom, the footer stays on the last row.",
    "/model, arrow to another model, Enter. Then /model again and Escape. "
    "The screen is not corrupted either time.",
    "Ask for a file edit: a diff card appears with +/- counts, and Ctrl-D "
    "expands and collapses it.",
    "Ask for something that needs permission: the question is VISIBLE in "
    "the composer, and answering it works.",
    "Press Ctrl-C at that permission prompt: it goes away, and the first "
    "character of your next message is not eaten.",
    "Start a long command and press Ctrl-C: it stops, and Task Manager "
    "shows no leftover child processes.",
    "Ask it to open an application by name, in your own words.",
    "Quit with /quit: the terminal is left exactly as it was -- no "
    "alternate screen, no stuck colours, no lost cursor.",
]


if __name__ == "__main__":
    sys.exit(main())
