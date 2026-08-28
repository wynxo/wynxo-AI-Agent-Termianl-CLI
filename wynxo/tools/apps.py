"""Launching the applications this machine actually has installed.

The model's job is intent -- *that* the user wants an application opened and
*what* they called it. This tool's job is everything after that: resolve the
name against the applications discovered on this machine, launch the real
shortcut, and report honestly which of those steps happened. It never
substitutes one application for another, never falls back to Explorer, and
never accepts a path from the model -- the OS catalog is the only source of
launch targets. See appcatalog.py for where the applications come from.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

from ..schema import Field, Schema
from .appcatalog import AppEntry, ApplicationCatalog
from .base import Tool, ToolResult


class LaunchApplicationInput(Schema):
    query = Field(
        str,
        "The application the user wants, as they described it, for example "
        "'Visual Studio Code', 'vscode', 'LibreWolf' or 'Steam'. A name, "
        "never a file path.",
    )


class LaunchApplication(Tool):
    name = "launch_application"
    description = (
        "Launch a GUI application installed on the user's computer. Provide "
        "the application name or query exactly as the user described it. The "
        "system searches the applications actually installed on this machine "
        "(Start Menu shortcuts on Windows, /Applications on macOS, .desktop "
        "entries on Linux) and launches a matching application. Never guess "
        "and never substitute another application; if nothing installed "
        "matches, the tool says so and you should tell the user.\n\n"
        "A successful launch completes the request: reply with a short "
        "confirmation and stop. Do not continue inspecting files, planning "
        "or running further tools unless the user asked for additional work "
        "in the same message -- if they did, do that work before launching, "
        "or launch last."
    )
    Input = LaunchApplicationInput
    mutating = True
    concurrency_safe = False

    def __init__(self, workspace, boundary=None, shield=None,
                 catalog: ApplicationCatalog | None = None):
        super().__init__(workspace, boundary, shield)
        self.catalog = catalog or ApplicationCatalog()

    async def run(self, args: LaunchApplicationInput) -> ToolResult:
        query = (args.query or "").strip()
        if not query:
            return ToolResult.failure(
                "Give the application's name, for example 'Visual Studio Code'.")

        resolution = self.catalog.resolve(query)
        if resolution.status == "not_found":
            # An application installed a minute ago should be findable
            # without restarting wynxo; one clean rescan on a miss buys
            # that without rescanning on every lookup.
            self.catalog.refresh()
            resolution = self.catalog.resolve(query)

        if resolution.status == "path_query":
            return ToolResult.failure(
                f"'{query}' looks like a file path. Give the application's "
                "name instead -- only applications discovered on this "
                "machine can be launched.",
                status="path_query", query=query)

        if resolution.status == "not_found":
            return ToolResult.failure(
                f"Could not find an installed application matching "
                f"'{query}'. Nothing was launched. Tell the user, rather "
                "than opening something else.",
                not_found=True, query=query)

        if resolution.status == "ambiguous":
            names = ", ".join(f"'{c.name}'" for c in resolution.candidates)
            return ToolResult.failure(
                f"'{query}' matches several installed applications: {names}. "
                "Ask the user which one they mean instead of choosing.",
                ambiguous=True, candidates=[c.name for c in resolution.candidates],
                query=query)

        return await self._launch(resolution.entry)

    async def _launch(self, entry: AppEntry) -> ToolResult:
        try:
            await _launch_entry(entry)
        except OSError as exc:
            return ToolResult.failure(
                f"Found '{entry.name}' but launching it failed: {exc}. "
                "Nothing was launched.",
                status="failed", application=entry.name,
                source=entry.source, path=str(entry.path))
        return ToolResult.success(
            f"Launched {entry.name}. This completes the request -- reply "
            "with a short confirmation and stop; do not perform further "
            "tool calls unless the user asked for additional work.",
            display=f"launched {entry.name}",
            terminal=True,
            status="launched", application=entry.name,
            source=entry.source, path=str(entry.path))


async def _launch_entry(entry: AppEntry) -> None:
    """Start the application the way the OS would from its own UI.

    A .lnk is started through the shell so its target, arguments and working
    directory arrive exactly as the installer wrote them -- the same thing a
    double-click does. This call returns once the launch has been handed to
    the OS; whether the application keeps running is the application's own
    affair. The launch is detached from wynxo's console: a GUI application's
    console children (VS Code's node processes are the loud example) must
    not be able to print into the UI mid-session.
    """
    path = entry.path
    if sys.platform == "win32" and path.suffix.lower() == ".lnk":
        await asyncio.to_thread(_startfile, str(path))
        return
    if sys.platform == "darwin" and path.suffix == ".app":
        await _shell_launch(["open", str(path)])
        return
    if entry.source == "linux_desktop":
        await _launch_desktop(path)
        return
    await _shell_launch([str(path)])


async def _launch_desktop(path) -> None:
    """A Linux .desktop entry, via whatever the desktop offers.

    ``gio launch`` runs the entry as the file manager would, field codes and
    all; ``gtk-launch`` takes the entry id instead. Neither reconstructs the
    command line by hand, which is where quoting bugs are born.
    """
    import shutil
    if shutil.which("gio"):
        await _shell_launch(["gio", "launch", str(path)])
        return
    if shutil.which("gtk-launch"):
        await _shell_launch(["gtk-launch", path.stem])
        return
    raise OSError(
        "no gio or gtk-launch available to start this application")


def _startfile(path: str) -> None:
    """Launch a shortcut, detached from wynxo's console.

    os.startfile hands the .lnk to ShellExecute -- exactly what a double-
    click does -- but on Windows the launched process tree can inherit
    wynxo's console and print into it, shredding the UI (VS Code's node
    extension host writes deprecation noise to stderr on every start).
    ``cmd /c start`` keeps the same ShellExecute semantics for the shortcut
    while handing the child null handles and a process group of its own, so
    nothing it prints can reach our screen. Kept as a module function so
    tests can stand in for it.
    """
    if os.name == "nt":
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            ["cmd", "/c", "start", "", path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation,
        )
        return
    os.startfile(path)      # noqa: S606 -- non-Windows, never reached for a .lnk


async def _shell_launch(argv: list[str]) -> None:
    """Start a launch command without waiting on the application.

    stdout/stderr are discarded rather than piped: some GUI applications
    write chatter forever, and an unread full pipe blocks them mid-launch.
    """
    await asyncio.to_thread(
        subprocess.Popen,
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
