"""Launching the applications this machine actually has installed.

The model's job is intent -- *that* the user wants an application opened and
*what* they called it. This tool's job is everything after that: resolve the
name against the applications discovered on this machine, launch the real
shortcut, and report honestly which of those steps happened. It never
substitutes one application for another and never falls back to Explorer:
the OS catalog is the only source of launch targets. The optional ``path``
argument is *not* a launch target -- the application still comes from the
catalog -- it is merely handed to the resolved application as an argument,
so the model can open a specific file in the app it launched. See
appcatalog.py for where the applications come from.
"""

from __future__ import annotations

import asyncio
import os
import shutil
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
    path = Field(
        str,
        "Optional absolute path of a file or folder to open in the "
        "application, for example 'C:\\\\Users\\\\elliot\\\\Desktop\\\\text.py'. "
        "The application is still chosen by `query`; this path is only "
        "handed to it as an argument. Use it when the user asks to open "
        "a specific file in the application.",
        default="",
    )
    command = Field(
        str,
        "A shell command to run inside the application, for terminal "
        "emulators only -- for example 'python3 main.py' when the user "
        "says 'open konsole and run main.py'. The terminal stays open "
        "afterwards so its output can be read. Leave empty for anything "
        "that is not a terminal.",
        default="",
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
            "If the user wants a specific file opened in the application (for "
            "example 'create text.py and open it in VS Code'), pass that file's "
            "absolute path in the `path` argument as well -- the application is "
            "still resolved from `query`, and the path is handed to it as an "
            "argument so it opens with that file loaded.\n\n"
            "To open a terminal already running something -- 'open konsole "
            "and run python3 main.py' -- pass the command in `command` as "
            "well. It only works on terminal emulators; anything else "
            "refuses rather than opening without it.\n\n"
            "Several applications in one message are several calls, in the "
            "order the user named them.\n\n"
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

        return await self._launch(resolution.entry, (args.path or "").strip(),
                                  (args.command or "").strip())

    async def _launch(self, entry: AppEntry, open_path: str = "",
                      command: str = "") -> ToolResult:
        argv = None
        if command:
            argv = terminal_argv(entry, command)
            if argv is None:
                # Refused rather than launched without it. Dropping the
                # command silently would open the application and report
                # success, and the user would be told their script was run
                # by something that never saw it.
                return ToolResult.failure(
                    f"'{entry.name}' is not a terminal this can run a "
                    f"command in, so nothing was launched. Tell the user, "
                    f"or use the shell tool to run the command here.",
                    status="not_a_terminal", application=entry.name,
                    query=entry.name)
        try:
            if argv is not None:
                await _shell_launch(argv)
            else:
                await _launch_entry(entry, open_path)
        except OSError as exc:
            return ToolResult.failure(
                f"Found '{entry.name}' but launching it failed: {exc}. "
                "Nothing was launched.",
                status="failed", application=entry.name,
                source=entry.source, path=str(entry.path))
        meta = {
            "status": "launched", "application": entry.name,
            "source": entry.source, "path": str(entry.path),
        }
        if open_path:
            meta["opened"] = open_path
        if command:
            meta["command"] = command
        did = (f"Launched {entry.name}"
               + (f" running `{command}`" if command
                  else f" with {open_path}" if open_path else ""))
        return ToolResult.success(
            did + ". This completes the request -- reply with a short "
            "confirmation and stop; do not perform further tool calls "
            "unless the user asked for additional work.",
            display=(f"launched {entry.name}"
                     + (f" running {command}" if command else "")),
            said=did + ".",
            terminal=True,
            **meta)


# -- terminals ---------------------------------------------------------------

_STAYS_OPEN = "; exec bash"
"""Kept alive after the command finishes.

"Open a terminal and run this" means the window is there to be read. Left
to exit, a command that takes a second flashes a window and closes it, and
the output -- the entire reason for asking -- is gone before anyone sees
it."""

TERMINALS: dict[str, tuple[str, ...]] = {
    # The flag that means "the rest of this is the program to run" differs
    # per terminal and there is no convention to fall back on: -e, -x, --,
    # or nothing at all. Guessing wrong does not fail cleanly -- it opens a
    # terminal that ignores the command, or one that treats it as a file
    # name -- so only terminals whose spelling is known are offered.
    "konsole": ("-e",),
    "gnome-terminal": ("--",),
    "kgx": ("--",),
    "ptyxis": ("--",),
    "xfce4-terminal": ("-x",),
    "mate-terminal": ("--",),
    "lxterminal": ("-e",),
    "terminator": ("-x",),
    "tilix": ("-e",),
    "deepin-terminal": ("-e",),
    "alacritty": ("-e",),
    "kitty": (),
    "foot": (),
    "wezterm": ("start", "--"),
    "xterm": ("-e",),
    "urxvt": ("-e",),
    "rxvt": ("-e",),
    "st": ("-e",),
    "qterminal": ("-e",),
}


def terminal_argv(entry: AppEntry, command: str) -> list[str] | None:
    """How to start ``entry`` running ``command``, or None if it cannot.

    The binary is looked up on PATH rather than taken from the entry: a
    .desktop file is a description, and `gio launch` has nowhere to put a
    command. The registry above is keyed on the executable's own name,
    which is what both the desktop entry and the PATH hit agree on.

    None rather than a guess. A terminal whose flag is not known here would
    be handed a command it silently ignores, and "I opened konsole and ran
    your script" would be a sentence about something that did not happen.
    """
    for candidate in (entry.path.stem, entry.name):
        key = str(candidate).strip().lower()
        if key not in TERMINALS:
            continue
        binary = shutil.which(key) or (
            str(entry.path) if entry.path.stem.lower() == key
            and entry.path.suffix.lower() not in (".desktop", ".lnk")
            else "")
        if not binary:
            return None
        return [binary, *TERMINALS[key], "bash", "-c", command + _STAYS_OPEN]
    return None


def is_a_terminal(entry: AppEntry) -> bool:
    return any(str(c).strip().lower() in TERMINALS
               for c in (entry.path.stem, entry.name))


async def _launch_entry(entry: AppEntry, open_path: str = "") -> None:
    """Start the application the way the OS would from its own UI.

    A .lnk is started through the shell so its target, arguments and working
    directory arrive exactly as the installer wrote them -- the same thing a
    double-click does. When ``open_path`` is given it is appended as an
    argument, so the application opens with that file loaded (ShellExecute
    and `open -a` both hand extra arguments to the target). This call
    returns once the launch has been handed to the OS; whether the
    application keeps running is the application's own affair. The launch
    is detached from wynxo's console: a GUI application's console children
    (VS Code's node processes are the loud example) must not be able to
    print into the UI mid-session.
    """
    arg = [open_path] if open_path else []
    path = entry.path
    if path.suffix.lower() == ".lnk":
        # A .lnk is a Windows shortcut: a binary description of a target,
        # not something to execute. Handing it to the shell is the only way
        # to start it, so it routes here on every platform rather than only
        # on win32 -- off Windows _startfile says so plainly, where the old
        # fallthrough tried to exec the shortcut's own bytes and reported
        # the target application as broken with a bare "Permission denied".
        await asyncio.to_thread(_startfile, str(path), arg[0] if arg else "")
        return
    if sys.platform == "darwin" and path.suffix == ".app":
        await _shell_launch(["open", "-a", str(path)] + arg)
        return
    if entry.source == "linux_desktop":
        await _launch_desktop(path, arg[0] if arg else "")
        return
    await _shell_launch([str(path)] + arg)


async def _launch_desktop(path, open_arg: str = "") -> None:
    """A Linux .desktop entry, via whatever the desktop offers.

    ``gio launch`` runs the entry as the file manager would, field codes and
    all; ``gtk-launch`` takes the entry id instead. Neither reconstructs the
    command line by hand, which is where quoting bugs are born. Both accept
    a file URI argument, so a requested file can be handed to the launched
    application.
    """
    arg = [_file_uri(open_arg)] if open_arg else []
    if shutil.which("gio"):
        await _shell_launch(["gio", "launch", str(path)] + arg)
        return
    if shutil.which("gtk-launch"):
        await _shell_launch(["gtk-launch", path.stem] + arg)
        return
    raise OSError(
        "no gio or gtk-launch available to start this application")


def _file_uri(path: str) -> str:
    """A file:// URI for gio/gtk-launch; the raw path if it cannot be
    expressed as one (relative paths, odd names)."""
    from pathlib import Path
    try:
        return Path(path).resolve().as_uri()
    except (OSError, ValueError):
        return path


def _startfile(path: str, arg: str = "") -> None:
    """Launch a shortcut, detached from wynxo's console.

    os.startfile hands the .lnk to ShellExecute -- exactly what a double-
    click does -- but on Windows the launched process tree can inherit
    wynxo's console and print into it, shredding the UI (VS Code's node
    extension host writes deprecation noise to stderr on every start).
    ``cmd /c start`` keeps the same ShellExecute semantics for the shortcut
    while handing the child null handles and a process group of its own, so
    nothing it prints can reach our screen. A trailing ``arg`` is handed to
    the shortcut's target, so the application opens with that file loaded.
    Kept as a module function so tests can stand in for it.
    """
    if os.name == "nt":
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        argv = ["cmd", "/c", "start", "", path]
        if arg:
            argv.append(arg)
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation,
        )
        return
    # A .lnk reaches here only when the catalog carries a Windows shortcut on
    # a machine that cannot start one -- a synced profile, a mounted Windows
    # drive. OSError rather than AttributeError from a missing os.startfile,
    # so the caller reports it as a launch failure and not as a crash.
    raise OSError(f"{path} is a Windows shortcut and this is not Windows")


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
