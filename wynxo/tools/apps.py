"""Launch installed applications, with reliable terminal command execution."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..schema import Field, Schema
from .appcatalog import AppEntry, ApplicationCatalog
from .base import Tool, ToolResult


class LaunchApplicationInput(Schema):
    query = Field(
        str,
        "The installed application the user wants, such as 'Konsole', "
        "'Visual Studio Code', 'vscode' or 'Steam'. A name, never a file path.",
    )
    path = Field(
        str,
        "Optional absolute path of a file or folder to open in the application.",
        default="",
    )
    command = Field(
        str,
        "For terminal emulators only: the shell command to execute in the new "
        "terminal. Use this when the user says 'open Konsole and run X'. "
        "Leave empty for normal GUI applications.",
        default="",
    )


class LaunchApplication(Tool):
    name = "launch_application"
    description = (
        "Launch an installed GUI application. Resolve the application from "
        "the machine's application catalog; never guess or substitute. "
        "If the user asks to open a terminal and run a command, pass the "
        "terminal name in `query` AND the command in `command`. For example, "
        "'open Konsole and run echo hello' means query='Konsole', "
        "command='echo hello'. The command must actually execute before the "
        "tool reports success. Keep the terminal open so the user can see "
        "the output. A successful launch completes that request; do not "
        "invent follow-up tool calls."
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
                "Give the application's name, for example 'Visual Studio Code'."
            )

        resolution = self.catalog.resolve(query)
        if resolution.status == "not_found":
            self.catalog.refresh()
            resolution = self.catalog.resolve(query)

        if resolution.status == "path_query":
            return ToolResult.failure(
                f"'{query}' looks like a file path. Give the application's name.",
                status="path_query", query=query)

        if resolution.status == "not_found":
            return ToolResult.failure(
                f"Could not find an installed application matching '{query}'. "
                "Nothing was launched.",
                not_found=True, query=query)

        if resolution.status == "ambiguous":
            names = ", ".join(f"'{c.name}'" for c in resolution.candidates)
            return ToolResult.failure(
                f"'{query}' matches several installed applications: {names}. "
                "Ask which one the user means.",
                ambiguous=True,
                candidates=[c.name for c in resolution.candidates],
                query=query)

        return await self._launch(
            resolution.entry,
            (args.path or "").strip(),
            (args.command or "").strip(),
        )

    async def _launch(self, entry: AppEntry, open_path: str = "",
                      command: str = "") -> ToolResult:
        try:
            if command:
                argv = terminal_argv(entry, command, self.workspace)
                if argv is None:
                    return ToolResult.failure(
                        f"'{entry.name}' is not a supported terminal emulator, "
                        "so the command was not run.",
                        status="not_a_terminal", application=entry.name,
                        query=entry.name)
                await _shell_launch(argv)
            else:
                await _launch_entry(entry, open_path)
        except OSError as exc:
            return ToolResult.failure(
                f"Found '{entry.name}' but launching it failed: {exc}. "
                "Nothing was launched.",
                status="failed", application=entry.name,
                source=entry.source, path=str(entry.path))

        did = f"Launched {entry.name}"
        if command:
            did += f" running `{command}`"
        elif open_path:
            did += f" with {open_path}"

        return ToolResult.success(
            did + ".",
            display=(f"launched {entry.name}"
                     + (f" running {command}" if command else "")),
            said=did + ".",
            terminal=bool(command),
            status="launched",
            application=entry.name,
            source=entry.source,
            path=str(entry.path),
            **({"opened": open_path} if open_path else {}),
            **({"command": command} if command else {}),
        )


# ---------------------------------------------------------------------------
# Terminal execution
# ---------------------------------------------------------------------------

TERMINALS: dict[str, tuple[str, ...]] = {
    # --separate prevents Konsole from handing the request to an existing
    # process. --hold keeps the session visible after the command exits.
    # -e is deliberately last because Konsole consumes every following arg.
    "konsole": ("--separate", "--hold", "-e"),
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


def terminal_argv(entry: AppEntry, command: str,
                  workspace: str = "") -> list[str] | None:
    """Build a terminal argv that executes command in WYNXO's workspace."""
    for candidate in (entry.path.stem, entry.name):
        key = str(candidate).strip().lower()
        if key not in TERMINALS:
            continue

        binary = shutil.which(key)
        if not binary and entry.path.suffix.lower() not in (".desktop", ".lnk"):
            binary = str(entry.path)
        if not binary:
            return None

        shell_command = command
        if workspace:
            try:
                workdir = str(Path(workspace).expanduser().resolve())
                if Path(workdir).is_dir():
                    shell_command = f"cd -- {shlex.quote(workdir)} && {command}"
            except (OSError, ValueError):
                pass

        # The shell receives the command as one argument after -lc. This keeps
        # pipes, quotes, &&, redirects and other normal shell syntax intact.
        return [binary, *TERMINALS[key], "bash", "-lc", shell_command]

    return None


def is_a_terminal(entry: AppEntry) -> bool:
    return any(
        str(c).strip().lower() in TERMINALS
        for c in (entry.path.stem, entry.name)
    )


# ---------------------------------------------------------------------------
# Normal application launching
# ---------------------------------------------------------------------------

async def _launch_entry(entry: AppEntry, open_path: str = "") -> None:
    arg = [open_path] if open_path else []
    path = entry.path

    if path.suffix.lower() == ".lnk":
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
    arg = [_file_uri(open_arg)] if open_arg else []

    if shutil.which("gio"):
        await _shell_launch(["gio", "launch", str(path)] + arg)
        return

    if shutil.which("gtk-launch"):
        await _shell_launch(["gtk-launch", path.stem] + arg)
        return

    raise OSError("no gio or gtk-launch available to start this application")


def _file_uri(path: str) -> str:
    try:
        return Path(path).resolve().as_uri()
    except (OSError, ValueError):
        return path


def _startfile(path: str, arg: str = "") -> None:
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

    raise OSError(f"{path} is a Windows shortcut and this is not Windows")


async def _shell_launch(argv: list[str]) -> None:
    """Launch detached from WYNXO's terminal without inheriting its streams."""
    await asyncio.to_thread(
        subprocess.Popen,
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
