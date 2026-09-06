"""Discover and launch installed applications without guessing.

The catalog is the source of truth. The tools in this module only act on
entries the operating system reported, so a model can search for an app or
ask for "any terminal" without inventing an executable name or silently
substituting some unrelated program.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..schema import Field, Schema
from .appcatalog import AppEntry, ApplicationCatalog, condense, normalize_name
from .base import Tool, ToolResult


# ---------------------------------------------------------------------------
# Terminal discovery and execution
# ---------------------------------------------------------------------------

# Terminal programs for which Wynxo knows a safe argv contract for running a
# shell command. Generic launching can recognise a wider set below; command
# execution is deliberately narrower because the switches differ by terminal.
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

# Canonical terminal ids -> names an OS catalog may actually report. The
# platform-native entries are launchable even when Wynxo intentionally does
# not claim a command-line contract for them.
_TERMINAL_ALIASES: dict[str, tuple[str, ...]] = {
    **{key: (key,) for key in TERMINALS},
    "windows-terminal": (
        "windows terminal",
        "windows terminal preview",
        "wt",
        "wt.exe",
    ),
    "terminal-app": ("terminal", "terminal.app"),
    "iterm2": ("iterm", "iterm2", "iterm.app", "iterm2.app"),
    "warp": ("warp", "warp.app"),
}

_GENERIC_TERMINAL_QUERIES = frozenset({
    "terminal",
    "a terminal",
    "any terminal",
    "terminal any",
    "some terminal",
    "default terminal",
    "terminal emulator",
    "a terminal emulator",
    "any terminal emulator",
})

_DESKTOP_TERMINAL_PREFERENCE = {
    "kde": ("konsole", "kitty", "wezterm", "alacritty", "foot", "xterm"),
    "plasma": ("konsole", "kitty", "wezterm", "alacritty", "foot", "xterm"),
    "gnome": ("ptyxis", "kgx", "gnome-terminal", "kitty", "wezterm", "xterm"),
    "xfce": ("xfce4-terminal", "kitty", "alacritty", "xterm"),
    "mate": ("mate-terminal", "kitty", "alacritty", "xterm"),
    "lxqt": ("qterminal", "kitty", "alacritty", "xterm"),
    "lxde": ("lxterminal", "kitty", "alacritty", "xterm"),
}


def _terminal_key(entry: AppEntry) -> str:
    """Canonical terminal id represented by an installed app entry."""
    forms = {condense(str(entry.name)), condense(str(entry.path.stem))}
    for key, aliases in _TERMINAL_ALIASES.items():
        alias_forms = {condense(key), *(condense(alias) for alias in aliases)}
        if forms & alias_forms:
            return key
    return ""


def _terminal_key_matches(key: str, wanted: str) -> bool:
    forms = {condense(key)}
    forms.update(condense(alias) for alias in _TERMINAL_ALIASES.get(key, ()))
    return condense(wanted) in forms


def is_a_terminal(entry: AppEntry) -> bool:
    return bool(_terminal_key(entry))


def terminal_entries(catalog: ApplicationCatalog) -> list[AppEntry]:
    """Recognised terminal emulators that are genuinely installed."""
    return [entry for entry in catalog.entries() if is_a_terminal(entry)]


def _terminal_preference() -> tuple[str, ...]:
    """Preference order for a generic terminal request.

    This is not an installed-app list: every returned candidate is still
    checked against the catalog. Desktop conventions only choose between
    several *real* matches, while $TERMINAL wins when the user configured it.
    """
    preferred: list[str] = []

    configured = (os.environ.get("TERMINAL") or "").strip()
    if configured:
        try:
            executable = shlex.split(configured)[0]
        except (ValueError, IndexError):
            executable = configured.split()[0] if configured.split() else ""
        if executable:
            preferred.append(Path(executable).name.lower())

    if sys.platform == "win32":
        preferred.append("windows-terminal")
    elif sys.platform == "darwin":
        preferred.extend(("terminal-app", "iterm2", "warp"))

    desktop = " ".join(filter(None, (
        os.environ.get("XDG_CURRENT_DESKTOP", ""),
        os.environ.get("DESKTOP_SESSION", ""),
    ))).lower()
    for marker, order in _DESKTOP_TERMINAL_PREFERENCE.items():
        if marker in desktop:
            preferred.extend(order)
            break

    preferred.extend(_TERMINAL_ALIASES)
    # Preserve priority while making repeated values harmless.
    return tuple(dict.fromkeys(preferred))


def preferred_terminal(catalog: ApplicationCatalog) -> AppEntry | None:
    """Choose one installed recognised terminal for an explicitly generic ask."""
    entries = terminal_entries(catalog)
    if not entries:
        return None

    by_key: dict[str, AppEntry] = {}
    for entry in entries:
        key = _terminal_key(entry)
        if key and key not in by_key:
            by_key[key] = entry

    for wanted in _terminal_preference():
        for key, entry in by_key.items():
            if _terminal_key_matches(key, wanted):
                return entry

    return sorted(entries, key=lambda item: item.name.casefold())[0]


def is_generic_terminal_query(query: str) -> bool:
    return normalize_name(query) in _GENERIC_TERMINAL_QUERIES


def matching_applications(catalog: ApplicationCatalog, query: str = "",
                          *, limit: int = 20) -> tuple[list[AppEntry], int]:
    """Return useful catalog matches and the uncapped match count.

    The same resolver used by launch_application gets first say, so
    abbreviations such as ``vscode`` and harmless typos behave consistently
    between discovery and launching. A simple word search is the fallback
    for broad browsing queries that intentionally match several apps.
    """
    limit = max(1, min(50, int(limit)))
    text = (query or "").strip()
    entries = list(catalog.entries())

    if not text:
        return entries[:limit], len(entries)

    if is_generic_terminal_query(text):
        matches = terminal_entries(catalog)
        return matches[:limit], len(matches)

    resolution = catalog.resolve(text)
    if resolution.status == "matched" and resolution.entry is not None:
        return [resolution.entry], 1
    if resolution.status == "ambiguous":
        matches = list(resolution.candidates)
        return matches[:limit], len(matches)
    if resolution.status == "path_query":
        return [], 0

    words = normalize_name(text).split()
    matches = []
    for entry in entries:
        haystack = normalize_name(f"{entry.name} {entry.path}")
        if words and all(word in haystack for word in words):
            matches.append(entry)
    return matches[:limit], len(matches)


def terminal_argv(entry: AppEntry, command: str,
                  workspace: str = "") -> list[str] | None:
    """Build an argv for terminals whose command contract Wynxo knows."""
    key = _terminal_key(entry)
    if not key or key not in TERMINALS:
        return None

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


# ---------------------------------------------------------------------------
# Tool contracts
# ---------------------------------------------------------------------------

class ListApplicationsInput(Schema):
    query = Field(
        str,
        "Optional application name or category to search for, such as "
        "'vscode', 'browser', or 'terminal'. Leave empty to list a sample.",
        default="",
    )
    limit = Field(
        int,
        "Maximum number of installed applications to return (1-50).",
        default=20,
        ge=1,
        le=50,
    )
    refresh = Field(
        bool,
        "Rescan the operating system's application catalog before searching.",
        default=False,
    )


class ListApplications(Tool):
    name = "list_applications"
    description = (
        "List applications actually installed on this machine. Use this when "
        "the user asks what apps are available, asks for a category such as "
        "'terminal', or when a launch target is unclear. It is read-only and "
        "uses the same operating-system catalog as launch_application. Do not "
        "call it before every launch when the user already named an exact app."
    )
    Input = ListApplicationsInput
    mutating = False
    # The catalog has a shared lazy cache and refresh operation. Serialising
    # calls avoids two concurrent model tool calls scanning the same OS tree.
    concurrency_safe = False

    def __init__(self, workspace, boundary=None, shield=None,
                 catalog: ApplicationCatalog | None = None):
        super().__init__(workspace, boundary, shield)
        self.catalog = catalog or ApplicationCatalog()

    async def run(self, args: ListApplicationsInput) -> ToolResult:
        if args.refresh:
            await asyncio.to_thread(self.catalog.refresh)

        query = (args.query or "").strip()
        matches, total = await asyncio.to_thread(
            matching_applications, self.catalog, query, limit=args.limit)

        if not matches:
            message = (f"No installed applications match '{query}'." if query
                       else "No installed applications were discovered.")
            return ToolResult.success(
                message,
                display="no matching installed applications",
                query=query,
                count=0,
                total=0,
                applications=[],
            )

        lines = [f"{entry.name} [{entry.where}]" for entry in matches]
        if total > len(matches):
            lines.append(f"... {total - len(matches)} more match(es)")
        output = "\n".join(lines)
        display = (f"{len(matches)} installed app"
                   f"{'s' if len(matches) != 1 else ''}"
                   + (f" matching {query}" if query else ""))
        return ToolResult.success(
            output,
            display=display,
            query=query,
            count=len(matches),
            total=total,
            applications=[entry.name for entry in matches],
        )


class LaunchApplicationInput(Schema):
    query = Field(
        str,
        "The installed application the user wants, such as 'Konsole', "
        "'Visual Studio Code', 'vscode' or 'Steam'. For an explicitly generic "
        "request such as 'any terminal', pass 'terminal' rather than guessing "
        "a terminal name. A name or generic terminal request, never a file path.",
    )
    path = Field(
        str,
        "Optional absolute path of a file or folder to open in the application.",
        default="",
    )
    command = Field(
        str,
        "For terminal emulators only: the shell command to execute in the new "
        "terminal. Use this when the user says 'open a terminal and run X'. "
        "Leave empty for normal GUI applications.",
        default="",
    )


class LaunchApplication(Tool):
    name = "launch_application"
    description = (
        "Launch an installed GUI application. Resolve named applications from "
        "the machine's application catalog; never guess or substitute. If the "
        "user explicitly says any/default terminal, pass query='terminal' and "
        "the tool deterministically chooses a recognised terminal that is "
        "actually installed. If the user asks to open a terminal and run a "
        "command, put that command in `command`. The command must actually "
        "execute before the tool reports success; if Wynxo cannot guarantee "
        "that terminal's command contract it fails honestly instead. Keep the "
        "terminal open so the user can see the output. A successful launch "
        "completes that request; do not invent follow-up tool calls."
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

        if is_generic_terminal_query(query):
            entry = await asyncio.to_thread(preferred_terminal, self.catalog)
            if entry is None:
                await asyncio.to_thread(self.catalog.refresh)
                entry = await asyncio.to_thread(preferred_terminal, self.catalog)
            if entry is None:
                return ToolResult.failure(
                    "Could not find a recognised installed terminal emulator. "
                    "Nothing was launched.",
                    not_found=True,
                    query=query,
                    status="not_found",
                )
            return await self._launch(
                entry,
                (args.path or "").strip(),
                (args.command or "").strip(),
            )

        resolution = await asyncio.to_thread(self.catalog.resolve, query)
        if resolution.status == "not_found":
            await asyncio.to_thread(self.catalog.refresh)
            resolution = await asyncio.to_thread(self.catalog.resolve, query)

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
                    if is_a_terminal(entry):
                        return ToolResult.failure(
                            f"'{entry.name}' is an installed terminal, but Wynxo "
                            "does not know a safe command-execution contract for "
                            "it on this platform. The terminal was not launched "
                            "and the command was not run.",
                            status="terminal_command_unsupported",
                            application=entry.name, query=entry.name)
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
            terminal=True,
            status="launched",
            application=entry.name,
            source=entry.source,
            path=str(entry.path),
            **({"opened": open_path} if open_path else {}),
            **({"command": command} if command else {}),
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
