"""The applications that are actually installed on this machine.

The computer already knows what it has installed; Wynxo asks it rather than
keeping a list. Start Menu shortcuts on Windows, application bundles on macOS
and .desktop entries on Linux are exactly what a desktop search would show a
person, and launching the shortcut the user's own Start Menu holds behaves
like double-clicking it -- target, arguments and working directory stay
whatever the installer wrote.

Nothing here knows an application by name. ``Visual Studio Code`` is
discovered because the machine has a shortcut called that, not because
somewhere in the source there is a line saying so. An application installed
tomorrow is discovered tomorrow, with no code change.

The developer never hearing of the application is the entire point, so the
matching is deliberately forgiving -- ``libre wolf`` still finds ``LibreWolf``
and ``vscode`` still finds ``Visual Studio Code`` -- but the *candidates* are
always real entries from this machine, and a query nobody matches comes back
as a clean miss rather than a guess.
"""

from __future__ import annotations

import difflib
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Discovery is capped so a machine with an enormous ProgramData tree cannot
# turn one lookup into a full-disk crawl. The caps are far above what a real
# Start Menu holds; they exist so the worst case is slow, not unbounded.
MAX_SHORTCUTS = 2_000
MAX_DESKTOP_ENTRIES = 2_000
MAX_PATH_ENTRIES = 500
AMBIGUOUS_LIMIT = 8

_FUZZY_CUTOFF = 0.75
_MIN_SUBSEQUENCE = 3
"""A two-letter query is a subsequence of half the catalog; three is the
shortest that still means something ('vsc', 'ff' aside)."""

RANK = {
    "start_menu": 3,
    "macos_app": 3,
    "linux_desktop": 3,
    "app_paths": 2,
    "path": 1,
}
"""Which source wins when two report the same application. A Start Menu
shortcut is the thing a person would double-click, so it outranks the same
name found as a bare executable on PATH."""

WHERE = {
    "start_menu": "start menu",
    "macos_app": "/Applications",
    "linux_desktop": ".desktop",
    "app_paths": "App Paths",
    "path": "PATH",
}


@dataclass(frozen=True)
class AppEntry:
    """One launchable application, exactly as the OS reports it."""

    name: str
    """The display name a person would recognise: the shortcut's name, the
    bundle's name, the desktop entry's Name= field."""

    path: Path
    """What gets launched: a .lnk, a .app bundle, a .desktop file, or an
    executable that was found on PATH."""

    source: str

    @property
    def where(self) -> str:
        return WHERE.get(self.source, self.source)


@dataclass(frozen=True)
class Sources:
    """The places to look, as explicit lists so tests can point at fakes."""

    shortcut_dirs: tuple[Path, ...] = ()
    """Directories searched recursively for .lnk shortcuts (Windows Start
    Menu, user and system)."""

    app_bundles: tuple[Path, ...] = ()
    """Directories whose children are .app bundles (macOS)."""

    desktop_dirs: tuple[Path, ...] = ()
    """Directories holding .desktop entries (Linux XDG data dirs)."""

    path_dirs: tuple[Path, ...] = ()
    """Directories on PATH to scan for executables."""

    use_app_paths: bool = False
    """Read the Windows App Paths registry. Kept as a flag rather than a
    directory list because the registry is not injectable like a folder."""

    @classmethod
    def for_platform(cls) -> "Sources":
        home = Path.home()
        if sys.platform == "win32":
            shortcuts = []
            for env in ("APPDATA", "PROGRAMDATA", "CSIDL_COMMON_START_MENU"):
                base = os.environ.get(env)
                if base:
                    shortcuts.append(
                        Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
            return cls(
                shortcut_dirs=tuple(dict.fromkeys(shortcuts)),
                path_dirs=_windows_path_dirs(),
                use_app_paths=True,
            )
        if sys.platform == "darwin":
            return cls(
                app_bundles=(
                    Path("/Applications"),
                    Path("/System/Applications"),
                    home / "Applications",
                ),
                path_dirs=_path_dirs(),
            )
        data_dirs = [Path("/usr/share/applications"),
                     Path("/usr/local/share/applications"),
                     home / ".local" / "share" / "applications"]
        for part in os.environ.get("XDG_DATA_DIRS", "").split(os.pathsep):
            if part:
                data_dirs.append(Path(part) / "applications")
        return cls(desktop_dirs=tuple(dict.fromkeys(data_dirs)),
                   path_dirs=_path_dirs())


def _path_dirs() -> tuple[Path, ...]:
    out = []
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            out.append(Path(part))
    return tuple(dict.fromkeys(out))


def _windows_path_dirs() -> tuple[Path, ...]:
    """PATH, minus the directory that lies.

    ``%LOCALAPPDATA%\\Microsoft\\WindowsApps`` holds zero-byte execution
    aliases whose only talent is opening the Microsoft Store when the real
    program is not installed. Launching 'python' from there does not run
    Python, so as launchable applications those aliases are noise.
    """
    return tuple(d for d in _path_dirs()
                 if "windowsapps" not in str(d).lower())


def normalize_name(text: str) -> str:
    """Comparison form: case, accents, punctuation and spacing all fold."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = "".join(ch.lower() if ch.isalnum() else " " for ch in plain)
    return " ".join(folded.split())


def condense(text: str) -> str:
    """The normalized name with the spaces gone: 'libre wolf' -> 'librewolf'."""
    return normalize_name(text).replace(" ", "")


def _words(text: str) -> list[str]:
    return normalize_name(text).split()


def _is_subsequence(needle: str, haystack: str) -> bool:
    """Whether needle's characters appear in haystack, in order, not
    necessarily adjacent -- how 'vscode' is an abbreviation of
    'visual studio code'."""
    if not needle:
        return False
    it = iter(haystack)
    return all(ch in it for ch in needle)


@dataclass(frozen=True)
class Resolution:
    """What a query resolved to, with the evidence either way."""

    status: str
    """'matched' | 'ambiguous' | 'not_found' | 'path_query'."""

    entry: AppEntry | None = None
    candidates: tuple[AppEntry, ...] = field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        return self.status == "matched"


class ApplicationCatalog:
    """The machine's installed applications, discovered once and cached.

    A lookup that finds nothing may call ``refresh()`` once and try again --
    an application installed a minute ago is not a reason to rescan the disk
    on every miss.
    """

    def __init__(self, sources: Sources | None = None):
        self.sources = sources or Sources.for_platform()
        self._entries: tuple[AppEntry, ...] | None = None

    # -- discovery ---------------------------------------------------------

    def entries(self) -> tuple[AppEntry, ...]:
        if self._entries is None:
            return self.refresh()
        return self._entries

    def refresh(self) -> tuple[AppEntry, ...]:
        self._entries = self._scan()
        return self._entries

    def _scan(self) -> tuple[AppEntry, ...]:
        found: dict[str, AppEntry] = {}

        def offer(entry: AppEntry) -> None:
            key = condense(entry.name)
            if not key:
                return
            current = found.get(key)
            if current is None or RANK.get(entry.source, 0) > RANK.get(current.source, 0):
                found[key] = entry

        self._scan_shortcuts(offer)
        self._scan_bundles(offer)
        self._scan_desktops(offer)
        self._scan_app_paths(offer)
        self._scan_path_dirs(offer)
        return tuple(sorted(found.values(), key=lambda e: e.name.lower()))

    def _scan_shortcuts(self, offer) -> None:
        if not self.sources.shortcut_dirs:
            return
        seen = 0
        for directory in self.sources.shortcut_dirs:
            if seen >= MAX_SHORTCUTS:
                break
            try:
                links = sorted(directory.rglob("*.lnk"))
            except OSError:
                continue
            for link in links:
                if seen >= MAX_SHORTCUTS:
                    break
                # The Start Menu is full of 'Uninstall X' shortcuts; they are
                # not applications and launching one by accident would be a
                # bad surprise.
                if "uninstall" in link.stem.lower():
                    continue
                offer(AppEntry(link.stem, link, "start_menu"))
                seen += 1

    def _scan_bundles(self, offer) -> None:
        for directory in self.sources.app_bundles:
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for child in children:
                if child.suffix == ".app" and child.is_dir():
                    offer(AppEntry(child.stem, child, "macos_app"))

    def _scan_desktops(self, offer) -> None:
        if not self.sources.desktop_dirs:
            return
        seen = 0
        for directory in self.sources.desktop_dirs:
            try:
                files = sorted(directory.glob("*.desktop"))
            except OSError:
                continue
            for file in files:
                if seen >= MAX_DESKTOP_ENTRIES:
                    break
                parsed = _parse_desktop(file)
                if parsed is None:
                    continue
                name, hidden = parsed
                if hidden or not name:
                    continue
                offer(AppEntry(name, file, "linux_desktop"))
                seen += 1

    def _scan_app_paths(self, offer) -> None:
        """The registry's App Paths keys: names software registers so
        ShellExecute can find it without being on PATH."""
        if not self.sources.use_app_paths or sys.platform != "win32":
            return
        try:
            import winreg
        except ImportError:
            return
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                try:
                    key = winreg.OpenKey(
                        hive,
                        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
                        0, winreg.KEY_READ | view)
                except OSError:
                    continue
                with key:
                    self._app_paths_from(key, offer)

    @staticmethod
    def _app_paths_from(key, offer) -> None:
        try:
            import winreg
        except ImportError:
            return
        index = 0
        while index < MAX_PATH_ENTRIES:
            try:
                subkey_name = winreg.EnumKey(key, index)
            except OSError:
                break
            index += 1
            stem = Path(subkey_name).stem
            if not stem:
                continue
            try:
                with winreg.OpenKey(key, subkey_name) as subkey:
                    target, _ = winreg.QueryValueEx(subkey, "")
            except OSError:
                continue
            if not target:
                continue
            target_path = Path(os.path.expandvars(str(target).strip('"')))
            offer(AppEntry(stem, target_path, "app_paths"))

    def _scan_path_dirs(self, offer) -> None:
        executable_suffixes = (".exe", ".cmd", ".bat") if sys.platform == "win32" \
            else (".sh", ".py", "")
        seen = 0
        for directory in self.sources.path_dirs:
            if seen >= MAX_PATH_ENTRIES:
                break
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for child in children:
                if seen >= MAX_PATH_ENTRIES:
                    break
                if not child.is_file() or child.name.startswith("."):
                    continue
                if sys.platform == "win32":
                    if child.suffix.lower() not in executable_suffixes:
                        continue
                else:
                    if executable_suffixes and child.suffix not in executable_suffixes:
                        continue
                    if not os.access(child, os.X_OK):
                        continue
                offer(AppEntry(child.stem, child, "path"))
                seen += 1

    # -- matching ----------------------------------------------------------

    def resolve(self, query: str) -> Resolution:
        """Match a human phrase against real installed applications.

        The tiers, tried in order and the first non-empty one wins:

        1. normalized equality      -- 'visual studio code'
        2. condensed equality       -- 'libre wolf'  ->  'LibreWolf'
        3. whole-word subphrase     -- 'steam' in 'Steam', 'studio code'
        4. fuzzy                    -- 'librewlof'   ->  'LibreWolf'
        5. character subsequence    -- 'vscode'      ->  'Visual Studio Code'

        Within the winning tier one hit is a match and several are
        ambiguity, never a guess.
        """
        text = (query or "").strip()
        if not text:
            return Resolution("not_found")

        # A path is not a name. Letting 'C:\\tools\\thing.exe' through would
        # turn the model's guess into arbitrary execution, and the whole
        # point of the catalog is that the OS decides what exists.
        if _looks_like_path(text):
            return Resolution("path_query")

        entries = self.entries()
        norm = normalize_name(text)
        cond = condense(text)
        query_words = _words(text)

        def collect(names: set[str]) -> tuple[AppEntry, ...]:
            by_cond = {condense(e.name): e for e in entries}
            hits = [by_cond[c] for c in sorted(names)
                    if c in by_cond]
            return tuple(hits[:AMBIGUOUS_LIMIT])

        # 1 + 2: exact after folding case, punctuation, spacing.
        exact = {condense(e.name) for e in entries
                 if normalize_name(e.name) == norm or condense(e.name) == cond}
        if exact:
            hits = collect(exact)
            if len(hits) == 1:
                return Resolution("matched", hits[0])
            return Resolution("ambiguous", candidates=hits)

        # 3: the query's words appear in order inside the candidate's name.
        # 'code' alone lands here for everything with a word 'code', which
        # is exactly the case that should come back as "which one?".
        word_hits = {condense(e.name) for e in entries
                     if _contains_words(query_words, _words(e.name))}
        if word_hits:
            hits = collect(word_hits)
            if len(hits) == 1:
                return Resolution("matched", hits[0])
            return Resolution("ambiguous", candidates=hits)

        # 4: typos, by condensed form. 'librewlof' -> 'librewolf'.
        close = difflib.get_close_matches(
            cond, [condense(e.name) for e in entries],
            n=AMBIGUOUS_LIMIT, cutoff=_FUZZY_CUTOFF)
        if close:
            hits = collect(set(close))
            if len(hits) == 1:
                return Resolution("matched", hits[0])
            return Resolution("ambiguous", candidates=hits)

        # 5: abbreviations. 'vscode' is a subsequence of
        # 'visualstudiocode'; 'vscode' is not a typo of anything.
        if len(cond) >= _MIN_SUBSEQUENCE:
            subs = {condense(e.name) for e in entries
                    if _is_subsequence(cond, condense(e.name))}
            if subs:
                hits = collect(subs)
                if len(hits) == 1:
                    return Resolution("matched", hits[0])
                return Resolution("ambiguous", candidates=hits)

        return Resolution("not_found")


def _contains_words(query: list[str], candidate: list[str]) -> bool:
    """Whether query's words appear consecutively and in order inside
    the candidate's words."""
    if not query or len(query) > len(candidate):
        return False
    for start in range(len(candidate) - len(query) + 1):
        if candidate[start:start + len(query)] == query:
            return True
    return False


def _looks_like_path(text: str) -> bool:
    if "\\" in text or "/" in text:
        return True
    if len(text) > 1 and text[1] == ":":
        return True
    return text.lower().endswith((".exe", ".lnk", ".bat", ".cmd", ".app",
                                  ".desktop", ".msi", ".bin"))


def _parse_desktop(file: Path) -> tuple[str, bool] | None:
    """(display name, hidden) from a .desktop file, without a full parser.

    Only the two facts discovery needs. Name= is what the user sees in their
    own application launcher, so it is the name to match against; the file's
    stem is only the fallback when the entry never named itself.
    """
    name = ""
    hidden = False
    try:
        with file.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line in ("[Desktop Entry]", ""):
                    continue
                if line.startswith("["):
                    break        # left the main section; stop reading
                if line.startswith("Name=") and not name:
                    name = line[len("Name="):].strip()
                elif line.startswith("Hidden=") and \
                        line[len("Hidden="):].strip().lower() == "true":
                    hidden = True
                elif line.startswith("NoDisplay=") and \
                        line[len("NoDisplay="):].strip().lower() == "true":
                    hidden = True
    except OSError:
        return None
    return name or file.stem, hidden
