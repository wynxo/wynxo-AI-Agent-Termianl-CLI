"""The rest of the computer: sound, power, clipboard, notifications, health.

Driving windows was half of it. The other half is everything people
actually ask a machine assistant for -- turn it down, pause that, lock the
screen, why is this thing slow, what's eating my disk -- and none of it is
about windows at all.

The shell could technically do all of this, and that is exactly the
problem: a local model handed a shell has one hammer and no idea which
nail. It does not know that this box runs PipeWire rather than PulseAudio,
that brightness needs brightnessctl here and `light` there, that the
clipboard is wl-copy under Wayland and xclip under X. So it guesses, and a
guess that fails silently -- `pactl` on a PipeWire-only system, `xclip`
with no DISPLAY -- reads as wynxo being broken.

Each action here is therefore three things, never fewer:

  named        in the words somebody would use, not the tool's spelling
  implemented  per platform, per stack, with what is actually installed
  checked      afterwards where the machine can be asked, so "muted" means
               it looked and it is muted -- not that a command exited zero

And where none of that is possible, it refuses and names the package. A
refusal somebody can act on is worth more than an attempt that might have
worked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

TIMEOUT = 8.0


class SystemError_(Exception):
    """Something could not be done; the message says what to install or do."""


def _run(argv: list[str], timeout: float = TIMEOUT,
         input_text: str | None = None) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, input=input_text)
    except FileNotFoundError:
        raise SystemError_(f"{argv[0]} is not installed.") from None
    except subprocess.TimeoutExpired:
        raise SystemError_(f"{argv[0]} did not answer within "
                           f"{timeout:.0f}s.") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise SystemError_(f"{argv[0]} failed: "
                           f"{detail[0][:160] if detail else proc.returncode}")
    return proc.stdout


def _which(*names: str) -> str:
    return next((n for n in names if shutil.which(n)), "")


# -- sound -------------------------------------------------------------------

class Audio:
    """Volume and mute, on whichever sound stack this machine runs.

    Three of them are still in the wild on Linux and they are not
    interchangeable: `pactl` exists on a PipeWire box as a compatibility
    shim and mostly works, `wpctl` is the native one and always does, and
    an ALSA-only machine has neither. Asking in that order is the whole
    trick.
    """

    def __init__(self, run=_run):
        self.run = run
        self.stack = self._detect()

    @staticmethod
    def _detect() -> str:
        if sys.platform == "darwin":
            return "macos"
        if sys.platform == "win32":
            return "windows"
        return _which("wpctl", "pactl", "amixer")

    @property
    def available(self) -> bool:
        return bool(self.stack)

    def missing(self) -> str:
        return ("No sound control here. Install pipewire's wpctl, or "
                "pulseaudio-utils for pactl, or alsa-utils for amixer.")

    def volume(self) -> int:
        """0-100, or -1 when it cannot be read."""
        if not self.available:
            raise SystemError_(self.missing())
        if self.stack == "wpctl":
            out = self.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
            # "Volume: 0.45" or "Volume: 0.45 [MUTED]"
            for word in out.split():
                try:
                    return round(float(word) * 100)
                except ValueError:
                    continue
            return -1
        if self.stack == "pactl":
            out = self.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
            for token in out.replace("/", " ").split():
                if token.endswith("%"):
                    try:
                        return int(token.rstrip("%"))
                    except ValueError:
                        continue
            return -1
        if self.stack == "amixer":
            out = self.run(["amixer", "get", "Master"])
            for token in out.replace("[", " ").replace("]", " ").split():
                if token.endswith("%"):
                    try:
                        return int(token.rstrip("%"))
                    except ValueError:
                        continue
            return -1
        if self.stack == "macos":
            out = self.run(["osascript", "-e", "output volume of (get volume settings)"])
            try:
                return int(out.strip())
            except ValueError:
                return -1
        return -1

    def set_volume(self, percent: int) -> int:
        """Set it, then read it back. Returns what it actually is.

        Read back because a sink can clamp, and because "set to 80" that
        silently did nothing is the failure this whole module exists to
        stop reporting as success.
        """
        percent = max(0, min(int(percent), 100))
        if not self.available:
            raise SystemError_(self.missing())
        if self.stack == "wpctl":
            self.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@",
                      f"{percent / 100:.2f}"])
        elif self.stack == "pactl":
            self.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])
        elif self.stack == "amixer":
            self.run(["amixer", "-q", "set", "Master", f"{percent}%"])
        elif self.stack == "macos":
            self.run(["osascript", "-e", f"set volume output volume {percent}"])
        elif self.stack == "windows":
            raise SystemError_(
                "Windows has no way to set an exact volume without an extra "
                "tool. Use up, down or mute, which send the media keys.")
        return self.volume()

    def nudge(self, step: int) -> int:
        """Up or down by a step. Returns the new volume."""
        if self.stack == "windows":
            # The media keys, which every Windows since 7 honours. 175 is
            # volume up, 174 down -- one press per five per cent, which is
            # what the keys themselves do.
            key = 175 if step > 0 else 174
            presses = max(1, abs(int(step)) // 5)
            self.run(["powershell", "-NoProfile", "-Command",
                      "$w=New-Object -ComObject WScript.Shell; "
                      + f"1..{presses} | ForEach-Object {{ $w.SendKeys([char]{key}) }}"])
            return -1
        current = self.volume()
        if current < 0:
            raise SystemError_("could not read the current volume.")
        return self.set_volume(current + step)

    def muted(self) -> bool | None:
        if self.stack == "wpctl":
            return "MUTED" in self.run(
                ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
        if self.stack == "pactl":
            return "yes" in self.run(
                ["pactl", "get-sink-mute", "@DEFAULT_SINK@"]).lower()
        if self.stack == "amixer":
            return "[off]" in self.run(["amixer", "get", "Master"])
        if self.stack == "macos":
            return "true" in self.run(
                ["osascript", "-e", "output muted of (get volume settings)"]).lower()
        return None

    def mute(self, on: bool | None = None) -> bool | None:
        """True to mute, False to unmute, None to toggle."""
        if not self.available:
            raise SystemError_(self.missing())
        word = "toggle" if on is None else ("1" if on else "0")
        if self.stack == "wpctl":
            self.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", word])
        elif self.stack == "pactl":
            self.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", word])
        elif self.stack == "amixer":
            self.run(["amixer", "-q", "set", "Master",
                      "toggle" if on is None else ("mute" if on else "unmute")])
        elif self.stack == "macos":
            value = ("not output muted of (get volume settings)" if on is None
                     else ("true" if on else "false"))
            self.run(["osascript", "-e", f"set volume output muted {value}"])
        elif self.stack == "windows":
            self.run(["powershell", "-NoProfile", "-Command",
                      "(New-Object -ComObject WScript.Shell)"
                      ".SendKeys([char]173)"])
            return None
        return self.muted()


# -- what is playing ---------------------------------------------------------

class Media:
    """Play, pause, skip -- through whatever exposes a player.

    playerctl on Linux speaks MPRIS, which every serious player implements:
    Spotify, VLC, Firefox, mpv. That is one integration rather than one per
    application, and it is the reason this is worth having at all.
    """

    def __init__(self, run=_run):
        self.run = run
        self.how = ("macos" if sys.platform == "darwin"
                    else "windows" if sys.platform == "win32"
                    else _which("playerctl"))

    @property
    def available(self) -> bool:
        return bool(self.how)

    def missing(self) -> str:
        return ("No media control here. Install playerctl -- it speaks "
                "MPRIS, which Spotify, VLC, Firefox and mpv all implement.")

    def command(self, what: str) -> str:
        """what: play, pause, playpause, next, previous. Returns what happened."""
        if not self.available:
            raise SystemError_(self.missing())
        if self.how == "playerctl":
            verb = {"play": "play", "pause": "pause", "playpause": "play-pause",
                    "next": "next", "previous": "previous"}.get(what)
            if verb is None:
                raise SystemError_(f"{what!r} is not a media command.")
            self.run(["playerctl", verb])
            return self.now_playing() or what
        if self.how == "macos":
            verb = {"play": "play", "pause": "pause", "playpause": "playpause",
                    "next": "next track", "previous": "previous track"}.get(what)
            if verb is None:
                raise SystemError_(f"{what!r} is not a media command.")
            self.run(["osascript", "-e", f'tell application "Music" to {verb}'])
            return what
        # Windows: the media keys again.
        key = {"play": 179, "pause": 179, "playpause": 179,
               "next": 176, "previous": 177}.get(what)
        if key is None:
            raise SystemError_(f"{what!r} is not a media command.")
        self.run(["powershell", "-NoProfile", "-Command",
                  f"(New-Object -ComObject WScript.Shell).SendKeys([char]{key})"])
        return what

    def now_playing(self) -> str:
        if self.how != "playerctl":
            return ""
        try:
            return self.run(
                ["playerctl", "metadata", "--format",
                 "{{artist}} - {{title}}"]).strip()
        except SystemError_:
            return ""


# -- clipboard ---------------------------------------------------------------

class Clipboard:
    def __init__(self, run=_run):
        self.run = run
        if sys.platform == "darwin":
            self.how = "macos"
        elif sys.platform == "win32":
            self.how = "windows"
        elif os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
            self.how = "wayland"
        else:
            self.how = _which("xclip", "xsel")

    @property
    def available(self) -> bool:
        return bool(self.how)

    def missing(self) -> str:
        return ("No clipboard access here. Install wl-clipboard on Wayland, "
                "or xclip on X11.")

    def read(self) -> str:
        if not self.available:
            raise SystemError_(self.missing())
        if self.how == "wayland":
            return self.run(["wl-paste", "--no-newline"])
        if self.how == "xclip":
            return self.run(["xclip", "-selection", "clipboard", "-o"])
        if self.how == "xsel":
            return self.run(["xsel", "--clipboard", "--output"])
        if self.how == "macos":
            return self.run(["pbpaste"])
        return self.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])

    def write(self, text: str) -> None:
        if not self.available:
            raise SystemError_(self.missing())
        if self.how == "wayland":
            self.run(["wl-copy"], input_text=text)
        elif self.how == "xclip":
            self.run(["xclip", "-selection", "clipboard"], input_text=text)
        elif self.how == "xsel":
            self.run(["xsel", "--clipboard", "--input"], input_text=text)
        elif self.how == "macos":
            self.run(["pbcopy"], input_text=text)
        else:
            self.run(["powershell", "-NoProfile", "-Command", "Set-Clipboard"
                      " -Value ([Console]::In.ReadToEnd())"], input_text=text)


# -- notifications -----------------------------------------------------------

class Notifier:
    def __init__(self, run=_run):
        self.run = run
        self.how = ("macos" if sys.platform == "darwin"
                    else "windows" if sys.platform == "win32"
                    else _which("notify-send"))

    @property
    def available(self) -> bool:
        return bool(self.how)

    def missing(self) -> str:
        return "No desktop notifications here. Install libnotify (notify-send)."

    def send(self, title: str, body: str = "") -> None:
        if not self.available:
            raise SystemError_(self.missing())
        if self.how == "notify-send":
            self.run(["notify-send", "--app-name=wynxo", "--", title, body])
        elif self.how == "macos":
            safe = lambda s: s.replace('"', '\\"')      # noqa: E731
            self.run(["osascript", "-e",
                      f'display notification "{safe(body)}" '
                      f'with title "{safe(title)}"'])
        else:
            escaped = (title + (" - " + body if body else "")).replace("'", "''")
            self.run(["powershell", "-NoProfile", "-Command",
                      "Add-Type -AssemblyName System.Windows.Forms; "
                      "$n=New-Object System.Windows.Forms.NotifyIcon; "
                      "$n.Icon=[System.Drawing.SystemIcons]::Information; "
                      "$n.Visible=$true; "
                      f"$n.ShowBalloonTip(5000,'wynxo','{escaped}',"
                      "[System.Windows.Forms.ToolTipIcon]::Info)"])


# -- power -------------------------------------------------------------------

@dataclass
class PowerAction:
    name: str
    argv: list[str]
    reversible: bool


class Power:
    """Lock, sleep, restart, shut down.

    Every one of these ends the session somebody is in the middle of, so
    none of them is ever a guess: an action whose command is not present
    refuses rather than falling back to something adjacent. "Suspend" that
    quietly became "shut down" would be a bug report about lost work.
    """

    def __init__(self, run=_run):
        self.run = run

    def actions(self) -> dict[str, PowerAction]:
        if sys.platform == "darwin":
            return {
                "lock": PowerAction("lock", [
                    "osascript", "-e",
                    'tell application "System Events" to keystroke "q" '
                    "using {command down, control down}"], True),
                "sleep": PowerAction("sleep", ["pmset", "sleepnow"], True),
                "restart": PowerAction("restart", [
                    "osascript", "-e",
                    'tell application "System Events" to restart'], False),
                "shutdown": PowerAction("shutdown", [
                    "osascript", "-e",
                    'tell application "System Events" to shut down'], False),
            }
        if sys.platform == "win32":
            return {
                "lock": PowerAction("lock", [
                    "rundll32.exe", "user32.dll,LockWorkStation"], True),
                "sleep": PowerAction("sleep", [
                    "rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], True),
                "restart": PowerAction("restart", ["shutdown", "/r", "/t", "0"], False),
                "shutdown": PowerAction("shutdown", ["shutdown", "/s", "/t", "0"], False),
            }
        found: dict[str, PowerAction] = {}
        if shutil.which("loginctl"):
            found["lock"] = PowerAction("lock", ["loginctl", "lock-session"], True)
        elif locker := _which("swaylock", "i3lock", "xdg-screensaver"):
            found["lock"] = PowerAction(
                "lock", [locker] + (["lock"] if locker == "xdg-screensaver" else []),
                True)
        if shutil.which("systemctl"):
            found["sleep"] = PowerAction("sleep", ["systemctl", "suspend"], True)
            found["restart"] = PowerAction("restart", ["systemctl", "reboot"], False)
            found["shutdown"] = PowerAction("shutdown", ["systemctl", "poweroff"], False)
        return found

    def do(self, what: str) -> str:
        action = self.actions().get(what)
        if action is None:
            raise SystemError_(
                f"cannot {what} on this machine: neither systemctl nor a "
                f"known alternative is here. Nothing was done.")
        self.run(action.argv, timeout=15.0)
        return action.name


# -- how the machine is doing ------------------------------------------------

@dataclass
class Health:
    """What somebody means by "why is this thing slow"."""

    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_pct: int = 0
    swap_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_pct: int = 0
    load: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cores: int = 0
    uptime: str = ""
    battery_pct: int = -1
    charging: bool | None = None
    top: list[tuple[str, float, float]] = None      # (name, cpu%, mem MB)

    def lines(self) -> list[str]:
        out = []
        if self.memory_total_gb:
            out.append(f"Memory: {self.memory_used_gb:.1f} of "
                       f"{self.memory_total_gb:.1f} GB used ({self.memory_pct}%)"
                       + (f", {self.swap_used_gb:.1f} GB swapped"
                          if self.swap_used_gb > 0.1 else ""))
        if self.disk_total_gb:
            out.append(f"Disk: {self.disk_free_gb:.0f} GB free of "
                       f"{self.disk_total_gb:.0f} ({self.disk_pct}% used)")
        if self.cores:
            # Load against cores, because "load 4" means nothing until you
            # know whether this is a laptop or a build server.
            busy = self.load[0] / self.cores
            how = ("idle" if busy < 0.3 else "busy" if busy < 0.9
                   else "overloaded")
            out.append(f"CPU: load {self.load[0]:.2f} over {self.cores} cores "
                       f"-- {how}")
        if self.battery_pct >= 0:
            state = ("charging" if self.charging else "on battery"
                     if self.charging is False else "")
            out.append(f"Battery: {self.battery_pct}%"
                       + (f", {state}" if state else ""))
        if self.uptime:
            out.append(f"Up: {self.uptime}")
        if self.top:
            out.append("Heaviest processes:")
            out += [f"  {name}  {cpu:.0f}% cpu  {mem:.0f} MB"
                    for name, cpu, mem in self.top]
        return out


def health(path: str = "/", top: int = 5) -> Health:
    """A snapshot of how the machine is doing, from what it will tell us.

    Everything here is read rather than shelled out for where the platform
    allows it: /proc and /sys are files, and reading a file cannot fail
    halfway through in the way parsing somebody's `free` output can. Each
    piece is independent, so a machine that will not answer one question
    still answers the others.
    """
    out = Health(cores=os.cpu_count() or 0, top=[])
    _memory(out)
    _disk(out, path)
    _load(out)
    _uptime(out)
    _battery(out)
    out.top = _heaviest(top)
    return out


def _memory(out: Health) -> None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            fields = {}
            for line in handle:
                key, _, rest = line.partition(":")
                fields[key.strip()] = float(rest.strip().split()[0]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        return _memory_elsewhere(out)
    total = fields.get("MemTotal", 0.0)
    available = fields.get("MemAvailable", fields.get("MemFree", 0.0))
    if not total:
        return
    out.memory_total_gb = total
    out.memory_used_gb = max(0.0, total - available)
    out.memory_pct = round(out.memory_used_gb / total * 100)
    out.swap_used_gb = max(0.0, fields.get("SwapTotal", 0.0)
                           - fields.get("SwapFree", 0.0))


def _memory_elsewhere(out: Health) -> None:
    """macOS and Windows, where there is no /proc."""
    try:
        if sys.platform == "darwin":
            total = int(_run(["sysctl", "-n", "hw.memsize"]).strip())
            out.memory_total_gb = total / 1024 ** 3
        elif sys.platform == "win32":
            raw = _run(["powershell", "-NoProfile", "-Command",
                        "$o=Get-CimInstance Win32_OperatingSystem; "
                        "\"$($o.TotalVisibleMemorySize) $($o.FreePhysicalMemory)\""])
            total_kb, free_kb = (float(x) for x in raw.split())
            out.memory_total_gb = total_kb / 1024 / 1024
            out.memory_used_gb = (total_kb - free_kb) / 1024 / 1024
            out.memory_pct = round(out.memory_used_gb / out.memory_total_gb * 100)
    except (SystemError_, ValueError, ZeroDivisionError):
        pass


def _disk(out: Health, path: str) -> None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return
    out.disk_total_gb = usage.total / 1024 ** 3
    out.disk_free_gb = usage.free / 1024 ** 3
    if usage.total:
        out.disk_pct = round((usage.total - usage.free) / usage.total * 100)


def _load(out: Health) -> None:
    try:
        out.load = os.getloadavg()
    except (OSError, AttributeError):
        pass          # Windows has no load average at all


def _uptime(out: Health) -> None:
    try:
        with open("/proc/uptime", encoding="utf-8") as handle:
            seconds = float(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        return
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    parts = ([f"{days}d"] if days else []) + ([f"{hours}h"] if hours or days
                                              else []) + [f"{minutes}m"]
    out.uptime = " ".join(parts)


def _battery(out: Health) -> None:
    """Percentage and whether it is charging, from /sys.

    A desktop has no battery and that is not an error -- the field stays
    at -1 and nothing is said about it, rather than "battery: unknown"
    appearing on every machine that has never had one.
    """
    if sys.platform == "darwin":
        try:
            raw = _run(["pmset", "-g", "batt"])
        except SystemError_:
            return
        for token in raw.replace(";", " ").split():
            if token.endswith("%"):
                try:
                    out.battery_pct = int(token.rstrip("%"))
                except ValueError:
                    pass
        out.charging = "AC Power" in raw
        return
    root = "/sys/class/power_supply"
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return
    for name in names:
        base = os.path.join(root, name)
        try:
            with open(os.path.join(base, "type"), encoding="utf-8") as handle:
                if handle.read().strip() != "Battery":
                    continue
            with open(os.path.join(base, "capacity"), encoding="utf-8") as handle:
                out.battery_pct = int(handle.read().strip())
            with open(os.path.join(base, "status"), encoding="utf-8") as handle:
                out.charging = handle.read().strip().lower() in (
                    "charging", "full")
        except (OSError, ValueError):
            continue
        return


def _heaviest(count: int) -> list[tuple[str, float, float]]:
    """The processes actually using the machine.

    `ps` rather than walking /proc: the kernel's own accounting of CPU
    share over a process's lifetime is what `ps` reports, and computing it
    by hand from two /proc samples means sleeping in the middle of somebody
    asking a question.
    """
    if not shutil.which("ps"):
        return []
    try:
        raw = _run(["ps", "-eo", "comm=,pcpu=,rss=", "--sort=-pcpu"])
    except SystemError_:
        return []
    found: list[tuple[str, float, float]] = []
    for line in raw.splitlines()[:count]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            found.append((parts[0][:24], float(parts[1]),
                          float(parts[2]) / 1024))
        except ValueError:
            continue
    return found
