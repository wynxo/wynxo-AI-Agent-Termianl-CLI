"""Operating the rest of the computer: sound, media, clipboard, power, health.

The shell could do all of this, and that is the problem: a local model
handed a shell has one hammer and no idea which nail. It does not know that
this box runs PipeWire rather than PulseAudio, or that the clipboard is
wl-copy under Wayland and xclip under X. So these tests are mostly about
two things -- parsing what the real commands really print, and refusing
with the package name rather than guessing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from wynxo import system as sysmod
from wynxo.scope import Boundary, Scope
from wynxo.system import (Audio, Clipboard, Health, Media, Power,
                          SystemError_, health)
from wynxo.tools.system_tool import (SystemControl, SystemControlInput,
                                     SystemStatus, SystemStatusInput)

# What these commands actually print, copied from real machines.
WPCTL = "Volume: 0.45\n"
WPCTL_MUTED = "Volume: 0.45 [MUTED]\n"
PACTL = ("Volume: front-left: 29491 /  45% / -18.75 dB,   front-right: "
         "29491 /  45% / -18.75 dB\n       balance 0.00\n")
AMIXER = ("Simple mixer control 'Master',0\n  Capabilities: pvolume pswitch\n"
          "  Front Left: Playback 45 [45%] [-18.75dB] [on]\n")
AMIXER_OFF = AMIXER.replace("[on]", "[off]")


class Recorder:
    def __init__(self, answer=""):
        self.calls: list[list[str]] = []
        self.inputs: list[str | None] = []
        self.answer = answer

    def __call__(self, argv, timeout=sysmod.TIMEOUT, input_text=None):
        self.calls.append(list(argv))
        self.inputs.append(input_text)
        return self.answer

    @property
    def last(self):
        return self.calls[-1]


def _audio(stack, answer=""):
    run = Recorder(answer)
    audio = Audio(run)
    audio.stack = stack
    return audio, run


class TestReadingTheVolume:
    @pytest.mark.parametrize("stack,output", [
        ("wpctl", WPCTL), ("pactl", PACTL), ("amixer", AMIXER)])
    def test_every_stack_reports_the_same_number(self, stack, output):
        """Three sound systems are still in the wild and none of them
        prints anything like the others."""
        audio, _run = _audio(stack, output)
        assert audio.volume() == 45

    def test_wpctl_reports_mute_in_the_same_line(self):
        audio, _run = _audio("wpctl", WPCTL_MUTED)
        assert audio.muted() is True
        audio, _run = _audio("wpctl", WPCTL)
        assert audio.muted() is False

    def test_amixer_reports_mute_as_a_switch(self):
        audio, _run = _audio("amixer", AMIXER_OFF)
        assert audio.muted() is True

    def test_nothing_installed_names_all_three_packages(self):
        audio, _run = _audio("")
        with pytest.raises(SystemError_) as caught:
            audio.volume()
        for package in ("wpctl", "pactl", "amixer"):
            assert package in str(caught.value)


class TestSettingTheVolume:
    def test_wpctl_takes_a_fraction_and_pactl_a_percentage(self):
        """Getting this backwards sets the volume to zero on one of them
        and to nothing at all on the other."""
        audio, run = _audio("wpctl", WPCTL)
        audio.set_volume(80)
        assert run.calls[0][-1] == "0.80"
        audio, run = _audio("pactl", PACTL)
        audio.set_volume(80)
        assert run.calls[0][-1] == "80%"

    def test_it_is_clamped_before_it_is_sent(self):
        audio, run = _audio("wpctl", WPCTL)
        audio.set_volume(500)
        audio.set_volume(-20)
        # Every set is followed by a read-back, so the sets are the calls
        # that carry set-volume rather than every other call.
        sets = [c[-1] for c in run.calls if "set-volume" in c]
        assert sets == ["1.00", "0.00"]

    def test_it_reads_back_what_actually_happened(self):
        """A sink can clamp, and "set to 80" that silently did nothing is
        the failure this module exists to stop reporting as success."""
        audio, run = _audio("wpctl", WPCTL)     # always answers 45%
        assert audio.set_volume(80) == 45
        assert any("get-volume" in " ".join(c) for c in run.calls)


class TestMedia:
    def test_playerctl_spellings(self):
        run = Recorder("")
        media = Media(run)
        media.how = "playerctl"
        media.command("playpause")
        assert run.calls[0][:2] == ["playerctl", "play-pause"]

    def test_an_invented_command_is_refused(self):
        run = Recorder("")
        media = Media(run)
        media.how = "playerctl"
        with pytest.raises(SystemError_, match="not a media command"):
            media.command("rewind")

    def test_the_refusal_explains_what_playerctl_is_for(self):
        media = Media(Recorder())
        media.how = ""
        assert "MPRIS" in media.missing()


class TestClipboard:
    def test_wayland_and_x11_are_not_the_same_command(self):
        run = Recorder("")
        board = Clipboard(run)
        board.how = "wayland"
        board.write("hi")
        assert run.last[0] == "wl-copy" and run.inputs[-1] == "hi"
        board.how = "xclip"
        board.write("hi")
        assert run.last[:3] == ["xclip", "-selection", "clipboard"]

    def test_text_goes_in_on_stdin_not_as_an_argument(self):
        """A clipboard can hold a whole document. As an argument it would
        hit the command-line length limit, and anything with a quote in it
        would be a shell injection waiting to happen."""
        run = Recorder("")
        board = Clipboard(run)
        board.how = "wayland"
        board.write("$(rm -rf ~)")
        assert "$(rm -rf ~)" not in " ".join(run.last)
        assert run.inputs[-1] == "$(rm -rf ~)"


class TestPower:
    def test_nothing_falls_back_to_something_adjacent(self):
        """"Suspend" that quietly became "shut down" would be a bug report
        about lost work."""
        power = Power(Recorder())
        power.actions = lambda: {}
        with pytest.raises(SystemError_, match="Nothing was done"):
            power.do("sleep")

    def test_the_action_it_runs_is_the_one_named(self, monkeypatch):
        monkeypatch.setattr(sysmod.sys, "platform", "linux")
        monkeypatch.setattr(sysmod.shutil, "which", lambda n: f"/usr/bin/{n}")
        run = Recorder()
        Power(run).do("restart")
        assert run.last == ["systemctl", "reboot"]

    def test_locking_is_reversible_and_shutting_down_is_not(self, monkeypatch):
        monkeypatch.setattr(sysmod.sys, "platform", "linux")
        monkeypatch.setattr(sysmod.shutil, "which", lambda n: f"/usr/bin/{n}")
        actions = Power(Recorder()).actions()
        assert actions["lock"].reversible
        assert not actions["shutdown"].reversible


class TestHealth:
    def test_it_reads_this_machine(self):
        snapshot = health()
        assert snapshot.memory_total_gb > 0
        assert snapshot.disk_total_gb > 0

    def test_load_is_reported_against_the_core_count(self):
        """"Load 4" means nothing until you know whether this is a laptop
        or a build server."""
        snapshot = Health(cores=4, load=(0.4, 0.3, 0.2))
        assert "over 4 cores" in " ".join(snapshot.lines())
        assert "idle" in " ".join(snapshot.lines())
        assert "overloaded" in " ".join(
            Health(cores=4, load=(8.0, 8.0, 8.0)).lines())

    def test_a_machine_with_no_battery_says_nothing_about_one(self):
        """Rather than "battery: unknown" on every desktop ever built."""
        assert not [ln for ln in Health(battery_pct=-1).lines()
                    if "Battery" in ln]

    def test_one_unreadable_thing_does_not_lose_the_others(self):
        """Each piece is read independently, so a machine that will not
        answer one question still answers the rest."""
        snapshot = health(path="/definitely/not/a/mount/point")
        assert snapshot.disk_total_gb == 0, "the unreadable one is absent"
        assert snapshot.memory_total_gb > 0, "and the others survived it"
        assert snapshot.cores > 0


# -- the tools ---------------------------------------------------------------

def _status(**kw):
    ws = Path("/tmp")
    tool = SystemStatus(ws, Boundary(Scope.REPO, ws))
    return asyncio.run(tool.run(SystemStatusInput(**kw)))


def _control(**kw):
    ws = Path("/tmp")
    tool = SystemControl(ws, Boundary(Scope.REPO, ws))
    return asyncio.run(tool.run(SystemControlInput(**kw)))


class TestTheStatusTool:
    def test_it_answers_why_is_my_machine_slow(self):
        out = _status()
        assert out.ok
        assert "Memory:" in out.output and "Disk:" in out.output

    def test_it_never_needs_permission(self):
        from wynxo.permissions import PermissionStore

        assert not PermissionStore().needs_prompt("system_status", False, {})

    def test_asking_about_one_thing_gets_one_thing(self):
        out = _status(about="disk")
        assert "Disk:" in out.output and "Memory:" not in out.output

    def test_a_missing_stack_is_named_not_hidden(self):
        out = _status(about="volume")
        if not out.ok:
            assert "Install" in out.error


class TestTheControlTool:
    def test_power_always_asks_even_in_auto(self):
        from wynxo.permissions import PermissionStore
        from wynxo.scope import Mode

        store = PermissionStore()
        store.mode = Mode.AUTO
        assert store.needs_prompt("system_control", True,
                                  {"action": "shutdown"})

    def test_a_volume_outside_the_range_is_refused_with_the_alternative(
            self, monkeypatch):
        import wynxo.tools.system_tool as tool_mod

        class Present(Audio):
            def __init__(self):
                super().__init__(Recorder(WPCTL))
                self.stack = "wpctl"

        monkeypatch.setattr(tool_mod, "Audio", Present)
        out = _control(action="volume", level=500)
        assert not out.ok and "volume_up" in out.error

    def test_a_machine_with_no_sound_says_that_first(self, monkeypatch):
        """Ahead of complaining about the argument. The missing package is
        the harder constraint: told "level must be 0-100" on a machine with
        no sound stack, the model goes and fixes the wrong thing."""
        import wynxo.tools.system_tool as tool_mod

        class Absent(Audio):
            def __init__(self):
                super().__init__(Recorder())
                self.stack = ""

        monkeypatch.setattr(tool_mod, "Audio", Absent)
        out = _control(action="volume", level=500)
        assert not out.ok and "Install" in out.error

    def test_copying_nothing_is_refused(self):
        assert not _control(action="copy", text="").ok

    def test_notifying_nothing_is_refused(self):
        assert not _control(action="notify", text="").ok

    def test_an_action_this_machine_cannot_do_names_the_package(self):
        out = _control(action="volume_up")
        if not out.ok:
            assert "Install" in out.error
            assert out.metadata.get("kind") == "unavailable"
