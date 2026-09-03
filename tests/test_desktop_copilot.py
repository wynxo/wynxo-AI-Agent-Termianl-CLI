"""Doing several things on the machine, in one sentence.

"open kcalc, then firefox, then a terminal running python3 main.py" is a
single request with three actions in it, and the last one is not really a
launch at all -- it is a command, in a window wynxo does not own, outside
every guard the shell tool has.

So there are two halves here: that all three happen, in order, and that the
command half is asked about on the same terms a shell command would be.
"""

from __future__ import annotations

import pathlib
import time

import pytest

from wynxo.intent import Intent, parse
from wynxo.permissions import Decision, PermissionStore
from wynxo.scope import Mode
from wynxo.tools import apps as apps_module
from wynxo.tools.appcatalog import AppEntry
from wynxo.tools.apps import TERMINALS, terminal_argv

from test_agent import RecordingCallbacks, make_agent


def entry(name: str, suffix: str = "") -> AppEntry:
    return AppEntry(name=name, path=pathlib.Path(f"/usr/bin/{name}{suffix}"),
                    source="path")


@pytest.fixture
def machine(monkeypatch):
    """A machine with three applications and nothing really launched."""
    record = {"started": [], "argv": []}

    async def started(app, open_path=""):
        record["started"].append((app.name, open_path))

    async def argv(command):
        record["argv"].append(command)

    monkeypatch.setattr(apps_module, "_launch_entry", started)
    monkeypatch.setattr(apps_module, "_shell_launch", argv)
    monkeypatch.setattr(apps_module.shutil, "which", lambda n: f"/usr/bin/{n}",
                        raising=False)
    return record


async def act(tmp_path, machine, targets, command="", decision=Decision.ALLOW,
              installed=("kcalc", "firefox", "konsole")):
    cb = RecordingCallbacks(permission=decision)
    agent, _, _ = make_agent(tmp_path, [{"content": "ok"}], callbacks=cb)
    catalog = agent.tools.get("launch_application").catalog
    # Stubbed at the scan, not the cache: a miss triggers one rescan (so an
    # application installed a minute ago is findable), and a fixture that
    # only set the cache would be wiped by it -- which looks exactly like
    # "a missing application stops the ones after it".
    catalog._scan = lambda: tuple(entry(n) for n in installed)
    catalog._entries = catalog._scan()
    out = await agent._system_action(
        "...", Intent(kind="system_action", targets=tuple(targets),
                      command=command), time.monotonic())
    return out, cb


class TestSeveralApplicationsInOneSentence:
    async def test_all_three_are_launched_in_order(self, tmp_path, machine):
        await act(tmp_path, machine, ["kcalc", "firefox", "konsole"])
        assert [name for name, _ in machine["started"]] == \
            ["kcalc", "firefox", "konsole"]

    async def test_what_the_user_is_told_is_not_the_model_s_instructions(
            self, tmp_path, machine):
        """launch_application's output ends with "reply and stop" -- that is
        scaffolding aimed at the model, and it was being shown verbatim as
        the answer. Three applications meant three copies of it."""
        out, _ = await act(tmp_path, machine, ["kcalc", "firefox"])
        assert "do not perform further tool calls" not in out.content
        assert out.content.count("Launched") == 2

    async def test_one_that_is_not_installed_does_not_stop_the_rest(
            self, tmp_path, machine):
        await act(tmp_path, machine, ["kcalc", "nothing-like-this", "firefox"])
        assert [name for name, _ in machine["started"]] == ["kcalc", "firefox"]


class TestATerminalCanBeGivenSomethingToRun:
    async def test_the_command_goes_to_the_last_target(self, tmp_path, machine):
        """Where the sentence puts it: "kcalc, then firefox, then a terminal
        running main.py"."""
        await act(tmp_path, machine, ["kcalc", "firefox", "konsole"],
                  command="python3 main.py")
        assert [name for name, _ in machine["started"]] == ["kcalc", "firefox"]
        assert machine["argv"] == [
            ["/usr/bin/konsole", "-e", "bash", "-c",
             "python3 main.py; exec bash"]]

    async def test_the_window_stays_open_afterwards(self, tmp_path, machine):
        """"Open a terminal and run this" means the window is there to be
        read. Left to exit, a command taking a second flashes a window and
        closes it, and the output is gone before anyone sees it."""
        await act(tmp_path, machine, ["konsole"], command="echo hello")
        assert machine["argv"][0][-1].endswith("; exec bash")

    async def test_the_user_is_told_what_was_run(self, tmp_path, machine):
        out, _ = await act(tmp_path, machine, ["konsole"],
                           command="python3 main.py")
        assert "python3 main.py" in out.content

    def test_every_terminal_has_a_spelling_that_was_looked_up(self):
        """The flag meaning "the rest is the program" is -e, -x, --, or
        nothing, and there is no convention. A guess does not fail cleanly:
        it opens a terminal that ignores the command."""
        for name, flags in TERMINALS.items():
            assert isinstance(flags, tuple), name

    def test_an_application_that_is_not_a_terminal_gets_no_command(self):
        assert terminal_argv(entry("firefox"), "rm -rf /") is None

    async def test_and_is_refused_rather_than_launched_without_it(
            self, tmp_path, machine):
        """Dropping the command silently would open the application and
        report success, and the user would be told their script was run by
        something that never saw it."""
        out, _ = await act(tmp_path, machine, ["firefox"],
                           command="python3 main.py")
        assert machine["started"] == []
        assert machine["argv"] == []
        assert out.errors

    def test_a_desktop_entry_resolves_to_its_binary(self, monkeypatch):
        """gio launch has nowhere to put a command, so the executable is
        looked up on PATH instead of the .desktop file being run."""
        monkeypatch.setattr(apps_module.shutil, "which",
                            lambda n: f"/usr/bin/{n}")
        argv = terminal_argv(entry("konsole", ".desktop"), "ls")
        assert argv is not None and argv[0].endswith("konsole")
        assert not argv[0].endswith(".desktop")


class TestACommandIsACommandWhateverLaunchedIt:
    """It runs outside every guard the shell tool has -- no output ceiling,
    no workspace, no read-only test -- in a window wynxo does not own."""

    @pytest.mark.parametrize("mode", [Mode.MANUAL, Mode.AUTO])
    def test_a_dangerous_one_is_asked_about_in_every_mode(self, mode):
        store = PermissionStore()
        store.mode = mode
        assert store.needs_prompt(
            "launch_application", True,
            {"query": "konsole", "command": "rm -rf ~"})

    def test_a_read_only_one_is_not(self):
        store = PermissionStore()
        store.mode = Mode.AUTO
        assert not store.needs_prompt(
            "launch_application", True, {"query": "konsole", "command": "ls"})

    def test_opening_an_application_is_still_just_opening_it(self):
        store = PermissionStore()
        store.mode = Mode.AUTO
        assert not store.needs_prompt(
            "launch_application", True, {"query": "firefox"})

    async def test_declining_it_runs_nothing(self, tmp_path, machine):
        out, cb = await act(tmp_path, machine, ["konsole"],
                            command="rm -rf ~/important",
                            decision=Decision.DENY)
        assert cb.permission_asks, "it never asked"
        assert machine["argv"] == []

    async def test_the_earlier_launches_still_happened(self, tmp_path, machine):
        """Declining the command is not declining the whole sentence."""
        await act(tmp_path, machine, ["kcalc", "konsole"],
                  command="rm -rf ~", decision=Decision.DENY)
        assert [name for name, _ in machine["started"]] == ["kcalc"]


class TestTheRouterCanSayIt:
    def test_several_targets_survive(self):
        got = parse('{"kind": "system_action", '
                    '"targets": ["a", "b", "c"], "command": ""}')
        assert got.targets == ("a", "b", "c")

    def test_a_command_survives(self):
        got = parse('{"kind": "system_action", "targets": ["t"], '
                    '"command": "python3 main.py"}')
        assert got.command == "python3 main.py"

    def test_a_command_on_any_other_kind_is_dropped(self):
        """A command on a conversation or a coding turn is the model
        filling in a field it was shown, and acting on it would run
        something nobody asked for."""
        for kind in ("conversation", "coding"):
            got = parse('{"kind": "%s", "targets": [], '
                        '"command": "rm -rf /"}' % kind)
            assert got.command == ""

    def test_a_missing_command_is_empty_not_an_error(self):
        assert parse('{"kind": "system_action", "targets": ["t"]}').command == ""
