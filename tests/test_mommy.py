"""The /mommy toggle, and the effort keys that must not print into the box.

`/mommy` flips between the doting mommy persona and the plain voice, and a
voice change has to land in four places at once: saved config, pet style,
talker persona and the agent's system prompt. The effort keys (Ctrl-E/B)
used to print their "effort: ..." line straight into the live prompt, which
wedged it between the box edge and the input; they now park the note in the
toolbar instead.
"""

from __future__ import annotations

import types

from wynxo import cli
from wynxo.config import Config
from wynxo.effort import resolve
from wynxo.pet import Pet


class _UI:
    """Captures what the Repl prints, without touching a real terminal."""

    def __init__(self):
        self.messages = []

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warn(self, message: str) -> None:
        self.messages.append(("warn", message))


class _Agent:
    def __init__(self):
        self.refreshes = 0
        self.set_efforts = []
        self.policy = None

    def refresh_system_prompt(self) -> None:
        self.refreshes += 1

    def set_effort(self, policy) -> None:
        self.set_efforts.append(policy.name)
        self.policy = policy


class _Talker:
    def __init__(self):
        self.voice_block = ""


def _repl(monkeypatch, voice: str = "mommy") -> "tuple[types.SimpleNamespace, dict]":
    """A bare Repl carrying only what cmd_mommy reaches for."""
    config = Config(verify_with_tests=False, voice=voice)
    saved = {}
    monkeypatch.setattr(config, "save", lambda path=None: (saved.update(
        {"voice": config.voice}), path)[1])
    agent = _Agent()
    talker = _Talker()
    repl = types.SimpleNamespace(
        config=config, agent=agent, talker=talker,
        pet=Pet(name="wyn", enabled=True), ui=_UI(),
        _prompt_note=None)
    repl._set_voice = cli.Repl._set_voice.__get__(repl, type(repl))
    return repl, saved


# -- /mommy ------------------------------------------------------------------


def test_mommy_without_an_argument_toggles_to_plain(monkeypatch):
    repl, saved = _repl(monkeypatch, voice="mommy")
    cli.Repl.cmd_mommy(repl, [])
    assert repl.config.voice == "plain"
    assert saved["voice"] == "plain"
    assert repl.agent.refreshes == 1
    assert repl.pet.style_name == "default"
    # Plain now carries a human-tone block: present, and free of the
    # support-bot filler it exists to forbid.
    assert repl.talker.voice_block != ""
    flat = " ".join(repl.talker.voice_block.split()).lower()
    assert "support bot" in flat
    assert "corporate filler" in flat
    assert repl.ui.messages[-1][0] == "success"


def test_mommy_toggles_back_on(monkeypatch):
    repl, saved = _repl(monkeypatch, voice="plain")
    cli.Repl.cmd_mommy(repl, [])
    assert repl.config.voice == "mommy"
    assert saved["voice"] == "mommy"
    assert repl.pet.style_name == "mommy"
    assert "mommy" in repl.talker.voice_block
    assert repl.agent.refreshes == 1


def test_mommy_on_and_off_are_explicit(monkeypatch):
    repl, _ = _repl(monkeypatch, voice="plain")
    cli.Repl.cmd_mommy(repl, ["on"])
    assert repl.config.voice == "mommy"
    cli.Repl.cmd_mommy(repl, ["off"])
    assert repl.config.voice == "plain"


def test_mommy_already_in_that_state_is_an_info_not_a_change(monkeypatch):
    repl, _ = _repl(monkeypatch, voice="mommy")
    cli.Repl.cmd_mommy(repl, ["on"])
    assert repl.agent.refreshes == 0
    assert repl.ui.messages[-1] == ("info", "mommy style already on")


def test_mommy_unknown_argument_is_a_warn(monkeypatch):
    repl, _ = _repl(monkeypatch)
    cli.Repl.cmd_mommy(repl, ["sideways"])
    assert repl.config.voice == "mommy"
    assert repl.ui.messages[-1][0] == "warn"


# -- the effort keys -----------------------------------------------------------


def test_shift_effort_parks_the_note_instead_of_printing(monkeypatch):
    """Ctrl-E must not print into the live prompt: the note belongs to the
    toolbar, which prompt_toolkit redraws underneath the input."""
    repl, _ = _repl(monkeypatch)
    repl.policy = resolve("medium")
    repl.pet = Pet(name="wyn", enabled=True)
    agent = repl.agent
    monkeypatch.setattr(repl, "ui", _UI())
    cli.Repl._shift_effort(repl, 1)
    assert repl.config.effort == "high"
    assert agent.set_efforts == ["high"]
    assert repl._prompt_note is not None
    assert "effort: high" in repl._prompt_note[0]
    assert repl.ui.messages == [], "a key binding must not print transcript lines"


def test_shift_effort_no_notes_when_level_unchanged(monkeypatch):
    repl, _ = _repl(monkeypatch)
    repl.policy = resolve("ultra")
    repl.pet = Pet(name="wyn", enabled=True)
    monkeypatch.setattr(repl, "ui", _UI())
    cli.Repl._shift_effort(repl, 1)       # already at the top
    assert repl._prompt_note is None


# -- the toolbar note -----------------------------------------------------------


def _toolbar_repl(monkeypatch) -> types.SimpleNamespace:
    """The same stand-in the ascii-terminal tests use for the box."""
    import io

    from wynxo.ui import UI, Glyphs

    ui = UI()
    ui.g = Glyphs(False)
    ui.width = 100
    ui.console.file = io.StringIO()
    ui.console._width = ui.width
    repl = types.SimpleNamespace(ui=ui)
    repl._status_line = lambda: "medium . 0 tok . ctx 0%"
    repl._bottom_toolbar = cli.Repl._bottom_toolbar.__get__(repl, type(repl))
    repl._prompt_note = None
    return repl


def test_a_fresh_note_appears_in_the_toolbar(monkeypatch):
    import time

    repl = _toolbar_repl(monkeypatch)
    repl._prompt_note = ("effort: high -- real thinking", time.monotonic() + 3)
    assert "effort: high -- real thinking" in repl._bottom_toolbar().value


def test_an_expired_note_is_dropped(monkeypatch):
    import time

    repl = _toolbar_repl(monkeypatch)
    repl._prompt_note = ("effort: high -- real thinking", time.monotonic() - 1)
    assert "effort: high" not in repl._bottom_toolbar().value
    assert repl._prompt_note is None
