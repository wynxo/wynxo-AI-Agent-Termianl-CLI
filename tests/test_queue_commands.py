"""Type-ahead, command abbreviations, and the quiet start."""

import inspect

import pytest

from wynxo.cli import ALIASES, COMMANDS, resolve_command
from wynxo.pet import Pet
from wynxo.queue import Pending
from wynxo.ui import UI, ActivityBar


class TestPending:
    def test_characters_build_a_draft(self):
        pending = Pending()
        for char in "fix the tests":
            assert pending.key(char) is None
        assert pending.draft == "fix the tests"

    def test_enter_queues_the_line(self):
        pending = Pending()
        for char in "run tests\r":
            result = pending.key(char)
        assert result == "run tests"
        assert pending.draft == ""
        assert len(pending) == 1

    def test_messages_come_back_oldest_first(self):
        pending = Pending()
        for line in ("first", "second", "third"):
            for char in line + "\r":
                pending.key(char)
        assert [pending.take() for _ in range(3)] == ["first", "second", "third"]
        assert pending.take() is None

    def test_backspace_edits(self):
        pending = Pending()
        for char in "helo":
            pending.key(char)
        pending.key("\x7f")
        for char in "lo":
            pending.key(char)
        assert pending.draft == "hello"

    def test_ctrl_u_clears_the_line(self):
        pending = Pending()
        for char in "throw this away":
            pending.key(char)
        pending.key("\x15")
        assert pending.draft == ""

    def test_blank_enter_queues_nothing(self):
        pending = Pending()
        assert pending.key("\r") is None
        assert len(pending) == 0

    def test_whitespace_only_queues_nothing(self):
        pending = Pending()
        for char in "   \r":
            pending.key(char)
        assert len(pending) == 0

    def test_control_characters_are_ignored(self):
        """Anything with a binding belongs to the key watcher, not here."""
        pending = Pending()
        for char in ("\x0f", "\x14", "\x01", "\x1b"):
            pending.key(char)
        assert pending.draft == ""

    def test_preview_shows_the_draft_then_the_count(self):
        pending = Pending()
        for char in "typing":
            pending.key(char)
        assert pending.preview() == "typing"
        pending.key("\r")
        assert pending.preview() == "1 queued"

    def test_preview_keeps_the_end_of_a_long_draft(self):
        """You care about the characters you just typed, not the first ones."""
        pending = Pending()
        for char in "a very long message that keeps going and going":
            pending.key(char)
        preview = pending.preview(width=20)
        assert len(preview) <= 20
        assert preview.endswith("going")

    def test_clear_reports_what_it_dropped(self):
        pending = Pending()
        for line in ("one", "two"):
            for char in line + "\r":
                pending.key(char)
        assert "2 queued" in pending.clear()
        assert not pending

    def test_truthiness_covers_draft_and_queue(self):
        pending = Pending()
        assert not pending
        pending.key("x")
        assert pending
        pending.key("\r")
        assert pending


class TestBarShowsTyping:
    def test_typing_replaces_the_detail(self):
        """Your own keystrokes matter more than what the tool is doing."""
        ui = UI()
        ui.width = 92
        bar = ActivityBar(ui, "medium", pet=Pet())
        bar.update(activity="writing", detail="src/transfer.py", tokens=50)
        assert "src/transfer.py" in bar._render().plain
        bar.queued = "next question"
        rendered = bar._render().plain
        assert "next question" in rendered
        assert "src/transfer.py" not in rendered

    @pytest.mark.parametrize("width", [56, 70, 92, 140])
    def test_typing_survives_at_every_width(self, width):
        ui = UI()
        ui.width = width
        bar = ActivityBar(ui, "medium", "^O thinking  ^T detail", pet=Pet())
        bar.update(activity="writing", detail="src/a.py", tokens=999)
        bar.queued = "second one"
        rendered = bar._render()
        assert "second one" in rendered.plain, f"lost at {width}"
        assert rendered.cell_len == width


class TestCommandAbbreviations:
    def test_exact_commands_win_over_aliases(self):
        """/mode must keep meaning /mode even though /mo means /model."""
        assert "/mode" in COMMANDS
        assert ALIASES["/mo"] == "/model"

    @pytest.mark.parametrize("short,full", [
        ("/mo", "/model"), ("/mod", "/model"), ("/m", "/model"),
        ("/th", "/theme"), ("/eff", "/effort"), ("/mem", "/memory"),
        ("/u", "/undo"), ("/q", "/quit"), ("/exit", "/quit"),
    ])
    def test_short_forms_resolve(self, short, full):
        assert resolve_command(short) == full

    def test_an_unambiguous_prefix_resolves_without_an_alias(self):
        assert resolve_command("/doct") == "/doctor"
        assert resolve_command("/comp") == "/compact"

    def test_an_ambiguous_prefix_resolves_to_nothing(self):
        """Silently picking one of two commands is worse than asking."""
        assert resolve_command("/xyz") is None

    def test_every_alias_points_at_a_real_command(self):
        for short, full in ALIASES.items():
            assert full in COMMANDS, f"{short} -> {full} does not exist"

    def test_no_alias_shadows_a_real_command(self):
        for short in ALIASES:
            assert short not in COMMANDS, f"{short} is both a command and an alias"


class TestQuietStart:
    def test_connect_only_prints_problems(self):
        """A wall of green OK lines on every start is noise."""
        from wynxo import cli

        source = inspect.getsource(cli.Repl._connect)
        assert "def note(" in source
        assert "if problems:" in source
        # No unconditional success lines left behind.
        assert "status.ok(" not in source


class TestThinkingDefault:
    def test_thinking_is_hidden_by_default(self):
        """It always thinks; this only controls whether you watch it."""
        from wynxo.config import Config

        assert Config().show_thinking is False

    def test_toggling_off_closes_the_open_block(self):
        from wynxo import cli

        source = inspect.getsource(cli.TerminalCallbacks.toggle_thinking)
        assert "_end_thinking" in source
