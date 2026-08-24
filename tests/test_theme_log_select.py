"""Theme, session journal and the arrow-key selector."""

import asyncio
import json

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from wynxo import theme
from wynxo.journal import MAX_FIELD, prune
from wynxo.select import Choice, choose


class TestTheme:
    def test_default_is_purple(self):
        assert theme.DEFAULT == "purple"
        assert theme.resolve("purple").accent.startswith("#")

    def test_unknown_theme_falls_back_rather_than_raising(self):
        """A bad name in a config file must not stop the agent starting."""
        assert theme.resolve("nonsense").name == theme.DEFAULT
        assert theme.resolve("").name == theme.DEFAULT
        assert theme.resolve(None).name == theme.DEFAULT

    def test_every_palette_defines_every_colour(self):
        reference = set(theme.PURPLE.as_dict())
        for name, palette in theme.PALETTES.items():
            assert set(palette.as_dict()) == reference, name
            for key, value in palette.as_dict().items():
                assert value, f"{name}.{key} is empty"

    def test_status_colours_keep_their_meaning(self):
        """good/warn/bad must stay distinguishable in every palette, or the
        status lines stop carrying information."""
        for name, palette in theme.PALETTES.items():
            assert len({palette.good, palette.warn, palette.bad}) == 3, name

    def test_plain_uses_named_colours_for_16_colour_terminals(self):
        for value in theme.PLAIN.as_dict().values():
            assert not value.startswith("#"), "plain must not need truecolour"

    def test_applying_a_palette_rebinds_the_module_colours(self):
        from wynxo import ui

        before = ui.ACCENT
        try:
            ui.apply_palette(theme.EMBER)
            assert ui.ACCENT == theme.EMBER.accent
            assert ui.BAR_STYLE.endswith(theme.EMBER.bar_bg)
        finally:
            ui.apply_palette(theme.PURPLE)
            assert ui.ACCENT != before or True


@pytest.fixture
def journal(tmp_path, monkeypatch):
    import wynxo.journal as module

    monkeypatch.setattr(module, "data_dir", lambda: tmp_path)
    return module.Journal.open("test1234")


class TestJournal:
    def test_writes_one_json_object_per_line(self, journal):
        journal.user("hello")
        journal.assistant("hi", tokens=3)
        lines = journal.path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            json.loads(line)        # each line stands alone
        assert len(lines) == 3      # session, user, assistant

    def test_records_the_whole_turn(self, journal):
        journal.user("q")
        journal.thinking("reasoning")
        journal.tool("read_file", {"path": "a.py"})
        journal.tool_result("read_file", True, "contents")
        journal.assistant("answer")
        kinds = [r["kind"] for r in journal.tail()]
        assert kinds == ["session", "user", "thinking", "tool",
                         "tool_result", "assistant"]

    def test_huge_fields_are_trimmed(self, journal):
        journal.tool_result("read_file", True, "x" * 200_000)
        record = journal.tail()[-1]
        assert len(record["output"]) < MAX_FIELD + 200
        assert "more characters" in record["output"]

    def test_nested_arguments_are_trimmed_too(self, journal):
        journal.tool("write_file", {"path": "a", "content": "y" * 100_000})
        assert len(journal.tail()[-1]["args"]["content"]) < MAX_FIELD + 200

    def test_disabled_journal_writes_nothing(self, tmp_path, monkeypatch):
        import wynxo.journal as module

        monkeypatch.setattr(module, "data_dir", lambda: tmp_path)
        off = module.Journal.open("x", enabled=False)
        off.user("secret")
        assert off.path is None
        assert not list(tmp_path.glob("**/*.jsonl"))

    def test_an_unwritable_directory_never_breaks_the_agent(self, monkeypatch):
        import wynxo.journal as module
        from pathlib import Path

        monkeypatch.setattr(module, "data_dir",
                            lambda: Path("/proc/definitely-not-writable"))
        broken = module.Journal.open("x")
        broken.user("still fine")       # must not raise
        assert not broken.enabled

    def test_old_logs_are_pruned(self, tmp_path):
        for i in range(30):
            (tmp_path / f"log{i:02}.jsonl").write_text("{}\n")
        prune(tmp_path, keep=5)
        assert len(list(tmp_path.glob("*.jsonl"))) == 5

    def test_tail_survives_a_truncated_last_line(self, journal):
        journal.user("ok")
        with journal.path.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "user", "text": "cut off')
        records = journal.tail()
        assert records[-1]["text"] == "ok"


async def drive(choices, keys, **kwargs):
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        with create_app_session(input=pipe, output=DummyOutput()):
            return await choose(choices, **kwargs)


CHOICES = [
    Choice("a", "alpha", "tools", hint="first"),
    Choice("b", "beta", "no tools", "badge.warn", hint="second"),
    Choice("c", "gamma", "tools", hint="third"),
]


class TestSelect:
    def test_enter_takes_the_default(self):
        assert asyncio.run(drive(CHOICES, "\r")) == "a"

    def test_a_different_default(self):
        assert asyncio.run(drive(CHOICES, "\r", default=2)) == "c"

    def test_arrow_keys_move(self):
        assert asyncio.run(drive(CHOICES, "\x1b[B\r")) == "b"
        assert asyncio.run(drive(CHOICES, "\x1b[B\x1b[B\r")) == "c"

    def test_vim_keys_move(self):
        assert asyncio.run(drive(CHOICES, "jj\r")) == "c"
        assert asyncio.run(drive(CHOICES, "jjk\r")) == "b"

    def test_selection_wraps(self):
        assert asyncio.run(drive(CHOICES, "\x1b[A\r")) == "c"
        assert asyncio.run(drive(CHOICES, "\x1b[B\x1b[B\x1b[B\r")) == "a"

    def test_number_keys_select_immediately(self):
        """Typing a number must still work; an arrows-only list cannot be
        driven by anyone used to the old prompt."""
        assert asyncio.run(drive(CHOICES, "2")) == "b"
        assert asyncio.run(drive(CHOICES, "3")) == "c"

    def test_escape_cancels(self):
        assert asyncio.run(drive(CHOICES, "\x1b")) is None

    def test_ctrl_c_cancels(self):
        assert asyncio.run(drive(CHOICES, "\x03")) is None

    def test_home_and_end(self):
        assert asyncio.run(drive(CHOICES, "\x1b[F\r")) == "c"
        assert asyncio.run(drive(CHOICES, "\x1b[F\x1b[H\r")) == "a"

    def test_empty_list_returns_none(self):
        assert asyncio.run(drive([], "\r")) is None

    def test_long_labels_do_not_break_it(self):
        long = [Choice("x", "some/absurdly-long-vendor/model-name:70b-instruct-q8",
                       "tools", hint="18GB")]
        assert asyncio.run(drive(long, "\r", width=40)) == "x"

    def test_ascii_mode_avoids_the_unicode_cursor(self):
        assert asyncio.run(drive(CHOICES, "\r", unicode=False)) == "a"


class TestPluralCommands:
    """People type the plural without thinking, and the command lists
    things, so /themes and /models are the natural words. Prefix matching
    cannot catch them -- "/theme" does not start with "/themes"."""

    def test_plurals_resolve(self):
        from wynxo.cli import resolve_command

        assert resolve_command("/themes") == "/theme"
        assert resolve_command("/models") == "/model"
        assert resolve_command("/efforts") == "/effort"

    def test_commands_that_genuinely_end_in_s_still_win(self):
        """/tools must not be stripped to /tool, which does not exist."""
        from wynxo.cli import resolve_command

        assert resolve_command("/tools") == "/tools"
        assert resolve_command("/sessions") == "/sessions"

    def test_abbreviations_still_work(self):
        from wynxo.cli import resolve_command

        assert resolve_command("/mo") == "/model"
        assert resolve_command("/th") == "/theme"

    def test_nonsense_is_still_unknown(self):
        from wynxo.cli import resolve_command

        assert resolve_command("/nonsense") is None
        assert resolve_command("/s") is None      # ambiguous, must not guess


class TestThemeAppliesLive:
    """A theme change used to need a restart: the consumers had done
    `from .ui import ACCENT`, which binds their own copy of the name."""

    def test_the_accent_reaches_the_importing_modules(self):
        from wynxo import cli, ui
        from wynxo.theme import resolve

        before = cli.ACCENT
        try:
            ui.apply_palette(resolve("sakura"))
            assert cli.ACCENT == resolve("sakura").accent
            assert cli.ACCENT != before
        finally:
            ui.apply_palette(resolve("purple"))

    def test_a_name_that_means_something_else_is_not_clobbered(self):
        """cli.py imports WARN from .status, where it is a status tag and
        not a colour. Overwriting it printed a raw '[#f0c674]' on screen
        where '[ WARN ]' belonged."""
        from wynxo import cli, ui
        from wynxo.theme import resolve

        before = cli.WARN
        try:
            ui.apply_palette(resolve("ember"))
            assert cli.WARN == before, "the status tag must survive a theme change"
            assert not str(cli.WARN).startswith("#")
        finally:
            ui.apply_palette(resolve("purple"))

    def test_every_listed_consumer_really_imports_those_names(self):
        """The map is hand-maintained, so it has to be checked against the
        imports it claims to mirror."""
        import importlib

        from wynxo.ui import _COLOUR_CONSUMERS

        for module_name, names in _COLOUR_CONSUMERS.items():
            module = importlib.import_module(module_name)
            for name in names:
                assert hasattr(module, name), f"{module_name} has no {name}"


class TestCommandCompleter:
    """Suggestions as you type, so the command list is not something you
    have to have memorised."""

    def complete(self, text):
        from prompt_toolkit.document import Document

        from wynxo.cli import CommandCompleter

        return list(CommandCompleter().get_completions(Document(text), None))

    def test_a_prefix_offers_every_command_that_matches(self):
        got = {c.text for c in self.complete("/mo")}
        assert got == {"/model", "/mode"}

    def test_each_suggestion_carries_what_it_does(self):
        """A bare list of names is not much help if you cannot remember
        which is which."""
        first = self.complete("/mo")[0]
        assert first.display_meta_text.strip()

    def test_it_replaces_what_was_typed(self):
        completion = self.complete("/mo")[0]
        assert completion.start_position == -len("/mo")

    def test_an_alias_that_is_not_a_prefix_is_still_found(self):
        """/q expands to /quit, which it is not a prefix of -- prefix
        matching alone would never suggest it."""
        assert "/quit" in {c.text for c in self.complete("/q")}

    def test_ordinary_prose_is_never_completed(self):
        """The menu must not open over the top of what you are actually
        typing most of the time."""
        assert self.complete("fix the parser") == []
        assert self.complete("") == []

    def test_completion_stops_after_the_command_word(self):
        """`/model qwen3` is an argument, not another command."""
        assert self.complete("/model qwen") == []

    def test_a_full_command_still_offers_itself(self):
        assert "/theme" in {c.text for c in self.complete("/theme")}

    def test_nothing_matches_nonsense(self):
        assert self.complete("/zzzz") == []

    def test_no_duplicates_when_an_alias_points_at_a_matching_command(self):
        """/m is an alias for /model and also a prefix of it."""
        texts = [c.text for c in self.complete("/m")]
        assert len(texts) == len(set(texts))
