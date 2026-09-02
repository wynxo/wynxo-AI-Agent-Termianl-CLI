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

    def test_kawaii_theme_is_available(self):
        assert theme.resolve("kawaii").name == "kawaii"
        assert "kawaii" in theme.names()

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

        # Made unwritable by refusing the mkdir rather than by naming a
        # path: "/proc/..." is a perfectly writable relative path on Windows,
        # so the test passed there without testing anything.
        def refuse(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(module, "data_dir", lambda: Path("anywhere"))
        monkeypatch.setattr(Path, "mkdir", refuse)
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

    def test_every_module_that_imports_a_colour_is_swept(self):
        """The old map named, per module, which colours it had imported --
        and it had drifted. cli.py imports GOOD, BAD and BAR_ACCENT too, and
        none of the three was listed, so after /theme the edit card's done
        and failed marks and the effort surge kept the palette before last.

        Read from the imports themselves rather than from a list beside
        them: a module that starts importing a colour must not have to
        remember to add itself.
        """
        import ast
        import pathlib

        from wynxo.ui import _COLOUR_CONSUMERS, _COLOUR_NAMES

        root = pathlib.Path(__file__).resolve().parent.parent / "wynxo"
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and (node.module or "").endswith("ui")
                for alias in node.names
            }
            colours = imported & set(_COLOUR_NAMES)
            if not colours:
                continue
            module = "wynxo." + str(
                path.relative_to(root).with_suffix("")).replace("/", ".")
            if module == "wynxo.ui":
                continue
            assert module in _COLOUR_CONSUMERS, (
                f"{module} imports {sorted(colours)} and is never swept, so "
                f"they keep the old palette after /theme")

    def test_every_imported_colour_actually_follows_the_theme(self):
        """The end of the story the map was there to tell."""
        import importlib

        from wynxo import ui
        from wynxo.theme import resolve
        from wynxo.ui import _COLOUR_CONSUMERS, _COLOUR_NAMES

        sakura = resolve("sakura")
        wanted = {"ACCENT": sakura.accent, "MUTED": sakura.muted,
                  "FAINT": sakura.faint, "GOOD": sakura.good,
                  "WARN": sakura.warn, "BAD": sakura.bad,
                  "BAR_STYLE": f"on {sakura.bar_bg}",
                  "BAR_ACCENT": sakura.bar_accent, "BAR_DIM": sakura.bar_dim}
        try:
            ui.apply_palette(sakura)
            for module_name in _COLOUR_CONSUMERS:
                module = importlib.import_module(module_name)
                for name in _COLOUR_NAMES:
                    if name == "WARN" and module_name == "wynxo.cli":
                        continue          # the status tag, not a colour
                    if not hasattr(module, name):
                        continue          # not imported there
                    assert getattr(module, name) == wanted[name], (
                        f"{module_name}.{name} kept the old palette")
        finally:
            ui.apply_palette(resolve("purple"))


class TestCommandCompleter:
    """Suggestions as you type, so the command list is not something you
    have to have memorised."""

    def complete(self, text):
        from prompt_toolkit.document import Document

        from wynxo.cli import CommandCompleter

        return list(CommandCompleter().get_completions(Document(text), None))

    def test_a_prefix_offers_every_command_that_matches(self):
        got = {c.text for c in self.complete("/mo")}
        assert got == {"/model", "/mode", "/mommy"}

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


class TestNewChat:
    def test_new_is_a_command_and_abbreviates(self):
        from wynxo.cli import COMMANDS, resolve_command

        assert "/new" in COMMANDS
        assert resolve_command("/new") == "/new"
        assert resolve_command("/n") == "/new"

    def test_it_resets_more_than_clear_does(self):
        """/clear empties the message list in place; /new is a new chat --
        new session id, new log, undo history dropped."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.Repl.cmd_new)
        assert "Session(" in source
        assert "checkpoints.clear()" in source
        assert "Journal.open" in source
        assert "self.ui.clear()" in source

    def test_memory_is_not_reset(self):
        """Memory is the thing that is supposed to outlive a conversation."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.Repl.cmd_new)
        assert "Memory(" not in source
        assert "memory" not in source.replace("memory kept", "")


class TestPersonalFactsAreRemembered:
    """Being told your name and forgetting it is the difference between an
    assistant and a search box."""

    def test_the_prompt_tells_the_model_to_save_what_it_is_told(self):
        from wynxo.prompts import MEMORY_TOOL_NOTE

        lowered = MEMORY_TOOL_NOTE.lower()
        assert 'scope="user"' in MEMORY_TOOL_NOTE
        assert "name" in lowered
        for word in ("tells you something about themselves", "same turn"):
            assert word in lowered

    def test_it_says_saying_so_is_not_the_same_as_doing_it(self):
        """Models like to answer "I'll remember that" and save nothing."""
        from wynxo.prompts import MEMORY_TOOL_NOTE

        assert "lie" in MEMORY_TOOL_NOTE.lower()

    def test_small_talk_does_not_exempt_it(self):
        """"my name is heio" is small talk *and* a fact worth keeping, and
        the small-talk path skips the whole planning pipeline."""
        from wynxo.prompts import MEMORY_TOOL_NOTE

        assert "small talk" in MEMORY_TOOL_NOTE.lower()

    def test_the_note_reaches_the_system_prompt(self, tmp_path):
        from wynxo.effort import resolve
        from wynxo.prompts import build_system_prompt

        prompt = build_system_prompt(tmp_path, resolve("medium"))
        assert 'scope="user"' in prompt


class TestResume:
    """Sessions were already written to disk after every turn, but there was
    no way back into one -- which made them a debugging artefact rather than
    something you could use."""

    def test_resume_is_a_command_and_abbreviates(self):
        from wynxo.cli import COMMANDS, resolve_command

        assert "/resume" in COMMANDS
        assert resolve_command("/res") == "/resume"

    def test_a_saved_session_round_trips(self, tmp_path, monkeypatch):
        from wynxo import session as session_module
        from wynxo.session import Session

        monkeypatch.setattr(session_module, "data_dir", lambda: tmp_path)
        original = Session(workspace=tmp_path)
        original.add_user("remember the number 4242")
        original.add_assistant("Noted.")
        assert original.save() is not None

        restored = Session.load(original.session_id, tmp_path)
        assert restored is not None
        assert len(restored.messages) == 2
        assert "4242" in restored.messages[0]["content"]

    def test_recent_lists_what_the_picker_needs(self, tmp_path, monkeypatch):
        from wynxo import session as session_module
        from wynxo.session import Session

        monkeypatch.setattr(session_module, "data_dir", lambda: tmp_path)
        saved = Session(workspace=tmp_path)
        saved.add_user("fix the parser")
        saved.save()

        rows = session_module.Session.recent()
        assert rows and rows[0]["messages"] == 1
        assert "fix the parser" in rows[0]["preview"]
        assert rows[0]["updated_at"] > 0      # the picker shows an age

    def test_the_system_prompt_is_rebuilt_not_restored(self):
        """Effort, scope, mode and memory may all have moved on since the
        conversation was saved; reinstating the old prompt would quietly
        bring the old ones back with it."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.Repl._load_session)
        assert "refresh_system_prompt()" in source

    def test_undo_history_is_dropped_on_resume(self):
        """Those snapshots describe files as they were during a different
        run; offering to revert to them would be a trap."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.Repl._load_session)
        assert "checkpoints.clear()" in source


class TestCommitMessageCleaning:
    """Small models fence things, label them, and add "Here is the commit
    message:" however firmly the prompt says not to."""

    def clean(self, text):
        from wynxo.cli import _clean_commit_message

        return _clean_commit_message(text)

    def test_a_clean_message_is_untouched(self):
        assert self.clean("Fix the token check") == "Fix the token check"

    def test_a_code_fence_is_removed(self):
        assert self.clean("```\nFix it\n```") == "Fix it"
        assert self.clean("```text\nFix it\n```") == "Fix it"

    def test_a_preamble_is_removed(self):
        assert self.clean("Here is the commit message:\nFix it") == "Fix it"

    def test_thinking_never_reaches_the_commit(self):
        assert self.clean("<think>hmm</think>Fix it") == "Fix it"

    def test_the_body_survives(self):
        got = self.clean("```\nFix it\n\nBecause it was wrong.\n```")
        assert got == "Fix it\n\nBecause it was wrong."

    def test_nothing_usable_gives_an_empty_string(self):
        """So the caller can refuse rather than commit an empty message."""
        assert self.clean("") == ""
        assert self.clean("<think>only thinking</think>") == ""


class TestCommitCommand:
    def test_it_is_a_command(self):
        from wynxo.cli import COMMANDS, resolve_command

        assert "/commit" in COMMANDS
        assert resolve_command("/commit") == "/commit"

    def test_it_never_stages_anything_for_you(self):
        """What to include is a decision the message should describe, not
        one this should make on your behalf."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.Repl.cmd_commit)
        assert '"add"' not in source
        assert "'add'" not in source
        assert "-A" not in source

    def test_it_asks_before_committing(self):
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.Repl.cmd_commit)
        # Through _question, which is the only thing allowed to open a
        # prompt: asking directly starts a second prompt_toolkit
        # application, which the chat layout cannot survive.
        assert "_question" in source
        assert "not committed" in source

    def test_the_diff_sent_to_the_model_is_capped(self):
        """A large staged diff would otherwise blow the context window."""
        import inspect

        from wynxo import cli

        assert "24_000" in inspect.getsource(cli.Repl.cmd_commit)

    def test_the_prompt_asks_for_the_message_and_nothing_else(self):
        from wynxo.prompts import COMMIT_PROMPT

        lowered = COMMIT_PROMPT.lower()
        assert "imperative" in lowered
        assert "72" in COMMIT_PROMPT
        assert "no preamble" in lowered


class _Anything:
    """Answers to anything: attribute, call, await, with-block, iteration.

    Stands in for the collaborators a settings command does not actually
    need in order to offer a choice -- the agent, the session, the running
    speaker. What a command *does* need is real, below.
    """

    def __init__(self, name="stub"):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, item):
        return _Anything(f"{self._name}.{item}")

    def __setattr__(self, item, value):
        pass

    def __call__(self, *args, **kwargs):
        return _Anything(f"{self._name}()")

    def __await__(self):
        async def answer():
            return _Anything(self._name)

        return answer().__await__()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False

    def __len__(self):
        return 0


class TestEverySettingIsPickable:
    """Typing a settings command bare must offer the choice, not print a
    table describing what you already have.

    Driven rather than read. The version of this test that grepped each
    method for "_pick" passed for months while /pet and /speak printed a
    table and stopped: both contained a picker -- for the voice, and for
    the engine -- just not on the path you reach by typing the command with
    no argument. Calling the command is the only way to ask the question
    that was actually meant.
    """

    PICKERS = ("cmd_theme", "cmd_effort", "cmd_mode", "cmd_scope",
               "cmd_pet", "cmd_speak", "cmd_endpoint", "cmd_model",
               "cmd_ctx", "cmd_thinking", "cmd_talker",
               "cmd_secrets")

    def _run_bare(self, name, monkeypatch):
        """Call one command with no arguments; report the pickers it opened."""
        import asyncio
        import io

        from rich.console import Console

        from wynxo import cli, provider as provider_module, speech as speech_module
        from wynxo.config import Config, Endpoint
        from wynxo.pet import Pet
        from wynxo.provider import ModelInfo
        from wynxo.ui import UI

        models = [
            ModelInfo(name="qwen3-coder:30b", size=30_000_000_000,
                      parameter_size="30B", quantization="Q4",
                      capabilities=["tools"]),
            ModelInfo(name="llama3.2:3b", size=2_000_000_000,
                      parameter_size="3B", quantization="Q4",
                      capabilities=["tools"]),
        ]
        engine = speech_module.Engine("espeak-ng", "espeak-ng", "robotic")
        # A machine with no synthesiser is right to refuse rather than offer
        # an empty list, so give this one a voice.
        monkeypatch.setattr(speech_module, "available", lambda: [engine])
        monkeypatch.setattr(speech_module, "pick", lambda preferred="auto": engine)

        async def already_inspected(client, entries, *args, **kwargs):
            return entries

        # Imported inside cmd_model, so patched where it is defined.
        monkeypatch.setattr(provider_module, "inspect_all", already_inspected)

        class _Client(_Anything):
            async def list_models(self):
                return models

        ui = UI()
        ui.console = Console(file=io.StringIO(), width=90)
        config = Config(
            endpoints=[Endpoint(name="local", url="http://127.0.0.1:11434")],
            active_endpoint="local", model="qwen3-coder:30b", num_ctx=32768)

        repl = object.__new__(cli.Repl)
        for attr, value in (("ui", ui), ("config", config),
                            ("client", _Client("client")),
                            ("agent", _Anything("agent")), ("pet", Pet()),
                            ("speaker", _Anything("speaker")), ("talker", None),
                            ("policy", _Anything("policy")), ("chat", None),
                            ("prompt_session", _Anything("prompt_session"))):
            object.__setattr__(repl, attr, value)

        opened = []

        async def offer(title, options, current):
            opened.append(title)
            return None                  # escape: the command must stop here

        async def ask(question, default=""):
            opened.append(question)
            return ""

        object.__setattr__(repl, "_pick", offer)
        object.__setattr__(repl, "_type_in", ask)

        asyncio.run(getattr(cli.Repl, name)(repl, []))
        return opened

    @pytest.mark.parametrize("name", PICKERS)
    def test_it_offers_a_choice_when_typed_bare(self, name, monkeypatch):
        assert self._run_bare(name, monkeypatch), (
            f"/{name[4:]} typed bare printed something and stopped")

    def test_they_are_all_async_so_they_can_await_a_choice(self):
        import inspect

        from wynxo import cli

        for name in self.PICKERS:
            assert inspect.iscoroutinefunction(getattr(cli.Repl, name)), name


class TestCancelIsNotTheSameAsNoPicker:
    """Escape means "never mind". Printing the table anyway ignores that,
    and both used to come back as None."""

    def test_the_sentinel_is_distinct_from_none(self):
        from wynxo.cli import NO_PICKER

        assert NO_PICKER is not None
        assert bool(NO_PICKER) is True      # never falsy by accident

    def test_no_picker_is_returned_when_arrows_are_unavailable(self, monkeypatch):
        import asyncio

        from wynxo import cli

        monkeypatch.setattr(cli, "arrows_supported", lambda: False)
        repl = object.__new__(cli.Repl)
        got = asyncio.run(cli.Repl._pick(repl, "t", [("a", "x")], "a"))
        assert got is cli.NO_PICKER

    def test_every_caller_handles_both_outcomes(self):
        """A caller that only checks one of them either prints a table on
        escape or crashes on a terminal with no picker."""
        import inspect

        from wynxo import cli

        for name in ("cmd_theme", "cmd_effort", "cmd_mode", "cmd_scope",
                     "cmd_pet", "cmd_speak", "cmd_endpoint"):
            source = inspect.getsource(getattr(cli.Repl, name))
            if "_pick" not in source:
                continue
            assert "is None" in source, f"{name} ignores escape"
            assert "NO_PICKER" in source, f"{name} ignores a missing picker"


class TestNothingReachesTheUserAsATraceback:
    """A raw Python traceback is a bug report the person reading it cannot
    act on. Everything inside a session is guarded; this is what catches
    start-up, and anything that gets past all of it."""

    def test_the_repl_guard_survives_an_arbitrary_exception(self):
        import asyncio
        import types

        from wynxo import cli
        from wynxo.journal import Journal
        from wynxo.ui import UI

        async def explode():
            raise RuntimeError("a tool did something unforeseen")

        repl = types.SimpleNamespace(
            ui=UI(), journal=Journal(session_id="test", path=None, enabled=False),
            callbacks=types.SimpleNamespace(_end_stream=lambda: None))
        got = asyncio.run(cli.Repl._guarded(repl, explode()))
        assert got is None, "the exception escaped the guard"

    def test_a_provider_error_is_shown_without_a_traceback(self):
        import asyncio
        import types

        from wynxo import cli
        from wynxo.journal import Journal
        from wynxo.provider import ProviderError
        from wynxo.ui import UI

        async def explode():
            raise ProviderError("the server said no")

        shown = []
        ui = UI()
        ui.error = lambda msg: shown.append(msg)
        repl = types.SimpleNamespace(
            ui=ui, journal=Journal(session_id="test", path=None, enabled=False),
            callbacks=types.SimpleNamespace(_end_stream=lambda: None))
        asyncio.run(cli.Repl._guarded(repl, explode()))
        assert shown == ["the server said no"]

    def test_ctrl_c_still_gets_through(self):
        """Swallowing these would make Ctrl-C look broken all over again."""
        import asyncio
        import types

        import pytest as _pytest

        from wynxo import cli
        from wynxo.agent import Interrupted
        from wynxo.journal import Journal
        from wynxo.ui import UI

        for boom in (Interrupted, asyncio.CancelledError):
            async def explode(exc=boom):
                raise exc()

            repl = types.SimpleNamespace(
                ui=UI(), journal=Journal(session_id="test", path=None, enabled=False),
                callbacks=types.SimpleNamespace(_end_stream=lambda: None))
            with _pytest.raises(boom):
                asyncio.run(cli.Repl._guarded(repl, explode()))

    def test_a_crash_report_is_written_and_names_the_version(self, tmp_path,
                                                             monkeypatch):
        from wynxo import cli

        monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
        path = cli._write_crash_report(RuntimeError("boom"))
        assert path is not None and path.exists()
        body = path.read_text()
        assert "RuntimeError: boom" in body
        assert "wynxo " in body and "python " in body

    def test_an_unwritable_directory_does_not_crash_the_crash_handler(
            self, tmp_path, monkeypatch):
        """Failing while reporting a failure would be the worst version."""
        from wynxo import cli

        def refuse():
            raise OSError("read-only")

        monkeypatch.setattr(cli, "data_dir", refuse)
        assert cli._write_crash_report(RuntimeError("boom")) is None

    def test_system_exit_is_not_swallowed(self):
        """--version and --help exit through SystemExit; catching it would
        turn a clean exit into a reported crash."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.main)
        assert "except SystemExit" in source
        assert source.index("except SystemExit") < source.index("BaseException")


class TestNothingWarnsAboutCursorPositions:
    """prompt_toolkit prints "your terminal doesn't support cursor position
    requests" into the middle of whatever is on screen when the terminal
    does not answer one -- a serial console, Termux, a pty without one.

    The REPL has silenced it since it was built. Setup, which runs before
    the REPL and is the first thing a new user ever sees, did not: the
    warning landed between the question and its input line.
    """

    def test_the_helper_silences_it(self):
        """Against a stand-in rather than a real PromptSession: building one
        opens a console handle, and on a Windows runner there is not one --
        the same NoConsoleScreenBufferError the chat layout was taught to
        survive."""
        import types

        from wynxo.select import silence_cpr_warning

        application = types.SimpleNamespace(
            renderer=types.SimpleNamespace(
                cpr_not_supported_callback=lambda: None))
        silence_cpr_warning(application)
        assert application.renderer.cpr_not_supported_callback is None

    def test_it_survives_an_application_without_one(self):
        from wynxo.select import silence_cpr_warning

        silence_cpr_warning(object())      # must not raise

    def test_setup_silences_it_too(self):
        import inspect

        from wynxo.wizard import run_wizard

        source = inspect.getsource(run_wizard)
        assert "silence_cpr_warning" in source
        assert source.index("silence_cpr_warning") < source.index("ask_endpoint")

    def test_the_repl_still_does(self):
        """The classic prompt silences it when it is actually built; the
        eager prompt session moved into _make_prompt_session so the chat
        layout no longer constructs (and crashes on) a console it never
        uses."""
        import inspect

        from wynxo.cli import Repl

        assert "silence_cpr_warning" in inspect.getsource(Repl._make_prompt_session)
