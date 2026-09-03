"""The table is the only place a command exists.

Two bugs came out of keeping the name, the description and the code in three
places. /status had a branch no input could reach for months: the resolver
did not know the name, so the code handling it was dead and nothing said so.
Every abbreviation in the alias table was dead for the same shape of reason
-- the guard in front of the dispatch chain skipped resolution in exactly
the case the table existed for.

Neither is representable now, and these are what keep it that way. They are
cheap and they walk the whole table, which is the point: the failure mode
was never one command being wrong, it was nobody checking all of them.
"""

from __future__ import annotations

import inspect

import pytest

from wynxo.cli import (ALIASES, COMMAND_LIST, COMMANDS, REGISTRY, Command,
                       Repl, resolve_command, suggest_commands)


def _advertised(does: str) -> set[str]:
    """The options a description offers, from its "subject: a | b | c" part.

    Every description that lists alternatives is written that way, which is
    what makes this readable by a test at all.
    """
    body = does.split(":", 1)[1] if ":" in does else does
    if "|" not in body:
        return set()
    words = set()
    for piece in body.split("|"):
        word = piece.strip().strip(",.").split(" ")[0].strip()
        if word and word.isalpha():
            words.add(word)
    return words


class TestTheTableIsWellFormed:
    def test_every_command_has_a_handler_on_the_repl(self):
        missing = [c.name for c in COMMAND_LIST
                   if not hasattr(Repl, c.handler)]
        assert missing == []

    def test_every_handler_takes_the_same_arguments(self):
        """The dispatcher calls them all the same way, so they all have to
        accept it. Two used to take none, which only worked because their
        branch in the chain knew that."""
        for command in COMMAND_LIST:
            handler = getattr(Repl, command.handler)
            taken = [p for p in inspect.signature(handler).parameters
                     if p != "self"]
            assert taken == ["args"], f"{command.name} -> {command.handler}"

    def test_no_command_is_listed_twice(self):
        names = [c.name for c in COMMAND_LIST]
        assert len(names) == len(set(names))

    def test_no_handler_serves_two_commands_by_accident(self):
        """/session and /sessions share a subject but not a method: one
        describes this conversation and the other lists the rest."""
        handlers = [c.handler for c in COMMAND_LIST]
        assert len(handlers) == len(set(handlers))

    def test_every_name_is_a_slash_command(self):
        for command in COMMAND_LIST:
            assert command.name.startswith("/"), command.name
            assert command.name == command.name.lower()
            assert " " not in command.name

    def test_every_description_is_written_the_same_way(self):
        """/help is a column of them, and one that shouts or trails a full
        stop is the one you notice instead of reading."""
        for command in COMMAND_LIST:
            assert command.does, command.name
            assert not command.does.endswith("."), command.name
            assert command.does[0].islower() or not command.does[0].isalpha(), \
                command.name

    def test_a_handler_is_named_after_its_command(self):
        """cmd_<name>, without exception, so a reader who has found one can
        find the other by guessing."""
        for command in COMMAND_LIST:
            assert command.handler == "cmd_" + command.name.lstrip("/"), \
                command.name


class TestTheDerivedTablesCannotDrift:
    def test_the_help_text_is_the_table(self):
        assert COMMANDS == {c.name: c.does for c in COMMAND_LIST}

    def test_the_lookup_is_the_table(self):
        assert set(REGISTRY) == {c.name for c in COMMAND_LIST}

    def test_completion_values_belong_to_real_commands(self):
        from wynxo.cli import _SUBCOMMAND_VALUES

        assert set(_SUBCOMMAND_VALUES) <= set(COMMANDS)

    def test_an_offered_value_is_never_empty(self):
        for command in COMMAND_LIST:
            for value in command.values:
                assert value and value == value.strip(), command.name

    def test_what_a_description_offers_is_what_completion_offers(self):
        """Argument-level drift, which is the same bug one level down.

        /queue advertised "show | run | clear" and rejected "show" -- the
        one word of its own three it did not accept. /animate advertised
        "list" and passed it through as a companion state, where it matched
        none of them, so the documented spelling was the one that failed.
        And /endpoint, /secrets, /speak and /talker each named options that
        Tab had never heard of.
        """
        for command in COMMAND_LIST:
            offered = _advertised(command.does)
            if not offered:
                continue
            assert offered == set(command.values), (
                f"{command.name}: says {sorted(offered)}, "
                f"completes {sorted(command.values)}")


class TestEveryAdvertisedCommandCanBeReached:
    def test_the_resolver_knows_every_command(self):
        """The /status bug: a branch existed, and no typed input could
        reach it because resolve_command had never heard of the name."""
        for command in COMMAND_LIST:
            assert resolve_command(command.name) == command.name, command.name

    def test_the_dispatcher_finds_a_handler_for_every_command(self):
        for command in COMMAND_LIST:
            assert REGISTRY.get(command.name) is not None, command.name

    def test_every_alias_lands_on_a_command_with_a_handler(self):
        for alias, target in ALIASES.items():
            assert target in REGISTRY, f"{alias} -> {target}"

    def test_every_suggestion_can_be_run(self):
        """A suggestion the dispatcher would then refuse is worse than
        none."""
        for probe in ("/m", "/s", "/co", "/th", "/d", "/re", "/mdoe"):
            for suggested in suggest_commands(probe):
                assert suggested in REGISTRY, f"{probe} -> {suggested}"


class TestDispatch:
    """The dispatcher's own contract, driven rather than read."""

    def _repl(self, handler_name: str, outcome):
        import io

        from wynxo.ui import UI

        repl = Repl.__new__(Repl)
        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = 100
        repl.ui = ui
        repl.seen = []

        def record(args):
            repl.seen.append(args)
            return outcome

        setattr(repl, handler_name, record)
        return repl

    @pytest.mark.asyncio
    async def test_a_command_reaches_its_handler_with_its_arguments(self):
        repl = self._repl("cmd_mode", True)
        assert await Repl.command(repl, "/mode plan") is True
        assert repl.seen == [["plan"]]

    @pytest.mark.asyncio
    async def test_an_alias_reaches_the_handler_it_names(self):
        repl = self._repl("cmd_stats", True)
        assert await Repl.command(repl, "/st") is True
        assert repl.seen == [[]]

    @pytest.mark.asyncio
    async def test_only_a_false_return_ends_the_session(self):
        """Handlers return True, None, or nothing at all. Treating anything
        falsy as "leave" would make a handler that forgot its return value
        quit the program."""
        for outcome in (True, None, "", 0, []):
            repl = self._repl("cmd_todo", outcome)
            assert await Repl.command(repl, "/todo") is True, repr(outcome)
        repl = self._repl("cmd_quit", False)
        assert await Repl.command(repl, "/quit") is False

    @pytest.mark.asyncio
    async def test_an_async_handler_is_awaited(self):
        import io

        from wynxo.ui import UI

        repl = Repl.__new__(Repl)
        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = 100
        repl.ui = ui
        ran = []

        async def handler(args):
            ran.append(args)
            return True

        repl.cmd_doctor = handler
        assert await Repl.command(repl, "/doctor") is True
        assert ran == [[]]

    @pytest.mark.asyncio
    async def test_the_name_is_matched_case_insensitively(self):
        repl = self._repl("cmd_help", True)
        assert await Repl.command(repl, "/HELP") is True
        assert repl.seen == [[]]


def test_the_table_holds_frozen_rows():
    """A command edited at runtime is a command that no longer matches the
    help beside it."""
    with pytest.raises(Exception):
        COMMAND_LIST[0].name = "/nope"
    assert isinstance(COMMAND_LIST[0], Command)
