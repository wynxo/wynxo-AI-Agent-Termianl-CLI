"""Half a command is a question, and it deserves an answer.

/mo is /model, /mode and /mommy. Resolving it silently to one of them, or
answering "unknown command", both leave you re-reading /help for a name you
nearly remembered.
"""

from __future__ import annotations

from wynxo.cli import ALIASES, COMMANDS, command_hints, suggest_commands


class TestPrefixesComeFirst:
    def test_an_ambiguous_prefix_offers_every_command_it_could_be(self):
        assert set(suggest_commands("/mo")) >= {"/mode", "/model", "/mommy"}

    def test_an_explicit_alias_is_offered_ahead_of_the_rest(self):
        """/mo is documented as /model. It stays the first answer, with the
        others named beside it rather than hidden behind it."""
        assert suggest_commands("/mo")[0] == "/model"

    def test_an_alias_that_is_not_a_prefix_of_its_target_is_still_found(self):
        assert "/quit" in suggest_commands("/q")

    def test_every_suggestion_is_a_real_command(self):
        for probe in ("/", "/s", "/se", "/th", "/co", "/m", "/g", "/re"):
            assert all(c in COMMANDS for c in suggest_commands(probe)), probe


class TestSpellingIsTheFallback:
    def test_a_transposition_still_finds_the_command(self):
        assert "/mode" in suggest_commands("/mdoe")

    def test_a_missing_letter_still_finds_the_command(self):
        assert "/session" in suggest_commands("/sesion")

    def test_nonsense_suggests_nothing_rather_than_anything(self):
        assert suggest_commands("/xyzzy") == []

    def test_spelling_is_not_consulted_when_a_prefix_matched(self):
        """Otherwise /co would offer /copy, /commit, /compact *and* /cd,
        which is a longer list that is less use."""
        assert all(c.startswith("/co") for c in suggest_commands("/co"))


class TestWhatTheComposerShows:
    def test_prose_never_puts_a_menu_under_the_prompt(self):
        for typed in ("hello", "fix the model loader", "what is /mode",
                      "", "  "):
            assert command_hints(typed) == [], typed

    def test_a_command_with_arguments_has_moved_on(self):
        assert command_hints("/mode plan") == []
        assert command_hints("/model ") == []

    def test_a_lone_slash_is_not_yet_a_question(self):
        assert command_hints("/") == []

    def test_a_complete_unambiguous_command_says_nothing(self):
        """/doctor is already /doctor. Repeating it back is noise."""
        assert command_hints("/doctor") == []

    def test_a_complete_command_that_is_also_a_prefix_still_helps(self):
        """/session is a command *and* the start of /sessions."""
        assert command_hints("/session") == ["/session", "/sessions"]

    def test_the_half_typed_case_is_the_whole_point(self):
        assert command_hints("/mo") == suggest_commands("/mo")
        assert len(command_hints("/mo")) >= 3


class TestTheAliasTableStaysHonest:
    def test_every_alias_expands_to_a_real_command(self):
        assert all(target in COMMANDS for target in ALIASES.values())

    def test_no_alias_shadows_a_real_command(self):
        """An alias that is also a command name would make the command
        unreachable, which is how /status used to be."""
        assert not set(ALIASES) & set(COMMANDS)
