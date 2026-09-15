"""Product-level help should be searchable without becoming a command wall."""

from __future__ import annotations

import subprocess
import sys
import textwrap

from wynxo import cli, help_ui


def test_exact_command_lookup_is_one_result():
    found = help_ui.find_commands(cli, "theme")
    assert [command.name for command in found] == ["/theme"]


def test_intent_word_uses_the_same_vocabulary_as_completion():
    found = help_ui.find_commands(cli, "llm")
    assert found and found[0].name == "/model"


def test_typo_lookup_reuses_command_recovery():
    found = help_ui.find_commands(cli, "hlep")
    assert found and found[0].name == "/help"


def test_enumerated_command_help_can_show_its_options():
    command = cli.REGISTRY["/theme"]
    text = help_ui.option_text(command)
    assert "purple" in text
    assert "minimal" in text


def test_core_help_is_deliberately_smaller_than_the_registry():
    assert len(help_ui.CORE) < len(cli.COMMAND_LIST)
    assert {"/chat", "/code", "/github", "/help"}.issubset(
        set(help_ui.CORE) | {"/help"}
    )


def test_install_is_idempotent_and_replaces_only_help_handler():
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            from wynxo import cli, help_ui
            before_code = cli.Repl.cmd_code
            help_ui.install()
            first = cli.Repl.cmd_help
            help_ui.install()
            assert cli.Repl.cmd_help is first
            assert cli.Repl.cmd_help.__module__ == 'wynxo.help_ui'
            assert cli.Repl.cmd_code is before_code
        """)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
