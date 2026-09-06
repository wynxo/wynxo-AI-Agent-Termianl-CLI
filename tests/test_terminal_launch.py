from pathlib import Path
from types import SimpleNamespace

from wynxo.tools.apps import terminal_argv


def test_konsole_command_uses_a_separate_instance_and_shell():
    entry = SimpleNamespace(name="Konsole", path=Path("/usr/bin/konsole"))
    argv = terminal_argv(entry, "echo helo")

    assert argv is not None
    assert argv[0].endswith("konsole")
    assert argv[1:4] == ["--separate", "--hold", "-e"]
    assert argv[4:6] == ["bash", "-lc"]
    assert argv[6] == "echo helo"


def test_konsole_command_preserves_shell_syntax():
    entry = SimpleNamespace(name="Konsole", path=Path("/usr/bin/konsole"))
    command = "echo 'hello world' && printf '%s\\n' ok | cat"
    argv = terminal_argv(entry, command)

    assert argv is not None
    assert argv[-1] == command


def test_konsole_command_uses_the_wynxo_workspace():
    entry = SimpleNamespace(name="Konsole", path=Path("/usr/bin/konsole"))
    workspace = Path("/tmp").resolve()
    argv = terminal_argv(entry, "pwd", str(workspace))

    assert argv is not None
    assert argv[-1] == f"cd -- {workspace} && pwd"


def test_unknown_terminal_is_rejected():
    entry = SimpleNamespace(
        name="Definitely Not A Terminal",
        path=Path("/usr/bin/not-a-terminal"),
    )
    assert terminal_argv(entry, "echo helo") is None
