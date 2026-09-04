from pathlib import Path
from types import SimpleNamespace

from wynxo.tools.apps import terminal_argv


def test_konsole_command_uses_a_separate_instance_and_shell():
    entry = SimpleNamespace(name="Konsole", path=Path("/usr/bin/konsole"))
    argv = terminal_argv(entry, "echo helo")

    assert argv is not None
    assert argv[0].endswith("konsole")
    assert argv[1:3] == ["--separate", "-e"]
    assert argv[3:5] == ["bash", "-lc"]
    assert "echo helo" in argv[5]
    assert "exec bash" in argv[5]
