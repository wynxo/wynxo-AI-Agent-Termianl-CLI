"""Running the project's own tests as part of verification.

The verify pass asks the model to review its own work, which is asking the
author whether the author was right -- and a 7B says yes. A failing test is
the one thing in that loop that does not come from the model.

Detection is deliberately narrow. Guessing wrong means running the wrong
command in someone's project, and a wrong command that happens to pass is
worse than no test run at all: it reports confidence nobody earned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wynxo.testing import detect, summarise


def project(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return tmp_path


class TestFindingTheRunner:
    def test_pytest_from_a_config_file(self, tmp_path):
        root = project(tmp_path, {"pytest.ini": "[pytest]\n"})
        assert detect(root).command.endswith(" -m pytest")

    def test_pytest_from_pyproject(self, tmp_path):
        root = project(tmp_path, {
            "pyproject.toml": "[tool.pytest.ini_options]\naddopts = '-q'\n"})
        assert detect(root).name == "pytest"

    def test_pytest_from_the_layout_alone(self, tmp_path):
        root = project(tmp_path, {"tests/test_thing.py": "def test_x(): pass\n"})
        assert detect(root).name == "pytest"

    def test_npm_from_a_test_script(self, tmp_path):
        root = project(tmp_path, {
            "package.json": '{"scripts": {"test": "jest"}}'})
        assert detect(root).command == "npm test"

    @pytest.mark.parametrize("lockfile,agent", [
        ("pnpm-lock.yaml", "pnpm"), ("yarn.lock", "yarn"),
        ("bun.lockb", "bun"), ("package-lock.json", "npm"),
    ])
    def test_it_uses_the_package_manager_the_project_uses(self, tmp_path,
                                                          lockfile, agent):
        root = project(tmp_path, {
            "package.json": '{"scripts": {"test": "jest"}}', lockfile: ""})
        assert detect(root).command == f"{agent} test"

    def test_cargo_go_and_mix(self, tmp_path):
        for name, expected in (("Cargo.toml", "cargo test"),
                               ("go.mod", "go test ./..."),
                               ("mix.exs", "mix test")):
            root = project(tmp_path / name, {name: ""})
            assert detect(root).command == expected

    def test_a_makefile_needs_an_actual_test_target(self, tmp_path):
        with_target = project(tmp_path / "a", {"Makefile": "test:\n\techo hi\n"})
        assert detect(with_target).command == "make test"
        without = project(tmp_path / "b", {"Makefile": "build:\n\techo hi\n"})
        assert detect(without) is None

    def test_it_says_why_so_the_user_can_check(self, tmp_path):
        root = project(tmp_path, {"Cargo.toml": ""})
        assert "Cargo.toml" in detect(root).why


class TestRefusingToGuess:
    def test_an_empty_directory_has_no_runner(self, tmp_path):
        assert detect(tmp_path) is None

    def test_a_project_with_no_tests_has_no_runner(self, tmp_path):
        root = project(tmp_path, {"app.py": "print('hi')\n",
                                  "README.md": "# thing\n"})
        assert detect(root) is None

    def test_the_npm_init_placeholder_is_not_a_test_command(self, tmp_path):
        """`npm init` writes a test script that exits 1. Running it would
        report a failure the user did not cause."""
        root = project(tmp_path, {"package.json": json_placeholder()})
        assert detect(root) is None

    def test_an_empty_test_script_is_not_a_test_command(self, tmp_path):
        root = project(tmp_path, {"package.json": '{"scripts":{"test":"  "}}'})
        assert detect(root) is None

    def test_a_corrupt_package_json_does_not_raise(self, tmp_path):
        root = project(tmp_path, {"package.json": "{not json"})
        assert detect(root) is None

    def test_a_package_json_that_is_not_an_object_does_not_raise(self, tmp_path):
        root = project(tmp_path, {"package.json": "[1,2,3]"})
        assert detect(root) is None

    def test_a_pyproject_without_pytest_is_not_pytest(self, tmp_path):
        root = project(tmp_path, {
            "pyproject.toml": "[project]\nname = 'thing'\n"})
        assert detect(root) is None


def json_placeholder() -> str:
    return ('{"scripts": {"test": "echo \\"Error: no test specified\\" '
            '&& exit 1"}}')


class TestSummarising:
    def test_short_output_is_kept_whole(self):
        assert summarise("one\ntwo") == "one\ntwo"

    def test_it_keeps_the_end_not_the_beginning(self):
        """A suite says what failed at the end; the start is collection
        noise, and local models have no context to spare for it."""
        body = "\n".join(str(i) for i in range(500))
        out = summarise(body, limit=10)
        assert "499" in out and "0\n1\n2" not in out
        assert "omitted" in out

    def test_blank_lines_are_dropped(self):
        assert summarise("a\n\n\n\nb") == "a\nb"

    def test_empty_output_is_survivable(self):
        assert summarise("") == "" and summarise(None) == ""


class TestWhenItRuns:
    """Running tests after a turn that changed nothing would be a slow way
    to learn that nothing changed."""

    def _agent(self, tmp_path, **overrides):
        from unittest.mock import MagicMock

        from wynxo.agent import Agent
        from wynxo.config import Config
        from wynxo.effort import resolve

        config = Config()
        for key, value in overrides.items():
            setattr(config, key, value)
        return Agent(client=MagicMock(), config=config,
                     policy=resolve("medium"), workspace=tmp_path)

    def _ran(self, agent, changed=None) -> bool:
        """Whether _verify_with_tests actually invoked the shell."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        if changed is None:
            changed = [type("S", (), {"path": agent.workspace / "app.py"})()]
        agent.checkpoints.changes_since = lambda _mark: changed
        shell = MagicMock()
        shell.invoke = AsyncMock(return_value=type("R", (), {
            "ok": True, "output": "", "metadata": {}})())
        agent.tools.get = lambda name: shell if name == "shell" else None
        asyncio.run(agent._verify_with_tests())
        return shell.invoke.called

    def test_it_does_not_run_when_the_setting_is_off(self, tmp_path):
        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        assert self._ran(self._agent(tmp_path, verify_with_tests=False)) is False

    def test_it_does_not_run_in_plan_mode(self, tmp_path):
        """Read-only means read-only, tests included."""
        from wynxo.scope import Mode

        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        agent = self._agent(tmp_path)
        agent.permissions.mode = Mode.PLAN
        assert self._ran(agent) is False

    def test_it_does_not_run_when_nothing_changed(self, tmp_path):
        import asyncio

        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        agent = self._agent(tmp_path)
        agent.checkpoints.changes_since = lambda _mark: []
        reached = []
        agent.tools.get = lambda name: reached.append(name)
        asyncio.run(agent._verify_with_tests())
        assert reached == []

    def test_it_runs_when_files_changed_and_a_runner_exists(self, tmp_path):
        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        assert self._ran(self._agent(tmp_path)) is True

    def test_it_does_not_run_without_a_detectable_runner(self, tmp_path):
        """No runner and no Python change: nothing to validate."""
        project(tmp_path, {"README.md": "# nothing to run\n"})
        changed = [type("S", (), {"path": tmp_path / "README.md"})()]
        assert self._ran(self._agent(tmp_path), changed) is False

    def test_without_a_runner_a_python_change_gets_a_syntax_gate(self, tmp_path):
        """No runner, but a changed .py file: the cheap compileall gate runs
        so a broken edit cannot sail through verification."""
        project(tmp_path, {"README.md": "# nothing to run\n"})
        assert self._ran(self._agent(tmp_path)) is True


class TestWhichInterpreterTheTestsRunUnder:
    """"python -m pytest" goes wrong two ways, and both report a failure the
    user did not cause -- which is worse than not running the tests at all,
    because the model then sets about fixing code that was fine."""

    def test_a_projects_virtualenv_wins(self, tmp_path):
        """That is where its pytest and its dependencies live. Run by
        whatever python is on PATH, the suite fails on imports installed
        three directories away."""
        from wynxo.testing import python_command

        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("#!/bin/sh\n")
        assert python_command(tmp_path) == str(venv / "python")

    def test_the_windows_layout_counts_too(self, tmp_path):
        from wynxo.testing import python_command

        scripts = tmp_path / ".venv" / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").write_text("")
        assert python_command(tmp_path) == str(scripts / "python.exe")

    def test_a_path_with_a_space_is_quoted(self, tmp_path):
        from wynxo.testing import python_command

        root = tmp_path / "my project"
        venv = root / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("")
        command = python_command(root)
        assert "my project" in command
        assert command != str(venv / "python")      # quoted somehow

    def test_no_virtualenv_falls_back_to_the_platform(self, tmp_path,
                                                     monkeypatch):
        """On Debian and Ubuntu `python` is not a command at all unless
        somebody installed python-is-python3."""
        import sys

        from wynxo.testing import python_command

        # Whatever interpreter Wynxo itself runs under is not a usable
        # candidate for this project, so the platform name must be chosen.
        monkeypatch.setattr(sys, "executable",
                            str(tmp_path / "no-such-python"))
        assert python_command(tmp_path) in ("python3", "python")

    def test_it_never_names_a_command_that_is_not_there(self, tmp_path,
                                                        monkeypatch):
        import shutil
        import sys

        from wynxo import testing

        monkeypatch.setattr(shutil, "which",
                            lambda name: "/usr/bin/python3"
                            if name == "python3" else None)
        # sys.executable must not leak into the decision on any platform.
        monkeypatch.setattr(sys, "executable",
                            str(tmp_path / "no-such-python"))
        assert testing.python_command(tmp_path) == "python3"

    def test_the_active_environment_wins_over_path_python(self, tmp_path,
                                                          monkeypatch):
        """The interpreter running Wynxo is the environment the user chose;
        a differently-named python on PATH must not override it."""
        import shutil
        import sys

        from wynxo.testing import python_command

        real = tmp_path / "real-env" / "python.exe"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"MZ\x90\x00" * 100)   # non-empty, like a real exe

        monkeypatch.setattr(sys, "executable", str(real))
        monkeypatch.setattr("wynxo.testing.os.name", "nt")
        monkeypatch.setattr(shutil, "which", lambda name: "python")

        command = python_command(tmp_path)
        assert str(real) in command
        assert "python.exe" in command

    def test_a_store_stub_is_not_the_active_environment(self, tmp_path,
                                                        monkeypatch):
        """A zero-byte WindowsApps alias is not an interpreter, so PATH
        wins rather than inheriting the alias into the test command."""
        import shutil
        import sys

        from wynxo.testing import python_command

        stub = tmp_path / "WindowsApps" / "python.exe"
        stub.parent.mkdir(parents=True)
        stub.write_bytes(b"")   # Store aliases are zero-byte reparse points

        monkeypatch.setattr(sys, "executable", str(stub))
        monkeypatch.setattr("wynxo.testing.os.name", "nt")
        monkeypatch.setattr(shutil, "which",
                            lambda name: "python" if name == "python" else None)

        assert python_command(tmp_path) == "python"

    def test_the_runner_uses_it(self, tmp_path):
        from wynxo.testing import detect

        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("")
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        assert detect(tmp_path).command == f"{venv / 'python'} -m pytest"
