"""Python-only hardening: traceback parsing, failure classification, async
test detection, environment discovery, focused test selection, and the agent
wiring that turns a failing suite into the model's next instruction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from wynxo import testing


PYTEST_FAILURE = """\
________________________ test_answer ________________________
    def test_answer():
        from bug import answer
>       assert answer() == 2
E       AssertionError: assert 1 == 2

bug.py:4: in answer
    return 1
E       AssertionError: assert 1 == 2
FAILED tests/test_bug.py::test_answer - AssertionError: assert 1 == 2
1 failed in 0.3s
"""


class TestTracebackParsing:
    def test_full_pytest_section_captures_raise_site(self):
        failures = testing.parse_failures(PYTEST_FAILURE)
        assert failures, "nothing parsed"
        by_file = {f.file: f for f in failures}
        assert "bug.py" in by_file
        assert by_file["bug.py"].kind == "AssertionError"
        assert by_file["bug.py"].line == 4
        assert by_file["bug.py"].message == "assert 1 == 2"

    def test_summary_lines_do_not_duplicate_sections(self):
        failures = testing.parse_failures(PYTEST_FAILURE)
        # The summary node id must not be captured again on top of the
        # section traceback that already has file/line information.
        assert not any(f.test == "tests/test_bug.py::test_answer"
                       for f in failures)
        assert sum(1 for f in failures if f.kind == "AssertionError") == 2

    def test_summary_only_output_still_parses(self):
        output = ("FAILED tests/test_x.py::test_y - "
                  "AssertionError: 1 != 2\n1 failed in 0.1s\n")
        failures = testing.parse_failures(output)
        assert failures[0].kind == "AssertionError"
        assert failures[0].test == "tests/test_x.py::test_y"

    def test_windows_paths_parse(self):
        output = ("C:\\Users\\me\\proj\\tests\\test_w.py:9: in test_win\n"
                  "E   TypeError: unsupported operand type(s)\n")
        failures = testing.parse_failures(output)
        assert failures[0].file == "C:\\Users\\me\\proj\\tests\\test_w.py"
        assert failures[0].line == 9

    def test_plain_python_traceback_parses(self):
        tb = ('Traceback (most recent call last):\n'
              '  File "bug.py", line 7, in answer\n'
              '    return 1 / x\n'
              'ZeroDivisionError: division by zero\n')
        failures = testing.parse_failures(tb)
        assert failures[0].kind == "ZeroDivisionError"
        assert failures[0].file == "bug.py"
        assert failures[0].line == 7

    def test_pytest9_location_line_format(self):
        """pytest 8+/9 prints the raise site as ``file:line: Type`` (no
        `` in func``) and assertion details as type-less E lines."""
        output = ("_____________________________ test_add "
                  "______________________________\n"
                  "    def test_add():\n"
                  "        from calc import add\n"
                  ">       assert add(2, 3) == 5\n"
                  "E       assert -1 == 5\n"
                  "E        +  where -1 = add(2, 3)\n"
                  "\n"
                  "test_calc.py:4: AssertionError\n"
                  "FAILED test_calc.py::test_add - assert -1 == 5\n")
        failures = testing.parse_failures(output)
        assert len(failures) == 1, failures
        assert failures[0].kind == "AssertionError"
        assert failures[0].file == "test_calc.py"
        assert failures[0].line == 4
        assert failures[0].message == "assert -1 == 5"

    def test_collection_error_mentions_the_module(self):
        output = ("________________________ ERROR collecting test_a.py "
                  "________________________\n"
                  "ImportError while importing test module\n"
                  "tests/test_a.py:1: in <module>\n"
                  "    import pytest_asyncio\n"
                  "E   ModuleNotFoundError: No module named 'pytest_asyncio'\n")
        failures = testing.parse_failures(output)
        assert failures[0].kind == "ModuleNotFoundError"
        assert "pytest_asyncio" in failures[0].message


class TestFailureClassification:
    def test_missing_dev_tool_is_an_environment_problem(self, tmp_path):
        failure = testing.Failure("ModuleNotFoundError",
                                  "No module named 'pytest_asyncio'")
        category, reason = testing.classify_failure(failure, tmp_path)
        assert category == "environment"
        assert "pytest-asyncio" in reason
        assert "pip install pytest-asyncio" in reason

    def test_specific_tool_beats_a_substring(self, tmp_path):
        failure = testing.Failure("ModuleNotFoundError",
                                  "No module named 'pytest_asyncio'")
        _, reason = testing.classify_failure(failure, tmp_path)
        assert "pytest-asyncio" in reason and "install pytest\"" not in reason

    def test_missing_project_module_is_a_structure_problem(self, tmp_path):
        failure = testing.Failure("ImportError",
                                  "cannot import name 'answer' from 'bug'")
        category, reason = testing.classify_failure(failure, tmp_path)
        assert category == "structure"
        assert "package layout" in reason

    def test_failure_inside_tests_is_a_test_problem(self, tmp_path):
        failure = testing.Failure("AssertionError", "assert 1 == 2",
                                  file="tests/test_bug.py", line=5)
        category, _ = testing.classify_failure(failure, tmp_path)
        assert category == "test"

    def test_syntax_error_is_code(self, tmp_path):
        failure = testing.Failure("SyntaxError", "invalid syntax",
                                  file="bug.py", line=3)
        category, _ = testing.classify_failure(failure, tmp_path)
        assert category == "code"

    def test_assertion_in_project_code_is_code(self, tmp_path):
        failure = testing.Failure("AssertionError", "assert 1 == 2",
                                  file="bug.py", line=4)
        category, _ = testing.classify_failure(failure, tmp_path)
        assert category == "code"

    def test_failure_report_carries_classification(self, tmp_path):
        report = testing.failure_report(PYTEST_FAILURE, tmp_path)
        assert "structured failure analysis" in report
        assert "AssertionError" in report
        assert "bug.py:4" in report
        assert "test_answer" in report


class TestAsyncTestDetection:
    def test_async_tests_are_found_by_shape(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_async.py").write_text(
            "import pytest\n\n@pytest.mark.asyncio\n"
            "async def test_fetch():\n    assert True\n")
        assert testing.async_tests_present(tmp_path) is True

    def test_no_async_tests(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_plain.py").write_text(
            "def test_plain():\n    assert True\n")
        assert testing.async_tests_present(tmp_path) is False

    def test_asyncio_mode_config_is_recognised(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\nasyncio_mode = \"auto\"\n")
        assert testing.pytest_asyncio_configured(tmp_path) is True
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = \"x\"\n")
        assert testing.pytest_asyncio_configured(tmp_path) is False


class TestPythonCommands:
    def test_pip_goes_through_the_project_interpreter(self, tmp_path):
        venv = tmp_path / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        (venv / "python.exe").write_text("")
        command = testing.pip_command(tmp_path)
        assert command.endswith("-m pip")
        assert "python.exe" in command

    def test_focused_command_selects_affected_tests(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_apps.py").write_text("def test_a(): pass\n")
        (tmp_path / "tests" / "test_other.py").write_text("def test_b(): pass\n")
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        command = testing.focused_command(tmp_path, [Path("wynxo/tools/apps.py")])
        assert command is not None
        assert "test_apps.py" in command
        assert "test_other.py" not in command
        assert command.startswith(f"{testing.python_command(tmp_path)} -m pytest")

    def test_focused_command_none_without_mapping(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_other.py").write_text("def test_b(): pass\n")
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        assert testing.focused_command(tmp_path, [Path("docs/readme.md")]) is None

    def test_focused_command_none_for_non_pytest(self, tmp_path):
        (tmp_path / "package.json").write_text(
            "{\"scripts\": {\"test\": \"vitest run\"}}\n")
        assert testing.focused_command(tmp_path, [Path("src/x.ts")]) is None

    def test_focused_command_quotes_paths_with_spaces(self, tmp_path):
        root = tmp_path / "my project"
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "test_apps.py").write_text("def test_a(): pass\n")
        (root / "pytest.ini").write_text("[pytest]\n")
        command = testing.focused_command(root, [Path("wynxo/tools/apps.py")])
        assert command is not None
        assert "test_apps.py" in command


class TestEnvironmentInfo:
    def test_virtualenv_detected_from_interpreter(self, tmp_path, monkeypatch):
        venv = tmp_path / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        (venv / "python.exe").write_text("")
        monkeypatch.setattr(testing, "_run_interpreter", lambda interp, code: "3.12.4")
        monkeypatch.setattr(testing, "_module_importable", lambda root, module: True)
        env = testing.environment_info(tmp_path)
        assert env.environment == "virtualenv"
        assert env.version == "3.12.4"
        assert env.pytest_installed is True

    def test_windows_store_alias_is_called_out(self, tmp_path, monkeypatch):
        stub = tmp_path / "WindowsApps"
        stub.mkdir()
        (stub / "python.exe").write_bytes(b"MZ\x90\x00")  # non-empty alias
        monkeypatch.setattr(sys, "executable", str(stub / "python.exe"))
        monkeypatch.setattr("wynxo.testing.os.name", "nt")
        monkeypatch.setattr(testing, "_run_interpreter", lambda interp, code: "")
        monkeypatch.setattr(testing, "_module_importable", lambda root, module: None)
        env = testing.environment_info(tmp_path)
        assert "windows-store-alias" in env.environment

    def test_package_manager_from_lockfiles(self, tmp_path, monkeypatch):
        (tmp_path / "uv.lock").write_text("")
        monkeypatch.setattr(testing, "_run_interpreter", lambda interp, code: "")
        monkeypatch.setattr(testing, "_module_importable", lambda root, module: None)
        assert testing.environment_info(tmp_path).package_manager == "uv"

    def test_config_files_are_listed(self, tmp_path, monkeypatch):
        for name in ("pyproject.toml", "pytest.ini", ".python-version"):
            (tmp_path / name).write_text("")
        monkeypatch.setattr(testing, "_run_interpreter", lambda interp, code: "")
        monkeypatch.setattr(testing, "_module_importable", lambda root, module: None)
        env = testing.environment_info(tmp_path)
        assert "pyproject.toml" in env.config_files
        assert "pytest.ini" in env.config_files
