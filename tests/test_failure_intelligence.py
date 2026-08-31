"""What the agent learns from a failing test run.

One bug routinely fails twenty tests. Reported as twenty findings it costs
twenty times the context to say one thing, and buries any second, actually
different cause underneath. These tests pin the grouping, and pin that
run_tests -- the tool the model calls itself -- gets the same treatment the
automatic verify pass always got.
"""

import asyncio
from pathlib import Path

import pytest

from wynxo import testing
from wynxo.tools.dev import RunTests

ONE_CAUSE = """\
=================================== FAILURES ===================================
__________________________________ test_empty __________________________________

>   def test_empty(): assert average([]) == 0
                             ^^^^^^^^^^^

tests/test_calc.py:6:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

numbers = []

    def average(numbers):
>       return sum(numbers) / len(numbers)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       ZeroDivisionError: division by zero

src/calc.py:2: ZeroDivisionError
__________________________________ test_other __________________________________

>   def test_other(): assert average([]) == 1
                             ^^^^^^^^^^^

tests/test_calc.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

numbers = []

    def average(numbers):
>       return sum(numbers) / len(numbers)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       ZeroDivisionError: division by zero

src/calc.py:2: ZeroDivisionError
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_empty - ZeroDivisionError: division by zero
FAILED tests/test_calc.py::test_other - ZeroDivisionError: division by zero
========================= 2 failed, 1 passed in 0.03s ==========================
"""
"""Real pytest output, in its default traceback style.

Written out rather than generated because the shape is the thing under
test: pytest prints the message on an ``E`` line with no frame above it,
and the file and line on a separate location line with no message. Getting
one finding with both halves out of that is the whole job.
"""

TWO_CAUSES = ONE_CAUSE.replace(
    "=========================== short test summary info",
    """\
__________________________________ test_name ___________________________________

    def label():
>       return NAME.upper()
E       AttributeError: 'NoneType' object has no attribute 'upper'

src/label.py:9: AttributeError
=========================== short test summary info""")


def causes(output: str):
    return testing.group_failures(testing.parse_failures(output))


# -- grouping -----------------------------------------------------------------


def test_two_tests_broken_by_one_bug_are_one_cause():
    grouped = causes(ONE_CAUSE)
    assert len(grouped) == 1
    failure, tests = grouped[0]
    assert (failure.file, failure.line, failure.kind) == \
           ("src/calc.py", 2, "ZeroDivisionError")
    assert sorted(tests) == ["test_empty", "test_other"]


def test_two_different_bugs_stay_two_causes():
    grouped = causes(TWO_CAUSES)
    assert {failure.kind for failure, _ in grouped} == \
           {"ZeroDivisionError", "AttributeError"}


def test_the_same_exception_at_two_places_is_two_causes():
    # Same type, different line: two bugs, not one.
    output = ONE_CAUSE.replace("src/calc.py:2: ZeroDivisionError",
                               "src/calc.py:9: ZeroDivisionError", 1)
    assert len({(f.file, f.line) for f, _ in causes(output)}) == 2


def test_a_cause_keeps_its_message_even_when_the_located_entry_lacks_one():
    # pytest states the type and the file in the traceback and the type and
    # the message in the summary. "(no message)" next to an output that
    # plainly contains the message is the tool looking broken.
    for failure, _ in causes(ONE_CAUSE):
        assert failure.message
        assert "no message" not in failure.message


def test_a_message_is_recovered_when_two_causes_share_an_exception_type(tmp_path):
    # With one located cause, the message-less traceback entry and the
    # message-carrying summary entry fold together. With two, there is no
    # single entry to fold into, and the message has to be recovered from
    # the run as a whole -- which is the shape of every real multi-cause
    # failure, and where "(no message)" showed up.
    output = ONE_CAUSE.replace(
        "=========================== short test summary info",
        """__________________________________ test_third __________________________________

    def median(numbers):
>       return numbers[len(numbers) // 2] / count
E       ZeroDivisionError: division by zero

src/calc.py:9: ZeroDivisionError
=========================== short test summary info""", 1)
    grouped = testing.group_failures(testing.parse_failures(output))
    assert {(f.file, f.line) for f, _ in grouped} == \
           {("src/calc.py", 2), ("src/calc.py", 9)}
    assert all(failure.message for failure, _ in grouped)
    assert "no message" not in testing.failure_report(output, tmp_path)


def test_a_failure_with_no_location_is_still_reported_once(tmp_path):
    output = ("Traceback (most recent call last):\n"
              "ImportError: cannot import name 'x'\n")
    report = testing.failure_report(output, tmp_path)
    assert report == "" or report.count("ImportError") == 1


# -- the report ---------------------------------------------------------------


def test_the_report_counts_causes_and_failing_tests(tmp_path):
    report = testing.failure_report(ONE_CAUSE, tmp_path)
    assert "1 root cause" in report
    assert "2 failing tests" in report


def test_the_report_names_the_tests_a_cause_breaks(tmp_path):
    report = testing.failure_report(ONE_CAUSE, tmp_path)
    assert "breaks 2 tests" in report
    assert "test_empty" in report


def test_the_report_states_the_source_location(tmp_path):
    assert "src/calc.py:2" in testing.failure_report(ONE_CAUSE, tmp_path)


def test_many_failures_of_one_cause_do_not_grow_the_report(tmp_path):
    # The whole point: 27 copies of one bug must not cost 27 findings.
    body = ONE_CAUSE
    for i in range(30):
        body = body.replace(
            "=========================== short test summary info",
            f"____ test_{i} ____\n"
            f"tests/t.py:{i}: in test_{i}\n    assert average([]) == 0\n"
            f"src/calc.py:2: in average\n    return sum(n) / len(n)\n"
            f"E   ZeroDivisionError: division by zero\n"
            f"src/calc.py:2: ZeroDivisionError\n"
            "=========================== short test summary info", 1)
    report = testing.failure_report(body, tmp_path)
    assert "1 root cause" in report
    assert len(report) < len(ONE_CAUSE) * 2


def test_a_long_list_of_causes_is_capped(tmp_path):
    body = ""
    for i in range(20):
        body += (f"____ test_{i} ____\nsrc/m{i}.py:{i}: in f\n    x()\n"
                 f"E   ValueError: bad {i}\nsrc/m{i}.py:{i}: ValueError\n")
    report = testing.failure_report(body, tmp_path)
    assert "more distinct causes" in report


def test_a_passing_run_produces_no_report(tmp_path):
    assert testing.failure_report("2 passed in 0.01s", tmp_path) == ""


# -- the tool the model calls -------------------------------------------------


def project(root: Path) -> Path:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "src" / "calc.py").write_text(
        "def average(numbers):\n    return sum(numbers) / len(numbers)\n")
    body = ["import sys, pathlib",
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))",
            "from src.calc import average"]
    body += [f"def test_case_{i}(): assert average([]) == {i}" for i in range(25)]
    (root / "tests" / "test_many.py").write_text("\n".join(body) + "\n")
    return root


@pytest.mark.parametrize("_", [0])
def test_run_tests_summarises_instead_of_dumping_the_whole_run(tmp_path, _):
    root = project(tmp_path)
    result = asyncio.run(RunTests(root).invoke({}))
    assert not result.ok
    # The raw pytest output for 25 identical failures is many times this.
    assert len(result.output) < 6000
    assert "structured failure analysis" in result.output


def test_run_tests_reports_one_cause_for_twenty_five_failures(tmp_path):
    root = project(tmp_path)
    result = asyncio.run(RunTests(root).invoke({}))
    assert "1 root cause" in result.output
    assert "src/calc.py:2" in result.output


def test_a_passing_run_is_not_rewritten(tmp_path):
    root = tmp_path
    (root / "tests").mkdir()
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "tests" / "test_ok.py").write_text("def test_ok(): assert True\n")
    result = asyncio.run(RunTests(root).invoke({}))
    assert result.ok
    assert "structured failure analysis" not in result.output


def test_a_long_passing_run_is_not_truncated(tmp_path):
    # Summarising exists to stop a failure flooding the context. A run that
    # passed has nothing to diagnose, and its output may carry warnings or
    # coverage the model asked for -- cutting it to the last sixty lines
    # would lose that for no benefit.
    root = tmp_path
    (root / "tests").mkdir()
    # -v so a passing run genuinely produces more lines than the summary
    # cap; without it pytest prints 120 tests as one row of dots.
    (root / "pytest.ini").write_text("[pytest]\naddopts = -v\n")
    body = "\n".join(f"def test_ok_{i}(): assert True" for i in range(120))
    (root / "tests" / "test_many_ok.py").write_text(body + "\n")
    result = asyncio.run(RunTests(root).invoke({}))
    assert result.ok
    assert "earlier lines omitted" not in result.output
    assert "120 passed" in result.output


def test_the_exit_code_and_command_survive_the_rewrite(tmp_path):
    root = project(tmp_path)
    result = asyncio.run(RunTests(root).invoke({}))
    assert result.metadata["exit_code"] == 1
    assert "pytest" in result.metadata["command"]
    assert result.metadata["duration"] >= 0
