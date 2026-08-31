"""How the parts of a project relate to each other.

find_symbols answers where a name is defined. Everything left over --
what calls this, what imports this, what extends this, what tests cover
this -- had only grep, which answers with text matches and states no
relationship at all. These tests pin the relations, and pin hardest on
the places where a confident wrong answer would be worse than grep.
"""

import asyncio
import textwrap
from pathlib import Path

import pytest

from wynxo import navigation
from wynxo.tools.references_tool import FindReferences


@pytest.fixture(autouse=True)
def fresh_index():
    navigation.forget()
    yield
    navigation.forget()


def project(root: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    return root


def run(tool, **arguments):
    return asyncio.run(tool.invoke(arguments))


@pytest.fixture
def tree(tmp_path):
    return project(tmp_path, {
        "app/core.py": """
            class Base:
                def handle(self): pass

            def helper(x):
                return x
        """,
        "app/impl.py": """
            from .core import Base, helper

            class Impl(Base):
                def handle(self):
                    return helper(1)

            class Other(Base):
                pass

            def go():
                return helper(2)
        """,
        "tests/test_core.py": """
            from app.core import helper

            def test_helper():
                assert helper(1) == 1
        """,
        "tests/test_unrelated.py": """
            def test_nothing():
                assert True
        """,
    })


# -- what calls this ----------------------------------------------------------


def test_callers_are_found_with_the_function_they_sit_in(tree):
    hits, total, _ = navigation.references(tree, "helper", kinds=("call",))
    assert total == 3
    assert ("app/impl.py", 6, "Impl.handle") in \
           {(h.path, h.line, h.context) for h in hits}


def test_a_call_at_module_level_has_no_enclosing_context(tmp_path):
    tree = project(tmp_path, {"a.py": "def f(): pass\nf()\n"})
    hits, _, _ = navigation.references(tree, "f", kinds=("call",))
    assert [(h.line, h.context) for h in hits] == [(2, "")]


def test_the_definition_is_not_counted_as_a_use_of_itself(tmp_path):
    # Otherwise "nothing calls this" and "one thing calls this" look the same.
    tree = project(tmp_path, {"a.py": "def lonely():\n    pass\n"})
    hits, total, _ = navigation.references(tree, "lonely", kinds=("call",))
    assert (hits, total) == ([], 0)


def test_a_method_call_is_attributed_to_the_method_name(tree):
    # x.handle() is a call to handle, whatever x turns out to be.
    tree = project(tree, {"app/run.py": "from .impl import Impl\nImpl().handle()\n"})
    hits, _, _ = navigation.references(tree, "handler", kinds=("call",))
    assert hits == []
    hits, _, _ = navigation.references(tree, "handle", kinds=("call",))
    assert [h.path for h in hits] == ["app/run.py"]


def test_a_name_used_but_never_called_is_a_use_not_a_call(tmp_path):
    tree = project(tmp_path, {
        "a.py": "class Thing: pass\n",
        "b.py": "from a import Thing\nx = isinstance(y, Thing)\n",
    })
    calls, call_total, _ = navigation.references(tree, "Thing", kinds=("call",))
    uses, use_total, _ = navigation.references(tree, "Thing")
    assert call_total == 0
    assert use_total > 0
    assert any(h.kind == "use" for h in uses)


# -- what subclasses this -----------------------------------------------------


def test_subclasses_are_found(tree):
    assert sorted(h.context for h in navigation.subclasses(tree, "Base")) == \
           ["Impl", "Other"]


def test_a_base_written_as_a_dotted_name_still_counts(tmp_path):
    # grep for "(Base)" misses this; the relation does not.
    tree = project(tmp_path, {
        "a.py": "class Base: pass\n",
        "b.py": "import a\nclass Sub(a.Base):\n    pass\n",
    })
    assert [h.context for h in navigation.subclasses(tree, "Base")] == ["Sub"]


def test_a_class_with_no_subclasses_reports_none(tree):
    assert navigation.subclasses(tree, "Impl") == []


# -- what imports this --------------------------------------------------------


def test_importers_of_a_module_are_found_by_path_or_by_dotted_name(tree):
    by_path = navigation.importers(tree, "app/core.py")
    by_name = navigation.importers(tree, "app.core")
    assert by_path == by_name
    assert set(by_path) == {"app/impl.py", "tests/test_core.py"}


def test_importing_a_symbol_counts_as_importing_its_module(tree):
    # `from app.core import helper` is an importer of app.core.
    assert "tests/test_core.py" in navigation.importers(tree, "app.core")


def test_importing_a_sibling_module_by_name_is_seen(tmp_path):
    # ``from . import core`` names the module in the alias, not the module
    # field. Reading only the module field made every ``from . import x``
    # in a package invisible, which in a package that imports itself the
    # way this one does is most of the graph.
    tree = project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "VALUE = 1\n",
        "pkg/user.py": "from . import core\n",
    })
    assert navigation.importers(tree, "pkg.core") == ["pkg/user.py"]


def test_importing_a_sibling_module_by_absolute_name_is_seen(tmp_path):
    # ``from pkg import core`` has the same shape as the relative form:
    # the module being imported is the alias, not the module field.
    tree = project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "VALUE = 1\n",
        "user.py": "from pkg import core\n",
    })
    assert navigation.importers(tree, "pkg.core") == ["user.py"]


def test_a_relative_import_resolves_to_the_real_module(tree):
    # `from .core import Base` inside app/impl.py means app.core. Left
    # unresolved, every intra-package import in a project is invisible.
    assert "app/impl.py" in navigation.importers(tree, "app.core")


def test_a_module_nobody_imports_has_no_importers(tree):
    assert navigation.importers(tree, "tests/test_unrelated.py") == []


# -- which tests cover this ---------------------------------------------------


def test_covering_tests_follows_imports_not_file_names(tree):
    # tests/test_core.py imports app.core; tests/test_unrelated.py does not.
    assert navigation.covering_tests(tree, [tree / "app/core.py"]) == \
           ["tests/test_core.py"]


def test_a_test_reaching_a_module_indirectly_is_still_selected(tmp_path):
    tree = project(tmp_path, {
        "pkg/low.py": "VALUE = 1\n",
        "pkg/high.py": "from .low import VALUE\n",
        "tests/test_high.py": "from pkg.high import VALUE\n",
    })
    # No test is named after low.py and none imports it directly.
    assert navigation.covering_tests(tree, [tree / "pkg/low.py"]) == \
           ["tests/test_high.py"]


def test_a_selection_covering_most_of_a_large_suite_is_discarded(tmp_path):
    # "Focused" that means "all of them" is the full suite with a longer
    # command line. The caller already falls back to the full suite.
    files = {"pkg/base.py": "VALUE = 1\n"}
    for i in range(12):
        files[f"tests/test_{i}.py"] = "from pkg.base import VALUE\n"
    tree = project(tmp_path, files)
    assert navigation.covering_tests(tree, [tree / "pkg/base.py"]) == []


def test_a_small_suite_is_never_discarded_for_being_wide(tmp_path):
    # With three test files, running all three IS the focused run.
    files = {"pkg/base.py": "VALUE = 1\n"}
    for i in range(3):
        files[f"tests/test_{i}.py"] = "from pkg.base import VALUE\n"
    tree = project(tmp_path, files)
    assert len(navigation.covering_tests(tree, [tree / "pkg/base.py"])) == 3


def test_a_file_no_test_reaches_selects_nothing(tmp_path):
    tree = project(tmp_path, {
        "lonely.py": "def f(): pass\n",
        "tests/test_other.py": "def test_x(): assert True\n",
    })
    assert navigation.covering_tests(tree, [tree / "lonely.py"]) == []


def test_the_name_heuristic_still_contributes(tmp_path):
    # test_gh.py is a test for gh.py whether or not it imports it. Losing
    # that signal when the import graph gained one would be a trade, not
    # an improvement.
    tree = project(tmp_path, {
        "gh.py": "def f(): pass\n",
        "tests/test_gh.py": "def test_x(): assert True\n",
    })
    assert navigation.covering_tests(tree, [tree / "gh.py"]) == ["tests/test_gh.py"]


def test_focused_command_uses_the_import_graph(tmp_path):
    from wynxo import testing

    tree = project(tmp_path, {
        "pytest.ini": "[pytest]\n",
        "pkg/thing.py": "VALUE = 1\n",
        "tests/test_unrelated_name.py": "from pkg.thing import VALUE\n",
    })
    command = testing.focused_command(tree, [tree / "pkg/thing.py"])
    assert command is not None
    assert "tests/test_unrelated_name.py" in command


# -- counts stay true ---------------------------------------------------------


def test_the_total_counts_references_past_the_storage_cap(monkeypatch):
    # The stored sample is capped to bound memory. Reporting the size of
    # the sample as the number of references is a quiet lie, and the whole
    # point of the tool is to be more trustworthy than grep.
    monkeypatch.setattr(navigation, "MAX_REFERENCES_PER_NAME", 5)
    import tempfile

    root = Path(tempfile.mkdtemp())
    body = "def f(): pass\n" + "".join(f"f()  # {i}\n" for i in range(40))
    project(root, {"a.py": body})
    hits, total, sampled = navigation.references(root, "f", kinds=("call",))
    assert total == 40
    assert len(hits) <= 5
    assert sampled is True


def test_an_uncapped_name_is_not_reported_as_a_sample(tree):
    _, _, sampled = navigation.references(tree, "helper", kinds=("call",))
    assert sampled is False


def test_filtering_by_kind_does_not_understate_the_total(monkeypatch):
    # Filtering a capped mixed sample down to its calls and reporting that
    # count was the original bug: 'run' claimed 100 calls out of 404.
    monkeypatch.setattr(navigation, "MAX_REFERENCES_PER_NAME", 4)
    import tempfile

    root = Path(tempfile.mkdtemp())
    uses = "".join(f"x{i} = f\n" for i in range(20))
    calls = "".join(f"f()  # {i}\n" for i in range(20))
    project(root, {"a.py": f"def f(): pass\n{uses}{calls}"})
    _, total, _ = navigation.references(root, "f", kinds=("call",))
    assert total == 20


# -- the tool -----------------------------------------------------------------


def tool(root: Path) -> FindReferences:
    return FindReferences(root)


def test_the_tool_answers_each_relation(tree):
    assert "app/impl.py" in run(tool(tree), relation="callers", name="helper").output
    assert "Impl" in run(tool(tree), relation="subclasses", name="Base").output
    assert "app/impl.py" in run(tool(tree), relation="importers", name="app.core").output
    assert "test_core" in run(tool(tree), relation="tests", name="app/core.py").output


def test_asking_for_callers_of_a_class_falls_back_to_uses(tree):
    # A class is never "called". Answering "nothing calls Base" would be
    # true and useless.
    result = run(tool(tree), relation="callers", name="Base")
    assert result.ok
    assert "app/impl.py" in result.output
    assert "Nothing calls" in result.output


def test_a_missing_name_is_an_error_that_says_what_to_pass(tree):
    result = run(tool(tree))
    assert not result.ok
    assert "name" in result.error


def test_an_unknown_relation_lists_the_real_ones(tree):
    result = run(tool(tree), relation="sideways", name="Base")
    assert not result.ok
    assert "callers" in result.error and "importers" in result.error


def test_nothing_found_is_a_successful_answer(tree):
    result = run(tool(tree), relation="callers", name="absent_entirely")
    assert result.ok
    assert result.metadata.get("hits") == 0
    assert "No calls" in result.output


def test_no_tests_found_says_to_run_the_suite_rather_than_assume_coverage(tmp_path):
    tree = project(tmp_path, {"lonely.py": "def f(): pass\n",
                              "tests/test_x.py": "def test_x(): assert True\n"})
    result = run(tool(tree), relation="tests", name="lonely.py")
    assert result.ok
    assert result.metadata.get("hits") == 0
    assert "run the whole suite" in result.output


def test_the_answer_is_far_smaller_than_grepping_the_name(tree):
    from wynxo.tools.search import Grep

    grepped = run(Grep(tree), pattern="helper").output
    answered = run(tool(tree), relation="callers", name="helper").output
    assert len(answered) <= len(grepped)


def test_a_credentials_file_is_not_reported_as_a_caller(tmp_path):
    tree = project(tmp_path, {
        "a.py": "def f(): pass\n",
        "creds.py": "from a import f\nf()\n",
    })
    blocking = tool(tree)
    blocking.shield.blocks = lambda path: path.name == "creds.py"
    result = run(blocking, relation="callers", name="f")
    assert "creds.py" not in result.output


def test_a_credentials_file_is_not_reported_as_an_importer(tmp_path):
    tree = project(tmp_path, {"a.py": "X = 1\n", "creds.py": "from a import X\n"})
    blocking = tool(tree)
    blocking.shield.blocks = lambda path: path.name == "creds.py"
    result = run(blocking, relation="importers", name="a")
    assert "creds.py" not in result.output


def test_the_tool_is_registered_for_the_model(tmp_path):
    from wynxo.tools import build_registry

    assert "find_references" in build_registry(tmp_path)
