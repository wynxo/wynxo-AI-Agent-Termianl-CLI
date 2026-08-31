"""Finding where something is defined, across a whole project.

Grep answers "where is X defined?" with every *use* of X. For a common
name that is dozens of call sites around a single definition, and the
model has to read all of them to find the one. These tests pin the index
that answers the question directly, and -- more importantly -- pin the
cases where a confident wrong answer would be worse than grep.
"""

import asyncio
import textwrap
from pathlib import Path

import pytest

from wynxo import navigation
from wynxo.tools.navigation_tool import NavigateSymbols


@pytest.fixture(autouse=True)
def fresh_index():
    # The index is cached against the tree's newest timestamp. A test that
    # writes a file and searches it in the same second would otherwise see
    # the previous test's tree.
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


# -- extracting definitions ---------------------------------------------------


def parse(source: str):
    return navigation.definitions_in(textwrap.dedent(source), "python")


def test_functions_classes_and_methods_are_all_found():
    found = parse("""
        FLAG = 1

        def top():
            pass

        class Thing:
            def method(self):
                pass

            async def slow(self):
                pass
    """)
    assert [(d.name, d.kind, d.parent) for d in found] == [
        ("FLAG", "constant", ""),
        ("top", "function", ""),
        ("Thing", "class", ""),
        ("method", "method", "Thing"),
        ("slow", "async method", "Thing"),
    ]


def test_a_method_records_the_class_that_owns_it():
    # Two classes defining run() is the normal case, not the exception.
    found = parse("""
        class A:
            def run(self): pass
        class B:
            def run(self): pass
    """)
    owners = sorted(d.parent for d in found if d.name == "run")
    assert owners == ["A", "B"]


def test_the_signature_carries_arguments_and_the_return_type():
    found = parse("""
        def add(a: int, b: int = 2) -> int:
            return a + b
    """)
    assert found[0].signature == "def add(a: int, b: int=2) -> int"


def test_a_private_name_is_indexed():
    # The project map hides underscored names because it is a summary. An
    # index that hid them would simply fail to answer half the questions.
    found = parse("def _helper(): pass")
    assert [d.name for d in found] == ["_helper"]


def test_a_lowercase_module_variable_is_not_a_definition():
    # Otherwise every `x = 1` in the project becomes a search hit.
    found = parse("x = 1\nCACHE = {}\n")
    assert [d.name for d in found] == ["CACHE"]


def test_a_file_that_does_not_parse_contributes_nothing():
    # And, crucially, does not raise: one broken file must not fail a
    # search across the whole tree.
    assert navigation.definitions_in("def (:", "python") == []


def test_line_numbers_point_at_the_definition():
    found = parse("""

        def first(): pass


        def second(): pass
    """)
    assert [(d.name, d.line) for d in found] == [("first", 3), ("second", 6)]


def test_other_languages_are_indexed_with_line_numbers():
    found = navigation.definitions_in(
        "const a = 1;\nexport function handler(req) {}\n", "clike")
    assert [(d.name, d.line) for d in found] == [("handler", 2)]


# -- searching the index ------------------------------------------------------


@pytest.fixture
def tree(tmp_path):
    return project(tmp_path, {
        "app/models.py": """
            class User:
                def save(self): pass

            def helper(): pass
        """,
        "app/views.py": """
            from .models import User

            def view():
                user = User()
                user.save()
                return helper(User, User, User)

            class Session:
                def save(self): pass
        """,
        "lib/util.py": "def helper_two(): pass\n",
    })


def test_it_finds_the_definition_and_not_the_uses(tree):
    hits = navigation.find(tree, "User")
    assert [(h.path, h.line) for h in hits] == [("app/models.py", 2)]


def test_every_class_defining_a_name_is_reported(tree):
    hits = navigation.find(tree, "save")
    assert sorted(h.parent for h in hits) == ["Session", "User"]


def test_a_qualified_name_selects_one_of_them(tree):
    hits = navigation.find(tree, "Session.save")
    assert [(h.parent, h.path) for h in hits] == [("Session", "app/views.py")]


def test_an_exact_match_beats_a_longer_name_containing_it(tree):
    # "helper" must not answer with helper_two ahead of helper.
    hits = navigation.find(tree, "helper")
    assert hits[0].name == "helper"


def test_a_substring_match_is_dropped_once_something_matches_exactly(tree):
    # Offering guesses alongside the answer is how a short result becomes a
    # long one again.
    assert [h.name for h in navigation.find(tree, "helper")] == ["helper"]


def test_a_substring_match_is_offered_when_nothing_matches_exactly(tree):
    # A half-remembered name should still get an answer.
    assert [h.name for h in navigation.find(tree, "helper_t")] == ["helper_two"]


def test_a_class_outranks_a_method_of_the_same_name(tmp_path):
    # Ordered so that file position alone would give the wrong answer: the
    # method is first, and the class must still lead.
    tree = project(tmp_path, {"a.py": """
        class Holder:
            def Thing(self): pass
        class Thing:
            def other(self): pass
    """})
    assert navigation.find(tree, "Thing")[0].kind == "class"


def test_a_name_that_is_defined_nowhere_returns_nothing(tree):
    assert navigation.find(tree, "not_here_at_all") == []


def test_an_empty_query_returns_nothing_rather_than_everything(tree):
    assert navigation.find(tree, "") == []
    assert navigation.find(tree, "   ") == []


def test_the_index_skips_the_noise_directories(tmp_path):
    tree = project(tmp_path, {
        "real.py": "def target(): pass\n",
        "node_modules/pkg/index.js": "export function target() {}\n",
        ".venv/lib/thing.py": "def target(): pass\n",
        "__pycache__/x.py": "def target(): pass\n",
    })
    assert [h.path for h in navigation.find(tree, "target")] == ["real.py"]


def test_paths_are_reported_with_forward_slashes(tree):
    assert all("\\" not in hit.path for hit in navigation.find(tree, "save"))


# -- the cache ----------------------------------------------------------------


def test_a_new_definition_is_found_after_the_tree_changes(tmp_path):
    tree = project(tmp_path, {"a.py": "def one(): pass\n"})
    assert navigation.find(tree, "two") == []
    project(tree, {"b.py": "def two(): pass\n"})
    # A stale index is a wrong answer, not a slow one.
    assert [h.path for h in navigation.find(tree, "two", refresh=True)] == ["b.py"]


def test_the_second_search_does_not_re_read_the_tree(tmp_path, monkeypatch):
    tree = project(tmp_path, {"a.py": "def one(): pass\n"})
    navigation.find(tree, "one")

    from wynxo import projectmap
    calls = []
    real = projectmap.walk
    monkeypatch.setattr(projectmap, "walk",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    navigation.find(tree, "one")
    assert calls == []


def test_forgetting_one_root_leaves_another_alone(tmp_path):
    one = project(tmp_path / "one", {"a.py": "def alpha(): pass\n"})
    two = project(tmp_path / "two", {"b.py": "def beta(): pass\n"})
    navigation.index(one)
    navigation.index(two)
    navigation.forget(one)
    assert str(one) not in navigation._cache
    assert str(two) in navigation._cache


# -- the tool -----------------------------------------------------------------


def tool(root: Path) -> NavigateSymbols:
    return NavigateSymbols(root)


def test_the_tool_answers_where_something_is_defined(tree):
    result = run(tool(tree), name="User")
    assert result.ok
    assert result.output.startswith("app/models.py:2")
    assert "class User" in result.output


def test_the_answer_is_far_smaller_than_grepping_for_the_name(tree):
    from wynxo.tools.search import Grep

    grepped = run(Grep(tree), pattern="User").output
    found = run(tool(tree), name="User").output
    assert len(found) < len(grepped) / 3


def test_the_tool_still_outlines_a_file_from_a_path_alone(tree):
    # The old shape of this tool, which existing prompts and habits use.
    result = run(tool(tree), path="app/models.py")
    assert result.ok
    assert "User" in result.output and "save" in result.output


def test_a_path_and_a_name_together_search_only_there(tree):
    result = run(tool(tree), name="save", path="app/views.py")
    assert "app/views.py" in result.output
    assert "app/models.py" not in result.output


def test_neither_argument_is_an_error_that_says_what_to_pass(tree):
    result = run(tool(tree))
    assert not result.ok
    assert "name" in result.error and "path" in result.error


def test_not_finding_it_is_a_successful_answer_not_a_failure(tree):
    # A failure reads as "the tool broke" and invites a retry. "It is not
    # defined here" is information.
    result = run(tool(tree), name="totally_absent")
    assert result.ok
    assert "No definition" in result.output
    assert result.metadata.get("hits") == 0


def test_a_missing_file_is_still_an_error(tree):
    result = run(tool(tree), path="nope.py")
    assert not result.ok


def test_too_many_matches_are_capped_and_say_how_to_narrow(tmp_path):
    body = "\n".join(f"class C{i}:\n    def go(self): pass\n" for i in range(60))
    tree = project(tmp_path, {"a.py": body})
    result = run(tool(tree), name="go")
    assert result.output.count("\n") <= 30
    assert "narrow it" in result.output


def test_a_definition_in_a_credentials_file_is_not_reported(tmp_path):
    # An index is a read with extra steps; a signature carries defaults.
    tree = project(tmp_path, {
        "safe.py": "def other(): pass\n",
        "secrets.py": "def token(key='AKIAIOSFODNN7EXAMPLE'): pass\n",
    })
    result = run(tool(tree), name="token")
    blocking = tool(tree)
    blocking.shield.blocks = lambda path: path.name == "secrets.py"
    blocked = run(blocking, name="token")
    assert result.ok
    assert blocked.metadata.get("hits") == 0
    assert "secrets.py" not in blocked.output


def test_outlining_a_credentials_file_is_refused(tmp_path):
    tree = project(tmp_path, {"creds.py": "def token(): pass\n"})
    blocking = tool(tree)
    blocking.shield.blocks = lambda path: path.name == "creds.py"
    result = run(blocking, path="creds.py")
    assert not result.ok
    assert "credentials" in result.error


def test_a_secret_in_a_signature_is_masked(tmp_path):
    tree = project(tmp_path, {
        "conf.py": "def connect(api_key='sk-abcdefghijklmnopqrstuvwxyz0123456789'): pass\n"})
    result = run(tool(tree), name="connect")
    assert result.ok
    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789" not in result.output


# -- languages that are not Python --------------------------------------------
#
# ast is exact; everything else is a regex, which guesses. A guess that
# points at a line still beats forty call sites, but a guess that says
# "not defined anywhere" when it is, or points at the wrong line, is worse
# than the grep it replaced.


def clike(source: str):
    return navigation.definitions_in(textwrap.dedent(source), "clike")


def test_a_javascript_method_is_indexed():
    # The project map lists only `function` and `class`, which in a JS
    # project means every method is missing. An index that missed them
    # would answer "total is not defined in this project" -- confidently,
    # and wrongly.
    found = clike("""
        export class Cart {
          constructor() { this.items = []; }
          add(item) { this.items.push(item); }
          total() { return 0; }
        }
    """)
    assert {d.name for d in found} == {"Cart", "constructor", "add", "total"}


def test_a_declaration_after_a_blank_line_reports_the_right_line():
    # These patterns begin with ^\s*, and \s eats the preceding newline.
    # An off-by-one line number is a confident wrong answer.
    found = clike("""
        const x = 1;

        export function later(a) {
          return a;
        }
    """)
    assert [(d.name, d.line) for d in found] == [("later", 4)]


def test_control_flow_is_not_mistaken_for_a_definition():
    found = clike("""
        class S {
          go() {
            if (a) { b(); }
            for (const x of xs) { }
            while (t) { }
            switch (k) { case 1: break; }
            try { f(); } catch (err) { g(err); }
            return fetch(u).then((r) => r.json());
          }
        }
    """)
    assert {d.name for d in found} == {"S", "go"}


def test_a_typed_method_signature_is_indexed():
    found = clike("""
        public class Cart {
            public int total() {
                if (items == null) { return 0; }
                return 1;
            }
            private static String name(String a) { return a; }
        }
    """)
    assert {d.name for d in found} == {"Cart", "total", "name"}


def test_a_declaration_matched_by_two_patterns_is_reported_once():
    found = clike("export function once(a) {\n  return a;\n}\n")
    assert [d.name for d in found] == ["once"]


def test_a_javascript_project_can_be_searched_end_to_end(tmp_path):
    tree = project(tmp_path, {
        "package.json": '{"name": "demo"}',
        "src/cart.js": """
            export class Cart {
              total() { return 0; }
            }
        """,
    })
    assert [h.describe() for h in navigation.find(tree, "total")] == \
           ["src/cart.js:3  total()"]


# -- languages other than Python ----------------------------------------------
#
# Support is not equal and this pins where the line is. Python goes through
# ast and is exact. Everything else is a regex over the text, so a name in a
# comment or a string can produce a spurious entry. What must not happen is
# the opposite: a definition that exists and cannot be found, because
# "nowhere in this project" is a confident wrong answer.


LANGUAGE_SAMPLES = {
    "go": ("go", """
        package main

        type Cart struct {
        \tn int
        }

        type Priced interface {
        \tPrice() int
        }

        func (c Cart) Total() int {
        \treturn c.n
        }

        func helper() int {
        \treturn Cart{}.Total()
        }
    """),
    "rust": ("rust", """
        pub struct Cart {
            n: i32,
        }

        pub trait Priced {
            fn price(&self) -> i32;
        }

        impl Cart {
            pub fn total(&self) -> i32 {
                self.n
            }
        }

        pub fn helper() -> i32 {
            Cart { n: 0 }.total()
        }
    """),
    "java": ("clike", """
        public class Cart {
            public int total() { return 1; }
            public static int helper() { return new Cart().total(); }
        }
    """),
    "typescript": ("clike", """
        export class Cart {
          private total(): number { return 1; }
        }
        export function helper(): number { return new Cart().total(); }
    """),
    "ruby": ("ruby", """
        class Cart
          def total
            1
          end
        end

        def helper
          Cart.new.total
        end
    """),
}


@pytest.mark.parametrize("label", sorted(LANGUAGE_SAMPLES))
def test_each_language_resolves_a_type_a_method_and_a_function(label):
    language, source = LANGUAGE_SAMPLES[label]
    names = {d.name.lower()
             for d in navigation.definitions_in(textwrap.dedent(source), language)}
    assert "cart" in names, f"{label}: the type is missing"
    assert "total" in names, f"{label}: the method is missing"
    assert "helper" in names, f"{label}: the function is missing"


def test_a_go_type_is_indexed():
    # Nothing matched `type X struct` at all, so every struct, interface and
    # named type in a Go project was missing from the index.
    names = {d.name for d in
             navigation.definitions_in(LANGUAGE_SAMPLES["go"][1], "go")}
    assert {"Cart", "Priced"} <= names


def test_an_unexported_go_function_is_indexed():
    # The map matches only capitalised names, which is right for a summary
    # of a package's public surface and wrong for a lookup table.
    names = {d.name for d in
             navigation.definitions_in(LANGUAGE_SAMPLES["go"][1], "go")}
    assert "helper" in names


def test_the_project_map_still_summarises_go_by_public_surface():
    # The index gained unexported names; the map must not, or every Go
    # package's map fills up with internals.
    from wynxo import projectmap

    assert projectmap.symbols_in(LANGUAGE_SAMPLES["go"][1], "go") == ["Total"]
