"""What a glob pattern means.

The matcher was ``fnmatch`` against the relative path, or the file name, or
both -- and fnmatch has no notion of a path separator: its ``*`` matches
``/`` like any other character. Three consequences, all of which the agent
hit constantly:

    *.py            matched every Python file in the tree, not the ones
                    beside you
    src/*.py        matched src/deep/nested/thing.py
    **/*.py         matched *nothing* in the root directory, because ``**``
                    collapses to nothing and the literal ``/`` then has to
                    be there

Between them the answer was roughly "files whose name ends in .py,
somewhere", whatever was actually asked -- and grep's file filter had its
own copy of the same pair, so narrowing a search made it neither cheaper
nor more precise.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from wynxo.tools import build_registry
from wynxo.tools.search import _matcher

TREE = ("calc.py", "new.py", "notes.txt", "sub/deep.py", "sub/inner/far.py",
        "sub/inner/notes.md", "src/test_a.ts", "src/main.ts")


@pytest.fixture
def tree():
    root = Path(tempfile.mkdtemp())
    for relative in TREE:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("needle\n", encoding="utf-8")
    return root


def found(root: Path, pattern: str) -> list[str]:
    tool = build_registry(root).get("glob")
    result = asyncio.run(tool.invoke({"pattern": pattern}))
    if "No files match" in result.output:
        return []
    return sorted(result.output.split())


class TestSegments:
    """`*` and `?` stay inside one name; `**` crosses them."""

    def test_a_star_does_not_cross_a_separator(self, tree):
        assert found(tree, "sub/*.py") == ["sub/deep.py"]

    def test_a_double_star_does(self, tree):
        assert found(tree, "sub/**/*.py") == ["sub/deep.py", "sub/inner/far.py"]

    def test_a_double_star_matches_no_directories_at_all(self, tree):
        """The pattern people type. It found nothing in the root, which is
        where a project's files most often are."""
        assert found(tree, "**/*.py") == [
            "calc.py", "new.py", "sub/deep.py", "sub/inner/far.py"]

    def test_a_question_mark_is_one_character(self, tree):
        assert found(tree, "?ew.py") == ["new.py"]

    def test_a_middle_double_star_works(self, tree):
        assert found(tree, "src/**/test_*.ts") == ["src/test_a.ts"]

    def test_an_exact_path_matches_only_itself(self, tree):
        assert found(tree, "sub/inner/far.py") == ["sub/inner/far.py"]


class TestABarePatternSearchesTheTree:
    """No separator means the file name, anywhere -- what `fd '*.py'` and
    `git ls-files '*.py'` do, and what someone typing it means."""

    def test_a_bare_extension_finds_them_all(self, tree):
        assert found(tree, "*.py") == [
            "calc.py", "new.py", "sub/deep.py", "sub/inner/far.py"]

    def test_a_bare_name_finds_it_wherever_it_is(self, tree):
        assert found(tree, "far.py") == ["sub/inner/far.py"]

    def test_everything_means_everything(self, tree):
        assert found(tree, "**/*") == sorted(TREE)


class TestNoMatch:
    def test_a_pattern_that_matches_nothing_says_so(self, tree):
        assert found(tree, "*.rs") == []

    def test_a_directory_that_is_not_there(self, tree):
        assert found(tree, "nope/*.py") == []


class TestGrepNarrowsByTheSameRules:
    """It had its own copy of the broken pair."""

    def searched(self, root: Path, glob: str) -> list[str]:
        tool = build_registry(root).get("grep")
        result = asyncio.run(tool.invoke({"pattern": "needle", "glob": glob}))
        return sorted({line.split(":")[0]
                       for line in result.output.split("\n") if ":" in line})

    def test_a_directory_prefix_actually_narrows(self, tree):
        assert self.searched(tree, "sub/*.py") == ["sub/deep.py"]

    def test_a_bare_pattern_searches_the_tree(self, tree):
        assert self.searched(tree, "*.py") == [
            "calc.py", "new.py", "sub/deep.py", "sub/inner/far.py"]

    def test_no_filter_searches_everything(self, tree):
        assert self.searched(tree, "") == sorted(TREE)


class TestTheMatcherItself:
    @pytest.mark.parametrize("pattern,path,expected", [
        ("*.py", "a.py", True),
        ("*.py", "sub/a.py", True),            # bare: name anywhere
        ("sub/*.py", "sub/a.py", True),
        ("sub/*.py", "sub/deep/a.py", False),  # * must not cross /
        ("**/*.py", "a.py", True),             # ** may be no directories
        ("**/*.py", "x/y/a.py", True),
        ("**", "anything/at/all.txt", True),
        ("a?c.py", "abc.py", True),
        ("a?c.py", "abbc.py", False),
        ("[ab].py", "a.py", True),
        ("[ab].py", "c.py", False),
        ("*.py", "a.pyc", False),              # anchored at the end
        ("src/*", "src/a", True),
        ("src/*", "src/a/b", False),
    ])
    def test_cases(self, pattern, path, expected):
        assert _matcher(pattern)(path) is expected

    def test_an_unclosed_bracket_is_a_literal_not_a_crash(self):
        assert _matcher("a[b.py")("a[b.py") is True
