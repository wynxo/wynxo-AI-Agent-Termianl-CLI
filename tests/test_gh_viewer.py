"""What `/gh cat` and `/gh ls` actually put on screen.

They printed their lines as plain strings, and rich reads square brackets in
a plain string as markup. So the viewer edited the file on the way to the
screen: `x: list[int] = []` was drawn as `x: list = []`, a markdown `[ref]`
disappeared, and a line holding `[/anything]` -- a prompt template, a closing
tag in a document -- raised MarkupError and took the command out.

A viewer that silently changes what it shows you is worse than no viewer, so
these drive the real command through a real rich console and read the bytes.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from test_gh import StubClient, _FakePrompt

from wynxo.cli import Repl
from wynxo.gh import Blob, Tree
from wynxo.ui import UI

TYPED = "def f(xs: list[int]) -> dict[str, int]:\n    return {}\n"
CLOSING = "The model sees [/INST] at the end of the prompt.\n"
REFERENCE = "See [the docs] and [the guide] below.\n"


class Viewing(StubClient):
    """A repository whose files hold ordinary square brackets."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def read(self, owner, repo, path, branch, start=0, end=0):
        return Blob(text=self.text, sha="s", size=len(self.text), path=path,
                    total_lines=self.text.count("\n"))


class Named(StubClient):
    """A repository with brackets in its file names.

    No closing tag among them: a name cannot hold a slash, so `[/x]` is not
    a file name anyone can commit. What a name *can* hold is something rich
    reads as a style, and those were simply eaten.
    """

    def tree(self, owner, repo, branch):
        return Tree(truncated=False, entries=[
            {"path": "notes[draft].md", "type": "blob", "size": 5},
            {"path": "[bold].txt", "type": "blob", "size": 9},
            {"path": "prompts", "type": "tree", "size": 0},
            {"path": "prompts/tags[v2].json", "type": "blob", "size": 3},
        ])


def _drive(client, argv):
    """Run one /gh command against a real console and return what it wrote."""
    repl = Repl.__new__(Repl)
    repl.gh = client
    repl.gh_ws = None
    repl.ui = UI()
    repl.ui.console.file = io.StringIO()
    repl.ui.console.width = 200
    repl.prompt_session = _FakePrompt("y")
    asyncio.run(repl.cmd_gh(["open", "o/r"]))
    asyncio.run(repl.cmd_gh(argv))
    return repl.ui.console.file.getvalue()


class TestCatShowsTheFileItWasGiven:
    def test_a_type_annotation_survives(self):
        """The one that costs a reader real time: `list[int]` was silently
        drawn as `list`, so the file on screen was not the file."""
        out = _drive(Viewing(TYPED), ["cat", "a.py"])
        assert "list[int]" in out
        assert "dict[str, int]" in out

    def test_a_closing_tag_does_not_take_the_command_out(self):
        out = _drive(Viewing(CLOSING), ["cat", "prompt.txt"])
        assert "[/INST]" in out

    def test_reference_style_brackets_survive(self):
        out = _drive(Viewing(REFERENCE), ["cat", "README.md"])
        assert "[the docs]" in out and "[the guide]" in out

    def test_ordinary_content_is_unaffected(self):
        out = _drive(Viewing("hello\nworld\n"), ["cat", "README.md"])
        assert "hello" in out and "world" in out


class TestLsShowsTheNamesItWasGiven:
    def test_a_bracket_in_a_file_name_survives(self):
        out = _drive(Named(), ["ls"])
        assert "notes[draft].md" in out

    def test_a_name_that_looks_like_a_style_is_not_eaten(self):
        """`[bold].txt` rendered as `.txt` -- a file listing that does not
        list the file."""
        out = _drive(Named(), ["ls"])
        assert "[bold].txt" in out

    def test_a_name_under_a_prefix_survives_too(self):
        out = _drive(Named(), ["ls", "prompts"])
        assert "tags[v2].json" in out


class TestTheHelpersOnTheUntrustedList:
    """Everything that draws somebody else's words is on one list in the
    escape-sequence regression test. Markup is the other way a plain string
    is read as instructions rather than as text."""

    PAYLOAD = "x [/nope] [bold]y[/] list[int] [#zz] z"

    def _drawn(self, draw):
        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = 200
        draw(ui)
        return ui.console.file.getvalue()

    @pytest.mark.parametrize("label,draw", [
        ("info", lambda u: u.info(TestTheHelpersOnTheUntrustedList.PAYLOAD)),
        ("warn", lambda u: u.warn(TestTheHelpersOnTheUntrustedList.PAYLOAD)),
        ("error", lambda u: u.error(TestTheHelpersOnTheUntrustedList.PAYLOAD)),
        ("success", lambda u: u.success(TestTheHelpersOnTheUntrustedList.PAYLOAD)),
        ("detail_line",
         lambda u: u.detail_line(TestTheHelpersOnTheUntrustedList.PAYLOAD, "")),
        ("tool_result",
         lambda u: u.tool_result("read_file", True,
                                 TestTheHelpersOnTheUntrustedList.PAYLOAD,
                                 TestTheHelpersOnTheUntrustedList.PAYLOAD)),
        ("table", lambda u: u.table(["c"],
                                    [[TestTheHelpersOnTheUntrustedList.PAYLOAD]])),
        ("rule", lambda u: u.rule(TestTheHelpersOnTheUntrustedList.PAYLOAD)),
    ])
    def test_markup_is_neither_obeyed_nor_fatal(self, label, draw):
        drawn = self._drawn(draw)
        assert "list[int]" in drawn, f"{label} ate a bracket"
        assert "[bold]" in drawn, f"{label} read a style tag as an instruction"
