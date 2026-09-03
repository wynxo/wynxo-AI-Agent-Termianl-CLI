"""A tool call that was cut off is not a tool call.

Local models are cut off constantly -- num_predict, a context that ran out,
a dropped socket -- and the parser's last resort was to close whatever
braces and quotes were still open and see if the result parsed. It usually
did, and what it parsed to was a guess: the last value is however much of it
arrived.

For a tool call that guess is acted on. A write_file cut off inside its
content parsed as a complete call whose content was the first half of the
file. Cut a little earlier, it parsed as a call whose content was the empty
string -- a request to empty the file. Neither could be told apart from a
real call downstream, and neither is what the model meant.
"""

from __future__ import annotations

import pytest

from wynxo.parsing import looks_truncated, parse_turn, repair_json

CALL = ('<tool_call>{"name":"write_file","arguments":{"path":"a.py",'
        '"content":"def f():\\n    return 1\\n"}}</tool_call>')


def call_at(cut: int):
    return parse_turn(CALL[:cut])


class TestATruncatedCallIsMalformed:
    def test_the_whole_thing_still_works(self):
        turn = call_at(len(CALL))
        assert not turn.malformed
        assert [c.name for c in turn.tool_calls] == ["write_file"]
        assert turn.tool_calls[0].arguments["content"].endswith("return 1\n")

    @pytest.mark.parametrize("cut", [55, 60, 70, 80, 90, 100])
    def test_cut_anywhere_inside_the_content(self, cut):
        """Every one of these used to produce a call that would be run."""
        turn = call_at(cut)
        assert turn.malformed
        assert turn.tool_calls == []

    def test_the_case_that_would_empty_a_file(self):
        """The worst of them: content parsed as "" -- write nothing over
        whatever is there."""
        for cut in range(50, 78):
            turn = call_at(cut)
            for written in turn.tool_calls:
                assert written.arguments.get("content") != "", cut


class TestRecoveryStopsAtAGuess:
    """A brace left open after a complete value costs nothing to close:
    every value the model wrote, it finished writing. A string left open is
    a different thing."""

    def test_a_missing_brace_is_still_recovered(self):
        assert repair_json('{"path": "x.py", "content": "print(1)"') == {
            "path": "x.py", "content": "print(1)"}

    def test_nested_missing_braces_are_recovered(self):
        assert repair_json('{"a": {"b": 1}') == {"a": {"b": 1}}

    def test_an_unterminated_string_is_not(self):
        assert repair_json('{"path": "x.p') is None

    def test_unless_the_caller_asks_for_a_fragment(self):
        """Kept for a caller that wants to look rather than act."""
        assert repair_json('{"path": "x.p', allow_truncated=True) == {"path": "x.p"}

    @pytest.mark.parametrize("raw", [
        '{"a": 1,}', "{'a': 1}", '{"a": True}', '{"c": "l1\nl2"}',
        '{"msg": "it\'s fine"}',
    ])
    def test_ordinary_repairs_are_untouched(self, raw):
        assert repair_json(raw) is not None


class TestTellingThemApart:
    """Asked to check its quotes, a model that was cut off re-emits the same
    over-long content and is cut off in the same place."""

    @pytest.mark.parametrize("raw,expected", [
        ('{"a": 1}', False),
        ('{"a": 1,}', False),
        ('this is not json', False),
        ('', False),
        ('{"a": "unterminat', True),
        ('{"a": {"b": 1}', True),
        ('{"name": "write_file", "arguments": {"content": "half a fi', True),
    ])
    def test_looks_truncated(self, raw, expected):
        assert looks_truncated(raw) is expected

    def test_the_repair_prompt_says_which_it_is(self):
        import inspect

        from wynxo.agent import Agent

        source = inspect.getsource(Agent._repair_tool_calls)
        assert "looks_truncated" in source
        assert "cut off" in source
