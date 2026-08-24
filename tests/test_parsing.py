"""Parsing is the module that decides whether a local model is usable at all."""

import random

from wynxo.parsing import (
    LiveContentFilter,
    parse_turn,
    repair_json,
    split_thinking,
    strip_soft_switches,
)


class TestThinking:
    def test_extracts_and_removes_block(self):
        content, thinking = split_thinking("<think>hmm</think>The answer.")
        assert content == "The answer."
        assert thinking == "hmm"

    def test_handles_unterminated_block(self):
        # Happens whenever the model hits its token limit mid-thought.
        content, thinking = split_thinking("<think>reasoning cut off")
        assert content == ""
        assert "reasoning cut off" in thinking

    def test_multiple_blocks_are_joined(self):
        content, thinking = split_thinking("<think>a</think>one<think>b</think>two")
        assert content == "onetwo"
        assert "a" in thinking and "b" in thinking

    def test_alternate_tag_names(self):
        _, thinking = split_thinking("<reasoning>x</reasoning>hi")
        assert thinking == "x"

    def test_plain_text_untouched(self):
        content, thinking = split_thinking("just an answer")
        assert content == "just an answer"
        assert thinking == ""


class TestRepairJson:
    def test_valid_json(self):
        assert repair_json('{"a": 1}') == {"a": 1}

    def test_trailing_comma(self):
        assert repair_json('{"a": 1,}') == {"a": 1}

    def test_python_literals(self):
        assert repair_json('{"a": True, "b": None}') == {"a": True, "b": None}

    def test_code_fence(self):
        assert repair_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_raw_newline_inside_string(self):
        # The classic failure when a model writes file contents.
        assert repair_json('{"c": "l1\nl2"}') == {"c": "l1\nl2"}

    def test_truncated_object_is_closed(self):
        out = repair_json('{"path": "x.py", "content": "print(1)"')
        assert out == {"path": "x.py", "content": "print(1)"}

    def test_apostrophe_in_double_quoted_string_survives(self):
        # The single-quote fixup must not fire here and mangle the string.
        assert repair_json('{"msg": "it\'s fine"}') == {"msg": "it's fine"}

    def test_hopeless_returns_none(self):
        assert repair_json("this is not json at all") is None

    def test_non_object_returns_none(self):
        assert repair_json("[1, 2, 3]") is None


class TestToolCalls:
    def test_hermes_block(self):
        turn = parse_turn('<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>')
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"
        assert turn.tool_calls[0].arguments == {"path": "a.py"}

    def test_call_is_stripped_from_visible_content(self):
        turn = parse_turn('Reading it.<tool_call>{"name":"read_file","arguments":{}}</tool_call>')
        assert turn.content == "Reading it."

    def test_multiple_calls_in_one_turn(self):
        turn = parse_turn(
            '<tool_call>{"name":"a","arguments":{}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{}}</tool_call>'
        )
        assert [c.name for c in turn.tool_calls] == ["a", "b"]

    def test_unclosed_tag_still_parses(self):
        turn = parse_turn('<tool_call>{"name":"shell","arguments":{"command":"ls"}}')
        assert turn.tool_calls[0].name == "shell"

    def test_native_calls_are_used(self):
        turn = parse_turn("", native_tool_calls=[
            {"function": {"name": "grep", "arguments": {"pattern": "x"}}}
        ])
        assert turn.tool_calls[0].name == "grep"

    def test_native_and_text_duplicate_is_collapsed(self):
        # Some templates emit both. Running the tool twice would be wrong.
        turn = parse_turn(
            '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>',
            native_tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
        )
        assert len(turn.tool_calls) == 1

    def test_same_tool_different_args_is_not_collapsed(self):
        turn = parse_turn(
            '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
            '<tool_call>{"name":"read_file","arguments":{"path":"b.py"}}</tool_call>'
        )
        assert len(turn.tool_calls) == 2

    def test_arguments_as_json_string(self):
        turn = parse_turn(
            '<tool_call>{"function":{"name":"read_file","arguments":"{\\"path\\":\\"b.py\\"}"}}</tool_call>'
        )
        assert turn.tool_calls[0].arguments == {"path": "b.py"}

    def test_alternate_argument_keys(self):
        for key in ("parameters", "args", "input"):
            turn = parse_turn('<tool_call>{"name":"t","%s":{"x":1}}</tool_call>' % key)
            assert turn.tool_calls[0].arguments == {"x": 1}, key

    def test_unsalvageable_call_is_reported_not_dropped(self):
        turn = parse_turn("<tool_call>total nonsense here</tool_call>")
        assert turn.tool_calls == []
        assert turn.malformed  # so the agent can ask for a repair

    def test_no_tool_call_is_a_plain_answer(self):
        turn = parse_turn("The function lives in main.py:42.")
        assert turn.tool_calls == []
        assert turn.malformed == []
        assert turn.content.startswith("The function")

    def test_thinking_and_call_together(self):
        turn = parse_turn(
            '<think>need the file</think><tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'
        )
        assert turn.thinking == "need the file"
        assert turn.tool_calls[0].name == "read_file"


def test_strip_soft_switches():
    assert strip_soft_switches("do the thing /no_think") == "do the thing"
    assert strip_soft_switches("think hard /think") == "think hard"


class TestLiveContentFilter:
    """A model with no native thinking/tools support writes <think> and
    <tool_call> straight into plain content -- streaming that unfiltered
    means raw protocol markup shows up in the middle of the answer instead
    of prose. This is what stops it, one chunk at a time, live."""

    def feed_all(self, filt: LiveContentFilter, chunks: list[str]) -> str:
        out = "".join(filt.feed(c) for c in chunks)
        return out + filt.finish()

    def test_plain_text_passes_through_unchanged(self):
        filt = LiveContentFilter()
        assert self.feed_all(filt, ["The answer ", "is 42."]) == "The answer is 42."

    def test_a_whole_tool_call_in_one_chunk_is_hidden(self):
        filt = LiveContentFilter()
        out = self.feed_all(filt, [
            'Before.<tool_call>{"name":"x"}</tool_call>After.'])
        assert out == "Before.After."

    def test_a_whole_think_block_in_one_chunk_is_hidden(self):
        filt = LiveContentFilter()
        out = self.feed_all(filt, ["<think>reasoning here</think>The answer."])
        assert out == "The answer."

    def test_tag_split_across_chunk_boundaries_is_still_hidden(self):
        """The exact failure this exists for: a chunk boundary landing
        inside the opening tag must not leak the fragment before the tag
        is recognised."""
        filt = LiveContentFilter()
        out = self.feed_all(filt, [
            "Before.<tool_c", "all>", '{"name":"x"}', "</tool_", "call>", "After."])
        assert out == "Before.After."

    def test_closing_tag_split_across_chunks_is_hidden(self):
        filt = LiveContentFilter()
        out = self.feed_all(filt, [
            "<think>a very long thought that keeps go", "ing</thi", "nk>Done."])
        assert out == "Done."

    def test_unterminated_tool_call_at_stream_end_is_dropped_not_leaked(self):
        """A model cut off by the token limit mid-call -- the same case
        parse_turn()'s UNCLOSED_TOOL_CALL regex exists for."""
        filt = LiveContentFilter()
        out = self.feed_all(filt, ['Before.<tool_call>{"name":"x", "arg'])
        assert out == "Before."

    def test_unterminated_think_at_stream_end_is_dropped_not_leaked(self):
        filt = LiveContentFilter()
        out = self.feed_all(filt, ["<think>cut off mid-thought"])
        assert out == ""

    def test_uppercase_and_mixed_case_tags_are_still_recognised(self):
        filt = LiveContentFilter()
        out = self.feed_all(filt, ["<THINK>loud</THINK>", "<Tool_Call>{}</Tool_Call>done"])
        assert out == "done"

    def test_multiple_calls_in_one_answer_are_all_hidden(self):
        filt = LiveContentFilter()
        out = self.feed_all(filt, [
            'First <tool_call>{"name":"a"}</tool_call> then '
            '<tool_call>{"name":"b"}</tool_call> done.'])
        assert out == "First  then  done."

    def test_a_lone_angle_bracket_that_never_becomes_a_tag_is_not_lost(self):
        """Ordinary text like `if x < 5:` must not be swallowed just
        because it starts with the character a tag would."""
        filt = LiveContentFilter()
        out = self.feed_all(filt, ["if x < 5: return threshold"])
        assert out == "if x < 5: return threshold"

    def test_a_near_miss_prefix_is_recovered_once_it_diverges(self):
        filt = LiveContentFilter()
        out = self.feed_all(filt, ["<th", "reshold>ok"])
        assert out == "<threshold>ok"

    def test_final_content_matches_the_unfiltered_join_minus_hidden_tags(self):
        """Property check: chopping the same text into different chunk
        boundaries must never change what comes out the other side."""
        text = ('Reading the file first.\n<tool_call>\n{"name": "read_file", '
                '"arguments": {"path": "a.py"}}\n</tool_call>\nDone, it prints hi.')
        rng = random.Random(0)
        for _ in range(20):
            cuts = sorted(rng.sample(range(1, len(text)), 4))
            chunks, start = [], 0
            for cut in cuts:
                chunks.append(text[start:cut])
                start = cut
            chunks.append(text[start:])
            filt = LiveContentFilter()
            assert self.feed_all(filt, chunks) == (
                "Reading the file first.\n\nDone, it prints hi.")
