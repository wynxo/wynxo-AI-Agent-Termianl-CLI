"""Watching a tool call being written.

Ollama's native tool-calling parses the model's output before handing it
over. The arguments on the wire are a parsed map, not a string, so a
half-written call cannot be represented -- the server's parser buffers the
whole thing and emits it once, complete. There is nothing partial to show,
and revealing a finished string slowly would be an animation pretending to
be a stream.

Asked for as text instead -- the Hermes-style <tool_call> block a model
without native support uses anyway -- the call is ordinary content, and
ordinary content streams. That is what /stream turns on, and what these
tests are about.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from wynxo.agent import Callbacks
from wynxo.parsing import CODE_KEYS, LiveContentFilter
from wynxo.provider import Chunk

from test_agent import make_agent


def call(name: str, arguments: dict) -> str:
    return ("<tool_call>\n" + json.dumps({"name": name,
                                          "arguments": arguments})
            + "\n</tool_call>")


def pieces(text: str, size: int = 3):
    """Fed the way a model writes: a few characters at a time."""
    filter = LiveContentFilter()
    out = []
    for i in range(0, len(text), size):
        filter.feed(text[i:i + size])
        if delta := filter.code_delta():
            out.append((filter.call_name(), filter.call_path(), delta))
    return out


class TestTheLongArgumentArrivesInPieces:
    @pytest.mark.parametrize("size", [1, 3, 12])
    def test_a_file_is_written_a_piece_at_a_time(self, size):
        body = "import asyncio\n\nasync def go():\n    return 1\n"
        got = pieces(call("write_file", {"path": "a.py", "content": body}),
                     size)
        assert "".join(d for _, _, d in got) == body
        assert len(got) > 1, "arrived whole, which is not streaming"

    def test_a_call_that_arrives_whole_has_nothing_to_watch(self):
        """And that is not a loss. Nothing was generated in between, so
        there is no moment at which half of it was the truth -- the block
        is drawn from the parsed call instead, as it always was."""
        body = "import asyncio\n"
        got = pieces(call("write_file", {"path": "a.py", "content": body}),
                     size=10_000)
        assert got == []

    @pytest.mark.parametrize("size", [1, 3, 12])
    def test_a_command_is_composed_a_piece_at_a_time(self, size):
        """The other half of the ask. A command is short, but it is the
        thing you most want to read before it runs."""
        got = pieces(call("shell", {"command": "which firefox"}), size)
        assert "".join(d for _, _, d in got) == "which firefox"

    def test_command_is_one_of_the_watched_arguments(self):
        assert "command" in CODE_KEYS

    def test_short_arguments_are_not_watched(self):
        """A path, a pattern, a line number arrive in a flash and watching
        them says nothing. What earns the screen is the long value the
        model is actually composing."""
        for quiet in ("path", "pattern", "query", "line"):
            assert quiet not in CODE_KEYS

    def test_nothing_is_shown_on_a_key_name_alone(self):
        """Half of `"content"` is not content."""
        filter = LiveContentFilter()
        filter.feed('<tool_call>{"name": "write_file", "arguments": {"cont')
        assert filter.code_delta() == ""


class TestTheDisplayIsToldWhatItIsWatching:
    def test_the_name_is_known_before_the_first_fragment(self):
        got = pieces(call("shell", {"command": "which firefox"}))
        assert got and got[0][0] == "shell"

    def test_the_path_is_known_before_the_first_fragment(self):
        """The path comes before the content in the arguments, so a block
        need never be headed "(unnamed)" while it fills up."""
        got = pieces(call("write_file", {"path": "demo.py",
                                         "content": "x = 1\n"}))
        assert got and got[0][1] == "demo.py"

    def test_a_half_written_name_is_not_reported(self):
        """A name read out of a value still arriving would be the wrong
        name, and a path shown as "dem" is worse than no path."""
        filter = LiveContentFilter()
        filter.feed('<tool_call>{"name": "write_f')
        assert filter.call_name() == ""

    def test_nothing_is_reported_outside_a_call(self):
        filter = LiveContentFilter()
        filter.feed('just prose about "name": "write_file"')
        assert filter.call_name() == ""
        assert filter.call_path() == ""


class Watcher(Callbacks):
    def __init__(self):
        self.told, self.code = [], ""

    async def on_code(self, text):
        self.code += text

    async def on_code_target(self, name, path=""):
        self.told.append((name, path))

    async def on_content(self, text):
        pass


def chunked(text: str, size: int = 4):
    def chat(*args, **kwargs):
        async def gen():
            for i in range(0, len(text), size):
                yield Chunk(content=text[i:i + size])
                await asyncio.sleep(0)
            yield Chunk(done=True, prompt_tokens=1, completion_tokens=1)
        return gen()
    return chat


class TestItReachesTheDisplay:
    async def _run(self, tmp_path, body):
        cb = Watcher()
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}], callbacks=cb)
        agent.permissions.yolo = True
        agent.native_tools = False
        agent.backend.chat = chunked(body)
        await agent._call_model(messages=[{"role": "user", "content": "go"}])
        return cb

    async def test_a_command_streams_through_the_agent(self, tmp_path):
        cb = await self._run(tmp_path,
                             call("shell", {"command": "which firefox"}))
        assert cb.code == "which firefox"
        assert ("shell", "") in cb.told

    async def test_a_file_streams_with_its_name(self, tmp_path):
        cb = await self._run(
            tmp_path, call("write_file", {"path": "demo.py",
                                          "content": "x = 1\ny = 2\n"}))
        assert cb.code == "x = 1\ny = 2\n"
        assert ("write_file", "demo.py") in cb.told

    async def test_the_display_is_told_once_per_call(self, tmp_path):
        cb = await self._run(
            tmp_path, call("write_file", {"path": "demo.py",
                                          "content": "x = 1\n" * 40}))
        assert len(cb.told) <= 2, cb.told


class TestWhatIsDrawn:
    async def _drawn(self, tool, path, fragments):
        from wynxo.cli import TerminalCallbacks
        from wynxo.ui import UI

        ui = UI()
        ui.console.file = io.StringIO()
        ui.console.width = ui.width = 80
        cb = TerminalCallbacks(ui)
        await cb.on_code_target(tool, path)
        for fragment in fragments:
            await cb.on_code(fragment)
        return cb, ui.console.file.getvalue()

    async def test_a_command_goes_to_the_transcript_under_its_own_verb(self):
        """A diff card is a picture of a file changing, and a command is
        not that. It used to open one anyway, headed with a filename it did
        not have and a verb it had not earned."""
        cb, drawn = await self._drawn("shell", "", ["whic", "h fi", "refox"])
        assert cb.card is None
        assert "running" in drawn
        assert "which firefox" in drawn

    async def test_a_file_goes_to_its_card_with_its_name(self):
        cb, drawn = await self._drawn("write_file", "demo.py",
                                      ["x = ", "1\n"])
        assert cb.card is not None and cb.card.live
        assert cb.card.path == "demo.py"
        assert "x = 1" in cb.card.streamed
        assert drawn.strip() == "", "the card draws in the live region"

    async def test_an_unknown_tool_is_still_shown(self):
        """A fragment shown nowhere is the one outcome worth avoiding. With
        no name yet it goes where it always went -- the card -- rather than
        being dropped for want of a label."""
        cb, drawn = await self._drawn("", "", ["something"])
        assert "something" in (drawn + (cb.card.streamed if cb.card else ""))


class TestTheSettingChoosesThePath:
    async def test_it_asks_for_text_even_from_a_model_with_native_tools(
            self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.config.stream_tool_calls = True
        await agent.detect_capabilities()
        assert agent.native_tools is False

    async def test_off_it_uses_the_server_parser(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.config.stream_tool_calls = False
        await agent.detect_capabilities()
        assert agent.native_tools is True

    async def test_no_schemas_are_sent_on_the_watched_path(self, tmp_path):
        """Which is the other half of what it buys: the same tools cost
        about 3,400 tokens as JSON schemas and 1,300 described in prose,
        and on a model reading its prompt at CPU speed that is real time."""
        agent, fake, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.config.stream_tool_calls = True
        await agent.detect_capabilities()
        await agent.run("hello")
        assert not any(body.get("tools") for body in fake.requests)

    def test_the_very_first_request_honours_it(self, tmp_path):
        """Set at construction as well as in detection, or the first turn
        of a session would go out on the other path."""
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.config.stream_tool_calls = True
        from wynxo.agent import Agent

        again = Agent(agent.client, agent.config, agent.policy,
                      tmp_path, agent.cb, registry=agent.tools)
        assert again.native_tools is False
