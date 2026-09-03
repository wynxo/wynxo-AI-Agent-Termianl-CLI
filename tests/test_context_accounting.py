"""What wynxo thinks it is sending, against what it sends.

Native tool calling puts the schemas in their own wire field, not in the
prompt -- so nothing here counted them, and they are not small. Seventeen
tools is around 3,900 tokens.

That is not a rounding error at a small window. Measured: wynxo reported
4,266 tokens used while actually sending 8,301 into a window of 8,192.
Ollama does not refuse an over-long prompt, it drops the front of it -- the
system prompt -- and answers anyway, so the model quietly stops being able
to see its instructions or its tools. Compaction did not fire because the
count said there was room, and the warning built for exactly this could not
fire either, for the same reason.
"""

from __future__ import annotations

import json
import pathlib


from wynxo.agent import Callbacks
from wynxo.session import Session, estimate_tokens, message_tokens

from test_agent import make_agent


def wire_tokens(body: dict) -> int:
    return (estimate_tokens(json.dumps(body.get("messages", [])))
            + estimate_tokens(json.dumps(body.get("tools") or [])))


class TestTheEstimateCountsTheWholeRequest:
    def test_overhead_is_part_of_the_count(self):
        session = Session(workspace=pathlib.Path("/tmp"))
        session.system_prompt = "hello"
        before = session.token_estimate()
        session.overhead = 500
        assert session.token_estimate() == before + 500

    async def test_it_matches_what_goes_on_the_wire(self, tmp_path):
        agent, fake, _ = make_agent(tmp_path, [{"content": "ok"}] * 4)
        await agent.detect_capabilities()
        for i in range(6):
            agent.session.add_user(f"q{i} " + "word " * 300)
            agent.session.add_assistant(f"a{i} " + "word " * 300)
        await agent.run("and now this")
        estimate = agent.session.token_estimate()
        real = wire_tokens(fake.requests[-1])
        # Framing per message is approximated, not counted exactly; what
        # matters is that the gap is a rounding error and not a whole set
        # of tool schemas.
        assert abs(real - estimate) < 400, (estimate, real)

    async def test_the_schemas_are_what_was_missing(self, tmp_path):
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        await agent.detect_capabilities()
        schemas = estimate_tokens(json.dumps(agent.tools.ollama_schemas()))
        assert schemas > 2000, "the thing being counted is not small"
        assert abs(agent.session.overhead - schemas) < 50

    async def test_nothing_is_double_counted_on_the_prompted_path(
            self, tmp_path):
        """There the tools are described inside the system prompt, which is
        already counted -- adding them again would compact a conversation
        that has room."""
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.config.stream_tool_calls = True
        await agent.detect_capabilities()
        assert agent.session.overhead == 0

    async def test_a_new_conversation_keeps_counting(self, tmp_path):
        """A fresh Session has no idea the schemas exist, so /clear and
        /resume would go back to under-counting until something happened to
        rebuild the prompt."""
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        await agent.detect_capabilities()
        agent.session = Session(workspace=tmp_path)
        assert agent.session.overhead > 2000


class Ears(Callbacks):
    def __init__(self):
        self.warnings: list[str] = []

    async def on_warning(self, message):
        self.warnings.append(message)

    async def on_content(self, text):
        pass


class TestTheSafetyNetCanActuallyFire:
    async def _session_at(self, tmp_path, window, counted=True):
        long = "The retry loop sleeps a fixed backoff between attempts. " * 30
        cb = Ears()
        agent, fake, _ = make_agent(tmp_path, [{"content": long}] * 40,
                                    callbacks=cb)
        agent.permissions.yolo = True
        agent.config.num_ctx = window
        await agent.detect_capabilities()
        if not counted:
            agent.session.overhead = 0
            agent.refresh_system_prompt = lambda: None
        for i in range(6):
            await agent.run(f"turn {i}: " + "explain at length. " * 20)
        return agent, fake, cb

    async def test_an_overflowing_window_is_reported(self, tmp_path):
        _, _, cb = await self._session_at(tmp_path, 8192)
        assert any("window" in w for w in cb.warnings), cb.warnings

    async def test_and_was_silent_before(self, tmp_path):
        """The same conversation, with the schemas uncounted: nothing said,
        while the model was losing its instructions every turn."""
        _, _, cb = await self._session_at(tmp_path, 8192, counted=False)
        assert not [w for w in cb.warnings if "window" in w]

    async def test_a_roomy_window_is_not_nagged(self, tmp_path):
        _, _, cb = await self._session_at(tmp_path, 65536)
        assert not [w for w in cb.warnings if "window" in w]

    async def test_compaction_still_reclaims(self, tmp_path):
        agent, _, _ = await self._session_at(tmp_path, 8192)
        assert agent.session.compactions > 0


class TestWhatTheStripShows:
    async def test_the_percentage_is_of_the_whole_request(self, tmp_path):
        """It reads the same estimate, so it was under-reporting by the
        same 3,900 tokens -- 52% while actually over the window."""
        agent, _, _ = make_agent(tmp_path, [{"content": "ok"}])
        agent.config.num_ctx = 8192
        await agent.detect_capabilities()
        used = agent.session.token_estimate()
        assert used > 1000, "an empty session already costs the schemas"
        assert 100 * used / 8192 > 10


class TestASmallWindowIsOnlyBigEnoughOneWay:
    """The two things a small machine is told to do are not independent.

    At num_ctx=8192 the native path's fixed cost -- system prompt plus tool
    schemas -- is around 5,600 tokens, which is most of the window before a
    word is said. The prompted path describes the same tools in prose for
    about 3,200. Measured over ten turns: native peaks at 8,616 (over), the
    prompted path at 6,445 (inside, with room).
    """

    async def _peak(self, tmp_path, window, stream):
        long = "The retry loop sleeps a fixed backoff between attempts. " * 30
        agent, fake, _ = make_agent(tmp_path, [{"content": long}] * 60)
        agent.permissions.yolo = True
        agent.config.num_ctx = window
        agent.config.stream_tool_calls = stream
        await agent.detect_capabilities()
        peak = 0
        for i in range(10):
            await agent.run(f"turn {i}: " + "explain at length. " * 20)
            peak = max(peak, wire_tokens(fake.requests[-1]))
        return peak, agent.session.compactions

    async def test_the_prompted_path_fits_a_small_window(self, tmp_path):
        peak, _ = await self._peak(tmp_path, 8192, stream=True)
        assert peak <= 8192, peak

    async def test_the_fixed_cost_is_what_decides_it(self, tmp_path):
        """Not the conversation -- compaction handles that. The prompt and
        the schemas are the part no amount of summarising reclaims."""
        native, _ = await self._peak(tmp_path, 8192, stream=False)
        prompted, _ = await self._peak(tmp_path, 8192, stream=True)
        assert native - prompted > 1500, (native, prompted)

    async def test_compaction_runs_either_way(self, tmp_path):
        for stream in (False, True):
            _, compactions = await self._peak(tmp_path, 8192, stream)
            assert compactions > 0, stream


class TestWhatCompactionIsWorthIsAboutTheConversation:
    def test_the_fixed_prompt_does_not_raise_the_bar(self):
        """"Is the older half worth summarising" is a question about the
        messages. Counting the prompt and the schemas in that comparison
        set the bar with the weight of the one thing compaction cannot
        move -- so at a small window it was over every turn and never
        compacted, because the reason it was over was the reason it would
        not.

        Sized so the two rules disagree: a conversation of about 3,000
        tokens with 1,500 in its older half, behind 9,000 tokens of prompt
        and schemas. Against the conversation the bar is 450 and it
        compacts; against everything it is 1,800 and it never does.
        """
        session = Session(workspace=pathlib.Path("/tmp"))
        for i in range(12):
            session.add_user(f"question {i} " + "word " * 25)
            session.add_assistant(f"answer {i} " + "word " * 25)
        conversation = sum(message_tokens(m) for m in session.messages)
        older, _ = session.slice_for_summary()
        removable = sum(message_tokens(m) for m in older)
        session.overhead = 9000

        everything = session.token_estimate()
        assert removable > conversation * 0.15, "the older half is worth it"
        assert removable < everything * 0.15, \
            "and would not have been, measured against the fixed prompt"
        assert session.should_compact(8000, 8000)

    def test_a_short_conversation_is_still_not_worth_it(self):
        session = Session(workspace=pathlib.Path("/tmp"))
        session.overhead = 8000
        session.add_user("hi")
        session.add_assistant("hello")
        assert not session.should_compact(16000, 16000)
