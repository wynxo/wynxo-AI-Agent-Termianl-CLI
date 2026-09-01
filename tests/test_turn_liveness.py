"""The coding turn, from the user's side: is anything happening?

The reported symptom was a turn that wrote a file, went silent, and came
back with "the conversation may have outgrown the context window" -- an
agent that looked dead while it was working, and then blamed the wrong
thing. The message sequence turned out to be correct; what was missing was
any sign of life between a tool finishing and the next reply arriving, and
any evidence behind the diagnosis.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from wynxo.agent import Agent, Callbacks, ModelEvidence, TaskState
from wynxo.config import Config
from wynxo.effort import resolve
from wynxo.provider import Chunk
from wynxo.scope import Boundary, Scope
from wynxo.tools import build_registry


class Watch(Callbacks):
    def __init__(self):
        self.stages: list[tuple[str, str]] = []
        self.tools: list[tuple[str, bool]] = []
        self.warnings: list[str] = []
        self.order: list[str] = []

    async def on_stage(self, name, detail=""):
        self.stages.append((name, detail))
        self.order.append(f"stage:{name}")

    async def on_tool_start(self, name, summary, event=None):
        self.order.append(f"tool:{name}")

    async def on_tool_result(self, name, ok, display, output, event=None):
        self.tools.append((name, ok))
        self.order.append(f"result:{name}")

    async def on_warning(self, message):
        self.warnings.append(message)


class Scripted:
    """A write_file, then whatever the script says. Counts its calls."""

    def __init__(self, after: list[tuple[str, int]]):
        self.after = after
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        if self.calls == 1:
            return self._iter("", [{"function": {"name": "write_file", "arguments": {
                "path": "index.html", "content": "<html>hi</html>"}}}], 0)
        text, tokens = self.after[min(self.calls - 2, len(self.after) - 1)]
        return self._iter(text, [], tokens)

    async def _iter(self, content, calls, completion):
        yield Chunk(content=content, tool_calls=calls, done=True,
                    stop_reason="stop", prompt_tokens=900,
                    completion_tokens=completion)


def run(tmp_path: Path, after):
    watch = Watch()
    backend = Scripted(after)
    config = Config(verify_with_tests=False, allow_shell=False,
                    auto_approve=["*"])
    agent = Agent(backend, config, resolve("low"), tmp_path, watch,
                  registry=build_registry(tmp_path),
                  boundary=Boundary(scope=Scope.FOLDER, root=tmp_path))
    result = asyncio.run(agent.run("create index.html"))
    return agent, backend, watch, result


def pairing(session):
    announced = sum(len(m["tool_calls"]) for m in session.messages
                    if m.get("role") == "assistant" and m.get("tool_calls"))
    return announced, sum(1 for m in session.messages if m.get("role") == "tool")


class TestToolThenModelContinuation:
    """write_file succeeds -> the model is called again -> it answers."""

    def test_the_turn_completes_and_the_file_is_written_once(self, tmp_path):
        agent, backend, watch, result = run(tmp_path, [("Wrote it.", 12)])
        assert result.content == "Wrote it."
        assert result.answered is True
        assert agent.task_state.state is TaskState.COMPLETED
        assert (tmp_path / "index.html").read_text() == "<html>hi</html>"
        assert watch.tools == [("write_file", True)], watch.tools

    def test_the_tool_is_not_replayed(self, tmp_path):
        """A second write_file the model never asked for would silently
        overwrite the first."""
        _, _, watch, _ = run(tmp_path, [("Wrote it.", 12)])
        assert watch.order.count("tool:write_file") == 1

    def test_the_conversation_stays_well_formed(self, tmp_path):
        agent, _, _, _ = run(tmp_path, [("Wrote it.", 12)])
        assert pairing(agent.session) == (1, 1)

    def test_a_stage_is_announced_before_every_model_call(self, tmp_path):
        """The gap between a tool finishing and the next reply arriving is
        the whole complaint: on a local model it is tens of seconds, and
        the screen kept showing the last tool's label throughout."""
        _, backend, watch, _ = run(tmp_path, [("Wrote it.", 12)])
        thinking = [s for s, _ in watch.stages if s == "thinking"]
        assert len(thinking) >= backend.calls, (
            f"{backend.calls} model calls but only {len(thinking)} stages: "
            f"{watch.stages}")

    def test_the_gap_after_the_tool_is_covered(self, tmp_path):
        """Specifically: a stage between the tool result and the answer."""
        _, _, watch, _ = run(tmp_path, [("Wrote it.", 12)])
        after_tool = watch.order[watch.order.index("result:write_file") + 1:]
        assert after_tool and after_tool[0].startswith("stage:"), watch.order


class TestToolThenEmptyAnswer:
    """The reported failure. The work happened; the reply did not."""

    def test_it_retries_once_and_only_once(self, tmp_path):
        _, backend, _, result = run(tmp_path, [("", 50)] * 5)
        assert result.empty_retried is True
        assert backend.calls == 3, "one tool call and exactly one retry"

    def test_the_retry_recovers_when_the_model_does(self, tmp_path):
        _, _, watch, result = run(tmp_path, [("", 50), ("Wrote it.", 12)])
        assert result.content == "Wrote it."
        assert result.answered is True
        assert not watch.warnings, watch.warnings

    def test_the_retry_is_shown_as_a_state_not_a_warning(self, tmp_path):
        """It usually works. A self-healing retry that succeeded is not
        something to leave an exclamation mark in the transcript about --
        but the user still has to see why the wait doubled."""
        _, _, watch, _ = run(tmp_path, [("", 50), ("Wrote it.", 12)])
        assert any(s == "retrying" for s, _ in watch.stages), watch.stages

    def test_an_empty_turn_is_not_reported_as_a_success(self, tmp_path):
        agent, _, _, result = run(tmp_path, [("", 50)] * 5)
        assert result.answered is False
        assert agent.task_state.state is TaskState.FAILED

    def test_the_work_that_did_happen_is_kept(self, tmp_path):
        """The file was written before the model went quiet. Reporting the
        turn as failed must not suggest the write did not happen."""
        _, _, watch, _ = run(tmp_path, [("", 50)] * 5)
        assert (tmp_path / "index.html").exists()
        assert watch.tools == [("write_file", True)]

    def test_the_tool_is_still_not_replayed(self, tmp_path):
        _, _, watch, _ = run(tmp_path, [("", 50)] * 5)
        assert watch.order.count("tool:write_file") == 1

    def test_the_conversation_survives_for_the_next_turn(self, tmp_path):
        agent, _, _, _ = run(tmp_path, [("", 50)] * 5)
        assert pairing(agent.session) == (1, 1)

    def test_exactly_one_diagnosis_is_shown(self, tmp_path):
        _, _, watch, _ = run(tmp_path, [("", 50)] * 5)
        assert len(watch.warnings) == 1, watch.warnings

    def test_the_warning_the_user_gets_is_the_reasoned_one(self, tmp_path):
        """Not just that explain_empty_answer() is right, but that its
        answer is what actually reaches the screen. Tested through the real
        turn, because a hardcoded message put back at the call site would
        leave the classifier's own tests passing.

        The two runs differ only in what the provider reported, so two
        different warnings is the whole claim.
        """
        _, _, generated, _ = run(tmp_path, [("", 50)] * 5)
        _, _, silent, _ = run(tmp_path, [("", 0)] * 5)
        assert generated.warnings and silent.warnings
        assert generated.warnings[0] != silent.warnings[0], (
            "the same message for both, so it is not reading the evidence:\n"
            f"  {generated.warnings[0]}")
        assert "50 tokens" in generated.warnings[0]


class TestTheDiagnosisIsEvidenceBased:
    """One message used to cover every cause: "its chat template does not
    fit, or the conversation has outgrown the context window". Both can be
    wrong at once, and /compact for a conversation that fits sends somebody
    a long way in the wrong direction."""

    def explain(self, *, num_ctx=8000, **evidence) -> str:
        agent = Agent.__new__(Agent)
        agent.config = Config(num_ctx=num_ctx)
        agent._last_evidence = ModelEvidence(**evidence)
        return Agent.explain_empty_answer(agent)

    def test_a_full_window_is_named_only_with_the_numbers_to_prove_it(self):
        said = self.explain(chunks=2, prompt_tokens=9000, num_ctx=8000)
        assert "9000" in said and "8000" in said and "/compact" in said

    def test_a_conversation_that_fits_is_not_blamed_on_context(self):
        said = self.explain(chunks=2, prompt_tokens=900, num_ctx=8000)
        assert "/compact" not in said and "context window" not in said

    def test_a_zero_reply_budget_says_so(self):
        said = self.explain(chunks=2, stop_reason="length", completion_tokens=0)
        assert "num_predict" in said
        assert "/compact" not in said

    def test_a_truncated_stream_says_so(self):
        said = self.explain(chunks=3, truncated=True)
        assert "ran out of memory" in said or "unloaded" in said

    def test_a_server_that_sent_nothing_says_so(self):
        said = self.explain(chunks=0)
        assert "nothing at all" in said

    def test_a_model_still_loading_says_so(self):
        said = self.explain(chunks=2, stop_reason="load")
        assert "loading" in said

    def test_tokens_generated_but_no_answer_blames_the_template(self):
        said = self.explain(chunks=9, stop_reason="stop",
                            completion_tokens=120, had_thinking=True)
        assert "template" in said and "reasoning" in said
        assert "/compact" not in said

    def test_every_branch_offers_one_thing_to_do(self):
        for evidence in ({"chunks": 0},
                         {"chunks": 3, "truncated": True},
                         {"chunks": 2, "prompt_tokens": 9000},
                         {"chunks": 2, "stop_reason": "length"},
                         {"chunks": 2, "stop_reason": "load"},
                         {"chunks": 9, "completion_tokens": 40},
                         {"chunks": 2, "stop_reason": "stop"}):
            said = self.explain(**evidence)
            assert any(hint in said for hint in
                       ("/doctor", "/compact", "num_predict", "num_ctx",
                        "ollama ps", "different model", "Trying again")), said


class TestTheStopReasonSurvivesTheProvider:
    """It was being dropped, which is why there was no evidence to reason
    from. Pinned here because the diagnosis is built on it."""

    def test_ollama_done_reason_reaches_the_chunk(self):
        from wynxo.provider import OllamaClient

        chunk = OllamaClient._to_chunk({
            "message": {"content": "hi"}, "done": True,
            "done_reason": "length", "prompt_eval_count": 12,
            "eval_count": 3})
        assert chunk.stop_reason == "length"
        assert chunk.prompt_tokens == 12 and chunk.completion_tokens == 3

    def test_a_missing_done_reason_is_empty_not_an_error(self):
        from wynxo.provider import OllamaClient

        assert OllamaClient._to_chunk({"message": {}, "done": True}).stop_reason == ""


class TestTheNextTurnStillWorks:
    """Whatever happened, the session must not be poisoned."""

    @pytest.mark.parametrize("script,label", [
        ([("", 50)] * 5, "after an empty answer"),
        ([("Wrote it.", 12)], "after a normal answer"),
    ])
    def test_a_second_request_runs_normally(self, tmp_path, script, label):
        agent, backend, watch, _ = run(tmp_path, script)
        backend.after = [("Second answer.", 9)]
        backend.calls = 99          # past the write_file branch
        second = asyncio.run(agent.run("say something else"))
        assert second.content == "Second answer.", label
        assert second.answered is True, label
        announced, answered = pairing(agent.session)
        assert announced == answered, label
