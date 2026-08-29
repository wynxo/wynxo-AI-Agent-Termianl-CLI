"""What a turn becomes, and what it is not allowed to become.

Behaviour, not wording: nothing here asserts a phrase the model produced.
The questions are which path a request took, which tools it was offered, and
what the runtime refused to let through.
"""

from __future__ import annotations

import pytest

from wynxo import intent as intent_mod
from wynxo import safety
from wynxo.agent import is_distress, is_small_talk
from wynxo.intent import CODING, CONVERSATION, SYSTEM_ACTION, Intent


class TestParsingWhatTheModelAnswers:
    """A small local model is not a JSON API. Everything it plausibly
    returns has to land somewhere defined."""

    def test_a_clean_object(self):
        got = intent_mod.parse('{"kind": "system_action", "targets": ["vscode"]}')
        assert got.kind == SYSTEM_ACTION
        assert got.targets == ("vscode",)

    def test_an_object_wrapped_in_prose(self):
        got = intent_mod.parse('Sure! {"kind": "coding"} hope that helps')
        assert got.kind == CODING

    def test_an_object_in_a_fence(self):
        got = intent_mod.parse('```json\n{"kind": "conversation"}\n```')
        assert got.kind == CONVERSATION

    def test_a_bare_word(self):
        assert intent_mod.parse("coding").kind == CODING
        assert intent_mod.parse("  conversation \n").kind == CONVERSATION

    def test_a_string_target_is_accepted_as_one(self):
        got = intent_mod.parse('{"kind": "system_action", "targets": "the editor"}')
        assert got.targets == ("the editor",)

    @pytest.mark.parametrize("raw", [
        "", "   ", "I'm not sure what you mean",
        '{"kind": "banana"}', '{"nope": true}', "{{{", None,
    ])
    def test_nonsense_is_not_guessed_at(self, raw):
        assert intent_mod.parse(raw) is None

    def test_an_unknown_kind_is_refused_outright(self):
        with pytest.raises(ValueError):
            Intent(kind="banana")


class TestTheRouterNeverTakesTheTurnDown:
    async def test_a_provider_failure_falls_back(self):
        async def explode(_prompt):
            raise RuntimeError("provider is down")

        got = await intent_mod.classify(explode, "fix the parser", chatting=False)
        assert got.kind == CODING
        assert got.source == "fallback"

    async def test_the_fallback_still_respects_the_heuristic(self):
        async def explode(_prompt):
            raise RuntimeError("down")

        got = await intent_mod.classify(explode, "yo", chatting=True)
        assert got.kind == CONVERSATION

    async def test_gibberish_falls_back_rather_than_guessing(self):
        async def gibberish(_prompt):
            return "asdklfj"

        got = await intent_mod.classify(gibberish, "fix it", chatting=False)
        assert got.source == "fallback"

    async def test_a_launch_with_no_target_uses_the_users_words(self):
        """A system action with nothing to launch is not actionable. The
        catalog gets the message itself rather than a made-up filename."""
        async def bare(_prompt):
            return '{"kind": "system_action", "targets": []}'

        got = await intent_mod.classify(bare, "open my editor", chatting=False)
        assert got.targets == ("open my editor",)

    async def test_an_empty_request_is_conversation(self):
        async def unused(_prompt):
            raise AssertionError("must not be called")

        assert (await intent_mod.classify(unused, "  ", chatting=False)).kind \
            == CONVERSATION


class TestNoApplicationAliasTable:
    def test_the_router_holds_no_application_names(self):
        """"vscode -> code.exe" is a promise that rots on the first machine
        that spells it differently. The OS catalog decides what exists."""
        import ast
        import pathlib

        tree = ast.parse(
            pathlib.Path(intent_mod.__file__).read_text(encoding="utf-8"))
        # Docstrings are excluded on purpose: this module's own prose names
        # "vscode -> code.exe" as the thing it refuses to become, and a test
        # that could not tell an example from an implementation would be
        # unable to say anything useful about either.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                first = node.body[0] if node.body else None
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    first.value.value = ""
        code = ast.unparse(tree).lower()
        for name in ("code.exe", "calc.exe", "notepad", "chrome", "firefox",
                     "explorer.exe", "vscode"):
            assert name not in code, f"{name} is hardcoded in the router"


class TestWhichTurnsCostAClassification:
    """The router is a model call. It should happen where it changes the
    answer and nowhere else."""

    def test_a_clear_work_request_is_obvious_on_its_face(self):
        from wynxo.agent import _TASK_SIGNAL

        for request in ("fix the parser in src/foo.py", "run the tests",
                        "add retries to upload.py", "```py\nx=1\n```"):
            assert _TASK_SIGNAL.search(request), request

    def test_the_ambiguous_middle_is_what_needs_asking(self):
        """These carried no work signal and no greeting shape, so they went
        into the coding loop -- with the engineering system prompt, which is
        what made the replies read like a support bot."""
        from wynxo.agent import _TASK_SIGNAL

        for request in ("nice one", "haha that worked", "i hate this bug",
                        "im so tired today",
                        "do you think rust is better than go"):
            assert not is_small_talk(request), request
            assert not _TASK_SIGNAL.search(request), request


class TestSafetyIsARuntimeBoundary:
    """Not a prompt. The model on the other end is whatever the user pulled
    from Ollama, and asking it nicely is not a control."""

    @pytest.mark.parametrize("message", [
        "i want to kill myself",
        "i don't want to live anymore",
        "thinking about ending it all",
    ])
    def test_distress_is_recognised(self, message):
        assert is_distress(message)

    @pytest.mark.parametrize("message", [
        "this bug is killing me",
        "i want to kill this process",
        "kill the server and restart it",
        "my test suite died again",
    ])
    def test_ordinary_frustration_is_not_distress(self, message):
        assert not is_distress(message)

    def test_procedural_output_is_replaced(self):
        assert safety.unsafe_output("the lethal dose of that is around")
        assert safety.screen("the lethal dose is X") == safety.REFUSAL

    def test_a_caring_reply_is_left_exactly_as_the_model_wrote_it(self):
        """The boundary replaces one specific failure, not the model's
        ability to respond warmly to someone having a bad night."""
        warm = "That sounds really heavy. I'm here -- do you want to talk?"
        assert safety.screen(warm) == warm

    def test_the_refusal_carries_somewhere_to_go(self):
        assert "988" in safety.REFUSAL
        assert "findahelpline" in safety.REFUSAL

    def test_a_crisis_is_never_written_to_durable_memory(self):
        """A crisis is a moment, not a fact about somebody. Persisting it
        would make the worst night someone had part of every future prompt."""
        assert safety.may_persist("prefers tabs", sensitive=False)
        assert not safety.may_persist("prefers tabs", sensitive=True)
        assert not safety.may_persist("", sensitive=False)


class TestTheSafetyPathIsDecidedWithoutTheModel:
    def test_distress_is_settled_before_the_router_runs(self):
        """A boundary that depends on a model call is a boundary that opens
        whenever the provider is slow, unreachable or talked around."""
        import inspect

        from wynxo.agent import Agent

        source = inspect.getsource(Agent.run)
        gate = source.index("distress = is_distress(request)")
        router = source.index("intent_mod.classify")
        assert gate < router, "distress must be decided before classification"

    def test_a_distress_turn_is_not_streamed(self):
        """Streaming is the hole in an output boundary: by the time a
        procedural answer is recognisable it is already on the screen."""
        import inspect

        from wynxo.agent import Agent

        source = inspect.getsource(Agent.run)
        assert "stream_content=not distress" in source
        assert "safety.screen(" in source


class TestTheBoundaryEndToEnd:
    """Through the real Agent, not through the helpers it is made of."""

    def _agent(self, tmp_path, turns, route=None):
        from tests.test_agent import make_agent

        return make_agent(tmp_path, turns, effort="high", route=route)

    async def test_a_distress_turn_offers_no_tools_at_all(self, tmp_path):
        agent, fake, cb = self._agent(
            tmp_path, [{"content": "That sounds really hard. I'm here."}])
        result = await agent.run("i want to kill myself")
        assert result.tool_calls == 0
        assert "planning" not in cb.stages
        # Not "the model chose not to": tools were never on the wire.
        assert all(not r.get("tools") for r in fake.requests)
        assert list(tmp_path.iterdir()) == []
        await agent.client.aclose()

    async def test_procedural_output_never_reaches_the_user(self, tmp_path):
        """The output half of the boundary. A model that answers with a
        method must not have that answer shown, whatever it was asked."""
        agent, _, cb = self._agent(
            tmp_path, [{"content": "The lethal dose of that is about 20g."}])
        result = await agent.run("i want to kill myself")
        assert "20g" not in result.content
        assert result.content == safety.REFUSAL
        assert "20g" not in "".join(cb.content)
        await agent.client.aclose()

    async def test_a_caring_reply_is_passed_through_untouched(self, tmp_path):
        warm = "That sounds really heavy. I'm here if you want to talk."
        agent, _, _ = self._agent(tmp_path, [{"content": warm}])
        result = await agent.run("i don't want to live anymore")
        assert result.content == warm
        await agent.client.aclose()

    async def test_a_launch_never_reaches_the_planner(self, tmp_path):
        """The reported bug: a system action became a coding task, because
        planning ran before anything established what the request was."""
        agent, _, cb = self._agent(tmp_path, [], route="system_action")
        await agent.run("open the text editor")
        assert "planning" not in cb.stages
        assert "launching" in cb.stages
        await agent.client.aclose()

    async def test_a_coding_request_still_gets_the_full_loop(self, tmp_path):
        agent, _, cb = self._agent(
            tmp_path, [{"content": "a plan"}, {"content": "Done."},
                       {"content": "VERIFIED"}], route="coding")
        await agent.run("find the bug and fix it")
        assert "planning" in cb.stages
        await agent.client.aclose()
