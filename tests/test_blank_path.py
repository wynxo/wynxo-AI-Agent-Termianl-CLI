"""A path the model got wrong must not end the turn.

Two helpers run before a write tool does -- the checkpoint that makes the
edit undoable, and the diff shown when asking permission -- and both resolve
the model's path themselves. Neither expected resolve_path to raise
ValueError, which it does for a path that is empty once stripped. A model
padding the field, or a template that rendered to nothing, killed the whole
turn on a mistake the tool itself reports in one sentence.
"""

from __future__ import annotations

import pytest
from test_agent import RecordingCallbacks, make_agent

from wynxo.permissions import Decision

BLANK = ["   ", "", "\t", "\n"]


def _call(name, path):
    args = {"path": path, "content": "x"}
    if name == "edit_file":
        args = {"path": path, "old_text": "a", "new_text": "b"}
    if name == "multi_edit":
        args = {"path": path, "edits": [{"old_text": "a", "new_text": "b"}]}
    return [{"tool_calls": [{"function": {"name": name, "arguments": args}}]},
            {"content": "Done."}]


class TestABlankPathIsAnsweredNotFatal:
    @pytest.mark.parametrize("path", BLANK)
    @pytest.mark.parametrize("name", ["write_file", "edit_file", "multi_edit"])
    async def test_the_turn_survives_every_write_tool(self, tmp_path, name, path):
        agent, _fake, _cb = make_agent(tmp_path, _call(name, path))
        result = await agent.run("write it")
        assert result.content == "Done."

    @pytest.mark.parametrize("path", BLANK)
    async def test_the_model_is_told_what_was_wrong(self, tmp_path, path):
        """Surviving is not enough -- a turn that carries on having said
        nothing leaves the model to repeat the same call."""
        agent, _fake, _cb = make_agent(tmp_path, _call("write_file", path))
        await agent.run("write it")
        told = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert told, "the model was told nothing"
        assert "path" in told[-1]["content"].lower()

    async def test_nothing_is_written(self, tmp_path):
        agent, _fake, _cb = make_agent(tmp_path, _call("write_file", "  "))
        await agent.run("write it")
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("path", BLANK)
    async def test_it_survives_the_permission_preview_too(self, tmp_path, path):
        """The other half. Asking permission renders a diff first, and that
        resolves the path as well -- so in manual mode the turn died one
        step earlier than the checkpoint, before anyone was even asked."""
        cb = RecordingCallbacks(Decision.ALLOW)
        agent, _fake, _cb = make_agent(tmp_path, _call("write_file", path),
                                       callbacks=cb)
        result = await agent.run("write it")
        assert result.content == "Done."

    async def test_a_path_outside_the_project_is_still_refused(self, tmp_path):
        """The boundary is not what was loosened here."""
        agent, _fake, _cb = make_agent(tmp_path, _call("write_file",
                                                       "../../etc/passwd"))
        await agent.run("write it")
        told = [m for m in agent.session.messages if m.get("role") == "tool"]
        assert "outside the project" in told[-1]["content"]

    async def test_a_real_path_is_still_undoable(self, tmp_path):
        """The checkpoint still has to happen for the paths that work --
        skipping the snapshot is the cost of a bad path, not of every one."""
        target = tmp_path / "a.py"
        target.write_text("before\n")
        agent, _fake, _cb = make_agent(tmp_path, _call("write_file", "a.py"))
        await agent.run("write it")
        assert target.read_text() == "x"
        assert agent.checkpoints.undo()[0] is True
        assert target.read_text() == "before\n"
