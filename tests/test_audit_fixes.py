"""Faults found by reading the repository rather than by driving it.

Each was reproduced before it was fixed; the reproduction is the test.
"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile

from wynxo.memory import Memory
from wynxo.tools.memory_tool import Remember


class TestTheModelCannotDeleteWhatItCannotWrite:
    """The guard sat on remember() alone.

    So the model could not add a personal fact and was free to delete one --
    and deletion is the more destructive of the two, is silent, and has no
    undo. A confused turn tidying up could erase everything the user had
    ever asked wynxo to remember about them.
    """

    def _memory(self):
        workspace = pathlib.Path(tempfile.mkdtemp())
        memory = Memory(workspace, user_dir=pathlib.Path(tempfile.mkdtemp()))
        memory.remember("my name is Sam", "user", explicit=True)
        memory.remember("I prefer tabs", "user", explicit=True)
        return workspace, memory

    def _tool(self, workspace, memory):
        return Remember(workspace, memory=memory)

    def test_the_model_cannot_write_user_memory(self):
        """The half that already worked. Kept, because the fix must not
        loosen it while making the other half match."""
        workspace, memory = self._memory()
        tool = self._tool(workspace, memory)
        asyncio.run(tool.run(tool.Input(note="Sam is 34", scope="user")))
        assert "Sam is 34" not in "\n".join(memory.user.entries())

    def test_the_model_cannot_delete_user_memory(self):
        workspace, memory = self._memory()
        tool = self._tool(workspace, memory)
        result = asyncio.run(tool.run(
            tool.Input(note="Sam", scope="user", forget=True)))
        assert any("my name is Sam" in e for e in memory.user.entries()), \
            "the model deleted a personal fact"
        assert "explicit user request" in result.output

    def test_the_person_still_can(self):
        """/memory forget is an explicit request and must keep working."""
        _workspace, memory = self._memory()
        count, _message = memory.forget("Sam", "user", explicit=True)
        assert count == 1
        assert not any("my name is Sam" in e for e in memory.user.entries())

    def test_project_memory_is_still_the_models_to_manage(self):
        """The restriction is about personal facts, not about notes on the
        codebase -- narrowing it further would make the tool useless."""
        workspace, memory = self._memory()
        tool = self._tool(workspace, memory)
        asyncio.run(tool.run(tool.Input(note="tests run with uv")))
        assert any("uv" in e for e in memory.project.entries())
        result = asyncio.run(tool.run(
            tool.Input(note="uv", forget=True)))
        assert result.ok
        assert not any("uv" in e for e in memory.project.entries())

    def test_the_two_halves_use_the_same_rule(self):
        """They disagreed once. A test is cheaper than finding out again."""
        import inspect

        source = inspect.getsource(Memory.forget)
        assert "_agent_write" in source and "explicit" in source


class TestTheTurnRemembersWhereItHasLooked:
    """add_file() kept two lists of the same paths in lockstep, and neither
    reached the model or the screen -- bookkeeping for nobody. The one worth
    keeping now reaches the recovery block, which is exactly where a turn
    that is repeating itself benefits from being told where it has been."""

    def _stuck(self):
        from wynxo.task_state import TaskStateMachine

        machine = TaskStateMachine()
        machine.begin("fix the parser")
        machine.add_file("wynxo/parsing.py")
        machine.add_file("wynxo/agent.py")
        machine.add_file("wynxo/parsing.py", changed=True)
        return machine

    def test_inspected_files_reach_the_recovery_block(self):
        block = self._stuck().recovery_block()
        assert "already inspected" in block
        assert "wynxo/agent.py" in block

    def test_changed_and_inspected_stay_separate(self):
        machine = self._stuck()
        assert machine.changed_files == ["wynxo/parsing.py"]
        assert "wynxo/agent.py" in machine.inspected_files

    def test_the_old_name_still_answers(self):
        """Call sites that ask for "relevant files" get the same data rather
        than a second copy of it."""
        machine = self._stuck()
        assert machine.relevant_files is machine.inspected_files

    def test_a_reset_clears_it(self):
        machine = self._stuck()
        machine.reset()
        assert machine.inspected_files == []
        assert machine.relevant_files == []
