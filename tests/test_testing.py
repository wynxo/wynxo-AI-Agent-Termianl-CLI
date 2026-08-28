
class TestWhenItRuns:
    """Running tests after a turn that changed nothing would be a slow way
    to learn that nothing changed."""

    def _agent(self, tmp_path, **overrides):
        from unittest.mock import MagicMock

        from wynxo.agent import Agent
        from wynxo.config import Config
        from wynxo.effort import resolve

        config = Config()
        for key, value in overrides.items():
            setattr(config, key, value)
        return Agent(client=MagicMock(), config=config,
                     policy=resolve("medium"), workspace=tmp_path)

    def _ran(self, agent) -> bool:
        import asyncio

        called = []
        agent.checkpoints.changes_since = lambda _mark: [object()]
        shell = agent.tools.get("shell")
        if shell is None:
            return False
        original_invoke = shell.invoke

        async def spy_invoke(*args, **kwargs):
            called.append(True)
            return await original_invoke(*args, **kwargs)

        shell.invoke = spy_invoke
        # Avoid depending on a real project test command; the helper only
        # cares whether verification reached shell execution.
        from wynxo.testing import detect
        original_detect = __import__("wynxo.agent", fromlist=["testing"]).testing.detect
        __import__("wynxo.agent", fromlist=["testing"]).testing.detect = lambda _root: type("Runner", (), {"command": "echo test"})()
        try:
            asyncio.run(agent._verify_with_tests())
        finally:
            __import__("wynxo.agent", fromlist=["testing"]).testing.detect = original_detect
        return bool(called)

    def test_it_does_not_run_when_the_setting_is_off(self, tmp_path):
        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        assert self._ran(self._agent(tmp_path, verify_with_tests=False)) is False

    def test_it_does_not_run_in_plan_mode(self, tmp_path):
        """Read-only means read-only, tests included."""
        from wynxo.scope import Mode

        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        agent = self._agent(tmp_path)
        agent.permissions.mode = Mode.PLAN
        assert self._ran(agent) is False

    def test_it_does_not_run_when_nothing_changed(self, tmp_path):
        import asyncio

        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        agent = self._agent(tmp_path)
        agent.checkpoints.changes_since = lambda _mark: []
        reached = []
        agent.tools.get = lambda name: reached.append(name)
        asyncio.run(agent._verify_with_tests())
        assert reached == []

    def test_it_runs_when_files_changed_and_a_runner_exists(self, tmp_path):
        project(tmp_path, {"pytest.ini": "[pytest]\n"})
        assert self._ran(self._agent(tmp_path)) is True

    def test_it_does_not_run_without_a_detectable_runner(self, tmp_path):
        project(tmp_path, {"README.md": "# nothing to run\n"})
        agent = self._agent(tmp_path)
        agent.checkpoints.changes_since = lambda _mark: [object()]
        import asyncio
        reached = []
        shell = agent.tools.get("shell")
        original_invoke = shell.invoke
        async def spy(*args, **kwargs):
            reached.append(True)
            return await original_invoke(*args, **kwargs)
        shell.invoke = spy
        asyncio.run(agent._verify_with_tests())
        assert reached == []


class TestWhichInterpreterTheTestsRunUnder:
    """"python -m pytest" goes wrong two ways, and both report a failure the
    user did not cause -- which is worse than not running the tests at all,
    because the model then sets about fixing code that was fine."""

    def test_a_projects_virtualenv_wins(self, tmp_path):