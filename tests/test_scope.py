"""Scope is the wall; mode is only how often it knocks. A mode must never be
able to widen a scope."""

import subprocess

import pytest

from wynxo.permissions import PermissionStore
from wynxo.scope import Mode, Scope, resolve
from wynxo.tools.files import ReadFile, WriteFile


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "top.py").write_text("top\n")
    (tmp_path / "sub" / "mid.py").write_text("mid\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True,
                   capture_output=True, timeout=10)
    return tmp_path


class TestParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("folder", Scope.FOLDER), ("dir", Scope.FOLDER), ("cwd", Scope.FOLDER),
        ("repo", Scope.REPO), ("git", Scope.REPO), ("repository", Scope.REPO),
        ("machine", Scope.MACHINE), ("pc", Scope.MACHINE), ("all", Scope.MACHINE),
        ("MACHINE", Scope.MACHINE),
    ])
    def test_scope_aliases(self, raw, expected):
        assert Scope.parse(raw) is expected

    @pytest.mark.parametrize("raw,expected", [
        ("plan", Mode.PLAN), ("readonly", Mode.PLAN), ("safe", Mode.PLAN),
        ("manual", Mode.MANUAL), ("ask", Mode.MANUAL), ("default", Mode.MANUAL),
        ("auto", Mode.AUTO), ("accept-edits", Mode.AUTO), ("edit", Mode.AUTO),
        ("yolo", Mode.YOLO), ("bypass", Mode.YOLO),
    ])
    def test_mode_aliases(self, raw, expected):
        assert Mode.parse(raw) is expected

    def test_unknown_values_list_the_options(self):
        with pytest.raises(KeyError, match="folder"):
            Scope.parse("everywhere-ish")
        with pytest.raises(KeyError, match="manual"):
            Mode.parse("turbo")


class TestResolution:
    def test_folder_scope_is_the_cwd(self, repo):
        boundary = resolve(repo / "sub", Scope.FOLDER)
        assert boundary.root == (repo / "sub").resolve()

    def test_repo_scope_walks_up_to_the_git_root(self, repo):
        boundary = resolve(repo / "sub" / "deep", Scope.REPO)
        assert boundary.root == repo.resolve()
        assert boundary.scope is Scope.REPO

    def test_repo_scope_outside_a_repo_falls_back_to_folder(self, tmp_path):
        """Never silently grant more than was asked for."""
        boundary = resolve(tmp_path, Scope.REPO)
        assert boundary.scope is Scope.FOLDER
        assert boundary.root == tmp_path.resolve()

    def test_machine_scope_is_unrestricted(self, tmp_path):
        boundary = resolve(tmp_path, Scope.MACHINE)
        assert boundary.unrestricted
        assert boundary.contains(tmp_path.parent.parent / "anything")


class TestContainment:
    def test_folder_scope_rejects_a_parent(self, tmp_path):
        boundary = resolve(tmp_path / "sub", Scope.FOLDER)
        (tmp_path / "sub").mkdir(exist_ok=True)
        assert not boundary.contains(tmp_path / "outside.py")

    def test_repo_scope_allows_a_sibling_directory(self, repo):
        boundary = resolve(repo / "sub", Scope.REPO)
        assert boundary.contains(repo / "top.py")

    def test_rejection_names_the_way_out(self, tmp_path):
        boundary = resolve(tmp_path, Scope.FOLDER)
        message = boundary.reject("../x")
        assert "--scope repo" in message and "--scope machine" in message


class TestToolEnforcement:
    async def test_write_outside_folder_scope_is_refused(self, tmp_path):
        (tmp_path / "work").mkdir()
        boundary = resolve(tmp_path / "work", Scope.FOLDER)
        tool = WriteFile(tmp_path / "work", boundary)
        result = await tool.invoke({"path": "../escaped.txt", "content": "x"})
        assert not result.ok
        assert not (tmp_path / "escaped.txt").exists()

    async def test_repo_scope_permits_what_folder_scope_refused(self, repo):
        folder = ReadFile(repo / "sub", resolve(repo / "sub", Scope.FOLDER))
        assert not (await folder.invoke({"path": "../top.py"})).ok

        wide = ReadFile(repo / "sub", resolve(repo / "sub", Scope.REPO))
        assert (await wide.invoke({"path": "../top.py"})).ok

    async def test_machine_scope_reaches_outside(self, tmp_path):
        (tmp_path / "work").mkdir()
        (tmp_path / "elsewhere.txt").write_text("hello\n")
        tool = ReadFile(tmp_path / "work", resolve(tmp_path / "work", Scope.MACHINE))
        result = await tool.invoke({"path": str(tmp_path / "elsewhere.txt")})
        assert result.ok and "hello" in result.output

    async def test_yolo_mode_does_not_widen_scope(self, tmp_path):
        """The important invariant: approving everything is not the same as
        being allowed everywhere."""
        (tmp_path / "work").mkdir()
        boundary = resolve(tmp_path / "work", Scope.FOLDER)
        tool = WriteFile(tmp_path / "work", boundary)
        store = PermissionStore(mode=Mode.YOLO)
        assert not store.needs_prompt("write_file", True, {})   # would not ask
        result = await tool.invoke({"path": "../nope.txt", "content": "x"})
        assert not result.ok, "yolo must not defeat the scope boundary"
        assert not (tmp_path / "nope.txt").exists()


class TestModeBehaviour:
    def test_plan_blocks_every_mutation(self):
        store = PermissionStore(mode=Mode.PLAN)
        for tool in ("write_file", "edit_file", "multi_edit", "shell"):
            assert store.blocked(tool, True)

    def test_plan_allows_reads(self):
        store = PermissionStore(mode=Mode.PLAN)
        assert store.blocked("read_file", False) is None
        assert store.blocked("grep", False) is None

    def test_plan_message_says_how_to_leave(self):
        message = PermissionStore(mode=Mode.PLAN).blocked("write_file", True)
        assert "/mode" in message

    def test_auto_writes_without_asking_but_shell_still_asks(self):
        store = PermissionStore(mode=Mode.AUTO)
        assert not store.needs_prompt("write_file", True, {"path": "a"})
        assert store.needs_prompt("shell", True, {"command": "make"})

    def test_manual_asks_for_everything_mutating(self):
        store = PermissionStore(mode=Mode.MANUAL)
        assert store.needs_prompt("write_file", True, {"path": "a"})
        assert store.needs_prompt("shell", True, {"command": "ls -la"}) is False  # read-only
        assert store.needs_prompt("shell", True, {"command": "make"})


class TestInternalTools:
    """A tool that only writes wynxo's own state should not need approval,
    and should still work in plan mode -- a read-only session that cannot
    write down what it learned is worse than useless."""

    def test_remember_is_marked_internal(self, tmp_path):
        from wynxo.tools import build_registry
        registry = build_registry(tmp_path)
        assert registry.get("remember").internal
        assert not registry.get("write_file").internal
        assert not registry.get("shell").internal

    def test_manual_mode_does_not_prompt_for_internal_writes(self):
        store = PermissionStore(mode=Mode.MANUAL)
        assert not store.needs_prompt("remember", True, {}, internal=True)
        assert store.needs_prompt("write_file", True, {"path": "a"})

    def test_plan_mode_allows_internal_writes(self):
        store = PermissionStore(mode=Mode.PLAN)
        assert store.blocked("remember", True, internal=True) is None
        assert store.blocked("write_file", True)

    async def test_remember_works_in_plan_mode_end_to_end(self, tmp_path):
        from wynxo.memory import Memory
        from wynxo.tools.memory_tool import Remember

        memory = Memory(tmp_path, tmp_path / "u")
        tool = Remember(tmp_path, None, memory)
        store = PermissionStore(mode=Mode.PLAN)
        assert store.blocked(tool.name, tool.mutating, tool.internal) is None
        result = await tool.invoke({"note": "Learned during a plan-mode session"})
        assert result.ok
        assert memory.counts()[0] == 1
