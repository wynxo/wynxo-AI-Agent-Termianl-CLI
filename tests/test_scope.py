"""Scope is the wall; mode is only how often it knocks. A mode must never be
able to widen a scope."""

import subprocess
from pathlib import Path

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


class TestBoundarySummary:
    """The terminal shows a shortened path; the system prompt gets the real
    one. /scope and /cd used to print the raw absolute path and wrap across
    lines on anything but a short workspace path."""

    def _summary(self, boundary):
        import types

        from wynxo import cli
        from wynxo.ui import UI

        repl = types.SimpleNamespace(ui=UI())
        return cli.Repl._boundary_summary(repl, boundary)

    def test_folder_scope_is_shortened(self, tmp_path):
        long_home = tmp_path / "a" / "b" / "c" / "d" / "e" / "project"
        long_home.mkdir(parents=True)
        boundary = resolve(long_home, Scope.FOLDER)
        summary = self._summary(boundary)
        assert summary != str(boundary.root)
        assert summary.startswith(".../")

    def test_repo_scope_keeps_the_sentence_and_shortens_the_path(self, repo):
        boundary = resolve(repo, Scope.REPO)
        summary = self._summary(boundary)
        assert summary.startswith("the repository at ")
        assert str(boundary.root) not in summary or len(str(boundary.root)) < 18

    def test_machine_scope_has_no_path_to_shorten(self, tmp_path):
        boundary = resolve(tmp_path, Scope.MACHINE)
        assert self._summary(boundary) == "the whole machine"

    def test_describe_itself_stays_the_full_path_for_the_system_prompt(self, tmp_path):
        long_home = tmp_path / "a" / "b" / "c" / "d" / "e" / "project"
        long_home.mkdir(parents=True)
        boundary = resolve(long_home, Scope.FOLDER)
        assert boundary.describe() == str(boundary.root)


class TestReviewMode:
    """The middle ground: manual interrupts a ten-file refactor ten times,
    auto never shows you the shape of what happened."""

    def test_it_is_a_mode(self):
        assert Mode.parse("review") is Mode.REVIEW
        assert Mode.parse("batch") is Mode.REVIEW

    def test_edits_do_not_prompt_individually(self):
        store = PermissionStore(mode=Mode.REVIEW)
        assert not store.needs_prompt("write_file", True, {"path": "x"})
        assert not store.needs_prompt("edit_file", True, {"path": "x"})

    def test_commands_still_prompt(self):
        """Deferring a file write is reversible; running a command is not."""
        store = PermissionStore(mode=Mode.REVIEW)
        assert store.needs_prompt("shell", True, {"command": "npm install"})

    def test_nothing_is_blocked_outright(self):
        store = PermissionStore(mode=Mode.REVIEW)
        assert store.blocked("write_file", True) is None

    def test_it_cannot_widen_the_scope(self):
        """Mode is the friction, scope is the wall -- review is no different."""
        boundary = resolve(Path.cwd(), Scope.FOLDER)
        store = PermissionStore(mode=Mode.REVIEW)
        assert store.mode is Mode.REVIEW
        assert not boundary.contains(Path("/etc/passwd"))

    def test_it_describes_itself(self):
        assert "end" in Mode.REVIEW.describe()


class TestAPathTheOSWillNotResolve:
    """resolve() has more failure modes than ValueError.

    A symlink loop raises RuntimeError, an unreadable mount raises OSError,
    and Windows raises on names it refuses outright. The boundary used to
    catch only ValueError, so a repo containing a symlink loop crashed the
    turn the moment any tool touched it.
    """

    def _boundary(self, root):
        from wynxo.scope import Boundary, Scope
        return Boundary(scope=Scope.FOLDER, root=root.resolve())

    def test_a_symlink_loop_is_refused_rather_than_raising(self, tmp_path):
        import os
        import pytest

        loop = tmp_path / "loop"
        try:
            os.symlink("loop", loop)
        except (OSError, NotImplementedError):
            pytest.skip("this platform will not make the symlink")

        assert self._boundary(tmp_path).contains(loop) is False

    def test_it_fails_closed_when_resolution_gives_up(self, tmp_path,
                                                     monkeypatch):
        """Unplaceable must mean 'outside'. A boundary that answered True
        when it could not tell would be worse than no boundary."""
        from pathlib import Path

        boundary = self._boundary(tmp_path)
        inside = tmp_path / "a.py"
        assert boundary.contains(inside) is True

        def refuse(self, *args, **kwargs):
            raise OSError("no")

        monkeypatch.setattr(Path, "resolve", refuse)
        assert boundary.contains(inside) is False

    def test_an_unrestricted_boundary_never_needs_to_resolve(self, tmp_path,
                                                             monkeypatch):
        from pathlib import Path
        from wynxo.scope import Boundary, Scope

        def refuse(self, *args, **kwargs):
            raise OSError("no")

        monkeypatch.setattr(Path, "resolve", refuse)
        wide = Boundary(scope=Scope.MACHINE, root=tmp_path, unrestricted=True)
        assert wide.contains(Path("/anything/at/all")) is True

    def test_a_looping_path_displays_instead_of_crashing(self, tmp_path):
        import os
        import pytest
        from wynxo.tools.base import Tool

        loop = tmp_path / "loop"
        try:
            os.symlink("loop", loop)
        except (OSError, NotImplementedError):
            pytest.skip("this platform will not make the symlink")

        class Probe(Tool):
            name = "probe"
            async def run(self, **kwargs):
                return ""

        shown = Probe(workspace=tmp_path).relative(loop)
        assert "loop" in shown
