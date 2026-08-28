"""The uninstaller has to be exactly as careful as the installer.

Two ways to get this wrong, and both are worse than leaving a file behind:
rewriting someone's shell profile, and deleting a cloned repository that
holds the only copy of their work. Those are what most of this covers.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wynxo_uninstall", ROOT / "uninstall.py")
uninstall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uninstall)


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, check=True, capture_output=True, timeout=30)


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q")
    (work / "a.txt").write_text("hello\n")
    git(work, "add", "-A")
    git(work, "commit", "-qm", "first")
    return work


class TestShellProfile:
    """Only the installer's own two lines come out. Everything else in the
    file is the user's and is not ours to touch."""

    def write(self, tmp_path, text):
        rc = tmp_path / ".bashrc"
        rc.write_text(text)
        return rc

    def test_removes_only_the_marker_and_its_line(self, tmp_path):
        rc = self.write(tmp_path, (
            "# my config\n"
            "export EDITOR=vim\n"
            "\n"
            f"{uninstall.MARKER}\n"
            'export PATH="/home/me/.local/bin:$PATH"\n'
            "\n"
            "export TOKEN=keepme\n"
        ))
        assert uninstall.strip_path_line(rc, dry_run=False) is True
        after = rc.read_text()
        assert uninstall.MARKER not in after
        assert ".local/bin" not in after
        # Every line the user wrote survives, in order.
        assert "# my config" in after
        assert "export EDITOR=vim" in after
        assert "export TOKEN=keepme" in after

    def test_leaves_exactly_one_blank_between_the_neighbours(self, tmp_path):
        """Install/uninstall cycles must not stack up blank lines."""
        rc = self.write(tmp_path, (
            "alias a='b'\n"
            "\n"
            f"{uninstall.MARKER}\n"
            'export PATH="/x:$PATH"\n'
            "\n"
            "alias c='d'\n"
        ))
        uninstall.strip_path_line(rc, dry_run=False)
        assert rc.read_text() == "alias a='b'\n\nalias c='d'\n"

    def test_a_profile_without_the_marker_is_untouched(self, tmp_path):
        original = "export PATH=\"/somewhere/else:$PATH\"\n# not ours\n"
        rc = self.write(tmp_path, original)
        assert uninstall.strip_path_line(rc, dry_run=False) is False
        assert rc.read_text() == original

    def test_dry_run_changes_nothing(self, tmp_path):
        original = f"x=1\n\n{uninstall.MARKER}\nexport PATH=\"/x:$PATH\"\n"
        rc = self.write(tmp_path, original)
        assert uninstall.strip_path_line(rc, dry_run=True) is True
        assert rc.read_text() == original

    def test_a_binary_or_unreadable_profile_is_survivable(self, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_bytes(b"\xff\xfe\x00binary")
        assert uninstall.strip_path_line(rc, dry_run=False) is False

    def test_missing_profile_is_survivable(self, tmp_path):
        assert uninstall.strip_path_line(tmp_path / "nope", dry_run=False) is False


class TestUnsavedWork:
    """A cloned repo in wynxo's data directory can hold the only copy of
    something. Deleting it silently would be unforgivable."""

    def test_uncommitted_changes_are_flagged(self, repo):
        (repo / "a.txt").write_text("changed\n")
        assert "uncommitted" in uninstall.unsaved_work(repo)

    def test_an_untracked_file_is_flagged(self, repo):
        (repo / "new.txt").write_text("scratch\n")
        assert "uncommitted" in uninstall.unsaved_work(repo)

    def test_commits_that_exist_nowhere_else_are_flagged(self, repo):
        """The one people forget: a clean tree is not a safe tree."""
        assert "unpushed" in uninstall.unsaved_work(repo)

    def test_a_fully_pushed_repo_is_clean(self, repo, tmp_path):
        """No false positives, or the guard becomes noise people --force past."""
        upstream = tmp_path / "upstream.git"
        subprocess.run(["git", "init", "-q", "--bare", str(upstream)],
                       check=True, capture_output=True, timeout=30)
        git(repo, "remote", "add", "origin", str(upstream))
        git(repo, "push", "-q", "origin", "HEAD:refs/heads/master")
        git(repo, "fetch", "-q", "origin")
        assert uninstall.unsaved_work(repo) == ""

    def test_a_plain_directory_is_not_treated_as_a_repo(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert uninstall.unsaved_work(plain) == ""


class TestDiscovery:
    def test_cloned_repos_are_found_two_levels_down(self, tmp_path, monkeypatch):
        """They are stored as repos/<owner>/<name>."""
        monkeypatch.setattr(uninstall, "data_dir", lambda: tmp_path)
        (tmp_path / "repos" / "someone" / "proj").mkdir(parents=True)
        (tmp_path / "repos" / "someone" / "other").mkdir(parents=True)
        names = sorted(p.name for p in uninstall.cloned_repos())
        assert names == ["other", "proj"]

    def test_no_repos_directory_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uninstall, "data_dir", lambda: tmp_path)
        assert uninstall.cloned_repos() == []

    def test_macos_collapses_config_and_data_into_one_entry(self, monkeypatch):
        """They are the same directory there; reporting it twice, and
        deleting it twice, both look like bugs."""
        monkeypatch.setattr(uninstall.sys, "platform", "darwin")
        assert uninstall.config_dir() == uninstall.data_dir()
        collapsed = list(dict.fromkeys([uninstall.config_dir(), uninstall.data_dir()]))
        assert len(collapsed) == 1

    def test_termux_launcher_is_looked_for_under_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(uninstall.sys, "platform", "linux")
        monkeypatch.setenv("TERMUX_VERSION", "0.118")
        monkeypatch.setenv("PREFIX", str(tmp_path))
        assert tmp_path / "bin" / "wynxo" in uninstall.launcher_candidates()

    def test_every_known_shell_profile_is_checked(self):
        """Someone can install under bash and uninstall under zsh; a stale
        PATH line in the profile they are not using right now is exactly the
        leftover this is meant to prevent."""
        names = {p.name for p in uninstall.rc_candidates()}
        assert {".bashrc", ".zshrc", ".profile", "config.fish"} <= names


class TestRemoval:
    def test_dry_run_removes_nothing(self, tmp_path):
        target = tmp_path / "thing"
        target.mkdir()
        (target / "f").write_text("x")
        assert uninstall.remove_tree(target, dry_run=True) is True
        assert target.exists()

    def test_missing_paths_report_nothing_removed(self, tmp_path):
        assert uninstall.remove_tree(tmp_path / "nope", dry_run=False) is False
        assert uninstall.remove_file(tmp_path / "nope", dry_run=False) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require admin on Windows")
    def test_a_broken_symlink_launcher_is_still_removed(self, tmp_path):
        """A launcher pointing at an already-deleted venv is the most likely
        thing to be left behind, so exists() alone is not enough to test."""
        link = tmp_path / "wynxo"
        link.symlink_to(tmp_path / "gone")
        assert not link.exists()      # dangling
        assert link.is_symlink()
        assert uninstall.remove_file(link, dry_run=False) is True
        assert not link.is_symlink()


class TestSelfDeletion:
    """Windows will not delete a directory holding a running program."""

    def test_detects_running_from_inside_the_tree(self, tmp_path, monkeypatch):
        venv_python = tmp_path / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("")
        monkeypatch.setattr(uninstall.sys, "executable", str(venv_python))
        assert uninstall.running_from(tmp_path) is True

    def test_a_system_interpreter_is_not_inside_the_tree(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uninstall.sys, "executable", "/usr/bin/python3")
        assert uninstall.running_from(tmp_path) is False


class TestRemovingTheMarksLeftInProjects:
    """wynxo writes .wynxo/ into every project it works in.

    Those live in the user's own directories rather than under the config or
    data directory the uninstaller otherwise clears, so removing wynxo used
    to leave a folder behind in every repository the agent had ever opened
    -- which is not the clean removal the installer promises.
    """

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        import uninstall

        sessions = tmp_path / "data" / "sessions"
        sessions.mkdir(parents=True)
        monkeypatch.setattr(uninstall, "data_dir", lambda: tmp_path / "data")
        return tmp_path

    def record(self, home, name, workspace) -> None:
        import json

        (home / "data" / "sessions" / f"{name}.json").write_text(
            json.dumps({"session_id": name, "workspace": str(workspace)}),
            encoding="utf-8")

    def project(self, home, name, marked=True):
        directory = home / name
        directory.mkdir(parents=True, exist_ok=True)
        if marked:
            (directory / ".wynxo").mkdir()
            (directory / ".wynxo" / "memory.md").write_text("# notes\n")
        return directory

    def test_it_finds_a_project_wynxo_worked_in(self, home):
        import uninstall

        work = self.project(home, "repo")
        self.record(home, "s1", work)
        assert uninstall.touched_projects() == [work / ".wynxo"]

    def test_each_project_is_listed_once(self, home):
        import uninstall

        work = self.project(home, "repo")
        for i in range(4):
            self.record(home, f"s{i}", work)
        assert len(uninstall.touched_projects()) == 1

    def test_a_project_without_a_marker_is_not_listed(self, home):
        import uninstall

        work = self.project(home, "plain", marked=False)
        self.record(home, "s1", work)
        assert uninstall.touched_projects() == []

    def test_a_project_that_no_longer_exists_is_skipped(self, home):
        import uninstall

        self.record(home, "s1", home / "deleted-long-ago")
        assert uninstall.touched_projects() == []

    def test_one_corrupt_record_does_not_hide_the_others(self, home):
        """Same rule as everywhere else: a bad file costs you that file."""
        import uninstall

        work = self.project(home, "repo")
        self.record(home, "good", work)
        (home / "data" / "sessions" / "bad.json").write_text("{ truncated",
                                                             encoding="utf-8")
        (home / "data" / "sessions" / "list.json").write_text("[]",
                                                              encoding="utf-8")
        assert uninstall.touched_projects() == [work / ".wynxo"]

    def test_no_sessions_at_all_is_fine(self, tmp_path, monkeypatch):
        import uninstall

        monkeypatch.setattr(uninstall, "data_dir", lambda: tmp_path / "nope")
        assert uninstall.touched_projects() == []

    def test_removing_the_marker_leaves_the_project_alone(self, home):
        """It must reach into a repository for exactly one directory."""
        import uninstall

        work = self.project(home, "repo")
        (work / "main.py").write_text("code = 1\n")
        (work / "README.md").write_text("# repo\n")

        assert uninstall.remove_tree(work / ".wynxo", dry_run=False)
        assert not (work / ".wynxo").exists()
        assert (work / "main.py").exists() and (work / "README.md").exists()

    def test_a_dry_run_changes_nothing(self, home):
        import uninstall

        work = self.project(home, "repo")
        uninstall.remove_tree(work / ".wynxo", dry_run=True)
        assert (work / ".wynxo").exists()

    def test_it_describes_what_is_inside(self, home):
        """memory.md may have been edited by hand, so the user gets to see
        what they are agreeing to delete."""
        import uninstall

        work = self.project(home, "repo")
        (work / ".wynxo" / "map.md").write_text("# map\n")
        described = uninstall.describe_marker(work / ".wynxo")
        assert "memory.md" in described and "map.md" in described

    def test_clearing_projects_is_asked_separately(self):
        """'Remove wynxo' and 'reach into my repositories' are not the same
        permission."""
        import inspect

        import uninstall

        source = inspect.getsource(uninstall.main)
        assert "clear_markers" in source
        # It must be its own ask(), not folded into the main confirmation.
        assert source.count("ask(") >= 2

    def test_keep_data_leaves_projects_untouched(self):
        import inspect

        import uninstall

        assert "args.keep_data else touched_projects()" in \
            inspect.getsource(uninstall.main)


class TestDirectoriesWynxoMadeOnItsWayIn:
    """A machine that never had an XDG layout gets ~/.config and
    ~/.local/share created for it the first time wynxo saves anything.
    Leaving those behind is a mark, and this file exists to keep the promise
    that there are none."""

    def _prune(self, path, dry_run=False):
        import uninstall

        return uninstall.prune_empty_parents(path, dry_run)

    def test_an_empty_parent_goes(self, tmp_path, monkeypatch):
        import pathlib as _pathlib

        monkeypatch.setattr(_pathlib.Path, "home", staticmethod(lambda: tmp_path))
        target = tmp_path / ".config" / "wynxo"
        target.mkdir(parents=True)
        target.rmdir()                       # as remove_tree would have left it
        assert self._prune(target) == [(tmp_path / ".config").resolve()]
        assert not (tmp_path / ".config").exists()

    def test_a_parent_holding_anything_else_stays(self, tmp_path, monkeypatch):
        """Somebody else's configuration is not ours to remove."""
        import pathlib as _pathlib

        monkeypatch.setattr(_pathlib.Path, "home", staticmethod(lambda: tmp_path))
        (tmp_path / ".config" / "wynxo").mkdir(parents=True)
        (tmp_path / ".config" / "git").mkdir()
        (tmp_path / ".config" / "wynxo").rmdir()
        assert self._prune(tmp_path / ".config" / "wynxo") == []
        assert (tmp_path / ".config" / "git").exists()

    def test_it_walks_up_while_they_are_empty(self, tmp_path, monkeypatch):
        import pathlib as _pathlib

        monkeypatch.setattr(_pathlib.Path, "home", staticmethod(lambda: tmp_path))
        target = tmp_path / ".local" / "share" / "wynxo"
        target.mkdir(parents=True)
        target.rmdir()
        assert len(self._prune(target)) == 2
        assert not (tmp_path / ".local").exists()

    def test_it_never_touches_home_itself(self, tmp_path, monkeypatch):
        import pathlib as _pathlib

        monkeypatch.setattr(_pathlib.Path, "home", staticmethod(lambda: tmp_path))
        target = tmp_path / "wynxo"
        target.mkdir()
        target.rmdir()
        assert self._prune(target) == []
        assert tmp_path.exists()

    def test_it_never_goes_above_home(self, tmp_path, monkeypatch):
        """A config directory outside home -- XDG_CONFIG_HOME can point
        anywhere -- must not walk the uninstaller up someone's filesystem."""
        import pathlib as _pathlib

        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "wynxo").mkdir(parents=True)
        (elsewhere / "wynxo").rmdir()
        monkeypatch.setattr(_pathlib.Path, "home",
                            staticmethod(lambda: tmp_path / "home"))
        (tmp_path / "home").mkdir()
        assert self._prune(elsewhere / "wynxo") == []
        assert elsewhere.exists()

    def test_a_dry_run_changes_nothing(self, tmp_path, monkeypatch):
        import pathlib as _pathlib

        monkeypatch.setattr(_pathlib.Path, "home", staticmethod(lambda: tmp_path))
        target = tmp_path / ".config" / "wynxo"
        target.mkdir(parents=True)
        target.rmdir()
        assert self._prune(target, dry_run=True)
        assert (tmp_path / ".config").exists()
