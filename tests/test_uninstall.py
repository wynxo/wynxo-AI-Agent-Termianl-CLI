"""The uninstaller has to be exactly as careful as the installer.

Two ways to get this wrong, and both are worse than leaving a file behind:
rewriting someone's shell profile, and deleting a cloned repository that
holds the only copy of their work. Those are what most of this covers.
"""

import importlib.util
import subprocess
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
