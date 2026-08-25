"""@path in a message means "read this first".

The part that matters most is that a mention is user input and goes through
the same Boundary the tools do: @../../etc/passwd has to be refused for
exactly the reason read_file refuses it.
"""

import pytest

from wynxo.mentions import candidates, expand, find
from wynxo.scope import Scope, resolve as resolve_scope


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def check_token(t):\n    return t\n")
    (tmp_path / "README.md").write_text("# hi\n")
    return tmp_path


@pytest.fixture
def boundary(project):
    return resolve_scope(project, Scope.FOLDER)


class TestFind:
    def test_a_plain_mention(self):
        assert find("why does @src/auth.py fail?") == ["src/auth.py"]

    def test_several_in_order_without_duplicates(self):
        assert find("@a.py and @b.py and @a.py") == ["a.py", "b.py"]

    def test_an_email_address_is_not_a_mention(self):
        """The single most likely false positive."""
        assert find("mail me at bob@example.com") == []

    def test_a_trailing_full_stop_is_not_part_of_the_path(self):
        assert find("look at @README.md.") == ["README.md"]

    def test_no_mentions_in_ordinary_prose(self):
        assert find("fix the parser please") == []

    def test_a_bare_at_sign_is_not_a_mention(self):
        assert find("what @ even is this") == []


class TestCandidates:
    def test_it_lists_the_workspace(self, project):
        assert set(candidates(project)) == {"src/", "README.md"}

    def test_directories_are_marked_so_you_can_walk_into_them(self, project):
        assert "src/" in candidates(project)

    def test_it_walks_into_a_directory(self, project):
        assert candidates(project, "src/") == ["src/auth.py"]

    def test_a_prefix_filters(self, project):
        assert candidates(project, "READ") == ["README.md"]

    def test_noise_directories_are_skipped(self, project):
        found = candidates(project)
        assert ".git" not in found and "node_modules/" not in found

    def test_a_path_outside_the_workspace_returns_nothing(self, project):
        assert candidates(project, "../") == []

    def test_a_missing_directory_is_not_an_error(self, project):
        assert candidates(project, "nope/") == []


class TestExpand:
    def test_the_file_is_inlined(self, project, boundary):
        message, problems = expand("what does @src/auth.py do?", project, boundary)
        assert "check_token" in message
        assert problems == []

    def test_the_original_sentence_survives(self, project, boundary):
        """Strip the mention and the model reads a file with no idea why."""
        message, _ = expand("what does @src/auth.py do?", project, boundary)
        assert message.startswith("what does @src/auth.py do?")

    def test_text_without_mentions_is_untouched(self, project, boundary):
        message, problems = expand("fix the parser", project, boundary)
        assert message == "fix the parser"
        assert problems == []

    def test_escaping_the_scope_is_refused(self, project, boundary):
        """A mention is user input; the boundary is not waivable."""
        message, problems = expand("read @../../etc/passwd", project, boundary)
        assert "root:" not in message
        assert any("outside the current scope" in p for p in problems)

    def test_a_missing_file_is_reported_not_swallowed(self, project, boundary):
        """A mention that quietly did nothing is worse than one that says so."""
        _, problems = expand("read @nope.py", project, boundary)
        assert any("does not exist" in p for p in problems)

    def test_an_oversized_file_is_refused_with_a_reason(self, project, boundary):
        big = project / "big.txt"
        big.write_text("x" * 80_000)
        _, problems = expand("read @big.txt", project, boundary)
        assert any("too large" in p for p in problems)

    def test_a_directory_mention_gives_a_listing(self, project, boundary):
        message, _ = expand("what is in @src?", project, boundary)
        assert "auth.py" in message
        assert "directory" in message

    def test_too_many_mentions_are_capped_and_reported(self, project, boundary):
        for i in range(15):
            (project / f"f{i}.txt").write_text("x")
        text = " ".join(f"@f{i}.txt" for i in range(15))
        message, problems = expand(text, project, boundary)
        assert message.count("### ") <= 10
        assert any("first 10" in p for p in problems)

    def test_several_files_all_arrive(self, project, boundary):
        message, problems = expand("@src/auth.py and @README.md", project, boundary)
        assert "check_token" in message and "# hi" in message
        assert problems == []


class TestAMentionGoesThroughTheShieldToo:
    """The same file behaved two ways depending on how it was named.

    read_file refuses a .env and masks the key in a settings module. "@.env"
    inlined the whole thing, unmasked, into a message bound for a model that
    is often on another machine -- which is the exact thing the shield
    exists to stop.
    """

    def _project(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        (tmp_path / ".env").write_text("API_KEY=sk-live-abcdefghijklmnop\n")
        (tmp_path / "settings.py").write_text(
            'API_KEY = "sk-proj-abcdefghij1234567890"\n')
        return tmp_path

    def _expand(self, tmp_path, mention):
        from wynxo.mentions import expand
        from wynxo.scope import Scope, resolve
        from wynxo.secrets import Shield

        return expand(f"look at {mention}", tmp_path,
                      resolve(tmp_path, Scope.FOLDER), Shield(tmp_path))

    def test_a_credentials_file_is_not_inlined(self, tmp_path):
        self._project(tmp_path)
        text, problems = self._expand(tmp_path, "@.env")
        assert "sk-live" not in text
        assert any("credentials" in p for p in problems)

    def test_a_key_inside_an_ordinary_file_is_masked(self, tmp_path):
        self._project(tmp_path)
        text, problems = self._expand(tmp_path, "@settings.py")
        assert "sk-proj-abcdefghij1234567890" not in text
        assert "API_KEY" in text
        assert any("masked" in p for p in problems)

    def test_an_ordinary_file_is_untouched(self, tmp_path):
        self._project(tmp_path)
        text, problems = self._expand(tmp_path, "@app.py")
        assert "x = 1" in text
        assert problems == []

    def test_the_repl_passes_the_shield_in(self):
        import inspect

        from wynxo.cli import Repl

        source = inspect.getsource(Repl._expand_mentions)
        assert "shield" in source
