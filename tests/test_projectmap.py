"""A one-page map of the codebase, so the model does not start blind.

The property that matters most: it may give up detail to fit the budget,
but it must never silently omit files. A map that lies about what is in the
project sends the model looking in the wrong place with confidence.
"""

import pytest

from wynxo import projectmap


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "def check_token(t):\n    return t\n\nclass TokenError(Exception):\n    pass\n")
    (tmp_path / "src" / "upload.py").write_text("def put(u):\n    pass\n")
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("function noise(){}")
    (tmp_path / "README.md").write_text("# docs\n")
    return tmp_path


class TestSymbols:
    def test_python_top_level_definitions(self):
        found = projectmap.symbols_in(
            "def a():\n    pass\nclass B:\n    def c(self):\n        pass\n", "python")
        assert found == ["a", "B"], "nested names are not top level"

    def test_private_names_are_left_out(self):
        assert projectmap.symbols_in("def _hidden():\n    pass\n", "python") == []

    def test_a_python_file_that_does_not_parse_is_survivable(self):
        assert projectmap.symbols_in("def (((", "python") == []

    def test_javascript_functions_and_arrow_consts(self):
        found = projectmap.symbols_in(
            "export function foo(){}\nexport const bar = async () => {}\n"
            "class Baz {}\n", "clike")
        assert set(found) == {"foo", "bar", "Baz"}

    def test_go_exported_functions(self):
        found = projectmap.symbols_in(
            "func Handle(w int) {}\nfunc helper() {}\n", "go")
        assert found == ["Handle"]

    def test_rust_and_ruby(self):
        assert "parse" in projectmap.symbols_in("pub fn parse() {}", "rust")
        assert "Widget" in projectmap.symbols_in("class Widget\nend", "ruby")

    def test_duplicates_are_collapsed(self):
        found = projectmap.symbols_in("def a():\n pass\ndef a():\n pass\n", "python")
        assert found == ["a"]


class TestWalk:
    def test_it_finds_the_source_files(self, project):
        names = {p.name for p in projectmap.walk(project)}
        assert {"auth.py", "upload.py"} <= names

    def test_noise_directories_are_skipped(self, project):
        paths = [str(p) for p in projectmap.walk(project)]
        assert not any("node_modules" in p for p in paths)
        assert not any(".git" in p for p in paths)

    def test_an_unreadable_directory_is_not_fatal(self, tmp_path):
        assert projectmap.walk(tmp_path / "nope") == []


class TestBuild:
    def test_files_and_their_symbols_appear(self, project):
        out = projectmap.build(project)
        assert "src/auth.py" in out
        assert "check_token" in out and "TokenError" in out

    def test_an_empty_project_maps_to_nothing(self, tmp_path):
        assert projectmap.build(tmp_path) == ""

    def test_detail_is_dropped_before_files_are(self, tmp_path, monkeypatch):
        """The map may get plainer to fit, but every file must still be in
        it -- one that silently omits half the project is worse than none."""
        for i in range(60):
            (tmp_path / f"mod{i}.py").write_text(
                "".join(f"def fn{j}():\n    pass\n" for j in range(10)))
        monkeypatch.setattr(projectmap, "MAX_CHARS", 1500)

        out = projectmap.build(tmp_path)
        assert len(out) <= 1500
        for i in range(60):
            assert f"mod{i}.py" in out, f"mod{i}.py was dropped"


class TestCache:
    def test_it_is_written_where_it_can_be_read(self, project):
        projectmap.load(project)
        assert projectmap.cache_path(project).exists()
        assert "auth.py" in projectmap.cache_path(project).read_text()

    def test_an_unchanged_project_reuses_the_cache(self, project):
        first = projectmap.load(project)
        projectmap.cache_path(project).write_text(first + "\nSENTINEL\n")
        assert "SENTINEL" in projectmap.load(project)

    def test_editing_a_file_rebuilds_it(self, project):
        projectmap.load(project)
        projectmap.cache_path(project).write_text("# stale\nSENTINEL\n")
        import os
        import time

        target = project / "src" / "auth.py"
        target.write_text("def brand_new():\n    pass\n")
        os.utime(target, (time.time() + 10, time.time() + 10))

        out = projectmap.load(project)
        assert "SENTINEL" not in out
        assert "brand_new" in out

    def test_a_read_only_project_still_gets_a_map(self, project, monkeypatch):
        """An unwritable checkout should lose the cache, not the feature."""
        def refuse(*args, **kwargs):
            raise OSError("read-only")

        monkeypatch.setattr(projectmap.Path, "write_text", refuse)
        assert "auth.py" in projectmap.load(project)


class TestSummary:
    def test_it_counts_the_files(self, project):
        assert "2 files mapped" in projectmap.summarise(projectmap.build(project))

    def test_nothing_maps_to_nothing(self):
        assert projectmap.summarise("") == ""
