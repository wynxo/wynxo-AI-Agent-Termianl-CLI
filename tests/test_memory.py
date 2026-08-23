"""Memory must stay small and fast: it is inlined into every system prompt,
so anything that lets it grow is a context tax paid on every turn."""

import time

import pytest

from wynxo.memory import MAX_PROJECT_CHARS, Memory, _already_known, _content_words


@pytest.fixture
def memory(tmp_path):
    return Memory(tmp_path / "project", tmp_path / "user")


class TestRemembering:
    def test_adds_an_entry(self, memory):
        added, message = memory.remember("Tests run with pytest -q")
        assert added and "pytest" in message
        assert memory.counts() == (1, 0)

    def test_user_scope_is_separate(self, memory):
        memory.remember("Project fact")
        memory.remember("Prefers terse answers", scope="user")
        assert memory.counts() == (1, 1)
        assert "Prefers terse" in memory.user.body()
        assert "Prefers terse" not in memory.project.body()

    def test_persists_to_disk(self, tmp_path):
        first = Memory(tmp_path / "p", tmp_path / "u")
        first.remember("Uses ruff for linting")
        second = Memory(tmp_path / "p", tmp_path / "u")
        assert "ruff" in second.project.body()

    def test_empty_note_is_rejected(self, memory):
        added, _ = memory.remember("   ")
        assert not added

    def test_whitespace_is_normalised(self, memory):
        memory.remember("uses\n\n  pytest   heavily")
        assert "uses pytest heavily" in memory.project.body()

    def test_long_notes_are_capped(self, memory):
        memory.remember("x" * 5000)
        assert len(memory.project.entries()[0]) < 600


class TestDeduplication:
    def test_exact_repeat_is_rejected(self, memory):
        memory.remember("Tests run with pytest -q")
        added, message = memory.remember("Tests run with pytest -q")
        assert not added and "equivalent" in message
        assert memory.counts() == (1, 0)

    def test_restatement_is_rejected(self, memory):
        """Stopwords must not disguise a duplicate."""
        memory.remember("This project uses pytest, run with pytest -q")
        added, _ = memory.remember("the project uses pytest and is run with pytest -q")
        assert not added

    def test_genuinely_different_notes_are_kept(self, memory):
        memory.remember("Uses pytest for tests")
        added, _ = memory.remember("Uses ruff for linting")
        assert added
        assert memory.counts() == (2, 0)

    def test_stopwords_are_stripped(self):
        assert _content_words("the project is a thing") == {"project", "thing"}

    def test_overlap_threshold(self):
        assert _already_known("uses pytest for tests", ["uses pytest for tests"])
        assert not _already_known("uses pytest", ["uses docker"])


class TestForgetting:
    def test_removes_matching_entries(self, memory):
        memory.remember("Uses pytest")
        memory.remember("Uses docker for local dev")
        count, _ = memory.forget("docker")
        assert count == 1
        assert memory.counts() == (1, 0)
        assert "pytest" in memory.project.body()

    def test_is_case_insensitive(self, memory):
        memory.remember("Uses Docker")
        assert memory.forget("docker")[0] == 1

    def test_no_match_reports_clearly(self, memory):
        memory.remember("Uses pytest")
        count, message = memory.forget("kubernetes")
        assert count == 0 and "kubernetes" in message

    def test_forget_on_empty_memory(self, memory):
        assert memory.forget("anything")[0] == 0


class TestSizeCap:
    def test_file_stays_under_its_limit(self, memory):
        for i in range(400):
            memory.remember(f"Fact number {i} about some distinct subsystem {i}")
        assert len(memory.project.read()) <= MAX_PROJECT_CHARS

    def test_oldest_entries_are_dropped_first(self, memory):
        memory.remember("The very first fact about alpha subsystem")
        for i in range(400):
            memory.remember(f"Later fact {i} concerning distinct component {i}")
        body = memory.project.body()
        assert "very first fact" not in body
        assert "Later fact 399" in body

    def test_prompt_section_stays_bounded(self, memory):
        for i in range(400):
            memory.remember(f"Fact {i} about distinct area {i}")
        # This is inlined into every request, so it must not balloon.
        assert len(memory.prompt_section()) < MAX_PROJECT_CHARS + 2000


class TestPromptSection:
    def test_empty_when_nothing_remembered(self, memory):
        assert memory.prompt_section() == ""

    def test_includes_both_scopes(self, memory):
        memory.remember("Project uses pytest")
        memory.remember("User prefers short answers", scope="user")
        section = memory.prompt_section()
        assert "About the user" in section and "About this project" in section
        assert "pytest" in section and "short answers" in section

    def test_does_not_echo_the_boilerplate_header(self, memory):
        memory.remember("A real fact")
        section = memory.prompt_section()
        assert "Keep entries short and true" not in section
        assert "A real fact" in section

    def test_tells_the_model_memory_can_be_wrong(self, memory):
        memory.remember("Something")
        assert "contradicts" in memory.prompt_section()


class TestSpeed:
    def test_loading_is_fast(self, memory):
        """No index, no embeddings: reading must be trivially cheap."""
        for i in range(200):
            memory.remember(f"Fact {i} about area {i}")
        started = time.monotonic()
        for _ in range(50):
            memory.prompt_section()
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"50 loads took {elapsed:.2f}s"


class TestHeaderIsNotContent:
    """An earlier version stripped the boilerplate header by matching its
    wording, which silently ate any note containing the same phrases."""

    @pytest.mark.parametrize("note", [
        "Keep it short when writing commit messages",
        "Delete anything that stops being accurate in the changelog",
        "Facts about this codebase live in docs/architecture.md",
        "Preferences and working habits are documented in CONTRIBUTING.md",
        "Keep entries short in the release notes",
    ])
    def test_notes_echoing_the_header_survive(self, memory, note):
        added, _ = memory.remember(note)
        assert added
        assert note in memory.project.body()
        assert note in memory.prompt_section()

    def test_header_prose_never_reaches_the_model(self, memory):
        memory.remember("A real fact")
        section = memory.prompt_section()
        assert "Keep entries short and true" not in section
        assert "Facts about this codebase worth carrying" not in section

    def test_body_is_only_the_entries(self, memory):
        memory.remember("First")
        memory.remember("Second")
        assert memory.project.body().splitlines() == ["- First", "- Second"]
