"""Repository targets, /scope taking a path, and smooth streaming."""

import pytest

from wynxo.pet import FACES_ASCII, FACES_KAWAII, Mood, Pet
from wynxo.prompts import VOICES, build_system_prompt
from wynxo.repo import parse
from wynxo.scope import Scope
from wynxo.ui import UI, CodeStreamer, ThoughtStreamer


class TestRepoTargets:
    @pytest.mark.parametrize("raw,slug", [
        ("wynxo/agent", "wynxo/agent"),
        ("https://github.com/wynxo/agent", "wynxo/agent"),
        ("https://github.com/wynxo/agent.git", "wynxo/agent"),
        ("https://github.com/wynxo/agent/tree/main/src", "wynxo/agent"),
        ("git@github.com:wynxo/agent.git", "wynxo/agent"),
        ("https://github.com/wynxo/agent/", "wynxo/agent"),
    ])
    def test_shapes_people_paste(self, raw, slug):
        target = parse(raw)
        assert target is not None and target.slug == slug

    def test_ssh_url_is_kept_as_ssh(self):
        """Rewriting it to https would break key-based access."""
        assert parse("git@github.com:wynxo/agent.git").url.startswith("git@")

    def test_shorthand_becomes_https(self):
        assert parse("wynxo/agent").url == "https://github.com/wynxo/agent.git"

    @pytest.mark.parametrize("raw", ["", "   ", "not a repo at all/", "/", "https://"])
    def test_nonsense_is_rejected(self, raw):
        assert parse(raw) is None

    def test_cache_path_is_per_owner_and_name(self, monkeypatch, tmp_path):
        import wynxo.repo as module

        monkeypatch.setattr(module, "data_dir", lambda: tmp_path)
        assert module.parse("a/b").directory() == tmp_path / "repos" / "a" / "b"

    def test_clone_failure_explains_credentials(self):
        import wynxo.repo as module

        message = module._explain_clone_failure(
            parse("a/b"), "fatal: could not read Username for 'https://github.com'")
        assert "credential" in message.lower()
        assert "private" in message.lower()

    def test_clone_failure_explains_a_missing_repo(self):
        import wynxo.repo as module

        message = module._explain_clone_failure(
            parse("a/b"), "ERROR: Repository not found.")
        assert "not found" in message.lower()


class TestScopeTakesAPath:
    def test_the_three_words_still_parse(self):
        assert Scope.parse("folder") is Scope.FOLDER
        assert Scope.parse("repo") is Scope.REPO
        assert Scope.parse("machine") is Scope.MACHINE

    @pytest.mark.parametrize("raw", [
        r"C:\Users\elliot\Desktop\website",
        "/home/u/projects/site",
        "../sibling",
        "~/code",
    ])
    def test_a_path_is_not_mistaken_for_a_scope(self, raw):
        """It must raise so the caller can treat it as a directory instead of
        printing 'unknown scope' at someone who gave a perfectly good path."""
        with pytest.raises(KeyError):
            Scope.parse(raw)

    def test_cd_is_a_real_command(self):
        from wynxo.cli import COMMANDS

        assert "/cd" in COMMANDS and "/repo" in COMMANDS


class TestSmoothStreaming:
    def _stream(self, chunks, width=72, streamer=CodeStreamer):
        ui = UI()
        ui.width = width
        ui.console.width = width
        s = streamer(ui)
        for chunk in chunks:
            s.feed(chunk)
        s.finish()

    def test_a_long_paragraph_with_no_newline_still_prints(self, capsys):
        """It used to print nothing at all until a newline arrived."""
        words = [f"word{i} " for i in range(40)]
        self._stream(words)
        out = capsys.readouterr().out
        assert "word0" in out and "word39" in out

    def test_prose_wraps_at_the_terminal_width(self, capsys):
        self._stream([f"word{i} " for i in range(60)], width=60)
        for line in capsys.readouterr().out.splitlines():
            assert len(line) <= 60, repr(line)

    def test_words_split_across_chunks_are_rejoined(self, capsys):
        self._stream(["check", "_token", " handles ", "the None ", "case"])
        assert "check_token" in capsys.readouterr().out

    def test_code_is_still_highlighted_per_line(self, capsys):
        self._stream(["```python\n", "x = 1\n", "y = 2\n", "```\n"])
        out = capsys.readouterr().out
        assert out.count("x = 1") == 1
        assert "```" not in out

    def test_thinking_does_not_treat_backticks_as_code(self, capsys):
        """A scratchpad is full of stray fences that are not code blocks."""
        self._stream(["There is a ``` in ", "this thought ", "and it is fine."],
                     streamer=ThoughtStreamer)
        out = capsys.readouterr().out
        assert "and it is fine." in out

    def test_thinking_is_indented(self, capsys):
        self._stream(["some reasoning here"], streamer=ThoughtStreamer)
        printed = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert printed[0].startswith("    ")


class TestKawaii:
    def test_it_is_a_voice(self):
        assert "kawaii" in VOICES

    def test_the_honesty_floor_still_applies(self):
        from pathlib import Path

        from wynxo.effort import resolve

        prompt = build_system_prompt(Path("."), resolve("low"), voice="kawaii")
        flat = " ".join(prompt.split())
        assert "never soften a failure" in flat
        assert "never imply something worked when it did not" in flat

    def test_the_voice_itself_forbids_sugar_coating(self):
        flat = " ".join(VOICES["kawaii"].split()).lower()
        assert "sugar-coating a failure" in flat
        assert "exactly as thorough" in flat

    def test_it_keeps_flourishes_out_of_machine_readable_text(self):
        flat = " ".join(VOICES["kawaii"].split()).lower()
        for target in ("code", "file paths", "commit messages"):
            assert target in flat

    def test_kawaii_faces_are_width_consistent(self):
        from rich.cells import cell_len

        for mood, frames in FACES_KAWAII.items():
            assert len({cell_len(f) for f in frames}) == 1, mood.value

    def test_every_mood_has_a_kawaii_face(self):
        for mood in Mood:
            assert FACES_KAWAII[mood]

    def test_the_style_switches_the_face_set(self):
        pet = Pet()
        pet.react(Mood.HAPPY)
        default = pet.face(advance=False)
        pet.style_name = "kawaii"
        assert pet.face(advance=False) != default

    def test_ascii_terminals_still_get_ascii(self):
        pet = Pet(unicode=False)
        pet.style_name = "kawaii"
        pet.react(Mood.HAPPY)
        assert pet.faces() is FACES_ASCII
        pet.face(advance=False).encode("ascii")
