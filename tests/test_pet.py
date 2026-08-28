"""The companion face, and the voice that shapes how the agent talks.

The face lives inside a width-exact status bar, so anything that mismeasures
it makes the bar jitter on every frame. The voice edits the system prompt, so
anything that lets it excuse work is worse than not having it.
"""

import pytest
from rich.cells import cell_len

from wynxo.pet import (ACTIVITY_MOODS, FACES, FACES_ASCII, Mood, Pet,
                       REMARKS_MOMMY, face_width)
from wynxo.prompts import VOICES, build_system_prompt
from wynxo.ui import UI, ActivityBar


class TestFaces:
    @pytest.mark.parametrize("table,label", [(FACES, "unicode"), (FACES_ASCII, "ascii")])
    def test_every_mood_has_frames(self, table, label):
        for mood in Mood:
            assert table[mood], f"{label}: {mood.value} has no frames"

    @pytest.mark.parametrize("table,label", [(FACES, "unicode"), (FACES_ASCII, "ascii")])
    def test_frames_of_a_mood_are_the_same_width(self, table, label):
        """Frames of differing width make the bar shift on every blink."""
        for mood, frames in table.items():
            widths = {cell_len(f) for f in frames}
            assert len(widths) == 1, f"{label}/{mood.value}: widths {widths}"

    def test_width_counts_cells_not_codepoints(self):
        """Combining marks take no cell; CJK punctuation takes two."""
        assert face_width("(•ᴗ•)") == 5
        assert face_width("à") == 1     # combining grave
        assert face_width("・") == 2           # fullwidth

    def test_ascii_frames_are_pure_ascii(self):
        for frames in FACES_ASCII.values():
            for frame in frames:
                frame.encode("ascii")     # raises if not

    def test_padded_is_always_the_mood_width(self):
        pet = Pet()
        for mood in Mood:
            pet.react(mood)
            for _ in range(12):
                assert cell_len(pet.padded()) == pet.width()


class TestMoods:
    def test_activity_maps_to_a_mood(self):
        pet = Pet()
        pet.set_activity("reading")
        assert pet.mood is Mood.READING
        pet.set_activity("editing")
        assert pet.mood is Mood.WORKING
        pet.set_activity("running")
        assert pet.mood is Mood.RUNNING

    def test_unknown_activity_falls_back_to_thinking(self):
        """Honest default: something is happening and we did not label it."""
        pet = Pet()
        pet.set_activity("doing something novel")
        assert pet.mood is Mood.THINKING

    def test_empty_activity_rests(self):
        pet = Pet()
        pet.react(Mood.WORKING)
        pet.set_activity("")
        assert pet.mood is Mood.IDLE

    def test_every_mapped_activity_has_a_face(self):
        for mood in ACTIVITY_MOODS.values():
            assert FACES[mood] and FACES_ASCII[mood]

    def test_changing_mood_restarts_the_animation(self):
        pet = Pet()
        for _ in range(7):
            pet.face()
        pet.react(Mood.HAPPY)
        assert pet._frame == 0


class TestAnimationToggle:
    def test_a_still_pet_never_changes_frame(self):
        pet = Pet(animate=False)
        pet.react(Mood.THINKING)
        first = pet.face()
        assert all(pet.face() == first for _ in range(10))

    def test_an_animated_pet_does_change(self):
        pet = Pet(animate=True)
        pet.react(Mood.THINKING)
        seen = {pet.face() for _ in range(24)}
        assert len(seen) > 1


class TestBarIntegration:
    def test_pet_replaces_the_spinner(self):
        ui = UI()
        ui.width = 90
        bar = ActivityBar(ui, "medium", pet=Pet())
        bar.update(activity="reading", tokens=5)
        # Taken from the table rather than written out, so redesigning the
        # faces does not break a test about the bar.
        from wynxo.pet import FACES, Mood

        assert FACES[Mood.READING][0] in bar._render().plain

    def test_disabled_pet_falls_back_to_the_spinner(self):
        ui = UI()
        ui.width = 90
        bar = ActivityBar(ui, "medium", pet=Pet(enabled=False))
        bar.update(activity="reading", tokens=5)
        rendered = bar._render().plain
        assert "(" not in rendered.split("reading")[0]

    @pytest.mark.parametrize("width", [40, 60, 92, 160])
    def test_bar_stays_width_exact_with_a_pet(self, width):
        ui = UI()
        ui.width = width
        bar = ActivityBar(ui, "medium", "^O thinking", pet=Pet())
        for activity in ("thinking", "reading", "writing", "running", "verifying"):
            bar.update(activity=activity, detail="src/some/file.py", tokens=4321)
            for _ in range(8):
                assert bar._render().cell_len == width, f"{activity} at {width}"

    def test_tokens_survive_alongside_the_pet(self):
        ui = UI()
        ui.width = 48
        bar = ActivityBar(ui, "medium", "^O thinking  ^T detail", pet=Pet())
        bar.update(activity="editing", detail="a/long/path.py", tokens=1234)
        assert "1234 tok" in bar._render().plain


class TestVoice:
    def test_plain_adds_nothing(self):
        from pathlib import Path

        from wynxo.effort import resolve

        assert "## Voice" not in build_system_prompt(
            Path("."), resolve("low"), voice="plain")

    @pytest.mark.parametrize("voice", ["warm", "mentor", "blunt"])
    def test_a_voice_adds_a_block_and_the_floor(self, voice):
        from pathlib import Path

        from wynxo.effort import resolve

        prompt = build_system_prompt(Path("."), resolve("low"), voice=voice)
        assert "## Voice" in prompt
        # Whitespace-insensitive: the floor wraps across lines in the prompt.
        flat = " ".join(prompt.split())
        # The floor is what stops a personality becoming a liability.
        assert "never soften a failure" in flat
        assert "never imply something worked when it did not" in flat
        assert "never leave out what changed" in flat

    def test_an_unknown_voice_is_ignored_not_fatal(self):
        from pathlib import Path

        from wynxo.effort import resolve

        prompt = build_system_prompt(Path("."), resolve("low"), voice="pirate")
        assert "## Voice" not in prompt
        assert "You are wynxo" in prompt

    def test_no_voice_excuses_less_work(self):
        """A tone must not license skipping, softening or overclaiming."""
        for block in VOICES.values():
            lowered = block.lower()
            for forbidden in ("skip", "don't bother", "no need to check",
                              "shorter answer is fine"):
                assert forbidden not in lowered

    def test_mommy_is_a_voice(self):
        assert "mommy" in VOICES
        flat = " ".join(VOICES["mommy"].split()).lower()
        # The whole point of the voice: the user is her goodboy and she is
        # his mommy -- but the engineering floor is untouched.
        assert "goodboy" in flat
        assert "mommy" in flat
        assert "sugar-coating a failure" in flat

    def test_mommy_keeps_flourishes_out_of_machine_readable_text(self):
        flat = " ".join(VOICES["mommy"].split()).lower()
        for target in ("code", "file paths", "commit messages"):
            assert target in flat

    def test_mommy_voice_carries_the_honesty_floor(self):
        from pathlib import Path

        from wynxo.effort import resolve

        prompt = build_system_prompt(Path("."), resolve("low"), voice="mommy")
        flat = " ".join(prompt.split())
        assert "never soften a failure" in flat
        assert "never imply something worked when it did not" in flat

    def test_mommy_remarks_exist_for_every_event(self):
        from wynxo.pet import REMARKS

        assert set(REMARKS) == set(REMARKS_MOMMY)
        for event, lines in REMARKS_MOMMY.items():
            assert lines, event
            # The voice is for the user, not for the transcript noise:
            # remarks stay short enough for one line.
            assert all(len(line) < 60 for line in lines), event

    def test_mommy_pet_speaks_like_mommy(self):
        pet = Pet()
        pet.style_name = "mommy"
        greet = pet.remark("greet")
        assert greet
        assert "goodboy" in greet or "mommy" in greet

    def test_mommy_pet_uses_the_round_face_set(self):
        pet = Pet()
        pet.style_name = "mommy"
        assert pet.faces()[Mood.IDLE][0] == "₍ᐢ•ﻌ•ᐢ₎"


class TestConfig:
    def test_pet_settings_round_trip(self, tmp_path):
        import json

        from wynxo.config import Config

        config = Config(pet=False, pet_name="ada", voice="blunt", animations=False)
        path = config.save(tmp_path / "c.json")
        loaded = Config.validate(json.loads(path.read_text()))
        assert (loaded.pet, loaded.pet_name, loaded.voice, loaded.animations) == (
            False, "ada", "blunt", False)

    def test_an_invalid_voice_is_rejected(self):
        from wynxo.config import Config
        from wynxo.schema import ValidationError

        with pytest.raises(ValidationError):
            Config(voice="pirate")
