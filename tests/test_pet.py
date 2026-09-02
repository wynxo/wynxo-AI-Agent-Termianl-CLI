"""The companion, and the voice that shapes how the agent talks.

The cat is a three-row drawing in the header, so a frame of a different size
makes the header jump between moods. In the status strip it is a single mark
whose colour carries the state, so nothing there can shift the line's width.
The voice edits the system prompt, so anything that lets it excuse work is
worse than not having it.
"""

import pytest
from rich.cells import cell_len

from wynxo.pet import (ACTIVITY_MOODS, EARS, FRAMES, HEIGHT, MARKS,
                       MARKS_ASCII, Mood, Pet, REMARKS_MOMMY, WIDTH)
from wynxo.prompts import VOICES, build_system_prompt
from wynxo.ui import UI, ActivityBar


class TestTheDrawing:
    def test_every_mood_has_frames(self):
        for mood in Mood:
            assert FRAMES[mood], f"{mood.value} has no frames"

    def test_every_frame_is_the_same_size(self):
        """A frame of a different size makes the header jump.

        Every row of every frame of every mood is one width, so the cat
        occupies the same block whatever it is doing and the text set
        beside it never moves.
        """
        for mood, frames in FRAMES.items():
            for eyes, mouth in frames:
                for row in (EARS, eyes, mouth):
                    assert cell_len(row) == WIDTH, f"{mood.value}: {row!r}"

    def test_the_cat_is_drawn_in_ascii(self):
        """One drawing, not two.

        The face used to be a kaomoji with a whole second table beside it
        for terminals that could not render it -- two sets of frames to
        keep in step, and the fallback was always the worse of the two.
        Line art made of slashes and parentheses is legible everywhere and
        needs no second tier, so there is one cat to maintain and it is the
        same cat on every terminal.
        """
        EARS.encode("ascii")
        for frames in FRAMES.values():
            for eyes, mouth in frames:
                eyes.encode("ascii")      # raises if not
                mouth.encode("ascii")

    def test_rows_are_always_the_same_block(self):
        pet = Pet()
        for mood in Mood:
            pet.react(mood)
            for _ in range(12):
                rows = pet.rows()
                assert len(rows) == HEIGHT
                assert all(cell_len(r) == WIDTH for r in rows)

    def test_the_mark_is_one_cell_for_every_mood(self):
        """The status strip is width-exact: a two-cell mark shifts it."""
        for mood in Mood:
            assert cell_len(MARKS[mood]) == 1, mood.value
            assert cell_len(MARKS_ASCII[mood]) == 1, mood.value
            MARKS_ASCII[mood].encode("ascii")


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
            assert FRAMES[mood] and MARKS[mood]

    def test_changing_mood_restarts_the_animation(self):
        pet = Pet()
        for _ in range(7):
            pet.rows()
        pet.react(Mood.HAPPY)
        assert pet._frame == 0


class TestAnimationToggle:
    def test_a_still_pet_never_changes_frame(self):
        pet = Pet(animate=False)
        pet.react(Mood.THINKING)
        first = pet.rows()
        assert all(pet.rows() == first for _ in range(10))

    def test_an_animated_pet_does_change(self):
        pet = Pet(animate=True)
        pet.react(Mood.THINKING)
        seen = {tuple(pet.rows()) for _ in range(24)}
        assert len(seen) > 1


class TestBarIntegration:
    def test_pet_replaces_the_spinner(self):
        ui = UI()
        ui.width = 90
        bar = ActivityBar(ui, "medium", pet=Pet())
        bar.update(activity="reading", tokens=5)
        # The strip carries one cell, not the drawing: three rows of cat in
        # a one-row status line was never possible. While a tool runs that
        # cell breathes, so what is on the strip is a pulse frame; its
        # colour is the mood. Taken from the table rather than written out,
        # so redesigning the cat does not break a test about the bar.
        from wynxo.pet import PULSE

        assert any(f" {frame} " in bar._render().plain for frame in PULSE)

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
    def test_plain_is_a_voice_too(self):
        """Plain is no longer "no voice": it got a human tone block, but
        that block must forbid the support-bot filler, not add to it."""
        from pathlib import Path

        from wynxo.effort import resolve

        prompt = build_system_prompt(Path("."), resolve("low"), voice="plain")
        assert "## Voice" in prompt
        flat = " ".join(prompt.split()).lower()
        # The block's job: forbid the support-bot default, not become it.
        assert "support bot" in flat
        assert "corporate filler" in flat

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
        for target in ("code", "paths", "commit messages"):
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

    def test_every_voice_has_a_farewell(self):
        from wynxo.pet import REMARKS, REMARKS_KAWAII, REMARKS_MOMMY
        assert "bye" in REMARKS and REMARKS["bye"]
        assert "bye" in REMARKS_KAWAII and REMARKS_KAWAII["bye"]
        assert "bye" in REMARKS_MOMMY and REMARKS_MOMMY["bye"]

    def test_every_voice_has_a_proud_moment(self):
        from wynxo.pet import REMARKS, REMARKS_KAWAII, REMARKS_MOMMY
        assert "proud" in REMARKS and REMARKS["proud"]
        assert "proud" in REMARKS_KAWAII and REMARKS_KAWAII["proud"]
        assert "proud" in REMARKS_MOMMY and REMARKS_MOMMY["proud"]

    def test_mommy_farewell_keeps_the_affection(self):
        pet = Pet()
        pet.style_name = "mommy"
        bye = pet.remark("bye")
        assert bye
        assert "goodboy" in bye or "mommy" in bye

    def test_farewell_is_silent_when_the_pet_is_off(self):
        pet = Pet()
        pet.enabled = False
        assert pet.remark("bye") == ""
        assert pet.remark("proud") == ""

    def test_the_voice_does_not_change_the_face(self):
        """There is one cat. Voices change the words, not the animal."""
        pet = Pet()
        pet.react(Mood.IDLE)
        base = pet.rows(advance=False)
        for voice in ("mommy", "kawaii", "plain", "mentor", "blunt"):
            pet.style_name = voice
            assert pet.rows(advance=False) == base, voice


class TestConfig:
    def test_pet_settings_round_trip(self, tmp_path):
        import json

        from wynxo.config import Config

        config = Config(pet=False, pet_name="ada", voice="blunt", animations=False)
        path = config.save(tmp_path / "c.json")
        loaded = Config.validate(json.loads(path.read_text()))
        assert (loaded.pet, loaded.pet_name, loaded.voice, loaded.animations) == (
            False, "ada", "blunt", False)

    def test_mommy_is_the_default_voice(self):
        """The product's own voice: the user is the goodboy, she is the
        mommy -- and it is what you get without asking for it."""
        from wynxo.config import Config

        assert Config().voice == "mommy"

    def test_an_invalid_voice_is_rejected(self):
        from wynxo.config import Config
        from wynxo.schema import ValidationError

        with pytest.raises(ValidationError):
            Config(voice="pirate")
