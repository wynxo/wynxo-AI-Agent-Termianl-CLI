"""The companion, and the voice that shapes how the agent talks.

The cat is a three-row drawing in the header, so a frame of a different size
makes the header jump between moods. In the status strip it is a single mark
whose colour carries the state, so nothing there can shift the line's width.
The voice edits the system prompt, so anything that lets it excuse work is
worse than not having it.
"""

import pytest

from wynxo.pet import Pet, REMARKS_MOMMY
from wynxo.prompts import VOICES, build_system_prompt


class TestTheVoiceDoesNotDraw:
    """pet.py is the companion's name and the lines it says. Nothing else.

    It used to hold the picture too: a face, a mood, a frame counter and a
    pace that followed the effort level. The picture is ``sprite.py`` now
    and what it is doing is read from the agent, so the split here is not
    tidying -- it is what stops the presentation layer from keeping its own
    opinion about whether work is happening.
    """

    def test_it_has_no_frames_and_no_mood(self):
        pet = Pet()
        for gone in ("rows", "mark", "block", "style", "react", "rest",
                     "set_activity", "set_pace", "mood", "pace"):
            assert not hasattr(pet, gone), f"{gone} is back on Pet"

    def test_what_it_does_have_is_a_name_and_a_voice(self):
        pet = Pet(name="wyn", style_name="mommy")
        assert pet.name == "wyn"
        assert pet.remark("greet")
        assert pet.greeting().startswith("wyn — ")

    def test_the_states_belong_to_the_agent(self):
        """The companion's state is derived from the running tool and the
        task state, not stored on the character."""
        from wynxo.companion import State, state_for

        assert state_for("edit_file", "executing") is State.CODING
        assert state_for("read_file", "completed") is State.SUCCESS


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

    def test_the_voice_does_not_change_the_character(self):
        """There is one cat. Voices change the words, not the animal.

        The voice cannot reach the drawing at all now -- they are different
        modules and the sprite is not passed a voice -- so this asks the
        question the old test was really asking: does anything about the
        picture depend on how the companion talks."""
        import inspect

        from wynxo import sprite

        source = inspect.getsource(sprite)
        for word in ("kawaii", "mommy", "style_name", "voice", "Pet"):
            assert word not in source, f"{word} reaches the drawing"


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
