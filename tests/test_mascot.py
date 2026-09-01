"""One cat, one size, and glyphs a terminal can be trusted with.

The mascot is an animation on the bottom row of a live display, which makes
it the one place in the interface where a glyph's *measured* width has to
match its *drawn* width. Everywhere else a bad guess costs a ragged margin;
here it tears the status line on whichever frame the glyph appears in, and
then repairs itself on the next, which reads as the terminal glitching.

Three separate faults, all of which this pins shut:

  * Frames were padded to the widest frame of the *current mood*. Inside a
    mood nothing moved; between moods idle (7 cells) to running (8) shifted
    the whole line, and the ASCII set moved between 5 and 7.

  * The eyes were U+2022 BULLET, East Asian Width "Ambiguous" -- one cell in
    a Western locale, two in a CJK one. So were the breve, the ≧≦ squint,
    the ╥ tears and the ×; WORKING was built from combining accents.

  * The muzzle was U+FECC, an Arabic presentation form, whose bidi class is
    AL. A terminal implementing the bidirectional algorithm may reorder the
    neutrals either side of a strong RTL character -- which is to say, the
    eyes -- and the face comes apart.
"""

from __future__ import annotations

import unicodedata

import pytest
from rich.cells import cell_len

from wynxo.pet import (ACTIVITY_MOODS, BOX, FACES, FACES_ASCII, MOOD_ROLES,
                       Mood, Pet)

TABLES = [(FACES, "unicode"), (FACES_ASCII, "ascii")]


def _every_frame():
    for table, label in TABLES:
        for mood, frames in table.items():
            for index, frame in enumerate(frames):
                yield label, mood, index, frame


class TestTheBoxNeverChangesSize:
    @pytest.mark.parametrize("table,label", TABLES)
    def test_every_frame_is_the_same_width(self, table, label):
        widths = {cell_len(f) for frames in table.values() for f in frames}
        assert widths == {BOX}, f"{label}: {sorted(widths)}"

    @pytest.mark.parametrize("table,label", TABLES)
    def test_both_tiers_agree_on_the_box(self, table, label):
        """A terminal that falls back to ASCII must not get a different
        layout, only a different drawing."""
        assert {cell_len(f) for frames in table.values() for f in frames} \
            == {BOX}

    def test_padded_is_the_box_in_every_mood(self):
        for unicode_ok in (True, False):
            pet = Pet(unicode=unicode_ok)
            for mood in Mood:
                pet.react(mood)
                for _ in range(8):
                    assert cell_len(pet.padded()) == BOX, mood.value

    def test_the_line_after_the_mascot_never_moves(self):
        """The failure this is really about: everything to the right of the
        mascot sitting still while the mood changes."""
        pet = Pet()
        starts = set()
        for mood in Mood:
            pet.react(mood)
            starts.add(cell_len(pet.padded() + "  activity"))
        assert len(starts) == 1, starts


class TestEveryGlyphIsSafeInALineOfText:
    @pytest.mark.parametrize("label,mood,index,frame", list(_every_frame()))
    def test_no_ambiguous_width(self, label, mood, index, frame):
        for ch in frame:
            width = unicodedata.east_asian_width(ch)
            assert width not in ("A", "W", "F"), (
                f"{label}/{mood.value}[{index}] {ch!r} is width {width}: it "
                "measures one cell here and draws two in a CJK locale")

    @pytest.mark.parametrize("label,mood,index,frame", list(_every_frame()))
    def test_no_combining_marks(self, label, mood, index, frame):
        for ch in frame:
            assert not unicodedata.combining(ch), (
                f"{label}/{mood.value}[{index}] {ch!r} is a combining mark")

    @pytest.mark.parametrize("label,mood,index,frame", list(_every_frame()))
    def test_nothing_reverses_the_reading_direction(self, label, mood, index,
                                                    frame):
        for ch in frame:
            bidi = unicodedata.bidirectional(ch)
            assert bidi not in ("AL", "R", "AN"), (
                f"{label}/{mood.value}[{index}] {ch!r} is bidi {bidi}: the "
                "eyes either side of it may be reordered")

    def test_the_ascii_tier_is_actually_ascii(self):
        for frames in FACES_ASCII.values():
            for frame in frames:
                frame.encode("ascii")


class TestItIsOneCharacter:
    def test_the_voice_does_not_change_the_species(self):
        pet = Pet()
        base = pet.faces()
        for voice in ("mommy", "kawaii", "plain", "mentor", "blunt", ""):
            pet.style_name = voice
            assert pet.faces() is base, voice

    @pytest.mark.parametrize("table,label", TABLES)
    def test_the_body_is_identical_in_every_frame(self, table, label):
        """Ears, muzzle and body hold still; only the eyes and the one
        accessory cell change. That is what makes a frame change read as an
        expression rather than as a different animal."""
        skeletons = set()
        for frames in table.values():
            for frame in frames:
                face = frame[:-1] if label == "unicode" else frame[:-1]
                # Blank the two eye positions, keep everything else.
                chars = list(face)
                for slot in (2, 4):
                    chars[slot] = "_"
                skeletons.add("".join(chars))
        assert len(skeletons) == 1, f"{label}: {skeletons}"

    def test_every_mood_is_drawn_in_both_tiers(self):
        for mood in Mood:
            assert FACES[mood], mood.value
            assert FACES_ASCII[mood], mood.value

    def test_reading_and_testing_do_not_look_alike(self):
        """They are different things to be doing and used to share a face."""
        assert set(FACES[Mood.READING]) & set(FACES[Mood.TESTING]) == set()


class TestTheColourComesFromTheTheme:
    def test_every_mood_has_a_role(self):
        for mood in Mood:
            assert mood in MOOD_ROLES, mood.value

    def test_a_role_is_a_palette_field_not_a_colour(self):
        from wynxo.theme import PURPLE

        for mood, role in MOOD_ROLES.items():
            assert hasattr(PURPLE, role), f"{mood.value} -> {role!r}"

    def test_switching_theme_switches_the_mascot(self):
        """It used to name literal colours -- grey62, bright_cyan -- so the
        mascot was the one thing /theme could not reach."""
        from wynxo import theme

        pet = Pet()
        pet.react(Mood.THINKING)
        theme.use(theme.resolve("purple"))
        purple = pet.style()
        theme.use(theme.resolve("catboy"))
        assert pet.style() != purple
        theme.use(theme.resolve("purple"))

    def test_trouble_never_looks_like_ordinary_work(self):
        """Whatever the theme, a sad cat must not be the busy colour."""
        from wynxo import theme

        pet = Pet()
        for name in theme.names():
            theme.use(theme.resolve(name))
            pet.react(Mood.SAD)
            sad = pet.style()
            pet.react(Mood.THINKING)
            assert sad != pet.style(), name
            pet.react(Mood.HAPPY)
            assert sad != pet.style(), name
        theme.use(theme.resolve("purple"))


class TestTheMoodAlwaysHasAFace:
    def test_every_mapped_activity_lands_somewhere_drawable(self):
        for activity, mood in ACTIVITY_MOODS.items():
            assert FACES[mood], f"{activity} -> {mood.value}"

    def test_an_unknown_activity_is_survivable(self):
        pet = Pet()
        pet.set_activity("something nobody has mapped")
        assert pet.mood is Mood.THINKING
        assert cell_len(pet.padded()) == BOX


class TestTheCompanionDoesNotRepeatItself:
    """Three or four lines per event and two or three uses per session: a
    plain random choice repeats often enough to be noticed, and a companion
    that greets you with the identical sentence every time you open it reads
    as a string constant rather than as a character."""

    def _pet(self, voice="mommy"):
        pet = Pet()
        pet.style_name = voice
        return pet

    def test_never_the_same_line_twice_running(self):
        for voice in ("default", "kawaii", "mommy"):
            pet = self._pet(voice)
            for event in ("greet", "done", "error", "bye", "proud"):
                seen = [pet.remark(event) for _ in range(20)]
                pairs = list(zip(seen, seen[1:]))
                assert all(a != b for a, b in pairs), (voice, event)

    def test_it_still_uses_the_whole_set(self):
        """Avoiding a repeat must not collapse to alternating two lines."""
        pet = self._pet()
        from wynxo.pet import REMARKS_MOMMY

        seen = {pet.remark("greet") for _ in range(200)}
        assert seen == set(REMARKS_MOMMY["greet"])

    def test_a_single_option_is_survivable(self):
        pet = self._pet()
        pet.style_name = "default"
        from wynxo import pet as pet_module

        pet_module.REMARKS["solo"] = ["only one"]
        try:
            assert pet.remark("solo") == "only one"
            assert pet.remark("solo") == "only one"
        finally:
            del pet_module.REMARKS["solo"]

    def test_an_unknown_event_is_silent(self):
        assert self._pet().remark("nothing-maps-to-this") == ""

    def test_a_disabled_pet_says_nothing(self):
        pet = self._pet()
        pet.enabled = False
        assert pet.remark("greet") == ""

    def test_the_companion_is_not_chatty(self):
        """It speaks at three moments in a whole session -- hello, goodbye,
        and a commit worth being pleased about. Anything that comments on
        every tool stops being charming after ten minutes."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli)
        assert source.count(".remark(") <= 3, "the companion gained a voice"
