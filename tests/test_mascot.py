"""One cat, one size, and glyphs a terminal can be trusted with.

The cat is three rows of line art in the header, and the status strip
carries a single mark whose colour is the mood. That split is what these
pin: the drawing has to hold its block so the text beside it does not move
between moods, and the mark has to measure exactly one cell so the strip --
which is redrawn a dozen times a second on the bottom row of a live display
-- cannot tear on whichever frame a glyph appears in.

The drawing itself is ASCII, which is not a limitation but the point. The
face before it was a kaomoji, and it collected exactly the faults a
single-line Unicode face collects:

  * Frames were padded to the widest frame of the *current mood*, so idle
    (7 cells) to running (8) shifted the whole line at every mood change.

  * The eyes were U+2022 BULLET, East Asian Width "Ambiguous" -- one cell in
    a Western locale, two in a CJK one. So were the breve, the squint, the
    tears and the multiplication sign; one mood was built from combining
    accents.

  * The muzzle was U+FECC, an Arabic presentation form, whose bidi class is
    AL. A terminal implementing the bidirectional algorithm may reorder the
    neutrals either side of a strong RTL character -- which is to say, the
    eyes -- and the face comes apart.

Line art has none of those questions to answer, so what is left to check is
that the block holds and that the one remaining Unicode glyph, the status
mark, is safe.
"""

from __future__ import annotations

import unicodedata

import pytest
from rich.cells import cell_len

from wynxo.pet import (ACTIVITY_MOODS, EARS, FRAMES, HEIGHT, MARKS,
                       MARKS_ASCII, MOOD_ROLES, Mood, Pet, WIDTH)


def _every_row():
    for mood, frames in FRAMES.items():
        for index, (eyes, mouth) in enumerate(frames):
            yield mood, index, eyes
            yield mood, index, mouth


class TestTheBlockNeverChangesSize:
    def test_every_row_is_the_same_width(self):
        widths = {cell_len(row) for _mood, _i, row in _every_row()}
        assert widths == {WIDTH}, sorted(widths)
        assert cell_len(EARS) == WIDTH

    def test_the_drawing_is_the_block_in_every_mood(self):
        for unicode_ok in (True, False):
            pet = Pet(unicode=unicode_ok)
            for mood in Mood:
                pet.react(mood)
                for _ in range(8):
                    rows = pet.rows()
                    assert len(rows) == HEIGHT, mood.value
                    assert {cell_len(r) for r in rows} == {WIDTH}, mood.value

    def test_the_text_beside_the_mascot_never_moves(self):
        """The failure this is really about: everything to the right of the
        cat sitting still while the mood changes."""
        pet = Pet()
        starts = set()
        for mood in Mood:
            pet.react(mood)
            starts |= {cell_len(row + "  wynxo") for row in pet.rows()}
        assert len(starts) == 1, starts

    def test_the_status_mark_is_one_cell(self):
        """The strip is width-exact. A two-cell mark tears it."""
        for mood in Mood:
            assert cell_len(MARKS[mood]) == 1, mood.value
            assert cell_len(MARKS_ASCII[mood]) == 1, mood.value


class TestEveryGlyphIsSafeInALineOfText:
    @pytest.mark.parametrize("mood,index,row", list(_every_row()))
    def test_the_drawing_is_ascii(self, mood, index, row):
        """Which settles ambiguous width, combining marks and bidi at once:
        no ASCII character is any of those things."""
        row.encode("ascii")     # raises if not

    @pytest.mark.parametrize("mood", list(Mood))
    def test_the_status_mark_is_safe(self, mood):
        mark = MARKS[mood]
        for ch in mark:
            assert unicodedata.east_asian_width(ch) not in ("A", "W", "F"), (
                f"{mood.value}: {ch!r} measures one cell here and draws two "
                "in a CJK locale")
            assert not unicodedata.combining(ch), f"{mood.value}: {ch!r}"
            assert unicodedata.bidirectional(ch) not in ("AL", "R", "AN"), (
                f"{mood.value}: {ch!r} may reorder what sits beside it")
        MARKS_ASCII[mood].encode("ascii")


class TestItIsOneCharacter:
    def test_the_voice_does_not_change_the_species(self):
        pet = Pet()
        pet.react(Mood.IDLE)
        base = pet.rows(advance=False)
        for voice in ("mommy", "kawaii", "plain", "mentor", "blunt", ""):
            pet.style_name = voice
            assert pet.rows(advance=False) == base, voice

    def test_the_body_is_identical_in_every_frame(self):
        """Ears and outline hold still; only the middle three cells of each
        row change. That is what makes a frame change read as an expression
        rather than as a different animal: the parentheses that are the
        cheeks and the > < that is the muzzle are in the same place in
        every frame of every mood, so what moves is the face inside them.
        """
        skeletons = set()
        for frames in FRAMES.values():
            for eyes, mouth in frames:
                blanked = []
                for row in (eyes, mouth):
                    chars = list(row)
                    for slot in (2, 3, 4):
                        chars[slot] = "_"
                    blanked.append("".join(chars))
                skeletons.add(tuple(blanked))
        assert skeletons == {("( ___ )", " >___< ")}, skeletons

    def test_every_mood_is_drawn(self):
        for mood in Mood:
            assert FRAMES[mood], mood.value
            assert MARKS[mood], mood.value

    def test_reading_and_testing_do_not_look_alike(self):
        """They are different things to be doing and used to share a face."""
        assert set(FRAMES[Mood.READING]) & set(FRAMES[Mood.TESTING]) == set()


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
            assert FRAMES[mood], f"{activity} -> {mood.value}"

    def test_an_unknown_activity_is_survivable(self):
        pet = Pet()
        pet.set_activity("something nobody has mapped")
        assert pet.mood is Mood.THINKING
        assert {cell_len(row) for row in pet.rows()} == {WIDTH}


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
