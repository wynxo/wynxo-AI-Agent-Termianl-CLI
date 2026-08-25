"""Turning a picture into text.

The conversion is a measurement, not a drawing: each cell of the image is
reduced to a brightness and given a character of matching visual weight.
Decode and render are separate, so the rendering half is testable without
an image library at all.
"""

from __future__ import annotations

import pytest

from wynxo.asciiart import (BLOCKS, CELL_ASPECT, RAMP, from_image, normalise,
                            ramp_for, render)


def grid(rows):
    return [list(row) for row in rows]


class TestTheRamp:
    def test_dark_and_light_land_at_the_ends(self):
        out = render(grid([[0.0, 1.0]]), style="simple")
        assert out[0] == " " and out[-1] == "@"

    def test_brightness_increases_along_the_ramp(self):
        """Each step must weigh more than the last, or the picture reads as
        noise rather than shading."""
        ramp = ramp_for("simple")
        out = render(grid([[i / 9 for i in range(10)]]), style="simple")
        assert list(out) == list(ramp)

    def test_inverting_flips_it_for_a_light_terminal(self):
        assert render(grid([[0.0]]), style="simple", invert=True) == "@"

    def test_values_outside_the_range_are_clamped(self):
        """A stretched grid can overshoot slightly; that must not index off
        the end of the ramp."""
        assert render(grid([[-5.0, 5.0]]), style="simple") == " @"

    def test_the_block_ramp_is_available(self):
        assert ramp_for("blocks") == BLOCKS
        assert ramp_for("anything else") == RAMP

    def test_trailing_space_is_trimmed(self):
        """Otherwise every line carries the width of the image in blanks,
        which matters when this is pasted into a source file."""
        assert render(grid([[1.0, 0.0, 0.0]]), style="simple") == "@"

    def test_an_empty_grid_is_survivable(self):
        assert render([]) == ""


class TestStretchingTheRange:
    def test_a_dim_picture_is_opened_up(self):
        """A webcam photo in a dim room uses a fraction of the range, and
        mapped straight on it comes out as an even wash of mid-tones."""
        stretched = normalise(grid([[0.30, 0.35, 0.40]]))
        assert stretched[0][0] == pytest.approx(0.0)
        assert stretched[0][-1] == pytest.approx(1.0)

    def test_a_flat_image_is_left_alone(self):
        """Dividing by a zero range would be a crash, and there is nothing
        to reveal anyway."""
        flat = grid([[0.5, 0.5]])
        assert normalise(flat) == flat

    def test_an_empty_grid_is_survivable(self):
        assert normalise([]) == []


class TestReadingAPicture:
    def _write(self, path, width, height, pixel):
        body = bytearray()
        for y in range(height):
            for x in range(width):
                body += bytes(pixel(x, y))
        path.write_bytes(b"P6\n%d %d\n255\n" % (width, height) + bytes(body))
        return path

    def test_a_gradient_comes_out_as_a_gradient(self, tmp_path):
        source = self._write(tmp_path / "g.ppm", 64, 64,
                             lambda x, y: (x * 4, x * 4, x * 4))
        art = from_image(source, width=32, style="simple")
        first = art.splitlines()[0]
        assert first[0] == " " and first.rstrip()[-1] == "@"

    def test_the_picture_is_not_stretched_lengthways(self, tmp_path):
        """A terminal cell is about twice as tall as it is wide. Sampling on
        a square grid is the usual reason homemade ASCII art looks melted."""
        source = self._write(tmp_path / "sq.ppm", 100, 100,
                             lambda x, y: (128, 128, 128))
        art = from_image(source, width=100)
        rows = len(art.splitlines())
        assert rows == pytest.approx(100 / CELL_ASPECT, abs=1)

    def test_colour_is_weighted_the_way_an_eye_weighs_it(self, tmp_path):
        """A plain average makes a red shirt and a blue one the same grey."""
        red = self._write(tmp_path / "r.ppm", 8, 8, lambda x, y: (255, 0, 0))
        blue = self._write(tmp_path / "b.ppm", 8, 8, lambda x, y: (0, 0, 255))
        # Compared before stretching, which would flatten both to one value.
        from wynxo.asciiart import load

        assert load(red, 4)[0][0] > load(blue, 4)[0][0]

    def test_a_comment_in_the_header_is_skipped(self, tmp_path):
        source = tmp_path / "c.ppm"
        source.write_bytes(b"P6\n# written by something\n2 2\n255\n"
                           + bytes([200] * 12))
        assert from_image(source, width=2)

    def test_a_truncated_file_does_not_raise(self, tmp_path):
        """Half a download should give a poor picture, not a crash."""
        source = tmp_path / "t.ppm"
        source.write_bytes(b"P6\n64 64\n255\n" + bytes([100] * 30))
        assert isinstance(from_image(source, width=16), str)

    def test_a_missing_library_says_what_to_install(self, tmp_path,
                                                    monkeypatch):
        import builtins

        from wynxo.asciiart import ImageSupportMissing, load

        real = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("no PIL")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        source = tmp_path / "photo.jpg"
        source.write_bytes(b"not really a jpeg")
        with pytest.raises(ImageSupportMissing) as caught:
            load(source, 40)
        assert "pip install pillow" in str(caught.value)


class TestTheCommandLine:
    def test_the_flags_exist(self):
        from wynxo.cli import build_parser

        args = build_parser().parse_args(["--ascii", "p.png",
                                          "--ascii-width", "60",
                                          "--ascii-style", "blocks"])
        assert args.ascii == "p.png" and args.ascii_width == 60
        assert args.ascii_style == "blocks"

    def test_it_runs_before_the_configuration_gate(self):
        """Converting a local picture needs no model and no server, so being
        unconfigured is irrelevant -- asking the user to run setup first
        would be nonsense."""
        import inspect

        from wynxo import cli

        source = inspect.getsource(cli.amain)
        assert source.index("args.ascii") < source.index("is_configured")

    def test_a_missing_file_is_reported_not_raised(self, tmp_path, capsys):
        from wynxo.cli import _print_ascii
        from wynxo.ui import UI

        args = type("Args", (), {"ascii": str(tmp_path / "nope.png"),
                                 "ascii_width": 40, "ascii_style": "detail",
                                 "ascii_invert": False})()
        assert _print_ascii(args, UI()) == 1

    def test_a_silly_width_is_clamped(self, tmp_path, capsys):
        from wynxo.cli import _print_ascii
        from wynxo.ui import UI

        source = tmp_path / "g.ppm"
        source.write_bytes(b"P6\n4 4\n255\n" + bytes([120] * 48))
        args = type("Args", (), {"ascii": str(source), "ascii_width": 100000,
                                 "ascii_style": "detail",
                                 "ascii_invert": False})()
        assert _print_ascii(args, UI()) == 0
        assert len(capsys.readouterr().out.splitlines()[0]) <= 400
