"""Status lines must degrade cleanly: no colour, no tty, no spinner."""

import io

from wynxo.status import Status, Timer


def render(**kwargs) -> str:
    stream = io.StringIO()
    stream.isatty = lambda: False
    status = Status(colour=False, stream=stream, **kwargs)
    return status, stream


class TestPlainOutput:
    def test_tags_are_aligned(self):
        status, stream = render()
        status.ok("a"); status.warn("b"); status.fail("c"); status.skip("d")
        lines = stream.getvalue().splitlines()
        assert lines[0].startswith("[  OK  ] ")
        assert lines[1].startswith("[ WARN ] ")
        assert lines[2].startswith("[FAILED] ")
        # Every tag is the same width, so the message column lines up.
        assert len({line.index("]") for line in lines}) == 1

    def test_detail_is_appended(self):
        status, stream = render()
        status.ok("ollama 0.12.0", "127.0.0.1:11434")
        assert "ollama 0.12.0 127.0.0.1:11434" in stream.getvalue()

    def test_note_is_indented_under_the_message(self):
        status, stream = render()
        status.warn("context small")
        status.note("raise it with /ctx")
        lines = stream.getvalue().splitlines()
        assert lines[1].startswith(" " * (lines[0].index("]") + 2))

    def test_no_escape_codes_without_colour(self):
        status, stream = render()
        status.ok("x"); status.fail("y")
        assert "\033" not in stream.getvalue()


class TestColour:
    def test_codes_appear_when_enabled(self):
        stream = io.StringIO()
        stream.isatty = lambda: False
        status = Status(colour=True, stream=stream)
        status.ok("x")
        assert "\033[1;32m" in stream.getvalue()

    def test_no_color_env_disables_it(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        from wynxo.status import _supports_colour
        assert not _supports_colour()


class TestSpinner:
    def test_busy_without_a_tty_prints_a_plain_line(self):
        status, stream = render()
        status.busy("working")
        status.ok("done")
        text = stream.getvalue()
        assert "working" in text and "done" in text

    def test_close_is_safe_when_nothing_is_spinning(self):
        status, _ = render()
        status.close()
        status.close()

    def test_context_manager_closes(self):
        stream = io.StringIO()
        stream.isatty = lambda: False
        with Status(colour=False, stream=stream) as status:
            status.ok("x")
        assert "[  OK  ] x" in stream.getvalue()


class TestTimer:
    def test_formats_by_magnitude(self, monkeypatch):
        timer = Timer()
        monkeypatch.setattr(timer, "started", timer.started - 0.05)
        assert timer.elapsed().endswith("ms")
        monkeypatch.setattr(timer, "started", timer.started - 5)
        assert timer.elapsed().endswith("s")
