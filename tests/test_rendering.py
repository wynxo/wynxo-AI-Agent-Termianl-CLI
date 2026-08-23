"""What the terminal actually shows.

These exist because two rendering bugs shipped and neither was catchable by
the other tests: one only happened on a real terminal, and one turned every
colour code into visible garbage without failing anything.
"""

import inspect

import pytest

from wynxo.ui import UI, ActivityBar, CodeStreamer


class TestAnsiIsNotMangled:
    """prompt_toolkit's patch_stdout routes output through Vt100_Output.write,
    which replaces every ESC byte with "?" as an escape-injection guard. Under
    it, every colour code from rich and from the status lines rendered as
    literal "?[1;32m" on screen. raw=True uses write_raw instead."""

    def test_patch_stdout_is_used_in_raw_mode(self):
        from wynxo import cli

        source = inspect.getsource(cli.amain)
        assert "patch_stdout(raw=True)" in source, (
            "patch_stdout() without raw=True turns every escape code into "
            'literal "?[...m" text'
        )

    def test_prompt_toolkit_still_escapes_without_raw(self):
        """Pin the upstream behaviour this guards against, so the day it
        changes, this test says so rather than the fix silently being moot."""
        from prompt_toolkit.output.vt100 import Vt100_Output

        source = inspect.getsource(Vt100_Output.write)
        assert '"?"' in source or "'?'" in source


class TestCodeStreaming:
    """The earlier implementation printed a dim preview, then rewound the
    cursor to overwrite it. That crashed on rich 15 (no Control.clear_lines)
    and only on a real terminal, so every non-tty test passed."""

    def _render(self, chunks):
        ui = UI()
        streamer = CodeStreamer(ui)
        for chunk in chunks:
            streamer.feed(chunk)
        streamer.finish()

    def test_no_cursor_control_is_used(self):
        """Checked as code, not text -- the docstring explaining why this is
        avoided obviously mentions it."""
        import ast

        from wynxo import ui as ui_module

        tree = ast.parse(inspect.getsource(ui_module))
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        for forbidden in ("clear_lines", "console.control", "Control.move"):
            assert not any(forbidden in call for call in calls), forbidden
        imported = {
            alias.name
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "Control" not in imported

    def test_code_renders_once(self, capsys):
        self._render(["```python\n", "x = 1\n", "y = 2\n", "```\n"])
        out = capsys.readouterr().out
        assert out.count("x = 1") == 1
        assert out.count("y = 2") == 1
        assert "```" not in out

    def test_fences_never_reach_the_screen(self, capsys):
        self._render(["a\n", "```js\n", "let x = 1\n", "```\n", "b\n"])
        assert "```" not in capsys.readouterr().out

    def test_unclosed_block_still_prints(self, capsys):
        self._render(["```python\n", "z = 3\n"])
        assert "z = 3" in capsys.readouterr().out

    def test_language_aliases_normalise(self):
        from wynxo.ui import _language

        assert _language("py") == "python"
        assert _language("sh") == "bash"
        assert _language("") == "text"
        assert _language("rust") == "rust"

    def test_an_unknown_language_does_not_raise(self, capsys):
        self._render(["```nonsense-lang\n", "some text\n", "```\n"])
        assert "some text" in capsys.readouterr().out

    def test_code_survives_chunks_split_mid_token(self, capsys):
        self._render(["```py", "thon\n", "def f", "oo():\n", "    pa", "ss\n", "``", "`\n"])
        out = capsys.readouterr().out
        assert "def foo():" in out and "pass" in out
        assert "```" not in out


class TestPinnedBar:
    def test_bar_is_exactly_one_line(self):
        ui = UI()
        ui.width = 80
        bar = ActivityBar(ui, "medium")
        bar.update(tokens=42)
        assert "\n" not in bar._render().plain

    def test_bar_fills_the_terminal_width(self):
        for width in (40, 60, 80, 120, 200):
            ui = UI()
            ui.width = width
            bar = ActivityBar(ui, "medium")
            bar.update(activity="writing", tokens=1234)
            assert bar._render().cell_len == width, f"at {width}"

    def test_tokens_are_always_shown(self):
        """The live counter is the point; it must never be the thing dropped."""
        for width in (40, 50, 72, 100, 160):
            ui = UI()
            ui.width = width
            bar = ActivityBar(ui, "medium", "^O thinking  ^T detail")
            bar.update(activity="editing", detail="a/long/path/name.py", tokens=777)
            assert "777 tok" in bar._render().plain, f"dropped at {width}"

    def test_counter_advances_per_chunk(self):
        bar = ActivityBar(UI(), "low")
        for _ in range(5):
            bar.add_token()
        assert bar.tokens == 5

    @pytest.mark.parametrize("width", [30, 45, 80, 200])
    def test_never_wraps(self, width):
        ui = UI()
        ui.width = width
        bar = ActivityBar(ui, "ultra", "^O thinking  ^T detail")
        bar.update(activity="verifying", detail="round 2/4", tokens=99999)
        assert bar._render().cell_len <= width


class TestHeader:
    def test_header_is_two_lines(self, capsys):
        UI().banner("qwen3-coder:30b", "http://127.0.0.1:11434", "medium", "/tmp/p")
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == 2, "the header should be a line and a rule, not a box"

    def test_narrow_header_drops_parts_rather_than_truncating(self, capsys):
        ui = UI()
        ui.width = 44
        ui.banner("qwen3-coder:30b", "http://192.168.1.50:11434", "medium",
                  "/home/u/code/project")
        out = capsys.readouterr().out
        assert "qwen3-coder:30b" in out
        assert "192.168.1.50" not in out, "the server should go before the model does"

    def test_no_stray_escape_codes_in_plain_mode(self, capsys):
        UI().banner("m", "http://127.0.0.1:11434", "low", "/tmp/p")
        assert "?[" not in capsys.readouterr().out
