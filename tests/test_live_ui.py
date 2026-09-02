"""Streaming render and the activity bar."""

from wynxo.ui import ActivityBar, CodeStreamer, UI


class TestCodeStreamer:
    def _feed(self, chunks):
        ui = UI()
        streamer = CodeStreamer(ui)
        for chunk in chunks:
            streamer.feed(chunk)
        streamer.finish()
        return streamer

    def test_plain_prose_streams(self, capsys):
        self._feed(["Hello ", "there.\n"])
        assert "Hello there." in capsys.readouterr().out

    def test_fenced_code_is_highlighted_once(self, capsys):
        self._feed(["```python\n", "x = 1\n", "```\n"])
        out = capsys.readouterr().out
        # Exactly one rendering: the dim preview only exists on a real tty.
        assert out.count("x = 1") == 1
        assert "```" not in out

    def test_language_is_picked_up(self):
        streamer = CodeStreamer(UI())
        streamer.feed("```rust\nfn main() {}\n")
        assert streamer.language == "rust"

    def test_unterminated_block_still_flushes(self, capsys):
        """A turn cut off mid-block must not swallow the code."""
        self._feed(["```python\n", "y = 2\n"])
        assert "y = 2" in capsys.readouterr().out

    def test_prose_around_code(self, capsys):
        self._feed(["Before.\n", "```py\n", "z = 3\n", "```\n", "After.\n"])
        out = capsys.readouterr().out
        assert "Before." in out and "z = 3" in out and "After." in out

    def test_partial_chunks_reassemble(self, capsys):
        # Models split tokens anywhere, including mid-word and mid-fence.
        self._feed(["Th", "e ans", "wer is ", "42.\n"])
        assert "The answer is 42." in capsys.readouterr().out


class TestActivityBar:
    def test_renders_the_essentials(self):
        """What it is doing, what it is doing it to, and whether anything is
        still arriving."""
        bar = ActivityBar(UI(), "high", "^O thinking")
        bar.update(activity="editing", detail="src/auth.py", tokens=120)
        # The whole pinned region: the activity and the file it is acting on
        # sit in the scene above the strip where there is room for one, and
        # in the strip itself where there is not.
        text = "\n".join([t.plain for t in bar._scene()]
                         + [bar._render().plain])
        assert "editing" in text
        assert "src/auth.py" in text
        assert "120 tok" in text

    def test_hint_is_dropped_on_a_narrow_screen(self):
        ui = UI()
        ui.width = 46
        bar = ActivityBar(ui, "low", "^O thinking  ^T detail")
        bar.update(tokens=812)
        assert "^O thinking" not in bar._render().plain

    def test_tokens_survive_when_space_is_tight(self):
        """Live tokens are the point of the bar; they go last, not first."""
        ui = UI()
        ui.width = 46
        bar = ActivityBar(ui, "low", "^O thinking  ^T detail")
        bar.update(activity="editing", detail="src/a.py", tokens=812)
        assert "812 tok" in bar._render().plain

    def test_bar_fills_the_width(self):
        """A filled strip is what makes it read as pinned rather than as a
        line that scrolled past."""
        ui = UI()
        ui.width = 80
        bar = ActivityBar(ui, "medium")
        bar.update(tokens=100)
        assert bar._render().cell_len == 80

    def test_left_side_is_truncated_not_the_stats(self):
        ui = UI()
        ui.width = 50
        bar = ActivityBar(ui, "medium")
        bar.update(activity="editing", detail="a/very/long/path/that/keeps/going.py",
                   tokens=999)
        rendered = bar._render().plain
        assert "999 tok" in rendered
        assert rendered.rstrip().endswith(("tok", "s", "%", "tok/s"))

    def test_rate_is_not_reported_before_it_is_meaningful(self):
        bar = ActivityBar(UI(), "low")
        bar.tokens = 5
        assert bar.rate() == 0.0     # too early to divide by
        bar.started -= 10
        assert bar.rate() > 0

    def test_start_stop_without_a_terminal_is_safe(self):
        bar = ActivityBar(UI(), "low")
        bar.start()
        bar.update(activity="x")
        bar.refresh()
        bar.stop()
        bar.stop()

    def test_spinner_advances(self):
        """The strip's own sign of life, for when it is the only row.

        With a scene above it the companion is what moves, and a second
        moving thing six rows from the first is not reassurance."""
        ui = UI()
        ui.width = 40          # too narrow for a scene: the strip owns it
        bar = ActivityBar(ui, "low")
        first = bar._render().plain[:6]
        second = bar._render().plain[:6]
        assert first != second

    def test_long_detail_is_clipped(self):
        bar = ActivityBar(UI(), "low")
        bar.update(detail="x" * 300)
        assert len(bar._render().plain) < 200

    def test_settings_are_not_repeated_during_a_turn(self):
        """The model, the effort level and the context percentage do not
        change while a turn runs, and all three are on the idle strip under
        the prompt. Claiming space for them here -- which they did, *before*
        the activity did -- left "what is it doing" as one word adrift at
        the left of an eighty-column line."""
        bar = ActivityBar(UI(), "medium", model="qwen3-coder:30b")
        bar.update(activity="thinking", tokens=10, context_pct=12.0)
        text = bar._render().plain
        assert "qwen3-coder:30b" not in text
        assert "medium" not in text
        assert "ctx" not in text

    def test_a_context_window_filling_up_is_news(self):
        """The one setting that becomes news: past the threshold a
        compaction is coming, and that is worth the space."""
        bar = ActivityBar(UI(), "medium")
        bar.update(activity="thinking", context_pct=88.0)
        assert "ctx 88%" in bar._render().plain

    def test_the_clock_moves_in_the_first_second(self):
        """Whole seconds alone read as stopped for the first second of every
        turn, which is exactly when somebody is watching to see whether
        anything happened at all."""
        bar = ActivityBar(UI(), "medium")
        bar.started -= 0.4
        assert bar._clock() == "0.4s"
        bar.started -= 30
        assert bar._clock() == "30s", "a long turn does not need a decimal"

    def test_the_activity_survives_a_narrow_screen(self):
        """The one thing that may never be dropped."""
        for width in (40, 46, 60):
            ui = UI()
            ui.width = width
            bar = ActivityBar(ui, "low", "^O thinking  ^T detail",
                              model="qwen3-coder:30b")
            bar.update(activity="editing", detail="src/a.py", tokens=812)
            live = "\n".join([t.plain for t in bar._scene()]
                             + [bar._render().plain])
            assert "editing" in live, width
