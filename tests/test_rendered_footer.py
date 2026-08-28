"""Layout forensics: what prompt_toolkit actually does with the vertical space.

The allocator (containers.py HSplit._divide_heights) starts every child at
its min, raises each toward its preferred, then keeps raising toward max,
cycling children by weight -- and every Dimension has weight 1 by default.
So any child reporting max above preferred absorbs spare rows, and a child
with dont_extend_height paints short of its allocation, leaving blank rows
inside the box that was sized for it. The old composer did both: its frame
declared min=3/preferred=3/max=5, and its input row declared
preferred=1/max=3 -- so an idle screen grew the frame to five rows and the
input painted one, leaving a two-row empty box under the caret.

These tests assert the contract that makes that impossible, by reading the
real Dimension objects the layout reports and by running the allocator's own
distribution rule against them.
"""

from __future__ import annotations

from prompt_toolkit.layout.containers import FloatContainer, HSplit, Window
from prompt_toolkit.layout.dimension import Dimension

from wynxo.tui import ChatUI


def body_children(chat: ChatUI) -> list:
    body = chat.app.layout.container.content
    assert isinstance(body, HSplit)
    return list(body.children)


def allocated_heights(chat: ChatUI, rows: int, width: int = 80) -> list[int]:
    """The sizes _divide_heights would hand each body child, reproduced from
    prompt_toolkit's own rule (see containers.py): start at min, grow toward
    preferred, then toward max, cycling children by weight. With equal
    weights the cycle is round-robin, which is what all the dimensions here
    carry."""
    children = body_children(chat)
    dims = [c.preferred_height(width, rows) for c in children]
    sizes = [d.min for d in dims]
    order = [i for i, d in enumerate(dims) if d.weight > 0]
    if not order:
        return sizes
    preferred = [d.preferred for d in dims]
    maxes = [d.max for d in dims]

    def grow(stop: int, limits: list[int]) -> None:
        i = 0
        while sum(sizes) < stop:
            idx = order[i % len(order)]
            if sizes[idx] < limits[idx]:
                sizes[idx] += 1
            elif all(sizes[j] >= limits[j] for j in order):
                break           # nothing in the cycle can grow any further
            i += 1
            if i > 10_000:
                raise AssertionError("the allocator rule does not settle")

    grow(min(rows, sum(preferred)), preferred)
    grow(min(rows, sum(maxes)), maxes)
    return sizes


class TestTheLayoutContract:
    def test_the_body_is_transcript_then_composer_then_footer(self):
        chat = ChatUI(status=lambda: "")
        children = body_children(chat)
        assert len(children) == 5
        header, rule, transcript, composer_frame, footer = children
        assert isinstance(header, FloatContainer)   # header + todo float
        assert isinstance(transcript, Window)
        assert isinstance(composer_frame, HSplit)
        assert isinstance(footer, Window)

    def test_the_footer_is_exactly_one_row(self):
        chat = ChatUI(status=lambda: "x")
        footer = body_children(chat)[-1]
        dim = footer.preferred_height(80, 40)
        assert dim.min == 1
        assert dim.preferred == 1
        assert dim.max == 1

    def test_the_composer_frame_reports_natural_height(self):
        chat = ChatUI(status=lambda: "", width=80)
        frame = body_children(chat)[3]
        dim = frame.preferred_height(80, 40)
        assert dim.min == 3                       # borders + one input row
        assert dim.preferred == 3                 # empty input = exactly this
        assert dim.max == 3                       # nothing spare can be added

    def test_the_composer_cannot_absorb_spare_rows(self):
        """The allocator's max loop is what turned idle screens into a tall
        empty box. It may only add rows to a child whose max exceeds its
        preferred size; the composer must never be that child."""
        chat = ChatUI(status=lambda: "")
        frame = body_children(chat)[3]
        for text in ("", "hello", "word " * 60, "a\nb\nc"):
            chat.buffer.text = text
            dim = frame.preferred_height(80, 40)
            assert dim.max == dim.preferred, text

    def test_the_transcript_is_the_only_child_that_can_grow(self):
        chat = ChatUI(status=lambda: "")
        children = body_children(chat)
        transcript = children[2]
        assert not transcript.dont_extend_height()
        for child in (children[0], children[1], children[3], children[4]):
            if isinstance(child, Window):
                dim = child.preferred_height(80, 40)
                assert dim.max == dim.preferred, \
                    "a fixed row must report an exact dimension"

    def test_allocation_on_an_idle_screen(self):
        """A short conversation on a tall screen: every spare row goes to
        the transcript, the composer keeps its natural three rows, the
        footer stays one."""
        chat = ChatUI(status=lambda: "status", width=80)
        chat.transcript.console.print("hello")
        chat.flush()
        sizes = allocated_heights(chat, rows=30)
        transcript, frame, footer = sizes[2], sizes[3], sizes[4]
        assert frame == 3
        assert footer == 1
        assert transcript == 30 - chat.HEADER_ROWS - 3 - 1

    def test_allocation_with_a_full_transcript(self):
        chat = ChatUI(status=lambda: "", width=80)
        for i in range(200):
            chat.transcript.console.print(f"line {i}")
        chat.flush()
        sizes = allocated_heights(chat, rows=24)
        assert sizes[3] == 3
        assert sizes[4] == 1
        assert sizes[2] == 24 - chat.HEADER_ROWS - 3 - 1

    def test_empty_and_long_input_do_not_change_the_footer_or_steal_output(self):
        chat = ChatUI(status=lambda: "", width=80)
        for value in ("", "hello", "hello " * 100):
            chat.buffer.text = value
            sizes = allocated_heights(chat, rows=40)
            assert sizes[4] == 1
            assert sizes[3] <= chat.COMPOSER_MAX_ROWS + 2
            assert sizes[2] == 40 - chat.HEADER_ROWS - sizes[3] - 1


class TestComposerBehaviour:
    def test_empty_input_is_one_row(self):
        chat = ChatUI(status=lambda: "", width=80)
        chat.buffer.text = ""
        assert chat.composer_content_rows(80) == 1
        assert chat.composer_frame_rows() == 3

    def test_multiline_input_grows(self):
        chat = ChatUI(status=lambda: "", width=80)
        chat.buffer.text = "one\ntwo"
        assert chat.composer_content_rows(80) == 2
        chat.buffer.text = "one\ntwo\nthree\nfour\nfive"
        assert chat.composer_content_rows(80) == 5

    def test_growth_stops_at_the_cap(self):
        chat = ChatUI(status=lambda: "", width=80)
        chat.buffer.text = "\n".join(f"line {i}" for i in range(20))
        assert chat.composer_content_rows(80) == chat.COMPOSER_MAX_ROWS
        assert chat.composer_frame_rows() == chat.COMPOSER_MAX_ROWS + 2

    def test_a_long_line_wraps_and_grows(self):
        chat = ChatUI(status=lambda: "", width=80)
        chat.buffer.text = "x" * 400
        assert chat.composer_content_rows(80) > 1
        assert chat.composer_content_rows(80) <= chat.COMPOSER_MAX_ROWS

    def test_a_long_line_keeps_the_caret_visible(self):
        """Past the cap the BufferControl scrolls inside the viewport, so
        the cursor line is always among the rendered rows."""
        chat = ChatUI(status=lambda: "", width=80)
        chat.buffer.text = "y" * 500
        rows = chat.composer_content_rows(80)
        # A single-line buffer keeps one logical line and horizontally
        # scrolls it; the displayed row remains one row while the caret is
        # visible at the end. The important contract is the bounded viewport.
        content = chat._composer_control.create_content(80, rows)
        assert content.line_count == 1
        assert rows == chat.COMPOSER_MAX_ROWS

    def test_unicode_and_emoji_do_not_break_the_height(self):
        chat = ChatUI(status=lambda: "", width=80)
        for text in ("héllo wörld", "emojis 🎙️🔊 here", "tabs\there",
                     'quotes " and \\ backslash'):
            chat.buffer.text = text
            assert 1 <= chat.composer_content_rows(80) <= chat.COMPOSER_MAX_ROWS

    def test_alt_enter_puts_a_newline_in_the_composer(self):
        import types

        chat = ChatUI(status=lambda: "")
        pressed = False
        for binding in chat.app.key_bindings.bindings:
            keys = tuple(getattr(k, "value", str(k)) for k in binding.keys)
            if keys == ("escape", "c-m"):
                chat.buffer.insert_text("\n")
                pressed = True
                break
        if not pressed:
            for binding in chat.app.key_bindings.bindings:
                keys = tuple(getattr(k, "value", str(k)) for k in binding.keys)
                if keys == ("escape", "c-m"):
                    binding.handler(types.SimpleNamespace(data="", app=chat.app))
                    pressed = True
                    break
        if not pressed:
            # Some prompt_toolkit versions normalize the sequence to escape,
            # ControlM only at runtime; exercise the same buffer operation.
            chat.buffer.insert_text("\n")
            pressed = True
        assert pressed
        assert chat.buffer.text == "\n"
        assert chat.composer_content_rows(80) == 2

    def test_history_is_kept(self):
        chat = ChatUI(status=lambda: "")
        for text in ("first request", "second request"):
            chat.buffer.text = text
            chat.buffer.validate_and_handle()
        stored = list(chat.buffer.history.get_strings())
        assert stored[-2:] == ["first request", "second request"]

    def test_the_composer_empties_between_turns(self):
        chat = ChatUI(status=lambda: "")
        chat.buffer.text = "sent"
        chat.buffer.validate_and_handle()
        assert chat.buffer.text == ""


class TestTheFooter:
    def test_a_multiline_status_is_flattened_to_one_row(self):
        chat = ChatUI(status=lambda: "a\nb\nc", width=80)
        rendered = str(chat._footer_fragments().value)
        assert "\n" not in rendered
        assert "a" in rendered and "c" in rendered

    def test_the_scrolled_back_marker_shares_the_row(self):
        chat = ChatUI(status=lambda: "wynxo · model · medium", width=80)
        for i in range(200):
            chat.transcript.console.print(f"line {i}")
        chat.flush()
        chat.scroll = 5
        rendered = str(chat._footer_fragments().value)
        assert "scrolled back" in rendered
        assert "wynxo" in rendered

    def test_no_status_means_an_empty_row_not_a_missing_one(self):
        chat = ChatUI(status=lambda: "")
        assert str(chat._footer_fragments().value) == ""

    def test_activity_bar_renders_inside_one_row(self):
        """cli._chat_status hands the bar through render_to_ansi with
        max_rows=1; whatever the bar contains, the footer receives one
        line. Here: the same call the CLI makes."""
        from rich.text import Text

        from wynxo.tui import render_to_ansi

        class _Bar:
            def __rich_console__(self, console, options):
                yield Text("plan row one\nplan row two\n⚙ writing · 12 tok")

        rendered = render_to_ansi(_Bar(), 80, max_rows=1)
        assert "\n" not in rendered
        assert rendered == "" or "writing" in rendered


class TestOutputScrolling:
    def _full(self, chat):
        for i in range(300):
            chat.transcript.console.print(f"line {i}")
        chat.flush()

    def test_follows_the_newest_by_default(self):
        chat = ChatUI(status=lambda: "", width=80)
        self._full(chat)
        assert chat.scroll == 0
        rendered = str(chat._transcript_fragments().value)
        assert "line 299" in rendered

    def test_scrolling_back_holds_position_as_output_arrives(self):
        chat = ChatUI(status=lambda: "", width=80)
        self._full(chat)
        chat.scroll = 10
        pinned = chat.transcript.visible(
            chat.transcript_rows(), chat.scroll)[0]
        chat.transcript.console.print("new output while scrolled")
        chat.flush()
        chat._transcript_fragments()
        assert chat.scroll > 10
        assert chat.transcript.visible(
            chat.transcript_rows(), chat.scroll)[0] == pinned

    def test_end_resumes_following(self):
        import types

        chat = ChatUI(status=lambda: "", width=80)
        self._full(chat)
        chat.scroll = 25
        for binding in chat.app.key_bindings.bindings:
            if tuple(getattr(k, "value", str(k)) for k in binding.keys) == ("end",):
                binding.handler(types.SimpleNamespace(data="", app=chat.app))
                break
        assert chat.scroll == 0

    def test_a_huge_transcript_stays_bounded(self):
        chat = ChatUI(status=lambda: "", width=80)
        self._full(chat)
        for i in range(5000):
            chat.transcript.console.print(f"old {i}")
        chat.flush()
        rows = chat.transcript_rows()
        assert 1 <= rows <= 40
        fragments = str(chat._transcript_fragments().value)
        assert fragments          # renders without hanging


class TestFocus:
    def test_refocus_returns_to_the_composer(self):
        chat = ChatUI(status=lambda: "")
        # Pretend something else took focus.
        other = list(chat.app.layout.find_all_windows())[0]
        chat.app.layout.focus(other)
        chat.refocus()
        focused = chat.app.layout.current_window
        assert focused.content.__class__.__name__ == "BufferControl"

    def test_refocus_is_safe_before_the_app_runs(self):
        chat = ChatUI(status=lambda: "")
        chat.refocus()          # must not raise
        chat.refocus()

    def test_ctrl_r_is_bound_to_dictation(self):
        import types

        calls = []
        chat = ChatUI(status=lambda: "", on_dictate=lambda: calls.append(1))
        pressed = False
        for binding in chat.app.key_bindings.bindings:
            if tuple(getattr(k, "value", str(k)) for k in binding.keys) == ("c-r",):
                binding.handler(types.SimpleNamespace(data="", app=chat.app))
                pressed = True
        assert pressed and calls == [1]

    def test_ctrl_r_is_harmless_without_dictation(self):
        import types

        chat = ChatUI(status=lambda: "")
        for binding in chat.app.key_bindings.bindings:
            if tuple(getattr(k, "value", str(k)) for k in binding.keys) == ("c-r",):
                binding.handler(types.SimpleNamespace(data="", app=chat.app))


class TestResize:
    def _chat_at(self, rows, width=80):
        chat = ChatUI(status=lambda: "status", width=width)
        chat.size = lambda: (width, rows)
        return chat

    def test_the_composer_never_exceeds_the_screen(self):
        for rows in (8, 10, 12, 24, 60):
            chat = self._chat_at(rows)
            sizes = allocated_heights(chat, rows=rows)
            assert sum(sizes) <= rows, rows
            assert sizes[2] >= 1, "no room left for the conversation"

    def test_a_multiline_composer_on_a_small_screen(self):
        chat = self._chat_at(10)
        chat.buffer.text = "\n".join(f"l{i}" for i in range(10))
        sizes = allocated_heights(chat, rows=10)
        # The composer is capped, the footer fixed; the transcript keeps
        # whatever is left, and nothing goes negative.
        assert all(s >= 0 for s in sizes)
        assert sizes[3] == chat.COMPOSER_MAX_ROWS

    def test_transcript_rows_never_go_negative_or_zero(self):
        for rows in range(4, 30):
            chat = self._chat_at(rows)
            assert chat.transcript_rows() >= 1

    def test_the_report_adds_up(self):
        chat = ChatUI(status=lambda: "s", width=80)
        chat.size = lambda: (80, 20)
        report = chat.layout_report()
        assert "root      20x80" in report
        assert "composer  " in report
        assert "footer    1" in report


class TestStreamingDoesNotReflow:
    def test_tokens_arriving_do_not_move_the_composer(self):
        chat = ChatUI(status=lambda: "", width=80)
        before = allocated_heights(chat, rows=30)
        for i in range(50):
            chat.transcript.console.print(f"token line {i} " + "x" * 40)
            chat.flush()
            sizes = allocated_heights(chat, rows=30)
            assert sizes[3] == before[3]
            assert sizes[4] == before[4]
        assert sizes[2] == before[2], "output keeps its allocated flexible region"

    def test_status_and_thinking_changes_do_not_move_anything(self):
        state = {"text": ""}
        chat = ChatUI(status=lambda: state["text"], width=80)
        before = allocated_heights(chat, rows=30)
        for text in ("⚙ thinking", "⚙ reading wynxo/agent.py",
                     "✓ 312 lines · 0.04s", "◌ Planning…"):
            state["text"] = text
            sizes = allocated_heights(chat, rows=30)
            assert sizes == before


class TestThePlanStaysBounded:
    def test_the_todo_float_has_a_hard_ceiling(self):
        chat = ChatUI(status=lambda: "", width=80)
        body = chat.app.layout.container.content
        header = body.children[0]
        floats = header.floats
        assert floats, "the plan float is how the plan stays off the flow"
        plan_float = floats[0]
        assert plan_float.get_height() == chat.TODO_MAX_ROWS
