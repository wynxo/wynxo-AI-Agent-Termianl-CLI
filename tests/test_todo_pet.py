"""The top-right task panel, the living pet scene, and notifications.

The panel is a float: it reserves nothing, so none of this may ever change
the composer or the transcript height. The tests below pin that contract
alongside the rendering of each panel state.
"""

from __future__ import annotations

import time

from wynxo.tui import ChatUI
from wynxo import motion


def make_chat(pet_mood="idle", pet_on=True, pet_animate=True) -> ChatUI:
    return ChatUI(
        status=lambda: "",
        unicode=True,
        pet_state=lambda: pet_mood if pet_on else "",
        pet_enabled=lambda: pet_on,
        pet_animate=lambda: pet_animate,
    )


def text_of(fragments) -> str:
    """The visible text of (style, text) fragment lists, exactly as the
    renderer lays it out: fragments concatenate on one line, and every row
    but the last carries its own trailing newline."""
    return "".join(text for _, text in fragments).rstrip("\n")


class TestFragmentContract:
    """The top-right block feeds a FormattedTextControl, and prompt_toolkit
    treats a bare list as (style, text) tuples: every entry is unpacked into
    (style, text, *rest), so a plain string's first character becomes a
    style name. A "✓" toast used to crash the renderer with
    "Wrong color format '✓'"; the panel must always emit proper tuples."""

    def test_fragments_are_style_text_tuples(self):
        chat = make_chat(pet_mood="coding")
        chat.set_todos("[x] inspect\n[>] edit\n[ ] test", title="Fix")
        chat.notify("✓ saved diff")
        for style, text in chat._todo_fragments():
            assert isinstance(style, str) and isinstance(text, str)
            assert style, "every fragment carries a style class"

    def test_fragments_survive_to_formatted_text(self):
        from prompt_toolkit.formatted_text import to_formatted_text
        chat = make_chat(pet_mood="error")
        chat.set_todos("[!] pytest failed\n[>] investigate", title="Fix")
        chat.notify("✕ stopped")
        converted = to_formatted_text(chat._todo_fragments())
        assert converted, "renderer conversion must not raise"
        assert all(isinstance(style, str) and isinstance(text, str)
                   for style, text in converted)


class TestTheFloatRendersAsRows:
    """The float feeds a FormattedTextControl, which lays fragments out on
    one line unless the text carries newlines. A flat list used to render
    the pet, toast and panel as one wrapped blob in the real application;
    every row must land on its own screen line."""

    def _screen_rows(self, chat) -> list[str]:
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.renderer import Renderer

        class Capture(DummyOutput):
            def __init__(self):
                self.data = ""

            def write(self, data):
                self.data += data

            def flush(self):
                pass

        out = Capture()
        renderer = Renderer(chat.app.style, out, full_screen=True)
        renderer.render(chat.app, chat.app.layout, is_done=False)
        buf = renderer._last_screen.data_buffer
        rows = []
        for y in range(16):
            row = buf[y]
            cells = []
            for x in range(80):
                ch = row.get(x, None)
                cells.append(ch.char if ch else " ")
            rows.append("".join(cells))
        return rows

    def test_the_pet_and_panel_each_keep_their_own_rows(self):
        chat = make_chat(pet_mood="idle")
        chat.size = lambda: (80, 24)
        chat.set_todos("[x] inspect\n[>] edit\n[ ] test", title="Fix")
        rows = self._screen_rows(chat)
        # The pet face, its tail and the panel's box each occupy distinct
        # screen rows on the right-hand side -- never one wrapping line.
        joined = "\n".join(row.rstrip() for row in rows)
        pet_face = chat._pet_lines()[0]
        assert joined.index(pet_face) < joined.index("\u256d")  # ╭ above the box
        assert "\u256d" in joined and "\u256f" in joined       # ╭ and ╯ both present
        assert joined.index("\u256d") < joined.index("\u256f")   # box is upright

    def test_panel_rows_are_distinct_lines_not_a_blob(self):
        chat = make_chat(pet_mood="idle")
        chat.size = lambda: (80, 24)
        chat.set_todos("[x] inspect\n[>] edit\n[ ] test", title="Fix")
        rows = self._screen_rows(chat)
        for row in rows:
            assert len(row.rstrip()) <= 80, "no wrapped overflow past the edge"
        # Every panel marker sits on its own row: no row carries two markers.
        for row in rows:
            marker_count = sum(row.count(m) for m in ("\u2713", "\u22c6"))
            assert marker_count <= 1, f"two progress markers on one row: {row!r}"


class TestTodoPanelStates:
    def test_no_plan_means_no_panel(self):
        chat = make_chat()
        assert chat._todo_panel() == []
        # The living pet still sits there; the panel itself is gone.
        assert text_of(chat._todo_fragments()) == "\n".join(chat._pet_lines())

    def test_pet_disabled_and_no_plan_renders_nothing(self):
        chat = make_chat(pet_on=False)
        assert chat._todo_fragments() == []

    def test_progress_markers_render(self):
        chat = make_chat()
        chat.set_todos("[x] inspect\n[>] edit\n[ ] test\n[!] pytest")
        text = text_of(chat._todo_panel())
        assert "✓ inspect" in text
        assert "edit" in text and "✕ pytest" in text
        assert "1/4" in text          # one done of four
        assert "1 failed" in text

    def test_panel_is_a_bordered_box(self):
        chat = make_chat()
        chat.set_todos("[x] a\n[>] b\n[ ] c")
        lines = [text for _, text in chat._todo_panel()]
        assert lines[0].startswith("╭")
        assert lines[-1].startswith("╰")
        assert all(line.startswith("│") for line in lines[1:-1])

    def test_title_comes_from_task_objective(self):
        chat = make_chat()
        chat.set_todos("[>] edit", title="Fix the launcher")
        assert "Fix the launcher" in text_of(chat._todo_panel())

    def test_compact_mode_is_one_line(self):
        chat = make_chat()
        chat.set_todos("[x] a\n[>] b\n[ ] c", title="Fix")
        chat.set_todo_mode("compact")
        panel = chat._todo_panel()
        assert len(panel) == 1
        style, text = panel[0]
        assert style == "class:todo-title"
        assert "Fix" in text and "1/3" in text

    def test_hidden_mode_removes_the_panel(self):
        chat = make_chat()
        chat.set_todos("[>] b")
        chat.set_todo_mode("hidden")
        assert chat._todo_fragments() == []      # panel gone, pet hidden too

    def test_ascii_fallback_borders(self):
        chat = ChatUI(status=lambda: "", unicode=False,
                      pet_state=lambda: "", pet_enabled=lambda: False,
                      pet_animate=lambda: True)
        chat.set_todos("[x] a\n[>] b")
        lines = [text for _, text in chat._todo_panel()]
        assert lines[0].startswith("+")
        assert lines[-1].startswith("+")
        assert lines[1].startswith("|")

    def test_tight_screen_collapses_to_compact_instead_of_clipping(self):
        """The float's measured budget is smaller than the full block on a
        short terminal. The panel must fall back to its one-line form -- a
        box whose bottom edge was clipped off would read as a bug."""
        chat = make_chat(pet_mood="coding")
        chat.set_todos("\n".join(f"[x] step {i}" for i in range(8)),
                       title="Fix")
        chat._float_budget = 6      # what _float_height measured on a short screen
        frags = chat._todo_fragments()
        assert len(frags) <= 6, "block stays inside the measured budget"
        text = text_of(frags)
        assert "✦ Fix" in text, "compact summary appears"
        assert "╭" not in text, "no half-clipped box"


class TestPetScene:
    def test_pet_scene_follows_the_mood(self):
        chat = make_chat(pet_mood="working")
        lines = chat._pet_lines()
        assert lines, "working mood should render a scene"
        # The coding scene includes the little terminal box.
        assert any("┌" in line for line in lines) or any("⌨" in line for line in lines)

    def test_pet_scene_is_static_under_reduced_motion(self):
        chat = make_chat(pet_mood="working", pet_animate=False)
        first = chat._pet_lines()
        second = chat._pet_lines()
        assert first == second

    def test_pet_advances_frames_per_repaint(self):
        chat = make_chat(pet_mood="thinking")
        seen = set()
        for _ in range(6):
            seen.add(tuple(chat._pet_lines()))
        assert len(seen) > 1, "the scene should animate"

    def test_disabled_pet_renders_nothing(self):
        chat = make_chat(pet_on=False)
        assert chat._pet_lines() == []

    def test_every_mood_has_a_scene(self):
        from wynxo.pet import Mood
        for mood in Mood:
            scene = motion.scene_for(mood.value)
            assert scene.frames, f"{mood.value} has no scene"

    def test_unknown_mood_falls_back_safely(self):
        chat = make_chat(pet_mood="")
        assert chat._pet_lines() == []


class TestNotifications:
    def test_a_fresh_notification_is_shown(self):
        chat = make_chat()
        chat.notify("✦ model: qwen")
        assert "model: qwen" in text_of(chat._todo_fragments())

    def test_a_stale_notification_clears_itself(self, monkeypatch):
        chat = make_chat()
        now = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: now)
        chat.notify("✦ model: qwen")
        now += chat._toast_life + 1
        assert "model: qwen" not in text_of(chat._todo_fragments())
        assert chat._toast is None

    def test_notification_never_touches_the_composer(self):
        chat = make_chat()
        chat.notify("✦ theme: minimal")
        before = chat.composer_frame_rows()
        chat.notify("another one")
        chat.set_todos("[>] x", title="t")
        chat.set_todo_mode("compact")
        assert chat.composer_frame_rows() == before
        assert chat.status_rows() == 1


class TestToolEventsDriveThePet:
    """The pet is a visualisation of the agent state: the activity names the
    callbacks feed it must map to the right mood and the right scene."""

    def test_editing_maps_to_the_coding_scene(self):
        from wynxo.pet import Mood, Pet
        pet = Pet(unicode=True)
        pet.set_activity("editing")
        assert pet.mood is Mood.WORKING
        scene = motion.scene_for(pet.mood.value)
        assert scene.name == "coding"

    def test_searching_and_testing_have_their_own_moods(self):
        from wynxo.pet import Mood, Pet
        pet = Pet(unicode=True)
        pet.set_activity("searching")
        assert pet.mood is Mood.SEARCHING
        assert motion.scene_for(pet.mood.value).name == "searching"
        pet.set_activity("testing")
        assert pet.mood is Mood.TESTING
        assert motion.scene_for(pet.mood.value).name == "testing"

    def test_success_and_error_moods(self):
        from wynxo.pet import Mood, Pet
        pet = Pet(unicode=True)
        pet.react(Mood.HAPPY)
        assert motion.scene_for(pet.mood.value).name == "happy"
        pet.react(Mood.SAD)
        assert motion.scene_for(pet.mood.value).name == "error"


class TestAgentTodoFlowUpdatesThePanel:
    """The real agent loop: todo_write calls flow into the panel through the
    same on_todos callback the CLI uses, and the panel reflects each stage."""

    def test_plan_moves_from_in_progress_to_done(self, tmp_path):
        import asyncio
        from pathlib import Path

        from wynxo.agent import Agent, Callbacks
        from wynxo.config import Config
        from wynxo.effort import resolve
        from wynxo.provider import Chunk
        from wynxo.scope import Boundary, Scope
        from wynxo.tools import build_registry

        class Backend:
            def __init__(self, turns):
                self.turns = list(turns)

            def chat(self, messages, **options):
                if not self.turns:
                    return self._chunks("Done.", [])
                content, calls = self.turns.pop(0)
                return self._chunks(content, calls)

            async def _iter(self, content, calls):
                yield Chunk(content=content, tool_calls=calls, done=True)

            def _chunks(self, content, calls):
                return self._iter(content, calls)

        class PlanEvents(Callbacks):
            def __init__(self, chat):
                self.chat = chat
                self.renders = []

            async def on_todos(self, rendered):
                self.renders.append(rendered)
                self.chat.set_todos(rendered, title="Fix the calc bug")

        def tc(name, items):
            return {"function": {"name": name, "arguments": {"items": items}}}

        (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n",
                                          newline="\n")
        (tmp_path / "test_calc.py").write_text(
            "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            newline="\n")
        (tmp_path / "pytest.ini").write_text("[pytest]\n")

        plan_in_progress = tc("todo_write", [
            {"task": "inspect calc", "status": "in_progress"},
            {"task": "edit calc", "status": "pending"},
            {"task": "run tests", "status": "pending"}])
        plan_done = tc("todo_write", [
            {"task": "inspect calc", "status": "done"},
            {"task": "edit calc", "status": "done"},
            {"task": "run tests", "status": "done"}])
        chat = make_chat(pet_mood="thinking")
        events = PlanEvents(chat)
        backend = Backend([
            ("", [plan_in_progress]),
            ("", [plan_done]),
            ("Fixed and verified.", []),
        ])
        config = Config(verify_with_tests=False, allow_shell=True,
                        auto_approve=["*"])
        agent = Agent(backend, config, resolve("low"), tmp_path, events,
                      registry=build_registry(tmp_path),
                      boundary=Boundary(scope=Scope.FOLDER, root=tmp_path))
        agent.backend = backend
        asyncio.run(agent.run("Fix the failing test."))

        # The panel saw both plan states and settled on all-done.
        assert len(events.renders) == 2
        final = events.renders[-1]
        assert all(line.startswith("[x]") for line in final.splitlines())
        panel = text_of(chat._todo_panel())
        assert "Fix the calc bug" in panel
        assert "1/3" not in panel or "3/3" in panel
