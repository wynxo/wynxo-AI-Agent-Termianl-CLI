from wynxo.tui import ChatUI


def test_todo_panel_is_empty_until_set():
    chat = ChatUI(width=100)
    assert chat._todo_fragments() == []


def test_todo_panel_shows_progress_and_kawaii_active_marker():
    chat = ChatUI(width=100)
    chat.set_todos("[x] inspect\n[>] build\n[ ] verify")
    first = chat._todo_fragments()
    chat.set_todos("[x] inspect\n[>] build\n[ ] verify")
    second = chat._todo_fragments()
    # Fragments are (style, text); the text is the visible row.
    top, inspect_row, active_row, _, bottom = [t for _, t in first]
    assert top.startswith("╭")
    assert "plan · 1/3" in top
    assert "✓ inspect" in inspect_row
    assert active_row[2] in "✧⋆✦♡"
    assert [t for _, t in second][2][2] in "✧⋆✦♡"
    assert bottom.startswith("╰")


def test_todo_panel_caps_long_plans():
    chat = ChatUI(width=100)
    chat.set_todos("\n".join(f"[ ] step {i}" for i in range(100)))
    assert len(chat._todo_fragments()) <= chat.TODO_MAX_ROWS


def test_todo_panel_can_be_cleared():
    chat = ChatUI(width=100)
    chat.set_todos("[>] work")
    chat.set_todos("")
    assert chat._todo_fragments() == []
