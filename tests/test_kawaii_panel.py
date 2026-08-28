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
    assert first[0].startswith("♡ plan 1/3 ♡")
    assert first[1].endswith("inspect")
    assert first[2][0] in "✧⋆✦♡"
    assert second[2][0] in "✧⋆✦♡"


def test_todo_panel_caps_long_plans():
    chat = ChatUI(width=100)
    chat.set_todos("\n".join(f"[ ] step {i}" for i in range(100)))
    assert len(chat._todo_fragments()) <= chat.TODO_MAX_ROWS


def test_todo_panel_can_be_cleared():
    chat = ChatUI(width=100)
    chat.set_todos("[>] work")
    chat.set_todos("")
    assert chat._todo_fragments() == []
