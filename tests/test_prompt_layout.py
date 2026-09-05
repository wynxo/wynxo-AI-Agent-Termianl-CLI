"""The live composer must stay a composer, not become a full-screen pane."""

from prompt_toolkit.key_binding import KeyBindings

from wynxo.cli import Repl


def test_prompt_input_window_does_not_fill_unused_terminal_rows():
    """The border and toolbar should hug the one-line input."""
    repl = object.__new__(Repl)
    repl._prompt_bindings = KeyBindings()
    repl._model_names = []

    session = repl._make_prompt_session()
    input_windows = [
        window for window in session.app.layout.find_all_windows()
        if getattr(getattr(window, "content", None), "buffer", None)
        is session.default_buffer
    ]

    assert input_windows
    assert all(window.dont_extend_height() for window in input_windows)

    spacer = session.app.layout.container.children[0]
    assert spacer.height.weight == 1
