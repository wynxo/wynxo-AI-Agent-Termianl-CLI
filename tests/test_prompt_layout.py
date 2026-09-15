"""The live composer must stay a composer, not become a full-screen pane."""

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import DummyOutput

from wynxo.cli import Repl


def test_prompt_input_window_does_not_fill_unused_terminal_rows():
    """The composer should hug the one-line input even without a real TTY.

    CI on Windows has no Win32 console buffer. Give prompt_toolkit explicit
    pipe/dummy I/O so this remains a layout test rather than accidentally
    becoming a test of the GitHub runner's terminal host.
    """
    repl = object.__new__(Repl)
    repl._prompt_bindings = KeyBindings()
    repl._model_names = []

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
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
