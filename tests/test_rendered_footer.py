from __future__ import annotations

from prompt_toolkit.layout.containers import HSplit, Window

from wynxo.tui import ChatUI


def _children(container):
    return list(getattr(container, "children", ()))


def test_footer_windows_have_explicit_natural_height():
    chat = ChatUI(status=lambda: "")
    root = chat.app.layout.container
    body = root.content
    children = _children(body)
    footer = children[-1]
    assert isinstance(body, HSplit)
    assert isinstance(footer, HSplit)
    assert footer.height.min == 3
    assert footer.height.preferred == 3
    assert footer.height.max == 5
    assert footer.children[0].height == 1
    assert footer.children[2].height == 1


def test_output_is_the_only_flexible_region():
    chat = ChatUI(status=lambda: "")
    body = chat.app.layout.container.content
    children = _children(body)
    transcript = children[2]
    footer = children[-1]
    assert not transcript.dont_extend_height()
    assert footer.height.max == 5


def test_empty_and_long_input_do_not_change_footer_contract():
    chat = ChatUI(status=lambda: "", width=80)
    for value in ("", "hello", "hello " * 100, "line one\nline two"):
        chat.buffer.text = value
        assert chat.app.layout.container is not None
        footer = _children(chat.app.layout.container.content)[-1]
        assert footer.height.min == 3
        assert footer.height.max == 5
