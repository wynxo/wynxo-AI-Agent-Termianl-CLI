from __future__ import annotations

import asyncio
from types import SimpleNamespace

from wynxo.tui import ChatUI


def press(chat: ChatUI, key: str) -> bool:
    for binding in chat.app.key_bindings.bindings:
        names = tuple(getattr(item, "value", str(item)) for item in binding.keys)
        if names == (key,):
            binding.handler(SimpleNamespace(data="", app=chat.app))
            return True
    return False


def test_layout_diagnostics_expose_the_space_contract():
    chat = ChatUI(status=lambda: "working\nignored", width=80)
    report = chat.layout_report()
    assert "root" in report
    assert "output" in report
    assert "composer" in report
    assert "footer    1" in report
    assert "check" in report


def test_status_changes_cannot_change_footer_or_output_geometry():
    state = {"value": "idle"}
    chat = ChatUI(status=lambda: state["value"], width=80)
    initial = (chat.transcript_rows(), chat.composer_frame_rows())
    for value in ("◌ Thinking...", "→ reading a file", "✕ tests failed\nmore detail"):
        state["value"] = value
        assert (chat.transcript_rows(), chat.composer_frame_rows()) == initial
        assert "\n" not in chat._footer_fragments().value


def test_long_and_multiline_input_keep_a_bounded_natural_frame():
    chat = ChatUI(status=lambda: "", width=80)
    for text in ("", "one", "one\ntwo", "\n".join(str(i) for i in range(50)), "x" * 1000):
        chat.buffer.text = text
        assert 3 <= chat.composer_frame_rows() <= chat.COMPOSER_MAX_ROWS + 2


def test_end_returns_to_following_and_refocus_is_idempotent():
    chat = ChatUI(status=lambda: "")
    for i in range(200):
        chat.transcript.console.print(f"line {i}")
    chat.flush()
    chat.scroll = 10
    assert press(chat, "end")
    assert chat.scroll == 0
    chat.refocus()
    chat.refocus()
    assert chat.app.layout.current_window.content.__class__.__name__ == "BufferControl"


def test_dictation_binding_is_optional_and_does_not_submit():
    calls = []
    chat = ChatUI(status=lambda: "", on_dictate=lambda: calls.append(True))
    assert press(chat, "c-r")
    assert calls == [True]
    assert chat.submissions.empty()


def test_completion_is_a_float_not_a_flow_child():
    chat = ChatUI(status=lambda: "")
    root = chat.app.layout.container
    assert len(root.floats) == 1
    assert root.floats[0].content.__class__.__name__ == "CompletionsMenu"


class FakeSpeaker:
    def __init__(self):
        self.calls = []

    async def say_async(self, text):
        self.calls.append(text)
        await asyncio.sleep(0)
        return True


async def _call_speaker():
    speaker = FakeSpeaker()
    await speaker.say_async("answer")
    return speaker.calls


def test_async_speech_api_does_not_block_the_event_loop():
    assert asyncio.run(_call_speaker()) == ["answer"]
