from __future__ import annotations

import asyncio
from pathlib import Path

from wynxo.stt import SpeechConfig, SpeechSession, SpeechState
from wynxo.tui import ChatUI


class Recorder:
    def __init__(self, audio=b"audio"):
        self.audio = audio
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_seen = False

    async def record(self, *, cancel, max_duration, silence_timeout, device=None):
        self.started.set()
        while not self.release.is_set():
            if cancel.is_set():
                self.cancel_seen = True
                return b""
            await asyncio.sleep(0)
        return self.audio


class Transcriber:
    def __init__(self, text="find the bug"):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio, *, language=""):
        self.calls += 1
        await asyncio.sleep(0)
        return self.text


def test_stt_states_are_ordered_and_final_text_is_emitted_once():
    async def run():
        recorder = Recorder()
        transcriber = Transcriber("hello   hello world")
        states = []
        session = SpeechSession(
            recorder, transcriber, SpeechConfig(),
            on_state=lambda snapshot: states.append(snapshot.state),
        )
        task = asyncio.create_task(session.start())
        await recorder.started.wait()
        recorder.release.set()
        result = await task
        return result, states, transcriber.calls

    result, states, calls = asyncio.run(run())
    assert result == "hello world"
    assert states == [SpeechState.LISTENING, SpeechState.TRANSCRIBING,
                      SpeechState.COMPLETED]
    assert calls == 1


def test_stt_cancellation_is_terminal_and_does_not_transcribe():
    async def run():
        recorder = Recorder()
        transcriber = Transcriber()
        states = []
        session = SpeechSession(
            recorder, transcriber, SpeechConfig(),
            on_state=lambda snapshot: states.append(snapshot.state),
        )
        task = asyncio.create_task(session.start())
        await recorder.started.wait()
        await session.cancel()
        return task, session, states, transcriber.calls

    task, session, states, calls = asyncio.run(run())
    assert task.done()
    assert session.state is SpeechState.CANCELLED
    assert calls == 0
    assert states[-1] is SpeechState.CANCELLED


def test_stt_can_be_reused_for_a_second_recording():
    async def run():
        recorder = Recorder()
        transcriber = Transcriber("first")
        session = SpeechSession(recorder, transcriber)
        first = asyncio.create_task(session.start())
        await recorder.started.wait()
        recorder.release.set()
        first_text = await first
        recorder.release.clear()
        second = asyncio.create_task(session.start())
        await recorder.started.wait()
        recorder.release.set()
        second_text = await second
        return first_text, second_text, transcriber.calls

    assert asyncio.run(run()) == ("first", "first", 2)


def test_transcription_is_a_composer_draft_not_an_automatic_submission():
    async def run():
        recorder = Recorder()
        transcriber = Transcriber("fix the parser")
        states = []
        session = SpeechSession(
            recorder, transcriber, on_state=lambda snapshot: states.append(snapshot))
        task = asyncio.create_task(session.start())
        await recorder.started.wait()
        recorder.release.set()
        text = await task
        chat = ChatUI(status=lambda: "")
        chat.buffer.insert_text(text)
        return chat.buffer.text, chat.submissions.empty(), states[-1].state

    text, empty, state = asyncio.run(run())
    assert text == "fix the parser"
    assert empty is True
    assert state is SpeechState.COMPLETED


def test_layout_has_no_speech_specific_height():
    chat = ChatUI(status=lambda: "🎙 Listening...")
    before = (chat.transcript_rows(), chat.composer_frame_rows())
    chat._status = lambda: "◌ Transcribing..."
    after = (chat.transcript_rows(), chat.composer_frame_rows())
    assert before == after
    assert chat.FOOTER_ROWS == 1
