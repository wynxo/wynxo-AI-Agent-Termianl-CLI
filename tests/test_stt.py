import asyncio

import pytest

from wynxo.stt import SpeechConfig, SpeechSession, SpeechState, normalize_transcript


class Recorder:
    async def record(self, **kwargs):
        await asyncio.sleep(0)
        return b"audio"


class Transcriber:
    async def transcribe(self, audio, *, language=""):
        await asyncio.sleep(0)
        return "  hello hello   Wynxo  "


@pytest.mark.asyncio
async def test_speech_commits_one_normalized_final_result():
    states = []
    session = SpeechSession(Recorder(), Transcriber(), on_state=states.append)
    assert await session.start() == "hello Wynxo"
    assert [s.state for s in states] == [SpeechState.LISTENING, SpeechState.TRANSCRIBING, SpeechState.COMPLETED]
    assert states[-1].text == "hello Wynxo"


@pytest.mark.asyncio
async def test_speech_cancellation_returns_to_cancelled_without_result():
    class SlowRecorder:
        async def record(self, **kwargs):
            await asyncio.sleep(60)

    states = []
    session = SpeechSession(SlowRecorder(), Transcriber(), on_state=states.append)
    task = asyncio.create_task(session.start())
    await asyncio.sleep(0)
    await session.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.state is SpeechState.CANCELLED
    assert session.final_text == ""


@pytest.mark.asyncio
async def test_transcription_timeout_is_recoverable():
    class SlowTranscriber:
        async def transcribe(self, audio, *, language=""):
            await asyncio.sleep(60)

    session = SpeechSession(Recorder(), SlowTranscriber(), SpeechConfig(transcription_timeout=0.001))
    assert await session.start() == ""
    assert session.state is SpeechState.ERROR


def test_normalization_is_conservative():
    assert normalize_transcript("fix fix the tests") == "fix the tests"
    assert normalize_transcript("open calc.py") == "open calc.py"
    assert normalize_transcript("") == ""
