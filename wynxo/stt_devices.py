"""Microphone capture and transcription, from optional local backends.

Wynxo's install stays pure Python: the speech-to-text path here only lights
up when the user has installed a backend that can actually do the work --
``sounddevice`` for the microphone, and ``faster-whisper`` (offline) or
``SpeechRecognition`` (online) for the transcript. Until then
``create_session`` says what to install rather than pretending to listen.

The recorder's shape is the whole answer to the classic dictation bugs:

- A block is *speech* only when its RMS clears a floor, and the recorder
  stops on silence only after speech has actually begun -- so it cannot
  cut the first word off, and it cannot hang forever on a quiet room.
- A short hangover after the last voiced block keeps the tail of a word
  from being clipped by a breath between words.
- Cancellation is checked between blocks, so Ctrl-R mid-sentence stops
  within one block rather than at the end of the cap.
- The final result is one string, once -- partials never reach the UI, so
  nothing can be transcribed twice.
"""

from __future__ import annotations

import asyncio
import io
import math
import struct
import tempfile
import wave
from pathlib import Path

from .stt import SpeechConfig, SpeechSession, SpeechState

SAMPLE_RATE = 16_000
BLOCK = 800                 # samples per block: 50 ms at 16 kHz
RMS_FLOOR = 320             # int16 amplitude under which a block is quiet
MIN_SPEECH_SECONDS = 0.35   # a burst shorter than this is a click, not speech

RecorderFactoryError = str


class SoundDeviceRecorder:
    """Microphone capture with RMS silence detection, on one backend."""

    name = "sounddevice"

    def __init__(self, sample_rate: int = SAMPLE_RATE, floor: int = RMS_FLOOR):
        self.sample_rate = sample_rate
        self.floor = floor

    class _StopRecording(Exception):
        pass

    async def record(self, *, cancel: asyncio.Event, max_duration: float,
                     silence_timeout: float,
                     device: str | int | None = None) -> bytes:
        import array

        import sounddevice as sd

        block_seconds = BLOCK / self.sample_rate
        quiet_blocks_allowed = max(1, int(silence_timeout / block_seconds))
        max_blocks = max(1, int(max_duration / block_seconds))
        min_speech_blocks = max(1, int(MIN_SPEECH_SECONDS / block_seconds))

        pcm = array.array("h")
        speech_blocks = 0
        quiet_blocks = 0

        def callback(indata, frames, time_info, status) -> None:
            nonlocal speech_blocks, quiet_blocks
            if cancel.is_set():
                raise self._StopRecording
            samples = array.array("h")
            samples.frombytes(bytes(indata))
            pcm.extend(samples)
            if _rms(samples) >= self.floor:
                speech_blocks += 1
                quiet_blocks = 0
            elif speech_blocks:
                quiet_blocks += 1

        stream = sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=BLOCK, channels=1,
            dtype="int16", device=device, callback=callback)
        stream.start()
        try:
            while len(pcm) // BLOCK < max_blocks:
                if cancel.is_set():
                    return b""
                # The callback runs on PortAudio's thread; sleep here and
                # let it fill, checking the same conditions it does so a
                # silent room ends the recording without waiting out the
                # whole cap.
                await asyncio.sleep(block_seconds / 4)
                if speech_blocks >= min_speech_blocks and \
                        quiet_blocks >= quiet_blocks_allowed:
                    break
                if getattr(stream, "active", True) is False:
                    break
        except self._StopRecording:
            pass
        finally:
            with _suppress():
                stream.stop()
                stream.close()

        if speech_blocks < min_speech_blocks:
            # Nothing that counts as speech arrived; an empty recording
            # reads better than a transcription of room tone.
            return b""
        return _wav_bytes(pcm, self.sample_rate)


class _suppress:
    """contextlib.suppress, local, so the recorder imports nothing at module
    scope beyond sounddevice itself."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def _rms(samples: "array.array") -> float:
    if not samples:
        return 0.0
    total = 0
    for value in samples:
        total += value * value
    return math.sqrt(total / len(samples))


def _wav_bytes(pcm, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class WhisperTranscriber:
    """faster-whisper: local, offline, and the quality bar for this feature."""

    name = "faster-whisper"

    def __init__(self, model_size: str = "base.en"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="cpu",
                                       compute_type="int8")
        return self._model

    async def transcribe(self, audio: bytes, *, language: str = "") -> str:
        def work() -> str:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                fh.write(audio)
                path = Path(fh.name)
            try:
                model = self._load()
                segments, _info = model.transcribe(
                    str(path), language=language or None, beam_size=1)
                return " ".join(segment.text for segment in segments).strip()
            finally:
                with _suppress():
                    path.unlink()

        return await asyncio.get_running_loop().run_in_executor(None, work)


class GoogleWebTranscriber:
    """The SpeechRecognition package's web engine. Needs internet; used only
    when faster-whisper is not installed."""

    name = "speech-recognition"

    async def transcribe(self, audio: bytes, *, language: str = "") -> str:
        def work() -> str:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            clip = sr.AudioData(audio, SAMPLE_RATE, 2)
            try:
                return recognizer.recognize_google(
                    clip, language=language or "en-US")
            except sr.UnknownValueError:
                return ""       # heard nothing intelligible; not an error
            except sr.RequestError as exc:
                raise RuntimeError(f"speech service unreachable: {exc}") from exc

        return await asyncio.get_running_loop().run_in_executor(None, work)


def create_session(config: SpeechConfig | None = None,
                   on_state=None,
                   recorder=None, transcriber=None,
                   ) -> tuple[SpeechSession | None, str]:
    """A ready SpeechSession, or (None, what-to-install).

    The backends are detected, not configured: whichever transcriber is
    installed is the one used, offline preferred over online. Both halves
    are optional dependencies and the hint names exactly what is missing,
    so a user who presses Ctrl-R on a fresh machine learns the one command
    that makes it work.
    """
    config = config or SpeechConfig()

    if recorder is None:
        try:
            import sounddevice  # noqa: F401
            recorder = SoundDeviceRecorder()
        except ImportError:
            recorder = None
    if transcriber is None:
        try:
            import faster_whisper  # noqa: F401
            transcriber = WhisperTranscriber()
        except ImportError:
            try:
                import speech_recognition  # noqa: F401
                transcriber = GoogleWebTranscriber()
            except ImportError:
                transcriber = None

    if recorder is None and transcriber is None:
        return None, ("speech input needs a microphone backend and a "
                      "transcriber: pip install sounddevice faster-whisper")
    if recorder is None:
        return None, "speech input needs a microphone backend: pip install sounddevice"
    if transcriber is None:
        return None, ("a microphone was found, but nothing can transcribe: "
                      "pip install faster-whisper (offline, recommended) "
                      "or SpeechRecognition (online)")
    return SpeechSession(recorder, transcriber, config, on_state), \
        f"stt: {getattr(recorder, 'name', '?')} + {getattr(transcriber, 'name', '?')}"


__all__ = [
    "SoundDeviceRecorder", "WhisperTranscriber", "GoogleWebTranscriber",
    "create_session", "SAMPLE_RATE", "SpeechConfig", "SpeechSession",
    "SpeechState",
]
