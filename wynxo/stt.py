"""Cancellable speech-recognition orchestration with optional backends."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol


class SpeechState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class SpeechError(RuntimeError):
    """A recoverable microphone or transcription failure."""


class Recorder(Protocol):
    async def record(self, *, cancel: asyncio.Event, max_duration: float,
                     silence_timeout: float, device: str | int | None = None) -> Any: ...


class Transcriber(Protocol):
    async def transcribe(self, audio: Any, *, language: str = "") -> str: ...


@dataclass(frozen=True)
class SpeechConfig:
    language: str = ""
    device: str | int | None = None
    silence_timeout: float = 1.25
    max_duration: float = 30.0
    transcription_timeout: float = 60.0


@dataclass(frozen=True)
class SpeechSnapshot:
    state: SpeechState
    text: str = ""
    error: str = ""


def normalize_transcript(text: str) -> str:
    """Conservatively clean ASR artifacts without rewriting intent."""
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    return re.sub(r"\b([\w'-]+)(?:\s+\1){1,}\b", r"\1", text, flags=re.I)


class SpeechSession:
    """One microphone -> transcription -> final-text operation.

    Partial backend results are deliberately not exposed: callers receive one
    finalized string, preventing partials from becoming duplicate prompts.
    """

    def __init__(self, recorder: Recorder, transcriber: Transcriber,
                 config: SpeechConfig | None = None,
                 on_state: Callable[[SpeechSnapshot], None] | None = None):
        self.recorder = recorder
        self.transcriber = transcriber
        self.config = config or SpeechConfig()
        self.on_state = on_state
        self.state = SpeechState.IDLE
        self._cancel = asyncio.Event()
        self._task: asyncio.Task[str] | None = None
        self._final_text = ""

    def _set(self, state: SpeechState, *, text: str = "", error: str = "") -> None:
        self.state = state
        if self.on_state:
            self.on_state(SpeechSnapshot(state, text, error))

    async def start(self) -> str:
        if self._task and not self._task.done():
            raise SpeechError("speech recognition is already running")
        self._cancel = asyncio.Event()
        self._final_text = ""
        self._task = asyncio.create_task(self._run())
        try:
            return await self._task
        finally:
            self._task = None

    async def _run(self) -> str:
        try:
            self._set(SpeechState.LISTENING)
            audio = await self.recorder.record(
                cancel=self._cancel, max_duration=self.config.max_duration,
                silence_timeout=self.config.silence_timeout, device=self.config.device)
            if self._cancel.is_set():
                self._set(SpeechState.CANCELLED)
                return ""
            self._set(SpeechState.TRANSCRIBING)
            text = await asyncio.wait_for(
                self.transcriber.transcribe(audio, language=self.config.language),
                timeout=self.config.transcription_timeout)
            if self._cancel.is_set():
                self._set(SpeechState.CANCELLED)
                return ""
            self._final_text = normalize_transcript(text)
            self._set(SpeechState.COMPLETED, text=self._final_text)
            return self._final_text
        except asyncio.CancelledError:
            self._set(SpeechState.CANCELLED)
            raise
        except Exception as exc:
            self._set(SpeechState.ERROR, error=str(exc) or type(exc).__name__)
            return ""

    async def cancel(self) -> None:
        self._cancel.set()
        task = self._task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._set(SpeechState.CANCELLED)

    @property
    def final_text(self) -> str:
        return self._final_text
