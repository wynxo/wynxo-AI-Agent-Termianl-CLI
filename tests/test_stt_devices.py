"""The dictation backend, minus the hardware: RMS detection, WAV encoding
and the backend-detection logic in create_session are all pure Python and
testable without a microphone, PortAudio or a Whisper model."""

from __future__ import annotations

import array
import io
import sys
import wave

from wynxo import stt_devices
from wynxo.stt_devices import SAMPLE_RATE, _rms, _wav_bytes, create_session


class _FakeRecorder:
    name = "fake-recorder"


class _FakeTranscriber:
    name = "fake-transcriber"


class TestRms:
    def test_empty_is_silence(self):
        assert _rms(array.array("h")) == 0.0

    def test_flat_silence_is_zero(self):
        assert _rms(array.array("h", [0, 0, 0, 0])) == 0.0

    def test_constant_signal_is_that_level(self):
        assert _rms(array.array("h", [2, 2, 2])) == 2.0

    def test_sign_does_not_matter(self):
        # (-1, 1) has the same power as (1, -1): sqrt((1+1)/2) = 1.
        assert _rms(array.array("h", [1, -1])) == 1.0

    def test_louder_blocks_clear_the_floor(self):
        assert _rms(array.array("h", [1000] * 16)) > 320


class TestWavBytes:
    def _round_trip(self, pcm: array.array):
        data = _wav_bytes(pcm, SAMPLE_RATE)
        with wave.open(io.BytesIO(data), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == SAMPLE_RATE
            assert wav.getnframes() == len(pcm)
            assert wav.readframes(len(pcm)) == pcm.tobytes()

    def test_silence_round_trips(self):
        self._round_trip(array.array("h", [0] * 40))

    def test_audio_round_trips(self):
        self._round_trip(array.array("h", [0, 100, -100, 32767, -32768]))

    def test_header_is_written_for_empty_audio(self):
        self._round_trip(array.array("h"))


class TestCreateSession:
    def test_injected_recorder_and_transcriber_make_a_session(self):
        session, hint = create_session(recorder=_FakeRecorder(),
                                       transcriber=_FakeTranscriber())
        assert session is not None
        assert "fake-recorder" in hint and "fake-transcriber" in hint

    def test_recorder_missing_names_the_microphone_package(self):
        session, hint = create_session(recorder=None,
                                       transcriber=_FakeTranscriber())
        assert session is None
        assert "pip install sounddevice" in hint

    def test_transcriber_missing_names_faster_whisper(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        session, hint = create_session(recorder=_FakeRecorder(),
                                       transcriber=None, backend="offline")
        assert session is None
        assert "faster-whisper" in hint

    def test_nothing_installed_names_everything(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        monkeypatch.setitem(sys.modules, "speech_recognition", None)
        session, hint = create_session()
        assert session is None
        assert "sounddevice" in hint and "faster-whisper" in hint

    def test_online_backend_skips_the_offline_check(self, monkeypatch):
        def boom():
            raise AssertionError("_transcriber_offline must not be consulted")

        monkeypatch.setattr(stt_devices, "_transcriber_offline", boom)
        monkeypatch.setattr(stt_devices, "_transcriber_online",
                            lambda: _FakeTranscriber())
        session, hint = create_session(recorder=_FakeRecorder(),
                                       transcriber=None, backend="online")
        assert session is not None
        assert "fake-transcriber" in hint

    def test_offline_backend_skips_the_online_check(self, monkeypatch):
        def boom():
            raise AssertionError("_transcriber_online must not be consulted")

        monkeypatch.setattr(stt_devices, "_transcriber_online", boom)
        monkeypatch.setattr(stt_devices, "_transcriber_offline",
                            lambda: _FakeTranscriber())
        session, hint = create_session(recorder=_FakeRecorder(),
                                       transcriber=None, backend="offline")
        assert session is not None
        assert "fake-transcriber" in hint

    def test_auto_prefers_offline_over_online(self, monkeypatch):
        called: list[str] = []

        def offline():
            called.append("offline")
            return _FakeTranscriber()

        def online():
            called.append("online")
            return None

        monkeypatch.setattr(stt_devices, "_transcriber_offline", offline)
        monkeypatch.setattr(stt_devices, "_transcriber_online", online)
        session, hint = create_session(recorder=_FakeRecorder(),
                                       transcriber=None, backend="auto")
        # Offline is preferred: when it yields a transcriber, the online
        # fallback must never be consulted.
        assert called == ["offline"]
        assert session is not None
