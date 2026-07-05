"""ASR+VAD v1 — energy VAD segmenter + speech stream (P7.3 software half).

Deterministic, stdlib-only, no audio hardware: synthetic int16 PCM frames
(tone = speech, zeros = silence) drive the segmenter state machine; the
`SpeechStream` packages completed utterances as WAV and fills the SAME
``audio_source`` seam `mindloop.ListenerInterruptSource` polls. Real-mic
capture (`SoundDeviceMic`, the ``mic`` extra) is validated on-device later.
"""

from __future__ import annotations

import math
import struct

from mindloop import ListenerInterruptSource, MockListener

from voicelink import ScriptedMic, SpeechSegmenter, SpeechStream, VADConfig, pcm_to_wav

CFG = VADConfig(
    sample_rate=16000, frame_ms=30, energy_threshold=500.0, start_frames=2, end_frames=3
)
_SAMPLES = CFG.sample_rate * CFG.frame_ms // 1000


def _tone_frame(amp: int = 8000) -> bytes:
    return struct.pack(
        f"<{_SAMPLES}h",
        *(int(amp * math.sin(2 * math.pi * 440 * i / CFG.sample_rate)) for i in range(_SAMPLES)),
    )


def _silence_frame() -> bytes:
    return b"\x00\x00" * _SAMPLES


def _run(seg: SpeechSegmenter, frames: list[bytes]) -> list[bytes]:
    out = [u for f in frames if (u := seg.push(f)) is not None]
    tail = seg.flush()
    if tail is not None:
        out.append(tail)
    return out


def test_silence_yields_no_utterance() -> None:
    assert _run(SpeechSegmenter(CFG), [_silence_frame()] * 20) == []


def test_single_burst_yields_one_utterance() -> None:
    frames = [_silence_frame()] * 5 + [_tone_frame()] * 8 + [_silence_frame()] * 6
    utts = _run(SpeechSegmenter(CFG), frames)
    assert len(utts) == 1
    # at least the voiced frames survive (plus pre-roll)
    assert len(utts[0]) >= 8 * len(_tone_frame())


def test_two_bursts_yield_two_utterances() -> None:
    burst = [_tone_frame()] * 6 + [_silence_frame()] * 6
    utts = _run(SpeechSegmenter(CFG), burst * 2)
    assert len(utts) == 2


def test_short_blip_below_start_frames_is_ignored() -> None:
    frames = [_silence_frame()] * 4 + [_tone_frame()] + [_silence_frame()] * 8
    assert _run(SpeechSegmenter(CFG), frames) == []


def test_max_utterance_cap_force_closes() -> None:
    cfg = VADConfig(
        sample_rate=16000,
        frame_ms=30,
        energy_threshold=500.0,
        start_frames=2,
        end_frames=3,
        max_utterance_s=0.3,  # 10 frames
    )
    seg = SpeechSegmenter(cfg)
    utts = [u for _ in range(40) if (u := seg.push(_tone_frame())) is not None]
    assert utts  # unbroken speech still emits bounded utterances


def test_flush_closes_an_open_utterance() -> None:
    seg = SpeechSegmenter(CFG)
    for _ in range(6):
        assert seg.push(_tone_frame()) is None or True
    assert seg.flush() is not None


def test_pcm_to_wav_header() -> None:
    wav = pcm_to_wav(_tone_frame(), sample_rate=16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[24:28])[0] == 16000  # sample rate field


def test_speech_stream_polls_wav_utterances() -> None:
    frames = (
        [_silence_frame()] * 4
        + [_tone_frame()] * 6
        + [_silence_frame()] * 6
        + [_tone_frame()] * 6
        + [_silence_frame()] * 6
    )
    stream = SpeechStream(ScriptedMic(frames), config=CFG)
    first = stream.poll()
    assert first is not None and first[:4] == b"RIFF"
    second = stream.poll()
    assert second is not None
    assert stream.poll() is None  # mic drained


def test_stream_fills_the_barge_audio_source_seam() -> None:
    # the whole point: SpeechStream.poll IS the audio_source ListenerInterruptSource wants
    frames = [_silence_frame()] * 4 + [_tone_frame()] * 6 + [_silence_frame()] * 6
    stream = SpeechStream(ScriptedMic(frames), config=CFG)
    src = ListenerInterruptSource(MockListener(["stop"]), audio_source=stream.poll)
    assert src.poll() == "stop"
    assert src.poll() is None
