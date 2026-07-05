"""Energy VAD + utterance segmenter — the streaming half of the ears (P7.3 v1).

A microphone yields fixed-size PCM frames; `SpeechSegmenter` is the
deterministic state machine that turns them into **utterances**: it opens on
``start_frames`` consecutive voiced frames (energy above the threshold),
accumulates (including a small pre-roll so the first syllable survives), and
closes after ``end_frames`` of silence — or force-closes at
``max_utterance_s`` so unbroken noise can never grow an unbounded buffer.

Stdlib-only (`array`/`struct`/`wave`), no numpy, no audio hardware: tests
drive it with synthetic PCM. An energy gate is v1 — good enough to segment a
quiet home; a model VAD (silero etc.) can replace the ``is_voiced`` decision
behind the same segmenter later without touching callers. Real-microphone
capture lives in :mod:`voicelink.stream` behind the ``mic`` extra and is
validated on-device.

Units: PCM is 16-bit signed little-endian mono; ``sample_rate`` in Hz;
energy is RMS over one frame in int16 amplitude units.
"""

from __future__ import annotations

import array
import io
import math
import wave
from dataclasses import dataclass

__all__ = ["SpeechSegmenter", "VADConfig", "frame_rms", "pcm_to_wav"]


@dataclass(frozen=True, slots=True)
class VADConfig:
    """Segmenter tuning. Defaults suit a quiet room at 16 kHz mono int16.

    ``energy_threshold`` is RMS in int16 amplitude units (~500 ≈ soft speech;
    raise it in noisy rooms). ``start_frames``/``end_frames`` are consecutive
    frame counts (30 ms frames ⇒ defaults open after ~90 ms of speech and
    close after ~300 ms of silence). ``pre_roll_frames`` of leading audio are
    kept so the first syllable is not clipped.
    """

    sample_rate: int = 16000
    frame_ms: int = 30
    energy_threshold: float = 500.0
    start_frames: int = 3
    end_frames: int = 10
    pre_roll_frames: int = 3
    max_utterance_s: float = 15.0

    @property
    def frame_bytes(self) -> int:
        """Size of one PCM frame in bytes (int16 mono)."""
        return 2 * (self.sample_rate * self.frame_ms // 1000)

    @property
    def max_frames(self) -> int:
        """The force-close cap in frames (≥ 1)."""
        return max(1, int(self.max_utterance_s * 1000 / self.frame_ms))


def frame_rms(pcm: bytes) -> float:
    """RMS energy of one int16 little-endian PCM frame (0.0 for empty)."""
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def pcm_to_wav(pcm: bytes, *, sample_rate: int = 16000) -> bytes:
    """Wrap raw int16 mono PCM in a WAV container (what ASR endpoints expect)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class SpeechSegmenter:
    """Frames in, utterances out — the VAD state machine (deterministic).

    ``push(frame)`` returns a completed utterance's raw PCM when this frame
    closes one, else ``None``; ``flush()`` closes an utterance still open at
    end-of-stream. Trailing silence inside the ``end_frames`` hangover stays
    in the utterance (harmless to ASR); leading audio keeps up to
    ``pre_roll_frames`` before the trigger.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self._cfg = config or VADConfig()
        self._pre_roll: list[bytes] = []
        self._pending: list[bytes] = []  # candidate voiced frames while opening
        self._utterance: list[bytes] = []
        self._voiced_run = 0
        self._silent_run = 0
        self._open = False

    def is_voiced(self, frame: bytes) -> bool:
        """The v1 voice decision: RMS energy above the threshold."""
        return frame_rms(frame) >= self._cfg.energy_threshold

    def push(self, frame: bytes) -> bytes | None:
        """Feed one PCM frame; returns a finished utterance's PCM, or None."""
        cfg = self._cfg
        voiced = self.is_voiced(frame)

        if not self._open:
            if voiced:
                self._pending.append(frame)
                self._voiced_run += 1
                if self._voiced_run >= cfg.start_frames:
                    # open: pre-roll + the frames that triggered the open
                    self._utterance = [*self._pre_roll, *self._pending]
                    self._pre_roll = []
                    self._pending = []
                    self._open = True
                    self._silent_run = 0
            else:
                # a blip shorter than start_frames goes back to the pre-roll
                self._pre_roll.extend(self._pending)
                self._pending = []
                self._voiced_run = 0
                self._pre_roll.append(frame)
                del self._pre_roll[: -cfg.pre_roll_frames or None]
            return None

        self._utterance.append(frame)
        self._silent_run = 0 if voiced else self._silent_run + 1
        if self._silent_run >= cfg.end_frames or len(self._utterance) >= cfg.max_frames:
            return self._close()
        return None

    def flush(self) -> bytes | None:
        """Close and return an utterance still open at end-of-stream, if any."""
        if self._open and self._utterance:
            return self._close()
        return None

    def _close(self) -> bytes:
        out = b"".join(self._utterance)
        self._utterance = []
        self._open = False
        self._voiced_run = 0
        self._silent_run = 0
        return out
