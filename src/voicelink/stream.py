"""Speech stream — mic frames → VAD segments → the barge-in seam (P7.3 v1).

`SpeechStream.poll` is the whole point: it pumps whatever frames the mic has
captured through the `SpeechSegmenter` and returns the next completed
utterance as WAV bytes — exactly the ``audio_source`` signature
`mindloop.ListenerInterruptSource` wants, and the same bytes
`mindloop.VoiceLoop.turn` takes. Wire it once and the robot hears both
conversation turns and mid-task barge-ins from the same microphone:

    stream = SpeechStream(SoundDeviceMic(), config=VADConfig())
    interrupts = ListenerInterruptSource(make_listener(...), audio_source=stream.poll)

`MicSource` is the capture seam: ``read()`` returns the next PCM frame or
``None`` (nothing captured yet) — never blocks. `ScriptedMic` scripts frames
for tests; `SoundDeviceMic` (the ``mic`` extra, lazy-imported) captures a real
microphone and is **validated on-device** — everything else here is
deterministic and network/hardware-free.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .vad import SpeechSegmenter, VADConfig, pcm_to_wav

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["MicSource", "ScriptedMic", "SoundDeviceMic", "SpeechStream"]


@runtime_checkable
class MicSource(Protocol):
    """Non-blocking microphone seam: the next captured PCM frame, or None."""

    def read(self) -> bytes | None: ...


class ScriptedMic:
    """A `MicSource` that replays scripted frames (tests / file-driven demos)."""

    def __init__(self, frames: Iterable[bytes] = ()) -> None:
        self._q: deque[bytes] = deque(frames)

    def push(self, frame: bytes) -> None:
        self._q.append(frame)

    def read(self) -> bytes | None:
        return self._q.popleft() if self._q else None


class SpeechStream:
    """Pump mic frames through the VAD; hand out finished utterances as WAV.

    ``poll()`` drains every frame the mic currently has, runs them through the
    segmenter, and returns the oldest completed utterance (or ``None``) —
    at most one per call so a barge-in inbox sees utterances one at a time;
    extras queue for the next poll. ``close()`` flushes an utterance still
    open (end of session).
    """

    def __init__(
        self,
        mic: MicSource,
        *,
        config: VADConfig | None = None,
        segmenter: SpeechSegmenter | None = None,
        wav: bool = True,
    ) -> None:
        self._mic = mic
        self._cfg = config or VADConfig()
        self._seg = segmenter or SpeechSegmenter(self._cfg)
        self._wav = wav
        self._ready: deque[bytes] = deque()

    def poll(self) -> bytes | None:
        """The ``audio_source`` fill: the next finished utterance, or None."""
        while (frame := self._mic.read()) is not None:
            utt = self._seg.push(frame)
            if utt is not None:
                self._ready.append(utt)
        if not self._ready:
            return None
        pcm = self._ready.popleft()
        return pcm_to_wav(pcm, sample_rate=self._cfg.sample_rate) if self._wav else pcm

    def close(self) -> bytes | None:
        """Flush an utterance still open at end-of-session, if any."""
        utt = self._seg.flush()
        if utt is None:
            return None
        return pcm_to_wav(utt, sample_rate=self._cfg.sample_rate) if self._wav else utt


class SoundDeviceMic:  # pragma: no cover - real hardware; validated on-device
    """A real-microphone `MicSource` over ``sounddevice`` (the ``mic`` extra).

    Frames are captured by a background callback into a bounded queue;
    ``read()`` pops non-blockingly (oldest first; overflow drops the oldest —
    live speech beats stale audio). Install: ``pip install 'voicelink[mic]'``.
    On-device validation pending (real-mic latency/gain tuning).
    """

    def __init__(self, config: VADConfig | None = None, *, max_buffer_frames: int = 256) -> None:
        try:
            import sounddevice  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise RuntimeError(
                "SoundDeviceMic needs the 'mic' extra: pip install 'voicelink[mic]'"
            ) from e
        cfg = config or VADConfig()
        self._frames: deque[bytes] = deque(maxlen=max_buffer_frames)
        block = cfg.sample_rate * cfg.frame_ms // 1000

        def _on_audio(indata, frames, time_info, status) -> None:
            self._frames.append(bytes(indata))

        self._stream = sounddevice.RawInputStream(
            samplerate=cfg.sample_rate,
            blocksize=block,
            channels=1,
            dtype="int16",
            callback=_on_audio,
        )
        self._stream.start()

    def read(self) -> bytes | None:
        return self._frames.popleft() if self._frames else None

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
