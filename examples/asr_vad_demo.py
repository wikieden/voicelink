"""ASR+VAD v1 demo — mic frames → VAD → barge-in, the whole ear (P7.3).

Offline by default: a scripted mic replays synthetic PCM (tone = speech), the
VAD segments it into utterances, and the segment lands in the SAME barge-in
inbox the brain polls mid-task — a "stop" spoken mid-navigation cancels the
in-flight step through the T5-8 handshake. No audio hardware, no network.

Real-device path (validated on-device later):

    stream = SpeechStream(SoundDeviceMic())            # pip install 'voicelink[mic]'
    listener = make_listener(base_url="http://<edge>:8003/v1")   # Qwen3-ASR
    interrupts = ListenerInterruptSource(listener, audio_source=stream.poll)

Run:  uv run python voicelink/examples/asr_vad_demo.py
"""

from __future__ import annotations

import math
import struct

import numpy as np
from bodylink import MockBody
from mindloop import (
    Brain,
    Command,
    ListenerInterruptSource,
    MockListener,
    MockSpeaker,
    ScriptedIntentRouter,
    ScriptedPlanner,
    ScriptedReasoner,
)
from tempomem import Detection, TempoMem

from voicelink import ScriptedMic, SpeechStream, VADConfig

CFG = VADConfig(
    sample_rate=16000, frame_ms=30, energy_threshold=500.0, start_frames=2, end_frames=3
)
_SAMPLES = CFG.sample_rate * CFG.frame_ms // 1000


def _tone() -> bytes:
    return struct.pack(
        f"<{_SAMPLES}h",
        *(int(8000 * math.sin(2 * math.pi * 440 * i / CFG.sample_rate)) for i in range(_SAMPLES)),
    )


def _silence() -> bytes:
    return b"\x00\x00" * _SAMPLES


class _SpeakDuringStep(MockBody):
    """Feeds one spoken burst into the mic as the FIRST step starts (mid-task)."""

    def __init__(self, mic: ScriptedMic) -> None:
        super().__init__()
        self._mic = mic
        self._done = False

    def dispatch(self, subtask):
        if not self._done:
            self._done = True
            for f in [_silence()] * 2 + [_tone()] * 6 + [_silence()] * 4:
                self._mic.push(f)
        yield from super().dispatch(subtask)


def main() -> int:
    mem = TempoMem.open(":memory:", embedding_dim=4)
    v = np.ones(4, dtype="float32") / 2.0
    mem.add_detections([Detection("mug", v, (0.5, 0.5, 0.8), (0.4, 0.4, 0.7), (0.6, 0.6, 0.9))])
    mem.commit()

    mic = ScriptedMic()
    stream = SpeechStream(mic, config=CFG)
    # MockListener plays the ASR: the segmented WAV arrives, "stop" comes back.
    interrupts = ListenerInterruptSource(MockListener(["stop"]), audio_source=stream.poll)
    speaker = MockSpeaker()
    body = _SpeakDuringStep(mic)
    brain = Brain(
        mem,
        ScriptedReasoner(),
        router=ScriptedIntentRouter({}, default="task"),
        planner=ScriptedPlanner(
            {"bring the mug": (Command("navigate", ("mug",)), Command("grasp", ("mug",)))}
        ),
        executor=body,
        interrupts=interrupts,
        speaker=speaker,
    )

    print('1) task starts; user speaks mid-step (VAD segments the burst) …')
    r = brain.handle("bring the mug")
    statuses = [st.status.value for st in r.statuses]
    print(f"   step outcomes: {statuses}")
    for line in speaker.said:
        print(f"   🗣  {line}")

    ok = statuses == ["aborted"] and any("stopping" in s.lower() for s in speaker.said)
    print(f"\nmic → VAD → barge-in {'GO' if ok else 'NO-GO'} — "
          "spoken 'stop' cancelled the in-flight step")
    mem.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
