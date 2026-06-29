"""voicelink — the voice I/O layer for Chronotope (ears + mouth).

Fills `mindloop`'s `Listener` (ASR) and `Speaker` (TTS) protocol seams so the brain
can hear and speak while staying text-only. The default fills wrap an
OpenAI-compatible audio endpoint (Qwen3-ASR / Qwen3-TTS); on-device engines land
behind the `local` extra (P7.3). With these + a `FailoverVerbalizer` reasoner, the
whole AI chain runs on-body when the network drops (P7).

    from mindloop import VoiceLoop, Conversation
    from voicelink import make_listener, make_speaker

    loop = VoiceLoop(brain, make_listener(base_url="http://localhost:8003/v1"),
                     make_speaker(base_url="http://localhost:8003/v1"))
    loop.turn(mic_audio, Conversation())
"""

from __future__ import annotations

from .endpoint import (
    EndpointListener,
    EndpointSpeaker,
    VoiceError,
    make_listener,
    make_speaker,
)

__version__ = "0.1.0a0"

__all__ = [
    "EndpointListener",
    "EndpointSpeaker",
    "VoiceError",
    "make_listener",
    "make_speaker",
]
