# voicelink

> 🌐 **English** · [中文](README.zh.md)

Voice I/O for **Chronotope** — the **ears (ASR)** and **mouth (TTS)** of the
embodied brain. Fills `mindloop`'s `Listener` / `Speaker` protocol seams so the
brain can hear a spoken instruction and speak its reply while staying text-only.

The 6th workspace member, the speech sibling of `bodylink` (which fills the
executor seam): **engines live here, the brain does not change**.

## What it ships

- `EndpointListener` (ASR) + `EndpointSpeaker` (TTS) — fills backed by an
  **OpenAI-compatible audio endpoint** (`/v1/audio/transcriptions`,
  `/v1/audio/speech`), the Qwen-aligned edge stack (Qwen3-ASR / Qwen3-TTS). Stdlib
  only (`urllib`); a `transport` injection seam keeps them network-free testable.
- `make_listener` / `make_speaker` factories.
- On-device engines (sherpa-onnx / piper, no network) land behind the `local`
  extra in P7.3.

## Use

```python
from mindloop import VoiceLoop, Conversation
from voicelink import make_listener, make_speaker

loop = VoiceLoop(
    brain,
    make_listener(base_url="http://localhost:8003/v1", model="qwen3-asr"),
    make_speaker(base_url="http://localhost:8003/v1", model="qwen3-tts"),
)
loop.turn(mic_audio, Conversation())   # hear → think → speak
```

With these fills + a `FailoverVerbalizer` reasoner pointed at an on-body LLM, the
whole AI chain runs **offline** when the network drops (the P7 goal). See
`mindloop/docs/en/devplan/specs/P7-edge-fallback.md`.
