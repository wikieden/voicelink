# voicelink

> 🌐 [English](README.md) · **中文**

**Chronotope** 的语音 I/O —— 具身大脑的**耳（ASR）**与**嘴（TTS）**。填 `mindloop` 的
`Listener` / `Speaker` 协议 seam，让大脑能听到口语指令、说出回复，而大脑本身保持纯文本。

第 6 个工作区成员，`bodylink`（填执行器 seam）的语音姊妹：**引擎住这里，大脑不改**。

## 提供什么

- `EndpointListener`（ASR）+ `EndpointSpeaker`（TTS）—— 由 **OpenAI 兼容音频端点**
  （`/v1/audio/transcriptions`、`/v1/audio/speech`）支撑的 fill，对齐 Qwen 边缘栈
  （Qwen3-ASR / Qwen3-TTS）。仅 stdlib（`urllib`）；`transport` 注入 seam 使其网络隔离可测。
- `make_listener` / `make_speaker` 工厂。
- 本体引擎（sherpa-onnx / piper，无网络）在 P7.3 经 `local` extra 加入。

## 用法

```python
from mindloop import VoiceLoop, Conversation
from voicelink import make_listener, make_speaker

loop = VoiceLoop(
    brain,
    make_listener(base_url="http://localhost:8003/v1", model="qwen3-asr"),
    make_speaker(base_url="http://localhost:8003/v1", model="qwen3-tts"),
)
loop.turn(mic_audio, Conversation())   # 听 → 想 → 说
```

这些 fill + 指向本体 LLM 的 `FailoverVerbalizer` reasoner，断网时整条 AI 链路**离线**运行
（P7 目标）。见 `mindloop/docs/zh/devplan/specs/P7-edge-fallback.md`。
