"""OpenAI-compatible audio-endpoint fills for the voice seams (P7).

`voicelink` fills `mindloop`'s `Listener` (ASR, the ears) and `Speaker` (TTS, the
mouth) protocols. The default fills talk to an **OpenAI-compatible audio endpoint**
(`/v1/audio/transcriptions`, `/v1/audio/speech`) — the Qwen-aligned edge stack
(Qwen3-ASR / Qwen3-TTS) served by vLLM/NIM, on the workstation or on-body. Stdlib
only (`urllib`), so the package is dependency-light; on-device engines
(sherpa-onnx / piper) land behind the `local` extra in P7.3.

`transport` is the injection seam (same pattern as `CosmosReasonVerbalizer`): a
callable that does the actual ASR/TTS call, so the whole thing is network-free
testable. The default transport is a thin `urllib` POST — validate the exact wire
format against the chosen server during P7.3.

Security: `base_url` is caller-supplied trusted input (your own endpoint); the
bearer token rides along — do not point it at an untrusted URL (SSRF).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable

# ASR: audio bytes -> transcript. TTS: text -> audio bytes.
ASRTransport = Callable[[bytes], str]
TTSTransport = Callable[[str], bytes]

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VoiceError(RuntimeError):
    """A voice I/O transport failure (network down, server unreachable, bad reply)."""


class EndpointListener:
    """`Listener` (ASR) backed by an OpenAI-compatible transcription endpoint.

    `transcribe(audio)` returns the recognized text. The on-body fallback points
    `base_url` at a local Qwen3-ASR server; with a `FailoverVerbalizer`-style setup
    the whole chain runs offline.
    """

    def __init__(
        self,
        *,
        model: str = "qwen3-asr",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        transport: ASRTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._api_key = api_key or os.environ.get("VOICELINK_API_KEY")
        self._transport = transport or self._http

    def transcribe(self, audio: bytes) -> str:
        """Decode `audio` to a text instruction (empty string ⇒ silence/no speech)."""
        return self._transport(audio).strip()

    def _http(self, audio: bytes) -> str:  # pragma: no cover - host-validated in P7.3
        boundary = f"----voicelink{uuid.uuid4().hex}"
        body = _multipart(boundary, fields={"model": self.model}, file=("audio.wav", audio))
        req = urllib.request.Request(
            f"{self.base_url}/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key or 'EMPTY'}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["text"])
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            raise VoiceError(f"ASR request failed: {e}") from e


class EndpointSpeaker:
    """`Speaker` (TTS) backed by an OpenAI-compatible speech endpoint.

    `say(text)` synthesizes audio and hands the bytes to `sink` (an audio device /
    file writer); `synthesize(text)` returns the raw audio. `last_audio` keeps the
    most recent clip (handy for tests / replay).
    """

    def __init__(
        self,
        *,
        model: str = "qwen3-tts",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        voice: str = "default",
        sink: Callable[[bytes], None] | None = None,
        transport: TTSTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.voice = voice
        self.last_audio: bytes = b""
        self._sink = sink
        self._api_key = api_key or os.environ.get("VOICELINK_API_KEY")
        self._transport = transport or self._http

    def synthesize(self, text: str) -> bytes:
        """Return synthesized audio bytes for `text`."""
        return self._transport(text)

    def say(self, text: str) -> None:
        """Synthesize `text` and emit it to the sink (no-op sink just records it)."""
        audio = self.synthesize(text)
        self.last_audio = audio
        if self._sink is not None:
            self._sink(audio)

    def _http(self, text: str) -> bytes:  # pragma: no cover - host-validated in P7.3
        payload = {"model": self.model, "input": text, "voice": self.voice}
        req = urllib.request.Request(
            f"{self.base_url}/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key or 'EMPTY'}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise VoiceError(f"TTS request failed: {e}") from e


def _multipart(boundary: str, *, fields: dict[str, str], file: tuple[str, bytes]) -> bytes:
    """Minimal multipart/form-data encoder (one file + text fields)."""
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n".encode()
        out += f"{v}\r\n".encode()
    name, data = file
    out += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode()
    out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out)


def make_listener(kind: str = "endpoint", **kwargs: object) -> EndpointListener:
    """Build a `Listener`. ``"endpoint"`` → `EndpointListener`. On-device kinds land in P7.3."""
    if kind == "endpoint":
        return EndpointListener(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown listener kind {kind!r}; known: 'endpoint'")


def make_speaker(kind: str = "endpoint", **kwargs: object) -> EndpointSpeaker:
    """Build a `Speaker`. ``"endpoint"`` → `EndpointSpeaker`. On-device kinds land in P7.3."""
    if kind == "endpoint":
        return EndpointSpeaker(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown speaker kind {kind!r}; known: 'endpoint'")
