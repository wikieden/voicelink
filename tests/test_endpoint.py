"""voicelink — endpoint ASR/TTS fills (network-free via injected transports)."""

from __future__ import annotations

import numpy as np
import pytest
from bodylink import MockBody
from mindloop import (
    Brain,
    Command,
    Conversation,
    Listener,
    ScriptedIntentRouter,
    ScriptedPlanner,
    ScriptedReasoner,
    Speaker,
    TaskStatus,
    VoiceLoop,
)
from tempomem import Detection, SpatialMemory

from voicelink import (
    EndpointListener,
    EndpointSpeaker,
    VoiceError,
    make_listener,
    make_speaker,
)


def test_listener_fills_mindloop_seam() -> None:
    lis = EndpointListener(transport=lambda audio: "  bring the apple  ")
    assert isinstance(lis, Listener)  # fills mindloop's ASR protocol
    assert lis.transcribe(b"<wav>") == "bring the apple"  # trimmed


def test_speaker_fills_mindloop_seam_and_emits_to_sink() -> None:
    out: list[bytes] = []
    spk = EndpointSpeaker(transport=lambda text: f"AUDIO:{text}".encode(), sink=out.append)
    assert isinstance(spk, Speaker)
    spk.say("hello")
    assert out == [b"AUDIO:hello"]
    assert spk.last_audio == b"AUDIO:hello"


def test_factories_and_unknown_kind() -> None:
    assert isinstance(make_listener(transport=lambda a: "x"), EndpointListener)
    assert isinstance(make_speaker(transport=lambda t: b"y"), EndpointSpeaker)
    with pytest.raises(ValueError, match="unknown listener kind"):
        make_listener("local")
    with pytest.raises(ValueError, match="unknown speaker kind"):
        make_speaker("local")


def test_transport_failure_surfaces_voiceerror() -> None:
    def boom(_audio: bytes) -> str:
        raise VoiceError("server down")

    with pytest.raises(VoiceError):
        EndpointListener(transport=boom).transcribe(b"x")


def test_voiceloop_with_voicelink_fills(tmp_path) -> None:
    # cross-package: mindloop VoiceLoop driven by voicelink's ASR/TTS fills
    mem = SpatialMemory.open(tmp_path / "s.smem", embedding_dim=8)
    rng = np.random.default_rng(0).standard_normal(8).astype("float32")
    mem.add_detections([Detection("apple", rng / np.linalg.norm(rng), (1, 0, 0.9),
                                  (0.9, -0.1, 0.8), (1.1, 0.1, 1.0))])
    mem.commit()
    brain = Brain(
        mem,
        ScriptedReasoner(),
        router=ScriptedIntentRouter(default="task"),
        planner=ScriptedPlanner({"bring the apple": (Command("grasp", ("apple",)),
                                                     Command("speak", ("here you go",)))}),
        executor=MockBody(),
    )
    spoken: list[bytes] = []
    loop = VoiceLoop(
        brain,
        EndpointListener(transport=lambda audio: "bring the apple"),
        EndpointSpeaker(transport=lambda text: text.encode(), sink=spoken.append),
    )
    r = loop.turn(b"<mic>", Conversation())
    assert r.statuses[-1].status == TaskStatus.SUCCESS
    assert spoken == [b"here you go"]  # the plan's utterance was synthesized + emitted
