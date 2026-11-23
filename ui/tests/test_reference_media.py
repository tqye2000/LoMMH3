from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from ui.core.generation import build_command, media_labels
from ui.core.models import ImageBinding
from ui.core.reference_media import MediaRejected
from ui.core.service import Service, ServiceError

from .conftest import png_bytes
from .test_assembly import write_clip


def wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        t = np.arange(int(seconds * rate)) / rate
        handle.writeframes((3000 * np.sin(2 * np.pi * 440 * t)).astype("<i2").tobytes())
    return buffer.getvalue()


def video_bytes(tmp_path: Path, name: str, seconds: float = 3.0, sound: bool = True) -> bytes:
    path = write_clip(tmp_path / name, frames=int(seconds * 24), audio_rate=32000 if sound else None)
    return path.read_bytes()


@pytest.fixture
def project(service: Service, tmp_path: Path) -> dict:
    pid = service.create_project("Refs").id
    image, _ = service.add_image(pid, "face.png", png_bytes((200, 10, 10)))
    video, _ = service.add_media(pid, "motion.mp4", video_bytes(tmp_path, "motion.mp4"))
    silent, _ = service.add_media(pid, "silent.mp4", video_bytes(tmp_path, "silent.mp4", sound=False))
    voice, _ = service.add_media(pid, "voice.wav", wav_bytes(3.0))
    step = service.add_step(pid, "Scene")
    service.save_prompt(pid, step.id, "<Subject 1> speaks with the voice of <Audio 1>.")
    service.update_step(pid, step.id, mode="ref2va")
    return {"pid": pid, "step": step.id, "image": image.id, "video": video.id, "silent": silent.id,
            "voice": voice.id}


def test_upload_records_kind_duration_and_dedupes(service: Service, project: dict) -> None:
    pid = project["pid"]
    media = {m.id: m for m in service.get_project(pid).media}
    video, silent, voice = media[project["video"]], media[project["silent"]], media[project["voice"]]
    assert (video.kind, video.has_audio) == ("video", True)
    assert (silent.kind, silent.has_audio) == ("video", False)
    assert (voice.kind, voice.has_audio) == ("audio", True)
    assert voice.duration_s == pytest.approx(3.0, abs=0.05)
    assert video.duration_s == pytest.approx(3.0, abs=0.1)
    again, created = service.add_media(pid, "copy.wav", wav_bytes(3.0))
    assert not created and again.id == voice.id
    assert len(service.get_project(pid).media) == 3


def test_invalid_media_is_rejected(service: Service, project: dict) -> None:
    pid = project["pid"]
    with pytest.raises(MediaRejected):
        service.add_media(pid, "notes.wav", b"not audio")
    with pytest.raises(MediaRejected):
        service.add_media(pid, "voice.txt", wav_bytes(3.0))
    assert len(service.get_project(pid).media) == 3


def test_audio_needs_an_image_or_video(service: Service, project: dict) -> None:
    pid, sid = project["pid"], project["step"]
    service.update_step(pid, sid, ref_audios=[project["voice"]])
    assert any("cannot be used on its own" in e for e in service.validate_step(pid, sid))
    service.update_step(pid, sid, images=[ImageBinding(asset_id=project["image"])])
    assert service.validate_step(pid, sid) == []
    service.update_step(pid, sid, images=[], ref_videos=[project["video"]])
    assert service.validate_step(pid, sid) == []


def test_kind_rules_on_update(service: Service, project: dict) -> None:
    pid, sid = project["pid"], project["step"]
    with pytest.raises(ServiceError):
        service.update_step(pid, sid, ref_videos=[project["voice"]])
    with pytest.raises(ServiceError):
        service.update_step(pid, sid, ref_audios=[project["silent"]])
    with pytest.raises(ServiceError):
        service.update_step(pid, sid, ref_audios=[project["voice"]] * 2)
    service.update_step(pid, sid, ref_audios=[project["video"]])  # a video's soundtrack is allowed


def test_command_and_labels(service: Service, project: dict, tmp_path: Path) -> None:
    pid, sid = project["pid"], project["step"]
    service.update_step(pid, sid, images=[ImageBinding(asset_id=project["image"])],
                        ref_videos=[project["silent"], project["video"]], ref_audios=[project["voice"]])
    project_obj = service.get_project(pid)
    labels = media_labels(project_obj, project_obj.step(sid))
    assert labels == {("video", project["silent"]): "<Video 1>",
                      ("video", project["video"]): "<Audio 1> + <Video 2>",
                      ("audio", project["voice"]): "<Audio 2>"}

    job = service.run_step(pid, sid)
    project_dir = service.store.project_dir(pid)
    cmd = build_command(service.settings, job, project_dir, tmp_path / "j", tmp_path / "t")
    videos = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--reference-video"]
    audios = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--reference-audio"]
    media = {m.id: m for m in project_obj.media}
    assert videos == [str(project_dir / media[project["silent"]].path), str(project_dir / media[project["video"]].path)]
    assert audios == [str(project_dir / media[project["voice"]].path)]
    assert cmd[cmd.index("--reference-video-short-edge") + 1] == "384"

    service.update_step(pid, sid, mode="fl2va")
    job = service.run_step(pid, sid)
    assert job.request.videos == [] and job.request.audios == []


def test_clip_durations_are_checked(service: Service, project: dict) -> None:
    pid, sid = project["pid"], project["step"]
    long_clip, _ = service.add_media(pid, "long.wav", wav_bytes(16.0))
    service.update_step(pid, sid, images=[ImageBinding(asset_id=project["image"])], ref_audios=[long_clip.id])
    assert any("each must be 2-15 s" in e for e in service.validate_step(pid, sid))

    clips = [service.add_media(pid, f"c{i}.wav", wav_bytes(6.0 + i * 0.1))[0].id for i in range(3)]
    service.update_step(pid, sid, ref_audios=clips)
    assert any("total 18.3 s" in e for e in service.validate_step(pid, sid))
