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


@pytest.mark.parametrize("duration", [15.01, 16, 28.75, 29, 30, 60, 120.5])
def test_long_video_section_plan(duration: float) -> None:
    import math

    from generate_reference_video import FPS, MAX_FRAMES, MIN_FRAMES, plan_sections

    sections = plan_sections(duration)
    assert len(sections) == math.ceil(duration * FPS / MAX_FRAMES)
    assert sum(section.frames for section in sections) == math.ceil(duration * FPS)
    next_start = 0
    for section in sections:
        assert section.start_frame == next_start
        assert MIN_FRAMES <= section.frames <= section.generation_frames <= MAX_FRAMES
        assert (section.generation_frames - 5) % 17 == 0
        next_start += section.frames


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_long_video_section_plan_rejects_invalid_duration(duration: float) -> None:
    from generate_reference_video import plan_sections

    with pytest.raises(ValueError, match="duration"):
        plan_sections(duration)


@pytest.mark.parametrize("frames", [72, 360])
def test_short_reference_video_is_forwarded_unchanged(tmp_path: Path, monkeypatch, frames: int) -> None:
    import generate_reference_video as runner

    source = write_clip(tmp_path / "source.mp4", frames=frames)
    arguments = ["--reference-video", str(source), "--frames", "124", "--seed", "22"]
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    assert runner.main(arguments) == 0
    assert len(calls) == 1 and calls[0][2:] == arguments


@pytest.mark.parametrize("sound", [True, False])
@pytest.mark.parametrize("fail_after_first", [True, False])
def test_long_reference_video_split_generate_and_join(tmp_path: Path, monkeypatch, sound: bool,
                                                     fail_after_first: bool) -> None:
    import generate_reference_video as runner
    from .test_assembly import probe, rms

    source = write_clip(tmp_path / "long source.mp4", frames=720, audio_rate=32000 if sound else None)
    output_dir = tmp_path / "generated videos"
    actual_run = runner.subprocess.run
    calls = []

    def fake_generation(command, **kwargs):
        if len(command) > 1 and Path(command[1]).name == "LoMMH.py":
            calls.append(command)
            if fail_after_first and len(calls) == 2:
                raise runner.subprocess.CalledProcessError(1, command)
            reference = Path(command[command.index("--reference-video") + 1])
            reference_frames, samples, rate = probe(reference)
            assert 124 <= reference_frames <= 345
            assert bool(rate) == sound
            if sound:
                assert rms(samples, rate, 0.1, reference_frames / 24 - 0.1) > 0.1
            frame_arg = len(command) - 1 - command[::-1].index("--frames")
            generation_frames = int(command[frame_arg + 1])
            generated = Path(command[command.index("--output-dir") + 1]) / command[command.index("--output") + 1]
            write_clip(generated, frames=generation_frames, audio_rate=32000 if sound else None)
            generated.with_suffix(".json").write_text("{}", encoding="utf-8")
            return None
        return actual_run(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fake_generation)
    arguments = [
        "--reference-video", str(source), "--output-dir", str(output_dir), "--output", "joined",
        "--frames", "124", "--prompt", "Keep the motion", "--seed", "22",
    ]
    if fail_after_first:
        output_dir.mkdir()
        existing = output_dir / "joined.mp4"
        existing.write_bytes(b"previous output")
        with pytest.raises(runner.subprocess.CalledProcessError):
            runner.main(arguments)
        assert len(calls) == 2
        assert existing.read_bytes() == b"previous output"
        assert len(list(output_dir.glob("joined_parts_*/section_0001_generated.mp4"))) == 1
        assert len(list(output_dir.glob("joined_parts_*/sections.json"))) == 1
        return
    assert runner.main(arguments) == 0
    assert len(calls) == 3
    assert all(command[command.index("--prompt") + 1] == "Keep the motion" for command in calls)
    frames, samples, rate = probe(output_dir / "joined.mp4")
    assert frames == 720 and bool(rate) == sound
    if sound:
        for start in (0.1, 14.5, 25.0, 29.0):
            assert rms(samples, rate, start, start + 0.5) > 0.1
        assert len(samples) / rate == pytest.approx(30, abs=0.05)
    assert (output_dir / "joined.json").is_file()
    assert runner.video_duration(source) == 30


def test_reference_cut_converts_frame_rate_and_seeks(tmp_path: Path) -> None:
    import av
    import generate_reference_video as runner
    from .test_assembly import probe

    original = write_clip(tmp_path / "original.mp4", frames=240, audio_rate=32000)
    converted = tmp_path / "30fps.mp4"
    cut = tmp_path / "cut.mp4"
    executable = runner.ffmpeg_executable()
    runner.subprocess.run([
        executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(original),
        "-vf", "fps=30", "-c:v", "libx264", "-c:a", "aac", str(converted),
    ], check=True)
    with av.open(str(converted)) as container:
        assert container.streams.video[0].average_rate == 30
    runner.cut_video(executable, converted, cut, 1, 124)
    frames, samples, rate = probe(cut)
    assert frames == 124
    assert len(samples) / rate == pytest.approx(124 / 24, abs=0.05)
    with av.open(str(cut)) as container:
        assert container.streams.video[0].average_rate == 24
        first = next(container.decode(video=0)).to_ndarray(format="rgb24")
        assert float(first.mean()) == pytest.approx(24 * 20 % 255, abs=5)


def test_section_constants_match_model() -> None:
    import generate_reference_video as runner
    from .test_constants import _lommh_constants

    model = _lommh_constants()
    assert (runner.FPS, runner.MIN_FRAMES, runner.MAX_FRAMES) == (
        model["FPS"], model["MIN_FRAMES"], model["MAX_FRAMES"],
    )
    for section in runner.plan_sections(30):
        assert (section.generation_frames - model["_FRAME_BASE"]) % model["_FRAME_STEP"] == 0


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


def test_ui_accepts_one_long_reference_video(service: Service, project: dict, tmp_path: Path) -> None:
    pid, sid = project["pid"], project["step"]
    long_video, _ = service.add_media(pid, "Dance-P_white.mp4", video_bytes(tmp_path, "long.mp4", 23.5))
    service.update_step(pid, sid, ref_videos=[long_video.id])
    assert service.validate_step(pid, sid) == []
    job = service.run_step(pid, sid)
    assert [video.media_id for video in job.request.videos] == [long_video.id]
    command = build_command(service.settings, job, service.store.project_dir(pid), tmp_path / "job", tmp_path / "take")
    assert Path(command[1]).name == "generate_reference_video.py"
    assert command[command.index("--generator-script") + 1] == str(service.settings.generator_script)

    service.update_step(pid, sid, ref_videos=[long_video.id, project["silent"]])
    assert service.validate_step(pid, sid)
    service.update_step(pid, sid, ref_videos=[project["video"]], ref_audios=[long_video.id])
    assert any("each must be 2-15 s" in error for error in service.validate_step(pid, sid))


@pytest.mark.parametrize("seconds", [3, 23.5])
async def test_ui_reference_video_job_records_one_complete_take(
    service: Service, project: dict, tmp_path: Path, monkeypatch, seconds: float,
) -> None:
    import json
    from .conftest import run_until
    from .test_assembly import probe

    monkeypatch.setenv("FAKE_GEN_MODE", "full")
    pid, sid = project["pid"], project["step"]
    video, _ = service.add_media(pid, "dance.mp4", video_bytes(tmp_path, "dance.mp4", seconds))
    service.update_step(pid, sid, ref_videos=[video.id])
    job = service.run_step(pid, sid)
    await run_until(service, lambda: service.scheduler.jobs[job.id].state in ("succeeded", "failed"))
    finished = service.scheduler.jobs[job.id]
    assert finished.state == "succeeded", finished.error
    assert finished.progress.stage == "done"
    takes = service.get_project(pid).step(sid).takes
    assert len(takes) == 1
    expected_frames = int(seconds * 24) if seconds > 15 else job.request.params.frames
    assert takes[0].frames == expected_frames
    result = service.take_file(pid, sid, job.take_id, "video")
    assert probe(result)[0] == expected_frames
    logged = json.loads(result.with_suffix(".json").read_text(encoding="utf-8"))
    assert logged["run_id"] == job.id and logged["parameters"]["frames"] == expected_frames
    events = [json.loads(line) for line in
              (service.store.job_dir(pid, job.id) / "progress.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "done" for event in events) == 1
    assert [event["path"] for event in events if event["event"] == "saved"] == [str(result)]
    if seconds > 15:
        assert len(logged["sections"]) == 2
        assert any(event.get("name") == "generating section 2/2" for event in events)
        assert any(event.get("name") == "joining" for event in events)


@pytest.mark.parametrize("mode", ["fail", "child"])
async def test_ui_long_reference_video_failure_and_cancellation(
    service: Service, project: dict, tmp_path: Path, monkeypatch, mode: str,
) -> None:
    from ui.core import procs
    from .conftest import run_until

    monkeypatch.setenv("FAKE_GEN_MODE", mode)
    pid, sid = project["pid"], project["step"]
    video, _ = service.add_media(pid, "dance.mp4", video_bytes(tmp_path, "dance.mp4", 23.5))
    service.update_step(pid, sid, ref_videos=[video.id])
    job = service.run_step(pid, sid)
    take_dir = service.store.project_dir(pid) / "outputs" / sid / job.take_id
    if mode == "child":
        await run_until(service, lambda: bool(list(take_dir.glob("clip_parts_*/child.pid"))))
        process = service.scheduler.jobs[job.id].process
        assert process is not None
        service.cancel_job(job.id)
        await run_until(service, lambda: service.scheduler.jobs[job.id].state == "cancelled")
        assert not procs.group_alive(process.pgid)
    else:
        await run_until(service, lambda: service.scheduler.jobs[job.id].state == "failed")
    assert service.get_project(pid).step(sid).takes == []
    assert not (take_dir / "clip.mp4").exists()
