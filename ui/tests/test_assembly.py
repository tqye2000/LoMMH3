from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

import join_videos_with_audio as jv
from ui.core.preflight import load_repo_module
from ui.core.service import Service, ServiceError

from .conftest import REPO_ROOT, run_until
from .test_scheduler import _queued_step

FPS = 24


def write_clip(path: Path, frames: int = 12, size: tuple[int, int] = (64, 48), audio_rate: int | None = None,
               layout: str = "stereo", audio_seconds: float | None = None) -> Path:
    """Write a small test clip; audio is a 440 Hz tone lasting ``audio_seconds`` (default: video length)."""
    width, height = size
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("mpeg4", rate=FPS)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.time_base = Fraction(1, FPS)
        audio = None
        if audio_rate:
            audio = container.add_stream("aac", rate=audio_rate)
            audio.layout = layout
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), index * 20 % 255, np.uint8), "rgb24")
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            channels = 1 if layout == "mono" else 2
            total = round((audio_seconds if audio_seconds is not None else frames / FPS) * audio_rate)
            size_ = audio.codec_context.frame_size or 1024
            for start in range(0, total, size_):
                count = min(size_, total - start)
                t = (np.arange(start, start + count) / audio_rate).astype(np.float32)
                tone = np.tile(0.3 * np.sin(2 * math.pi * 440 * t), (channels, 1)).astype(np.float32)
                chunk = av.AudioFrame.from_ndarray(tone, format="fltp", layout=layout)
                chunk.sample_rate, chunk.pts = audio_rate, start
                for packet in audio.encode(chunk):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return path


def probe(path: Path) -> tuple[int, np.ndarray, int]:
    """Return (video frame count, mono audio samples, audio sample rate)."""
    with av.open(str(path)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return frames, np.zeros(0), 0
        rate = container.streams.audio[0].codec_context.sample_rate
        chunks = [f.to_ndarray().reshape(len(f.layout.channels), -1)[0] for f in container.decode(audio=0)]
    return frames, np.concatenate(chunks), rate


def rms(samples: np.ndarray, rate: int, start_s: float, end_s: float) -> float:
    segment = samples[int(start_s * rate):int(end_s * rate)]
    return float(np.sqrt(np.mean(segment ** 2)))


def test_join_aligns_each_clips_audio_to_its_video(tmp_path: Path) -> None:
    a = write_clip(tmp_path / "a.mp4", audio_rate=48000)
    b = write_clip(tmp_path / "b.mp4", audio_rate=44100, layout="mono", audio_seconds=1.0)  # 2x too long
    c = write_clip(tmp_path / "c.mp4")  # no audio
    result = jv.join_videos([a, b, c], tmp_path / "out" / "joined.mp4")

    assert result.output == (tmp_path / "out" / "joined.mp4").resolve() and result.has_audio
    assert result.video_frames == 36 and result.duration_s == pytest.approx(1.5)
    frames, samples, rate = probe(result.output)
    assert frames == 36 and rate == 48000
    assert len(samples) / rate == pytest.approx(1.5, abs=0.05)
    assert rms(samples, rate, 0.1, 0.4) > 0.1  # clip a
    assert rms(samples, rate, 0.6, 0.9) > 0.1  # clip b, trimmed to its 0.5 s of video
    assert rms(samples, rate, 1.05, 1.45) < 0.01  # clip c is silent, not b's overflow
    assert not list((tmp_path / "out").glob(".*partial*"))


def test_join_without_any_audio_writes_video_only(tmp_path: Path) -> None:
    a, b = write_clip(tmp_path / "a.mp4"), write_clip(tmp_path / "b.mp4", frames=6)
    result = jv.join_videos([a, b], tmp_path / "joined.mp4")
    assert not result.has_audio and result.audio_frames == 0
    assert probe(result.output)[0] == 18


def test_incompatible_inputs_are_rejected_before_writing(tmp_path: Path) -> None:
    a = write_clip(tmp_path / "a.mp4", audio_rate=48000)
    b = write_clip(tmp_path / "b.mp4", size=(96, 48), audio_rate=48000)
    out = tmp_path / "joined.mp4"
    with pytest.raises(ValueError, match="input 2 is 96x48, input 1 is 64x48"):
        jv.join_videos([a, b], out)
    assert sorted(tmp_path.iterdir()) == sorted([a, b])


def test_two_input_cli_still_works(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = write_clip(tmp_path / "a.mp4", audio_rate=48000)
    b = write_clip(tmp_path / "b.mp4", audio_rate=48000)
    out = tmp_path / "joined.mp4"
    assert jv.main([str(a), str(b), str(out)]) == 0
    assert "Encoded video frames: 24" in capsys.readouterr().out
    path, video_frames, _audio_frames = jv.join_videos_with_audio(a, b, tmp_path / "again.mp4")
    assert path.is_file() and video_frames == 24
    with pytest.raises(SystemExit):
        jv.main([str(a), str(out)])


def test_join_script_loads_by_path_without_prior_import(monkeypatch: pytest.MonkeyPatch) -> None:
    # The app loads it by path; @dataclass fails unless the module is registered in sys.modules.
    monkeypatch.delitem(sys.modules, "join_videos_with_audio", raising=False)
    module = load_repo_module.__wrapped__(REPO_ROOT, "join_videos_with_audio")
    assert module.JoinResult.__module__ == "join_videos_with_audio"


async def test_export_joins_selected_takes_in_chosen_order(service: Service) -> None:
    pid, first = _queued_step(service, "Scene one")
    second = service.add_step(pid, "Scene 2").id
    service.save_prompt(pid, second, "Scene two")
    jobs = [service.run_step(pid, first), service.run_step(pid, second)]
    await run_until(service, lambda: all(service.scheduler.jobs[j.id].state == "succeeded" for j in jobs))

    with pytest.raises(ServiceError, match="at least one step"):
        service.export_steps(pid, [])
    third = service.add_step(pid, "Scene 3").id
    with pytest.raises(ServiceError, match="'Scene 3' has no selected take"):
        service.export_steps(pid, [first, third])

    export = service.export_steps(pid, [second, first], "  Final cut ")
    assert export.name == "Final cut" and export.frames == 12 and not export.has_audio
    assert [(i.step_id, i.take_id) for i in export.inputs] == [(second, jobs[1].take_id), (first, jobs[0].take_id)]
    path, record = service.export_file(pid, export.id)
    assert record == export and path.stat().st_size > 0
    assert service.get_project(pid).exports == [export]
    assert service.activity(pid)[0].type == "export.created"

    service.delete_export(pid, export.id)
    assert service.get_project(pid).exports == [] and not path.exists()
    with pytest.raises(ServiceError):
        service.export_file(pid, export.id)
