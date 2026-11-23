"""Validate uploaded reference videos and audio clips and store them immutably."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import av

from . import constants as C
from .models import MediaAsset, new_id, utc_now


class MediaRejected(ValueError):
    pass


def _duration_s(container) -> float:
    if container.duration:
        return container.duration / av.time_base
    durations = [
        float(stream.duration * stream.time_base)
        for stream in (*container.streams.video, *container.streams.audio)
        if stream.duration and stream.time_base
    ]
    return max(durations, default=0.0)


def ingest_media(project_dir: Path, display_name: str, data: bytes) -> MediaAsset:
    """Validate ``data`` as a video or audio file, store it under ``media/``, and return its record."""
    if len(data) > C.MAX_MEDIA_UPLOAD_BYTES:
        raise MediaRejected(f"{display_name}: larger than {C.MAX_MEDIA_UPLOAD_BYTES // (1024 * 1024)} MB")
    suffix = Path(display_name).suffix.lower()
    if suffix not in (*C.VIDEO_SUFFIXES, *C.AUDIO_SUFFIXES):
        raise MediaRejected(f"{display_name}: unsupported file type {suffix or '(none)'}")
    try:
        with av.open(io.BytesIO(data)) as container:
            has_video = bool(container.streams.video)
            has_audio = bool(container.streams.audio)
            duration = _duration_s(container)
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise MediaRejected(f"{display_name}: not a readable media file ({exc})") from exc
    # Audio files may carry cover art as a video stream, so the suffix decides the kind.
    kind = "video" if suffix in C.VIDEO_SUFFIXES else "audio"
    if kind == "video" and not has_video:
        raise MediaRejected(f"{display_name}: no video stream")
    if kind == "audio" and not has_audio:
        raise MediaRejected(f"{display_name}: no audio stream")
    if duration <= 0:
        raise MediaRejected(f"{display_name}: unknown duration")

    media_id = new_id("med")
    folder = project_dir / "media"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{media_id}{suffix}").write_bytes(data)
    return MediaAsset(
        id=media_id,
        kind=kind,
        name=Path(display_name).name[:120] or f"{media_id}{suffix}",
        path=f"media/{media_id}{suffix}",
        duration_s=round(duration, 3),
        has_audio=has_audio,
        sha256=hashlib.sha256(data).hexdigest(),
        added_utc=utc_now(),
    )
