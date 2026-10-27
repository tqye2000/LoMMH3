#####################################################################
# Join videos end-to-end into one MP4 while preserving audio.
#
# Usage:
#   python join_videos_with_audio.py <video1> <video2> [<video3> ...] <output.mp4>
#   python join_videos_with_audio.py a.mp4 b.mp4 joined.mp4 --fps 24
#
# Inputs must share resolution and frame rate. Each input's audio is
# trimmed or padded with silence to its video duration so later clips
# stay in sync; inputs without audio contribute silence.
#####################################################################

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import av
from av.audio.fifo import AudioFifo
from av.audio.resampler import AudioResampler

AUDIO_FORMAT = "fltp"


@dataclass
class ClipInfo:
    path: Path
    width: int
    height: int
    rate: Fraction
    has_audio: bool
    sample_rate: int | None = None
    layout: str | None = None


@dataclass
class JoinResult:
    output: Path
    video_frames: int
    audio_frames: int
    width: int
    height: int
    rate: Fraction
    has_audio: bool

    @property
    def duration_s(self) -> float:
        return float(self.video_frames / self.rate)


def _video_rate_or_default(stream, fallback_fps: int = 24) -> Fraction:
    """Return a usable output frame rate from a source stream."""
    if stream.average_rate is not None:
        return Fraction(stream.average_rate)
    return Fraction(fallback_fps)


def _audio_layout_name(stream) -> str:
    """Pick a stable channel layout name for AAC encoding."""
    if stream.layout is not None and stream.layout.name:
        return stream.layout.name

    channels = stream.codec_context.channels or 2
    return "mono" if channels == 1 else "stereo"


def inspect_clip(path: Path) -> ClipInfo:
    """Read the stream properties of one input without decoding it."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in: {source}")
        video = container.streams.video[0]
        info = ClipInfo(
            path=source,
            width=video.codec_context.width,
            height=video.codec_context.height,
            rate=_video_rate_or_default(video),
            has_audio=bool(container.streams.audio),
        )
        if container.streams.audio:
            audio = container.streams.audio[0]
            info.sample_rate = audio.codec_context.sample_rate or 48000
            info.layout = _audio_layout_name(audio)
    return info


def check_compatible(clips: Sequence[ClipInfo]) -> None:
    """Raise ValueError listing every input whose resolution or frame rate differs from input 1."""
    if not clips:
        raise ValueError("At least one input video is required.")
    first = clips[0]
    problems = []
    for number, clip in enumerate(clips[1:], start=2):
        if (clip.width, clip.height) != (first.width, first.height):
            problems.append(
                f"input {number} is {clip.width}x{clip.height}, input 1 is {first.width}x{first.height}"
            )
        if clip.rate != first.rate:
            problems.append(
                f"input {number} is {float(clip.rate):g} fps, input 1 is {float(first.rate):g} fps"
            )
    if problems:
        raise ValueError("Incompatible inputs: " + "; ".join(problems) + ".")


class _AudioTrack:
    """AAC output track that keeps each clip's audio exactly as long as its video."""

    def __init__(self, out, sample_rate: int, layout: str) -> None:
        self.out = out
        self.sample_rate = sample_rate
        self.layout = layout
        self.time_base = Fraction(1, sample_rate)
        self.stream = out.add_stream("aac", rate=sample_rate)
        self.stream.layout = layout
        self.stream.time_base = self.time_base
        # AAC requires fixed-size frames, so audio is re-chunked through a FIFO.
        self.frame_size = self.stream.codec_context.frame_size or 1024
        self.fifo = AudioFifo()
        self.clip = AudioFifo()
        self.committed = 0  # samples handed to self.fifo so far
        self.pts = 0
        self.frames = 0

    def resampler(self) -> AudioResampler:
        return AudioResampler(format=AUDIO_FORMAT, layout=self.layout, rate=self.sample_rate)

    def add(self, frames: Iterable[av.AudioFrame]) -> None:
        for frame in frames:
            frame.pts = None
            frame.time_base = self.time_base
            self.clip.write(frame)

    def end_clip(self, end_sample: int) -> None:
        """Trim or pad the current clip's audio so the track ends at *end_sample*."""
        need = end_sample - self.committed
        if need > 0:
            have = min(need, self.clip.samples)
            if have:
                self._push(self.clip.read(have))
            if need > have:
                self._push(self._silence(need - have))
            self.committed = end_sample
        self.clip = AudioFifo()
        self._drain(final=False)

    def finish(self) -> None:
        self._drain(final=True)
        for packet in self.stream.encode():
            self.out.mux(packet)

    def _silence(self, samples: int) -> av.AudioFrame:
        frame = av.AudioFrame(format=AUDIO_FORMAT, layout=self.layout, samples=samples)
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.sample_rate = self.sample_rate
        return frame

    def _push(self, frame: av.AudioFrame | None) -> None:
        if frame is None:
            return
        # Frames read from the clip FIFO carry its timestamps, which the output FIFO would reject.
        frame.pts = None
        frame.time_base = self.time_base
        self.fifo.write(frame)

    def _drain(self, final: bool) -> None:
        while self.fifo.samples >= self.frame_size or (final and self.fifo.samples):
            chunk = self.fifo.read(min(self.frame_size, self.fifo.samples))
            if chunk is None:
                return
            chunk.pts = self.pts
            chunk.time_base = self.time_base
            self.pts += chunk.samples
            for packet in self.stream.encode(chunk):
                self.out.mux(packet)
            self.frames += 1


def _encode(clips: Sequence[ClipInfo], out_path: Path, rate: Fraction) -> JoinResult:
    first = clips[0]
    audio_source = next((clip for clip in clips if clip.has_audio), None)
    result = JoinResult(out_path, 0, 0, first.width, first.height, rate, audio_source is not None)
    video_time_base = 1 / rate

    with av.open(str(out_path), mode="w") as out:
        out_video = out.add_stream("libx264", rate=rate)
        out_video.width = first.width
        out_video.height = first.height
        out_video.pix_fmt = "yuv420p"
        out_video.time_base = video_time_base
        out_video.options = {"crf": "18", "preset": "medium"}

        audio = None
        if audio_source is not None:
            audio = _AudioTrack(out, audio_source.sample_rate or 48000, audio_source.layout or "stereo")

        for clip in clips:
            with av.open(str(clip.path)) as src:
                streams: list = [src.streams.video[0]]
                resampler = None
                if audio is not None and src.streams.audio:
                    streams.append(src.streams.audio[0])
                    resampler = audio.resampler()

                # Decode both streams in one interleaved pass; separate passes would
                # drain the demuxer on the first and leave the second empty.
                for frame in src.decode(*streams):
                    if isinstance(frame, av.VideoFrame):
                        out_frame = frame.reformat(format="yuv420p")
                        out_frame.pts = result.video_frames
                        out_frame.time_base = video_time_base
                        for packet in out_video.encode(out_frame):
                            out.mux(packet)
                        result.video_frames += 1
                    elif audio is not None and resampler is not None and isinstance(frame, av.AudioFrame):
                        audio.add(resampler.resample(frame))

            if audio is not None:
                if resampler is not None:
                    audio.add(resampler.resample(None))
                # Align to the cumulative video end so rounding never accumulates drift.
                audio.end_clip(round(result.video_frames * audio.sample_rate / rate))

        for packet in out_video.encode():
            out.mux(packet)
        if audio is not None:
            audio.finish()
            result.audio_frames = audio.frames

    return result


def join_videos(
    inputs: Sequence[Path],
    output: Path,
    fps: int | None = None,
) -> JoinResult:
    """Concatenate *inputs* in order into an H.264/AAC MP4 at *output*.

    Every input is inspected before anything is written, and incompatible
    inputs raise ValueError. The output path only appears once encoding has
    completed successfully.
    """
    clips = [inspect_clip(Path(path)) for path in inputs]
    check_compatible(clips)

    out_path = output.expanduser().resolve()
    if any(clip.path == out_path for clip in clips):
        raise ValueError(f"Output would overwrite an input: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    partial = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    rate = Fraction(fps) if fps is not None else clips[0].rate
    try:
        result = _encode(clips, partial, rate)
        os.replace(partial, out_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    result.output = out_path
    return result


def join_videos_with_audio(
    video1: Path,
    video2: Path,
    output: Path,
    fps: int | None = None,
) -> tuple[Path, int, int]:
    """Two-input form; returns ``(output_path, video_frame_count, audio_frame_count)``."""
    result = join_videos([video1, video2], output, fps)
    return result.output, result.video_frames, result.audio_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join videos end-to-end into one MP4 while preserving audio."
    )
    parser.add_argument(
        "inputs", type=Path, nargs="+", metavar="video",
        help="Input videos in playback order (at least two).",
    )
    parser.add_argument("output", type=Path, help="Path to the output MP4 file.")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Force output FPS (default: first video's FPS, fallback 24).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.inputs) < 2:
        parser.error("at least two input videos are required")

    try:
        result = join_videos(args.inputs, args.output, args.fps)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Output: {result.output}")
    print(f"Encoded video frames: {result.video_frames}")
    print(f"Encoded audio frames: {result.audio_frames}")
    print(f"Duration: {result.duration_s:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
