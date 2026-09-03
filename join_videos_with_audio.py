#####################################################################
# Join two videos end-to-end into one MP4 while preserving audio.
#
# Usage:
#   python join_videos_with_audio.py <video1> <video2> <output.mp4>
#   python join_videos_with_audio.py a.mp4 b.mp4 joined.mp4 --fps 24
#####################################################################

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import av
from av.audio.fifo import AudioFifo
from av.audio.resampler import AudioResampler


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


def join_videos_with_audio(
    video1: Path,
    video2: Path,
    output: Path,
    fps: int | None = None,
) -> tuple[Path, int, int]:
    """Concatenate *video1* + *video2* into *output*.

    Returns ``(output_path, video_frame_count, audio_frame_count)``.
    Both inputs must contain one decodable video stream and one decodable
    audio stream. The output is H.264 video + AAC audio in MP4.
    """
    src1 = video1.expanduser().resolve()
    src2 = video2.expanduser().resolve()
    out_path = output.expanduser().resolve()

    if not src1.is_file():
        raise FileNotFoundError(f"Video file not found: {src1}")
    if not src2.is_file():
        raise FileNotFoundError(f"Video file not found: {src2}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(src1)) as first:
        if not first.streams.video:
            raise ValueError(f"No video stream found in: {src1}")
        if not first.streams.audio:
            raise ValueError(f"No audio stream found in: {src1}")

        first_video = first.streams.video[0]
        first_audio = first.streams.audio[0]

        target_width = first_video.codec_context.width
        target_height = first_video.codec_context.height
        target_video_rate = (
            Fraction(fps) if fps is not None else _video_rate_or_default(first_video)
        )

        target_sample_rate = first_audio.codec_context.sample_rate or 48000
        target_layout = _audio_layout_name(first_audio)

    video_time_base = 1 / target_video_rate
    audio_time_base = Fraction(1, target_sample_rate)

    video_frames = 0
    audio_frames = 0

    with av.open(str(out_path), mode="w") as out:
        out_video = out.add_stream("libx264", rate=target_video_rate)
        out_video.width = target_width
        out_video.height = target_height
        out_video.pix_fmt = "yuv420p"
        out_video.time_base = video_time_base
        out_video.options = {"crf": "18", "preset": "medium"}

        out_audio = out.add_stream("aac", rate=target_sample_rate)
        out_audio.layout = target_layout
        out_audio.time_base = audio_time_base

        # AAC requires fixed-size frames (typically 1024 samples); buffer through
        # a FIFO so variable-size decoded/resampled frames are re-chunked.
        frame_size = out_audio.codec_context.frame_size or 1024
        fifo = AudioFifo()

        # Running output timestamps so the second clip continues after the first
        # instead of resetting to zero (which collapses the reported duration).
        video_pts = 0
        audio_pts = 0

        def drain_fifo(flush: bool) -> None:
            nonlocal audio_pts, audio_frames
            while True:
                if flush:
                    chunk = fifo.read(partial=True)
                else:
                    if fifo.samples < frame_size:
                        return
                    chunk = fifo.read(frame_size)
                if chunk is None:
                    return
                chunk.pts = audio_pts
                chunk.time_base = audio_time_base
                audio_pts += chunk.samples
                for packet in out_audio.encode(chunk):
                    out.mux(packet)
                audio_frames += 1
                if flush:
                    return

        def encode_source(source_path: Path) -> None:
            nonlocal video_pts, video_frames

            with av.open(str(source_path)) as src:
                if not src.streams.video:
                    raise ValueError(f"No video stream found in: {source_path}")
                if not src.streams.audio:
                    raise ValueError(f"No audio stream found in: {source_path}")

                in_video = src.streams.video[0]
                in_audio = src.streams.audio[0]

                audio_resampler = AudioResampler(
                    format="fltp",
                    layout=target_layout,
                    rate=target_sample_rate,
                )

                # Single interleaved pass: decoding video and audio separately
                # would drain the demuxer on the first pass and leave the second
                # stream empty. Decoding both streams together preserves order.
                for frame in src.decode(in_video, in_audio):
                    if isinstance(frame, av.VideoFrame):
                        out_frame = frame.reformat(
                            width=target_width,
                            height=target_height,
                            format="yuv420p",
                        )
                        out_frame.pts = video_pts
                        out_frame.time_base = video_time_base
                        video_pts += 1
                        for packet in out_video.encode(out_frame):
                            out.mux(packet)
                        video_frames += 1
                    elif isinstance(frame, av.AudioFrame):
                        for converted in audio_resampler.resample(frame):
                            converted.pts = None
                            fifo.write(converted)
                        drain_fifo(flush=False)

                # Flush any samples the resampler is still holding.
                for converted in audio_resampler.resample(None):
                    converted.pts = None
                    fifo.write(converted)
                drain_fifo(flush=False)

        encode_source(src1)
        encode_source(src2)

        # Emit any remaining buffered audio, then flush both encoders.
        drain_fifo(flush=True)
        for packet in out_video.encode():
            out.mux(packet)
        for packet in out_audio.encode():
            out.mux(packet)

    return out_path, video_frames, audio_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join two videos end-to-end into one MP4 while preserving audio."
    )
    parser.add_argument("video1", type=Path, help="Path to the first input video.")
    parser.add_argument("video2", type=Path, help="Path to the second input video.")
    parser.add_argument("output", type=Path, help="Path to the output MP4 file.")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Force output FPS (default: first video's FPS, fallback 24).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        out_path, video_frame_count, audio_frame_count = join_videos_with_audio(
            args.video1,
            args.video2,
            args.output,
            args.fps,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Output: {out_path}")
    print(f"Encoded video frames: {video_frame_count}")
    print(f"Encoded audio frames: {audio_frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
