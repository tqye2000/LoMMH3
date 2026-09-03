#####################################################################
# Save the first and last decoded frames of a video as PNG images.
#
# Usage:
#   python save_video_frames.py <video_path> [--output-dir <output_dir>]
#####################################################################

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import av
from PIL import Image


def save_first_and_last_frames(
    video_path: Path,
    output_dir: Path | None = None,
) -> tuple[Path, Path, int]:
    """Extract the first and last frames from *video_path*.

    Returns the two output paths and the number of decoded video frames.
    The entire stream is decoded so the final image is the true last frame,
    including for videos with variable frame rates or inaccurate metadata.
    """
    source = video_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")

    destination = (
        output_dir.expanduser().resolve() if output_dir is not None else source.parent
    )
    first_path = destination / f"{source.stem}_first.png"
    last_path = destination / f"{source.stem}_last.png"

    first_image: Image.Image | None = None
    last_image: Image.Image | None = None
    frame_count = 0

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in: {source}")

            for frame in container.decode(video=0):
                image = frame.to_image()
                if first_image is None:
                    first_image = image.copy()
                if last_image is not None:
                    last_image.close()
                last_image = image
                frame_count += 1

        if first_image is None or last_image is None:
            raise ValueError(f"No video frames could be decoded from: {source}")

        destination.mkdir(parents=True, exist_ok=True)
        first_image.save(first_path, format="PNG")
        last_image.save(last_path, format="PNG")
    finally:
        if first_image is not None:
            first_image.close()
        if last_image is not None:
            last_image.close()

    return first_path, last_path, frame_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save the first and last decoded frames of a video as PNG files."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: the video's directory).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        first_path, last_path, frame_count = save_first_and_last_frames(
            args.video,
            args.output_dir,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Decoded {frame_count} video frame(s).")
    print(f"First frame: {first_path}")
    print(f"Last frame:  {last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
