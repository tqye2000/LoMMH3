"""Stand-in for LoMMH.py used by tests: same CLI, no GPU.

Behaviour is selected with FAKE_GEN_MODE:
  ok (default) | fail | hang | child (spawns a grandchild, then hangs) | release (waits for FAKE_GEN_RELEASE file)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path


def _progress(progress_file: str | None, event: str, **fields) -> None:
    if progress_file:
        with open(progress_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")


def _write_video(path: Path, frames: int = 6) -> None:
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        stream.time_base = Fraction(1, 24)
        for index in range(frames):
            array = np.full((64, 64, 3), index * 40 % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> int:
    parser = argparse.ArgumentParser()
    for flag in ("--strategy", "--prompt-file", "--frames", "--width", "--height", "--steps", "--seed",
                 "--output-dir", "--output", "--run-id", "--progress-jsonl", "--image",
                 "--reference-short-edge"):
        parser.add_argument(flag)
    parser.add_argument("--reference-image", action="append", default=[])
    parser.add_argument("--no-auto-download", action="store_true")
    args = parser.parse_args()
    mode = os.environ.get("FAKE_GEN_MODE", "ok")
    print(f"[fake] mode={mode} run_id={args.run_id} refs={args.reference_image} image={args.image}", flush=True)
    print(f"[fake] prompt={Path(args.prompt_file).read_text(encoding='utf-8')!r}", flush=True)
    print(f"[fake] env MASTER_PORT={os.environ.get('MASTER_PORT')} "
          f"OPENAI_API_KEY={'set' if 'OPENAI_API_KEY' in os.environ else 'unset'}", flush=True)

    _progress(args.progress_jsonl, "stage", name="loading")
    if mode == "fail":
        print("RuntimeError: simulated failure", file=sys.stderr, flush=True)
        return 3
    if mode == "child":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        Path(args.output_dir, "child.pid").write_text(str(child.pid))
    if mode in ("hang", "child"):
        while True:
            time.sleep(0.1)
    if mode == "release":
        release = Path(os.environ["FAKE_GEN_RELEASE"])
        while not release.exists():
            time.sleep(0.05)

    steps = min(int(args.steps), 5)
    _progress(args.progress_jsonl, "stage", name="generating", steps=steps)
    for i in range(1, steps + 1):
        time.sleep(0.02)
        _progress(args.progress_jsonl, "step", i=i, n=steps)
    _progress(args.progress_jsonl, "stage", name="saving")
    out_dir = Path(args.output_dir)
    video = out_dir / args.output
    _write_video(video)
    video.with_suffix(".json").write_text(json.dumps({
        "run_id": args.run_id,
        "parameters": {"frames": int(args.frames), "seed": int(args.seed)},
    }), encoding="utf-8")
    _progress(args.progress_jsonl, "saved", path=str(video))
    _progress(args.progress_jsonl, "done", minutes=0.01)
    return 0


if __name__ == "__main__":
    sys.exit(main())
