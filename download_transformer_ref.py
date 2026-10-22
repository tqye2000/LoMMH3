"""Download MiniMax-H3's ref2va transformer partition into the local HF cache.

The download is resumable: running this script again reuses completed cache
blobs and fetches only files that are still missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_HF_HOME = Path(r"D:\hf_models")
REPO_ID = "MiniMaxAI/MiniMax-H3"
MODEL_CACHE_NAME = "models--MiniMaxAI--MiniMax-H3"
PARTITION = "transformer_ref"
MIN_RECOMMENDED_FREE_GIB = 65
DOWNLOAD_SIZE_LABEL = "~61.7 GiB / 66.3 GB"


def _active_revision(hf_home: Path) -> str | None:
    """Return the locally selected main commit, if one is already cached."""
    ref = hf_home / "hub" / MODEL_CACHE_NAME / "refs" / "main"
    if not ref.is_file():
        return None
    revision = ref.read_text(encoding="utf-8").strip()
    return revision or None


def _cache_dir(hf_home: Path) -> Path:
    """Return the actual HF hub cache root (``HF_HOME/hub``).

    ``snapshot_download`` treats ``cache_dir`` as this hub root directly, so it
    must include the ``hub`` segment to match where the base snapshot already
    lives and where ``LoMMH.py`` resolves the model.
    """
    return hf_home / "hub"


def _snapshot_dir(hf_home: Path, revision: str) -> Path:
    return _cache_dir(hf_home) / MODEL_CACHE_NAME / "snapshots" / revision


def partition_status(snapshot: Path) -> tuple[bool, str]:
    """Return whether ``transformer_ref`` has config and all weight shards."""
    partition = snapshot / PARTITION
    if not partition.is_dir():
        return False, f"{partition} does not exist"
    if not (partition / "config.json").is_file():
        return False, f"{partition / 'config.json'} is missing"

    index_files = sorted(partition.glob("*.index.json"))
    expected_weights: set[str] = set()
    for index_file in index_files:
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"cannot read {index_file}: {exc}"
        weight_map = index.get("weight_map", {})
        if isinstance(weight_map, dict):
            expected_weights.update(str(name) for name in weight_map.values())

    if expected_weights:
        missing = [
            name
            for name in sorted(expected_weights)
            if not (partition / name).is_file() or (partition / name).stat().st_size == 0
        ]
        if missing:
            preview = ", ".join(missing[:3])
            suffix = " …" if len(missing) > 3 else ""
            return False, f"{len(missing)} weight shard(s) missing: {preview}{suffix}"
        return True, f"complete ({len(expected_weights)} weight shard(s))"

    weights = [
        path
        for pattern in ("*.safetensors", "*.bin")
        for path in partition.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not weights:
        return False, f"no model weights found in {partition}"
    return True, f"complete ({len(weights)} weight file(s))"


def download_partition(hf_home: Path, revision: str, max_workers: int) -> Path:
    """Download only ``transformer_ref/`` and return its snapshot directory."""
    # This process may be launched by LoMMH.py, which deliberately runs the
    # generation pipeline offline. Clear those inherited flags before importing
    # huggingface_hub so this focused download can access the network.
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from huggingface_hub import snapshot_download

    cache_dir = _cache_dir(hf_home)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=[f"{PARTITION}/*"],
        max_workers=max_workers,
    )
    return Path(snapshot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MiniMax-H3's transformer_ref partition for reference-image "
            "(ref2va) generation. Existing and partial downloads are reused."
        )
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help="Hugging Face cache root",
    )
    parser.add_argument(
        "--revision",
        help="Model revision/commit (defaults to the cached refs/main, then main)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel download workers",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the partition without downloading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        print("[MiniMax-H3] ERROR: --max-workers must be at least 1.", file=sys.stderr)
        return 2

    hf_home = args.hf_home.expanduser().resolve()
    revision = args.revision or _active_revision(hf_home) or "main"
    snapshot = _snapshot_dir(hf_home, revision)
    complete, detail = partition_status(snapshot)
    if complete:
        print(f"[MiniMax-H3] {PARTITION} is already {detail}:\n  {snapshot / PARTITION}")
        return 0

    if args.check:
        print(f"[MiniMax-H3] {PARTITION} is not complete: {detail}", file=sys.stderr)
        return 1

    hf_home.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(hf_home).free / (1024**3)
    if free_gib < MIN_RECOMMENDED_FREE_GIB:
        print(
            f"[MiniMax-H3] WARNING: only {free_gib:.1f} GiB is free on "
            f"{hf_home.drive or hf_home}. The partition is {DOWNLOAD_SIZE_LABEL}.",
            file=sys.stderr,
        )

    print(f"[MiniMax-H3] {PARTITION} is missing or incomplete: {detail}")
    print(f"[MiniMax-H3] Downloading {REPO_ID}/{PARTITION} ({DOWNLOAD_SIZE_LABEL}) …")
    print(f"[MiniMax-H3] Cache:    {hf_home}")
    print(f"[MiniMax-H3] Revision: {revision}")
    print("[MiniMax-H3] Interrupted downloads can be resumed by running this again.")

    try:
        downloaded_snapshot = download_partition(hf_home, revision, args.max_workers)
    except KeyboardInterrupt:
        print("\n[MiniMax-H3] Download interrupted; partial files remain resumable.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[MiniMax-H3] Download failed: {exc}", file=sys.stderr)
        print(
            "[MiniMax-H3] Check the network connection and available disk space. "
            "If authentication is requested, run 'hf auth login'.",
            file=sys.stderr,
        )
        return 1

    complete, detail = partition_status(downloaded_snapshot)
    if not complete:
        print(
            f"[MiniMax-H3] Download returned but verification failed: {detail}",
            file=sys.stderr,
        )
        return 1

    print(f"[MiniMax-H3] Download complete and verified ({detail}):")
    print(f"  {downloaded_snapshot / PARTITION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())