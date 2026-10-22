"""Generation limits shared by validation and the UI.

Values mirror LoMMH.py; tests/test_constants.py checks they stay in sync.
"""

from __future__ import annotations

import math

FPS = 24
FRAME_STEP = 17
FRAME_BASE = 5
MIN_FRAMES = 124
MAX_FRAMES = 345

DIM_MULTIPLE = 32
MIN_DIM = 256
MAX_DIM = 2048
MAX_STEPS = 100
SEED_MAX = 2**32 - 1

# Per-generation UI limit; the model itself accepts up to 9 reference images.
IMAGE_SLOTS = 2
MODEL_MAX_REFERENCE_IMAGES = 9
MIN_REFERENCE_SHORT_EDGE = 64
MAX_REFERENCE_SHORT_EDGE = 4096

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
THUMB_SIZE = (320, 320)

MODES = ("t2va", "fl2va", "ref2va")
MODE_LABELS = {
    "t2va": "Text only",
    "fl2va": "First frame",
    "ref2va": "Reference images",
}
STRATEGIES = ("context_parallel", "auto_offload", "max_gpu", "bf16_single")
STRATEGY_LABELS = {
    "context_parallel": "Context parallel (multi-GPU, fastest)",
    "auto_offload": "Auto offload (single GPU)",
    "max_gpu": "Max GPU (needs ~64 GB per GPU)",
    "bf16_single": "BF16 single (80 GB GPU)",
}
MIN_CP_WORLD_SIZE = 2
MAX_CP_WORLD_SIZE = 8


def snap_frames(requested: int) -> int:
    """Round up to the nearest valid ``17*n+5`` frame count, clamped to the range."""
    n = max(0, math.ceil((requested - FRAME_BASE) / FRAME_STEP))
    return min(max(FRAME_STEP * n + FRAME_BASE, MIN_FRAMES), MAX_FRAMES)


def valid_frame_counts() -> list[int]:
    return [n for n in range(MIN_FRAMES, MAX_FRAMES + 1) if (n - FRAME_BASE) % FRAME_STEP == 0]
