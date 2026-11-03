from __future__ import annotations

import ast
from pathlib import Path

from ui.core import constants as C

LOMMH = Path(__file__).resolve().parents[2] / "LoMMH.py"


def _lommh_constants() -> dict[str, int]:
    values = {}
    for node in ast.parse(LOMMH.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, int):
                    values[target.id] = node.value.value
    return values


def test_constants_match_generator() -> None:
    lommh = _lommh_constants()
    assert lommh["FPS"] == C.FPS
    assert lommh["_FRAME_STEP"] == C.FRAME_STEP
    assert lommh["_FRAME_BASE"] == C.FRAME_BASE
    assert lommh["MIN_FRAMES"] == C.MIN_FRAMES
    assert lommh["MAX_FRAMES"] == C.MAX_FRAMES
    assert lommh["_MAX_REFERENCE_IMAGES"] == C.MODEL_MAX_REFERENCE_IMAGES
    assert C.IMAGE_SLOTS <= C.MODEL_MAX_REFERENCE_IMAGES


def test_frame_rules() -> None:
    counts = C.valid_frame_counts()
    assert counts[0] == 124 and counts[-1] == 345
    assert all((n - 5) % 17 == 0 for n in counts)
    assert C.snap_frames(125) == 141
    assert C.snap_frames(10) == 124
    assert C.snap_frames(400) == 345
