#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WSL2 / Linux launcher for MiniMax-H3 ref2va with a reference video and a
# reference image. Videos over 15 s are split, generated sequentially with
# single-clip context parallelism, and joined with audio.
#
# The prompt should use the official ref2va format with "<Picture 1>", "<Video 1>"
# and "<Audio 1>" (the video's soundtrack) labels.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu -- bash /mnt/.../run_ref2va_video.sh
# Extra flags are forwarded to LoMMH.py, e.g.:
#   wsl -d Ubuntu -- bash /mnt/.../run_ref2va_video.sh --steps 40
# Override inputs via env vars: REF_VIDEO, REF_IMAGE, PROMPTFILE, OUTPUT_DIR, OUTPUT.
# ---------------------------------------------------------------------------
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MINIMAX_VENV:-$HOME/minimax-venv}"
PYTHON="$VENV/bin/python"
SCRIPT="$ROOT/generate_reference_video.py"

[[ -x "$PYTHON" ]] || { echo "Python venv not found at: $PYTHON  (run wsl_setup.sh first)" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Generation script not found at: $SCRIPT" >&2; exit 1; }

# Prefer the ext4-staged model cache when present, else the Windows store over drvfs.
_STAGED="$HOME/hf_models/hub/models--MiniMaxAI--MiniMax-H3"
if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -d "$_STAGED" ]]; then
        export HF_HOME="$HOME/hf_models"
    else
        export HF_HOME="/mnt/d/hf_models"
    fi
fi
echo "[run_ref2va_video] HF_HOME=$HF_HOME"

export CP_WORLD_SIZE="${CP_WORLD_SIZE:-4}"

# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments: WSL2 lacks CUDA VMM.

REF_VIDEO="${REF_VIDEO:-$ROOT/project_3/outputs/part1.mp4}"
REF_IMAGE="${REF_IMAGE:-$ROOT/images/LL1.jpg}"
PROMPTFILE="${PROMPTFILE:-$ROOT/project_4/prompt_video_v4_edit.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/project_4/outputs}"
OUTPUT="${OUTPUT:-part1_video_ref_v4_edit.mp4}"

[[ -f "$REF_VIDEO" ]] || { echo "Reference video not found: $REF_VIDEO" >&2; exit 1; }
[[ -f "$REF_IMAGE" ]] || { echo "Reference image not found: $REF_IMAGE" >&2; exit 1; }
[[ -f "$PROMPTFILE" ]] || { echo "Prompt file not found: $PROMPTFILE" >&2; exit 1; }

exec "$PYTHON" "$SCRIPT" \
    --strategy context_parallel \
    --reference-image "$REF_IMAGE" \
    --reference-video "$REF_VIDEO" \
    --prompt-file "$PROMPTFILE" \
    --frames 345 \
    --width 704 \
    --height 384 \
    --steps 35 \
    --seed 22 \
    --reference-short-edge 1024 \
    --reference-video-short-edge 384 \
    --output-dir "$OUTPUT_DIR" \
    --output "$OUTPUT" \
    "$@"
