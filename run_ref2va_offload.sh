#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WSL2 / Linux launcher for MiniMax-H3 ref2va with CPU offload (auto_offload).
#
# Use this when a single clip conditions on reference images AND a long prompt,
# which OOMs the Qwen3-VL conditioner under context_parallel (CP replicates the
# ~31 GB int8 conditioner on every rank and only shards the *denoise*, so it
# gives no relief to the prompt/reference-encode stage). auto_offload streams
# the big weight stacks layer-by-layer instead of holding them resident, which
# frees the headroom the conditioner activation needs. Slower than CP, but it
# keeps the full workflow: both references, the uncropped last frame, all frames.
#
# Runs on ONE GPU (no torch.multiprocessing spawn), so CP_WORLD_SIZE is unused.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu -- bash /mnt/d/TQYE/Experiments/MiniMax/run_ref2va_offload.sh
# Extra flags are forwarded to LoMMH.py, e.g.:
#   wsl -d Ubuntu -- bash /mnt/.../run_ref2va_offload.sh --steps 40
# ---------------------------------------------------------------------------
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MINIMAX_VENV:-$HOME/minimax-venv}"
PYTHON="$VENV/bin/python"
SCRIPT="$ROOT/LoMMH.py"

[[ -x "$PYTHON" ]] || { echo "Python venv not found at: $PYTHON  (run wsl_setup.sh first)" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Generation script not found at: $SCRIPT" >&2; exit 1; }

# Point HuggingFace at the model store. Prefer the ext4-staged copy (fast local
# disk) when present, else fall back to the Windows store over drvfs (slow).
# LoMMH.py reads HF_HOME and derives the MiniMax-H3 cache path from it.
_STAGED="$HOME/hf_models/hub/models--MiniMaxAI--MiniMax-H3"
if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -d "$_STAGED" ]]; then
        export HF_HOME="$HOME/hf_models"
    else
        export HF_HOME="/mnt/d/hf_models"
    fi
fi
echo "[run_ref2va_offload] HF_HOME=$HF_HOME"

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here. WSL2's
# paravirtualized GPU lacks the CUDA VMM APIs that feature needs, so it fails
# with spurious OOM. LoMMH.py auto-selects a WSL-safe allocator config.

# Reference images for image-to-video generation (Ref2VA).
IMAGE1="$ROOT/project_3/outputs/part3_last.jpg"
IMAGE2="$ROOT/project_3/female_role_s.jpg"
IMAGE3="$ROOT/project_3/male_role_s.jpg"
PROMPTFILE="$ROOT/project_3/prompt_4.txt"

[[ -f "$IMAGE1" ]] || { echo "Input image not found: $IMAGE1" >&2; exit 1; }
[[ -f "$IMAGE2" ]] || { echo "Input image not found: $IMAGE2" >&2; exit 1; }
[[ -f "$IMAGE3" ]] || { echo "Input image not found: $IMAGE3" >&2; exit 1; }
[[ -f "$PROMPTFILE" ]] || { echo "Prompt file not found: $PROMPTFILE" >&2; exit 1; }

exec "$PYTHON" "$SCRIPT" \
    --strategy auto_offload \
    --reference-image "$IMAGE1" \
    --reference-image "$IMAGE3" \
    --prompt-file "$PROMPTFILE" \
    --frames 345 \
    --width 704 \
    --height 384 \
    --steps 35 \
    --seed 22 \
    --output-dir "$ROOT/project_3/outputs" \
    --output "part4.mp4" \
    "$@"
