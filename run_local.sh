#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WSL2 / Linux launcher for MiniMax-H3 single-clip context parallelism.
#
# Mirrors run_local_cp.ps1 but runs under Linux, where the PyPI torch wheel has
# NCCL, so Ulysses context parallelism actually works. LoMMH.py spawns one
# worker per GPU itself (torch.multiprocessing.spawn), so this is a plain python
# invocation, not torchrun.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu -- bash /mnt/d/TQYE/Experiments/MiniMax/run_local_cp.sh
# Extra flags are forwarded to LoMMH.py, e.g.:
#   wsl -d Ubuntu -- bash /mnt/.../run_local_cp.sh --steps 40
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
        export HF_HOME="$HOME/models"
    else
        export HF_HOME="/mnt/d/models"
    fi
fi
echo "[run_local_cp] HF_HOME=$HF_HOME"

# Number of GPUs to split one clip across. LoMMH.py spawns one worker per GPU.
export CP_WORLD_SIZE="${CP_WORLD_SIZE:-4}"

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here. WSL2's
# paravirtualized GPU lacks the CUDA VMM APIs that feature needs, so it fails
# with spurious OOM. LoMMH.py auto-selects a WSL-safe allocator config.

# Reference images for image-to-video generation (Ref2VA).
IMAGE1="$ROOT/project/ref1.jpg"
IMAGE2="$ROOT/project/ref2.jpg"
PROMPTFILE="$ROOT/project/prompt.txt"

[[ -f "$IMAGE1" ]] || { echo "Input image not found: $IMAGE1" >&2; exit 1; }
[[ -f "$IMAGE2" ]] || { echo "Input image not found: $IMAGE2" >&2; exit 1; }
[[ -f "$PROMPTFILE" ]] || { echo "Prompt file not found: $PROMPTFILE" >&2; exit 1; }

exec "$PYTHON" "$SCRIPT" \
    --strategy context_parallel \
    --reference-image "$IMAGE1" \
    --reference-image "$IMAGE2" \
    --prompt-file "$PROMPTFILE" \
    --frames 345 \
    --width 704 \
    --height 384 \
    --steps 35 \
    --seed 42 \
    --output-dir "$ROOT/project/outputs" \
    --output "output.mp4" \
    "$@"
