#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WSL2 / Linux launcher for MiniMax-H3 t2va with GPU context parallel (CP).
#
# Use this when a single clip conditions on reference images AND a long prompt,
# which OOMs the Qwen3-VL conditioner under context_parallel (CP replicates the
# ~31 GB int8 conditioner on every rank and only shards the *denoise*, so it
# gives no relief to the prompt/reference-encode stage). auto_offload streams
# the big weight stacks layer-by-layer instead of holding them resident, which
# frees the headroom the conditioner activation needs. Slower than CP, but it
# keeps the full workflow: both references, the uncropped last frame, all frames.
#
# Runs on MULTIPLE GPUs with context parallel (CP), so set CP_WORLD_SIZE accordingly.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu -- bash /mnt/.../run_t2va_cp.sh
# Extra flags are forwarded to LoMMH.py, e.g.:
#   wsl -d Ubuntu -- bash /mnt/.../run_t2va_cp.sh --steps 40
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
echo "[run_t2va_cp] HF_HOME=$HF_HOME"

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here. WSL2's
# paravirtualized GPU lacks the CUDA VMM APIs that feature needs, so it fails
# with spurious OOM. LoMMH.py auto-selects a WSL-safe allocator config.

PROMPTFILE="$ROOT/prompts/prompt_3.txt"
[[ -f "$PROMPTFILE" ]] || { echo "Prompt file not found: $PROMPTFILE" >&2; exit 1; }

exec "$PYTHON" "$SCRIPT" \
    --strategy context_parallel \
    --prompt-file "$PROMPTFILE" \
    --frames 346 \
    --width 1024 \
    --height 1024 \
    --steps 45 \
    --seed 32 \
    --output-dir "$ROOT/outputs" \
    --output "重返旧地.mp4" \
    "$@"
