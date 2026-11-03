#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time WSL2 / Ubuntu environment setup for MiniMax-H3 context parallelism.
#
# Native Windows PyTorch ships without NCCL, so Ulysses context parallelism
# (which needs all_to_all) cannot run there. This builds a Linux venv whose
# PyPI torch wheel is CUDA-enabled *with* NCCL, so the 4 GPUs can cooperate on
# a single clip.
#
# Notes on this corporate network:
#   * pypi.org / files.pythonhosted.org and github.com are reachable.
#   * download.pytorch.org is blocked, so we install torch from PyPI (its Linux
#     wheel bundles the CUDA runtime + NCCL) rather than the +cuXXX index.
#
# The venv lives in the Linux home (fast ext4), NOT on /mnt/d (drvfs is slow).
# Run once:  wsl -d Ubuntu -- bash /mnt/d/TQYE/Experiments/MiniMax/wsl_setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

VENV="${MINIMAX_VENV:-$HOME/minimax-venv}"
PY="${PYTHON_BIN:-python3}"

echo "==> Using base interpreter: $($PY --version)"

# Ubuntu ships python3 without ensurepip, so `python3 -m venv` can't bootstrap
# pip and installing python3-venv would need sudo. Instead we bootstrap the
# self-contained `virtualenv` tool with the system pip into the user site
# (no sudo), then use it to build a fully working venv with its own pip.
if [[ ! -d "$VENV" ]]; then
    if "$PY" -c 'import ensurepip' >/dev/null 2>&1; then
        echo "==> Creating venv at $VENV (stdlib venv)"
        "$PY" -m venv "$VENV"
    else
        echo "==> ensurepip missing; bootstrapping virtualenv (no sudo) ..."
        # --break-system-packages only affects the user site (~/.local); PEP 668
        # otherwise blocks the Debian-managed system pip. The real work happens
        # inside the isolated venv created below.
        "$PY" -m pip install --user --break-system-packages --upgrade virtualenv
        echo "==> Creating venv at $VENV (virtualenv)"
        "$PY" -m virtualenv "$VENV"
    fi
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch (CUDA + NCCL) from PyPI ..."
# PyPI Linux torch wheels are CUDA-enabled and bundle NCCL. We do NOT pin the
# +cu130 local version (that lives only on the blocked download.pytorch.org).
pip install "torch==2.13.0" "torchvision==0.28.0"

echo "==> Installing model stack ..."
pip install \
    "torchao==0.18.0" \
    "transformers==5.16.1" \
    "accelerate==1.14.0" \
    "safetensors==0.8.0" \
    "tokenizers==0.23.1" \
    "numpy==2.5.1" \
    "pillow==12.3.0" \
    "scipy==1.18.1" \
    "av==18.1.0" \
    "imageio-ffmpeg==0.6.0"

echo "==> Installing diffusers at the pinned commit ..."
pip install "git+https://github.com/huggingface/diffusers@7c79e7060750eb35c19bd8fcc7aaca57e692ae48"

echo "==> Verifying torch / CUDA / NCCL ..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("nccl available:", torch.distributed.is_nccl_available())
if torch.cuda.is_available():
    print("nccl version:", torch.cuda.nccl.version())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}:", torch.cuda.get_device_name(i))
PY

echo "==> Done. venv ready at: $VENV"
