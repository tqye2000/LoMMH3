#!/usr/bin/env bash
# Install SAM 3D Body inference deps into ~/minimax-venv without touching torch/numpy.
# Git sources are cloned from Windows into /mnt/d/hf_models/src (WSL git/pip can't reach GitHub/HF through Zscaler).
set -euo pipefail

VENV=~/minimax-venv
SRC=/mnt/d/hf_models/src
PY=$VENV/bin/python
CONSTRAINTS=$(mktemp)
$PY -m pip freeze | grep -iE '^(torch|torchvision|numpy)==' > "$CONSTRAINTS"
pip_install() { $PY -m pip install -c "$CONSTRAINTS" "$@"; }

pip_install pytorch-lightning pyrender opencv-python-headless yacs einops timm roma braceexpand \
    omegaconf hydra-core trimesh fvcore iopath pycocotools cloudpickle tabulate termcolor matplotlib \
    huggingface_hub pyrootutils loguru "setuptools<81"  # detectron2 model_zoo imports pkg_resources

# detectron2 (ViTDet human detector); CPU-only custom ops are enough for ViTDet inference.
pip_install --no-build-isolation --no-deps "$SRC/detectron2"

# MoGe-2 FOV estimator: only the v2 inference path, so skip its heavy v3/gradio deps.
pip_install --no-deps "$SRC/utils3d-moge" "$SRC/pipeline" "$SRC/MoGe"
pip_install click

# DINOv3 backbone is loaded via torch.hub from GitHub; seed the hub cache instead.
HUB=$($PY -c 'import torch; print(torch.hub.get_dir())')
mkdir -p "$HUB"
rm -rf "$HUB/facebookresearch_dinov3_main"
cp -r "$SRC/dinov3" "$HUB/facebookresearch_dinov3_main"

rm -f "$CONSTRAINTS"
echo "done"
