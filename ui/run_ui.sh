#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start MiniMax Studio inside WSL2. Open the printed login link in a Windows
# browser (WSL forwards localhost). Generations run with the GPU venv.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu -- bash /mnt/d/TQYE/Experiments/MiniMax/ui/run_ui.sh
#
# Optional environment overrides:
#   MINIMAX_UI_DATA     project data root     (default ~/minimax-ui-data)
#   MINIMAX_UI_PORT     web port              (default 8080)
#   MINIMAX_GEN_PYTHON  generator interpreter (default ~/minimax-venv/bin/python)
#   HF_HOME             model cache           (default ~/hf_models if staged, else /mnt/d/hf_models)
# ---------------------------------------------------------------------------
set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MINIMAX_UI_VENV:-$HOME/minimax-ui-venv}"

[[ -x "$VENV/bin/python" ]] || { echo "UI venv not found at $VENV (run ui/setup_ui.sh first)" >&2; exit 1; }

cd "$HERE/.."
exec "$VENV/bin/python" ui/main.py "$@"
