#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time setup of the MiniMax Studio venv inside WSL2.
#
# The UI gets its own venv (~/minimax-ui-venv) so its packages can never
# disturb the GPU venv (~/minimax-venv) that runs LoMMH.py.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu -- bash /mnt/d/TQYE/Experiments/MiniMax/ui/setup_ui.sh [--dev]
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MINIMAX_UI_VENV:-$HOME/minimax-ui-venv}"
PY="${PYTHON_BIN:-python3}"

if [[ ! -d "$VENV" ]]; then
    if "$PY" -c 'import ensurepip' >/dev/null 2>&1; then
        "$PY" -m venv "$VENV"
    else
        # Ubuntu's python3 lacks ensurepip; virtualenv works without sudo.
        "$PY" -m pip install --user --break-system-packages --upgrade virtualenv
        "$PY" -m virtualenv "$VENV"
    fi
fi

"$VENV/bin/python" -m pip install --upgrade pip
if [[ "${1:-}" == "--dev" ]]; then
    "$VENV/bin/python" -m pip install -r "$HERE/requirements-dev.txt"
else
    "$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"
fi
echo "==> MiniMax Studio venv ready at: $VENV"
