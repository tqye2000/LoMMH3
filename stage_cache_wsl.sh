#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stage_cache_wsl.sh — copy the MiniMax-H3 snapshot from the Windows drive
# (/mnt/d, drvfs 9p — slow) onto WSL's native ext4 home so repeated sequential
# runs load in ~1-2 min instead of ~30 min.
#
# The HuggingFace cache stores each snapshot file as a symlink into ../../blobs.
# We copy ONLY refs/ + snapshots/<rev>/ and DEREFERENCE the symlinks (-L / rsync
# -aL) so the ext4 copy holds real files and needs no blobs/ directory (avoids
# doubling the data). LoMMH.py's _resolve_snapshot only needs refs/main plus
# snapshots/<rev>/modular_model_index.json, both of which this reproduces.
#
# After staging, run generation with HF_HOME=$HOME/hf_models.
# ---------------------------------------------------------------------------
set -euo pipefail

REV="42ed227ee7df40d41602854ae760620d6eb651fe"
SRC="/mnt/d/hf_models/hub/models--MiniMaxAI--MiniMax-H3"
DEST="$HOME/hf_models/hub/models--MiniMaxAI--MiniMax-H3"

echo "[stage] source : $SRC"
echo "[stage] dest   : $DEST"

mkdir -p "$DEST/refs" "$DEST/snapshots"

# refs/main (tiny) — tells _resolve_snapshot which snapshot is 'main'.
cp -f "$SRC/refs/main" "$DEST/refs/main"

# The heavy component tree, symlinks dereferenced to real files.
if command -v rsync >/dev/null 2>&1; then
    echo "[stage] copying with rsync (resumable) …"
    rsync -aL --info=progress2 "$SRC/snapshots/$REV" "$DEST/snapshots/"
else
    echo "[stage] rsync not found; copying with cp -rL …"
    cp -rL "$SRC/snapshots/$REV" "$DEST/snapshots/"
fi

echo "[stage] done. Verifying size …"
du -shL "$DEST/snapshots/$REV"
echo "STAGED_OK $DEST/snapshots/$REV"
