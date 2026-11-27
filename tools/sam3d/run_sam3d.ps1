# SAM 3D Body: run from Windows PowerShell; inference runs in WSL Ubuntu with ~/minimax-venv.
#
# Usage:
#   .\tools\sam3d\run_sam3d.ps1
#   Edit the paths below first. They are relative to the repo root (forward slashes, no spaces).
#
# 1) Video -> white-model video (untextured grey meshes of every person, H.264 mp4, no audio).
#    Extra options for `tools/sam3d/sam3d_body.py video`:
#      --background black|white|video   background behind the meshes (default black; video = overlay on footage)
#      --max-frames N                   only the first N frames (quick test)
#      --per-frame-fov                  re-estimate FOV each frame (zooming shots)
#      --bbox-thresh 0.8                person detection threshold
#      --no-detector / --no-fov         whole frame = one person / default FOV
#    Speed: ~3 s per frame at 704x384 on one GPU.
#
# 2) Images (`tools/sam3d/sam3d_body.py image`) -> per image: <name>_vis.jpg (overlay + side view), <name>_mesh_NN.ply, <name>_params_NN.npz.
#    --images accepts a single file or a folder.
#
# Setup (once): checkpoint via `.venv\Scripts\python.exe tools\sam3d\sam3d_body.py download`, deps via tools/sam3d/install_sam3d_wsl.sh.

$input_video = "audios/Dance-P.mp4"
$output_video = "outputs/sam3d/Dance-P_white.mp4"

wsl -d Ubuntu -- bash -lc "cd /mnt/d/TQYE/Experiments/MiniMax && ~/minimax-venv/bin/python tools/sam3d/sam3d_body.py video --input $input_video --out $output_video"

#$input_images = "images"
#$output_images = "outputs/sam3d/images"

#wsl -d Ubuntu -- bash -lc "cd /mnt/d/TQYE/Experiments/MiniMax && ~/minimax-venv/bin/python tools/sam3d/sam3d_body.py image --images $input_images --out $output_images"