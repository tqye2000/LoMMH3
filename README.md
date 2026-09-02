# MiniMax-H3 Local Test Harness

This project is a minimal local test/experiment script for running the MiniMax-H3 model from a cached local snapshot under `D:\models`.

It is built around the Diffusers modular pipeline and is optimized for a multi-GPU setup, while also supporting single-GPU/offload modes.

## What this script does

- Loads the local MiniMax-H3 snapshot from `D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3`
- Uses an offline-style local modular index so components resolve from disk instead of the Hub
- Runs text-to-video (T2VA) and optional image-to-video (FL2VA)
- Supports multiple memory/loading strategies:
  - `max_gpu`
  - `auto_offload`
  - `multi_gpu`
  - `bf16_single`
- Saves output to `outputs/`

## Files

- `LoMMH.py` — main launcher and generation script
- `README.md` — project notes and usage

## Local model cache

The script expects the model snapshot to be present at:

```powershell
D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3
```

The script resolves the active revision from:

```powershell
D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3\refs\main
```

and then loads the model from the matching snapshot directory under:

```powershell
D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3\snapshots\<revision>
```

## Requirements

Use the local virtual environment in this workspace:

```powershell
& 'd:/Experiments/MiniMax/.venv/Scripts/python.exe' -m pip install -U diffusers transformers accelerate torchao av
```

If you already have the environment configured, just use the venv Python directly.

## Run examples

Default text-to-video generation:

```powershell
cd "D:\Experiments\MiniMax"
& '.venv\Scripts\python.exe' LoMMH.py
```

Short smoke test (fast):

```powershell
& '.venv\Scripts\python.exe' LoMMH.py --steps 2 --strategy max_gpu
```

Use a longer video (valid range is roughly 5 to 15 seconds at 24 fps):

```powershell
& '.venv\Scripts\python.exe' LoMMH.py --frames 345 --strategy max_gpu
```

Single-GPU/offload mode:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py --strategy auto_offload
```

Image-to-video mode:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py --image images\example_1.jpg --strategy auto_offload --frames 345 --prompt "A fashionable lady walking on a street with a handbag followed by a dog"
```

## Notes on frame counts

The model accepts video lengths in the range roughly 5 to 15 seconds at 24 fps, and valid frame counts are snapped to the form:

```text
17 * n + 5
```

So valid examples are:

- 124 frames (~5.2s)
- 345 frames (~14.4s)

This script automatically snaps the requested frame count to the nearest valid value and prints the adjustment.

## Recommended strategy

For a 4-GPU workstation like the RTX 6000 Ada setup, use:

```powershell
--strategy max_gpu
```

This keeps the transformer, conditioner, and decoder on dedicated GPUs and avoids the device-mismatch failures caused by sharding a single transformer across devices.

## Output

The generated MP4s are written under:

```powershell
D:\TQYE\Experiments\MiniMax\outputs
```

## Troubleshooting

- If a model component fails to load, verify the snapshot exists under `D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3\snapshots\...`
- If the output is too slow, reduce `--steps` or use a smaller frame count
- If you hit VRAM issues, use `--strategy auto_offload`
- If you see device mismatch errors, keep the default `max_gpu` layout rather than sharding the transformer across GPUs

## Credits

- MiniMax-H3 model: `MiniMaxAI/MiniMax-H3`
- Diffusers modular pipeline support: Hugging Face Diffusers

