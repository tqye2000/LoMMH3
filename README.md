# MiniMax-H3 Local Test Harness

This project runs the MiniMax-H3 video-and-audio model from the local Hugging Face cache. It uses the Diffusers modular pipeline and supports text-to-video, first-frame image-to-video, and reference-image generation.

Generation is offline after the required model files are present. The script rewrites the modular model index so components load from the cached snapshot instead of contacting the Hub.

## Features

- Text-to-video (`t2va`)
- First-frame image-to-video (`fl2va`) with `--image`
- Subject/style reference-image generation (`ref2va`) with one or more `--reference-image` arguments
- Int8 quantisation and CPU/group offloading for smaller GPUs
- Explicit multi-GPU placement for the `max_gpu` strategy
- MP4 output with the model-generated audio track
- UTF-8 prompt files for prompts containing Chinese or other non-ASCII text

## Files

- `LoMMH.py` — main command-line generation script
- `download_transformer_ref.py` — resumable downloader for the `ref2va` transformer partition
- `save_video_frames.py` — saves the first and last frames of a video as PNG images
- `run_local.ps1` — local PowerShell example using the reference-image workflow
- `prompts/` — example UTF-8 prompts
- `docs/` — prompt and reference-image notes
- `outputs/` — generated MP4 files

## Requirements

Use the virtual environment in this workspace. Diffusers from the main branch is required for the MiniMax-H3 modular pipeline:

```powershell
& '.venv\Scripts\python.exe' -m pip install -U git+https://github.com/huggingface/diffusers.git
& '.venv\Scripts\python.exe' -m pip install -U transformers accelerate torchao av pillow
```

The environment must also have a compatible CUDA-enabled PyTorch installation for GPU generation.

## Model cache

The scripts currently use this Hugging Face cache root and model cache:

```text
D:\hf_models
D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3
```

`LoMMH.py` reads the active revision from `refs\main` and loads the matching snapshot from:

```text
D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3\snapshots\<revision>
```

The base snapshot must contain `modular_model_index.json`, `transformer`, `text_encoder`, and the VAE components.

## Reference images and `ref2va`

Reference-image generation is selected by passing `--reference-image`. Up to 9 image references can be supplied, and the option can be repeated. The order of the images is preserved, so prompts should describe their roles as Picture 1, Picture 2, and so on.

Example with two references:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py `
  --strategy auto_offload `
  --reference-image images\hl_3.jpg `
  --reference-image images\chongda_a.jpg `
  --prompt-file prompts\prompt2.txt `
  --frames 345 `
  --width 704 `
  --height 384 `
  --steps 35 `
  --seed 52 `
  --output hl_output2.mp4
```

`--reference-image` and `--image` are different modes and cannot be combined. Reference-image generation is supported by `auto_offload`, `bf16_single`, and `max_gpu`; it is not supported by the experimental `multi_gpu` layout.

### Additional `ref2va` weights

The base MiniMax-H3 snapshot does not include the separate `transformer_ref` partition required by `ref2va`. When reference images are requested, `LoMMH.py` checks for that partition before loading the pipeline. If it is missing or incomplete, it automatically launches the resumable downloader, temporarily enables Hub access, verifies the downloaded shards, and then resumes generation in offline mode.

The partition is approximately 61.7 GiB, so at least 65 GiB of free disk space is recommended. To download it manually:

```powershell
& '.venv\Scripts\python.exe' download_transformer_ref.py
```

Check the local partition without downloading:

```powershell
& '.venv\Scripts\python.exe' download_transformer_ref.py --check
```

Interrupted downloads can be resumed by running the same command again. If the Hub requests authentication, authenticate with Hugging Face before starting the download.

## Running the generator

From the workspace root:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py
```

The default run is a 345-frame text-to-video generation using `max_gpu`. A short smoke test is:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py --steps 2 --frames 124 --strategy auto_offload
```

First-frame image-to-video (`fl2va`) on a single GPU:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py `
  --strategy auto_offload `
  --image images\100_1246_1245017111_o.jpg `
  --frames 345 `
  --prompt "A fashionable lady walking on a street with a handbag followed by a dog" `
  --output sample2.mp4
```

For non-ASCII prompts, prefer a UTF-8 file. `--prompt-file` overrides `--prompt` and avoids PowerShell/native-process code-page conversion:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py `
  --strategy auto_offload `
  --prompt-file prompts\prompt1.txt `
  --output prompt1.mp4
```

The included local example can also be run with:

```powershell
.\run_local.ps1
```

Extra arguments are forwarded by `run_local.ps1` to `LoMMH.py`.

## Saving the first and last video frames

Use `save_video_frames.py` with any video format supported by PyAV/FFmpeg:

```powershell
& '.venv\Scripts\python.exe' save_video_frames.py outputs\sample2.mp4
```

By default, the images are saved next to the video as
`sample2_first.png` and `sample2_last.png`. Existing images with those names
are replaced. To use another destination directory:

```powershell
& '.venv\Scripts\python.exe' save_video_frames.py outputs\sample2.mp4 `
  --output-dir images\sample2
```

The script decodes the complete video stream rather than relying on duration
metadata, ensuring the saved last image is the final decodable frame.

## Command-line options

| Option | Description |
| --- | --- |
| `--prompt TEXT` | Text prompt. |
| `--prompt-file PATH` | Read a UTF-8 prompt file; overrides `--prompt`. |
| `--image PATH` | First-frame image for `fl2va`. |
| `--reference-image PATH` | Reference image for `ref2va`; repeat up to 9 times. |
| `--frames N` | Requested frame count at 24 fps; automatically snapped to a valid value. |
| `--height N` / `--width N` | Output dimensions; use multiples of 32. |
| `--steps N` | Number of denoising steps. Lower values are faster but lower quality. |
| `--seed N` | Random seed. |
| `--strategy NAME` | `max_gpu`, `auto_offload`, `bf16_single`, or `multi_gpu`. |
| `--output-dir PATH` | Destination directory; defaults to `outputs`. |
| `--output NAME` | MP4 filename inside `--output-dir`. `.mp4` is added if omitted. |

Run `--help` for the full argument descriptions and defaults:

```powershell
& '.venv\Scripts\python.exe' LoMMH.py --help
```

## Memory strategies

| Strategy | Intended hardware | Notes |
| --- | --- | --- |
| `max_gpu` | At least 3 CUDA GPUs | Int8 models are kept whole on dedicated GPUs. The transformer or `transformer_ref`, conditioner, and decoder/reference VAEs are placed separately. This avoids sharding a single transformer across devices. |
| `auto_offload` | One 24–32 GB GPU plus system RAM | Int8 weights with block-level transformer and leaf-level text-encoder CPU streaming. Recommended for smaller VRAM. |
| `bf16_single` | One large, typically 80 GB GPU | Full BF16 components with automatic CPU offload. |
| `multi_gpu` | Experimental legacy layout | Places the conditioner on `cuda:1` and the remaining T2VA pipeline on `cuda:0`. Use `max_gpu` for the supported explicit multi-GPU runner. |

For `max_gpu`, the default placement is:

- `cuda:0` — transformer or `transformer_ref`
- `cuda:1` — text conditioner
- `cuda:2` — video/audio VAEs and reference encoder

Change the GPU indices in the `Config` defaults in `LoMMH.py` if the workstation uses a different layout.

## Frame counts and resolution

MiniMax-H3 generates approximately 5–15 seconds at 24 fps. The VAE accepts frame counts of the form:

$$
N = 17n + 5
$$

within the supported range of 124–345 frames. The script rounds a requested count up to the nearest valid value and clamps it to that range. For example:

- 124 frames ≈ 5.2 seconds
- 345 frames ≈ 14.4 seconds

The default resolution is 960×544. Both dimensions should be multiples of 32; reducing the resolution can substantially reduce runtime and memory use.

## Output

Generated files are written to `outputs\` by default. The default names are:

- `minimax_h3_t2va.mp4`
- `minimax_h3_fl2va.mp4`
- `minimax_h3_ref2va.mp4`

Each output is an MP4 containing the generated video and audio. A JSON run log with the same base filename is saved beside it (for example, `sample2.json`). The log records the prompt, input paths, requested and actual frame counts, generation settings, output path, command line, and runtime versions. Use `--output-dir` and `--output` to customize the destination and filename.

## Troubleshooting

- **Snapshot not found:** verify that the base snapshot exists under `D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3\snapshots\...` and contains `modular_model_index.json`.
- **`ref2va` partition missing:** run `download_transformer_ref.py`; allow approximately 61.7 GiB for `transformer_ref` and verify with `--check`.
- **VRAM or host-memory pressure:** use `--strategy auto_offload`, reduce `--frames`, lower `--width`/`--height`, or reduce `--steps`.
- **Device mismatch errors:** do not shard one transformer across GPUs. Use the provided `max_gpu` placement instead.
- **Prompt appears corrupted or uses unexpectedly high memory:** put non-ASCII text in a UTF-8 file and pass it with `--prompt-file`.
- **Slow generation:** reduce `--steps` and resolution, or use `max_gpu` when enough GPUs are available.

## Credits

- MiniMax-H3 model: `MiniMaxAI/MiniMax-H3`
- Diffusers modular pipeline: Hugging Face Diffusers
