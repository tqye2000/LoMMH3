"""
MiniMax-H3 Local Testing Script
================================
MiniMax-H3 is an omni-modal (video + audio) generation model.

Model reference : https://huggingface.co/MiniMaxAI/MiniMax-H3
GGUF quants     : https://huggingface.co/Abiray/MiniMax-H3-GGUF  (for ComfyUI)
Diffusers docs  : https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3

Quick start
-----------
    python LoMMH.py                       # default T2VA clip
    python LoMMH.py --frames 345          # ~14.4 s (model max)
    python LoMMH.py --image first.jpg     # image-to-video (FL2VA)
    python LoMMH.py --strategy auto_offload   # 24-32 GB VRAM
    python LoMMH.py --help                # all options

Hardware notes
--------------
- Full bf16 model: ~124 GB total (transformer 61.7 GB + conditioner 62.1 GB).
- The default "max_gpu" strategy int8-quantises each big model onto its own GPU
  (transformer/conditioner/decoder), so it needs three CUDA devices.
- On a single 24-32 GB card use ``--strategy auto_offload`` (int8 + CPU streaming).

Install (diffusers main branch required)
----------------------------------------
    pip install git+https://github.com/huggingface/diffusers
    pip install transformers accelerate torchao av
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace environment — MUST be set before importing any HF module.
# ---------------------------------------------------------------------------
HF_HOME = r"D:\hf_models"
os.environ["HF_HOME"] = HF_HOME
# Fully offline: the localized index (below) points every component at the local
# snapshot, so the Hub is never contacted.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from diffusers.modular_pipelines.components_manager import ComponentsManager
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.utils.export_utils import encode_video

# ---------------------------------------------------------------------------
# Model location
# ---------------------------------------------------------------------------
MODEL_CACHE = Path(r"D:\hf_models\hub\models--MiniMaxAI--MiniMax-H3")


def _resolve_snapshot(cache: Path) -> Path:
    """Return the on-disk snapshot folder for the cached ``main`` revision."""
    revision = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
    snapshot = cache / "snapshots" / revision
    if not (snapshot / "modular_model_index.json").is_file():
        raise FileNotFoundError(f"Incomplete MiniMax-H3 snapshot: {snapshot}")
    return snapshot


def _localize_model_index(snapshot: Path) -> str:
    """Rewrite the modular index so every component loads from local disk.

    The shipped ``modular_model_index.json`` points each component at the repo
    id ``MiniMaxAI/MiniMax-H3``, forcing a Hub lookup on ``load_components`` even
    though the weights are already cached. Rewriting those references to the
    snapshot folder makes loading work fully offline.
    """
    index = json.loads((snapshot / "modular_model_index.json").read_text(encoding="utf-8"))
    for value in index.values():
        if isinstance(value, list) and len(value) == 3 and isinstance(value[2], dict):
            if "pretrained_model_name_or_path" in value[2]:
                value[2]["pretrained_model_name_or_path"] = str(snapshot)
    local_dir = snapshot / "_local_index"
    local_dir.mkdir(exist_ok=True)
    (local_dir / "modular_model_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    return str(local_dir)


SNAPSHOT_DIR = _resolve_snapshot(MODEL_CACHE)
MODEL_ID = str(SNAPSHOT_DIR)                        # subfolder loads (transformer, vae, …)
MODEL_INDEX = _localize_model_index(SNAPSHOT_DIR)   # offline modular index

# ---------------------------------------------------------------------------
# Frame-count rules
# ---------------------------------------------------------------------------
# MiniMax-H3 renders 5-15 s at 24 fps, and the video VAE only accepts frame
# counts of the form ``17 * n + 5``. Valid counts run from 124 (~5.2 s) to
# 345 (~14.4 s); a request of 360 would round up to 362 and be rejected.
FPS = 24
_FRAME_STEP = 17
_FRAME_BASE = 5
MIN_FRAMES = 124
MAX_FRAMES = 345


def snap_num_frames(requested: int) -> int:
    """Round ``requested`` up to the nearest valid ``17*n+5`` count, clamped."""
    n = max(0, math.ceil((requested - _FRAME_BASE) / _FRAME_STEP))
    snapped = _FRAME_STEP * n + _FRAME_BASE
    return min(max(snapped, MIN_FRAMES), MAX_FRAMES)


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = (
    "A majestic red fox trotting through a snow-covered pine forest at golden hour, "
    "soft light filtering through the trees, cinematic wide shot"
)

# Per-strategy help, also surfaced in ``--help``.
STRATEGY_HELP = {
    "max_gpu": "4-stage int8 T2VA across 3 GPUs (no sharding/offload). Fastest multi-GPU.",
    "auto_offload": "Single GPU, int8 + block-level CPU streaming (24-32 GB VRAM).",
    "multi_gpu": "Conditioner on cuda:1, rest on cuda:0 (bf16 + CPU auto-offload).",
    "bf16_single": "Full bf16 on one 80 GB card with CPU auto-offload.",
}
STRATEGIES = tuple(STRATEGY_HELP)


@dataclass
class Config:
    """All user-tunable settings for a single generation run."""

    prompt: str = DEFAULT_PROMPT
    image: str | None = None          # first-frame image → FL2VA (else T2VA)
    num_frames: int = MAX_FRAMES      # 24 fps; snapped to 17*n+5 in [124, 345]
    height: int = 544                 # multiple of 32
    width: int = 960                  # multiple of 32; 960×544 ≈ 2.3× faster than 1344×768
    steps: int = 30                   # denoising steps (lower = faster / rougher)
    seed: int = 42
    strategy: str = "max_gpu"
    device: str = "cuda"              # "mps" for Apple Silicon
    output_dir: Path = Path("outputs")
    output_filename: str | None = None  # custom MP4 filename; defaults by mode
    # "max_gpu" only: each big model lives whole on its own GPU.
    transformer_gpu: int = 0
    conditioner_gpu: int = 1
    decoder_gpu: int = 2

    @property
    def duration_s(self) -> float:
        return self.num_frames / FPS


# ---------------------------------------------------------------------------
# int8 quantisation (shared by max_gpu and auto_offload)
# ---------------------------------------------------------------------------
# Layers kept in bf16: quantising them corrupts embeddings / projections.
_TRANSFORMER_SKIP = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
_TEXT_ENCODER_SKIP = [
    "model.visual", "model.language_model.embed_tokens",
    "model.language_model.norm", "lm_head",
]


def _load_int8_transformer(device_map: dict | None = None):
    """Load the denoising transformer int8-quantised (bf16 for skipped layers)."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from diffusers.quantizers.quantization_config import TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    return MiniMaxH3Transformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2), modules_to_not_convert=_TRANSFORMER_SKIP
        ),
        low_cpu_mem_usage=True,
        device_map=device_map,
    )


def _load_int8_text_encoder(device_map: dict | None = None):
    """Load the Qwen3-VL conditioner int8-quantised (bf16 for skipped layers)."""
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2), modules_to_not_convert=_TEXT_ENCODER_SKIP
        ),
        device_map=device_map,
    )


def move_conditioning_state(state, device: torch.device):
    """Move model embeddings while leaving layout metadata on the CPU."""
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to(device)
    return state


def move_latent_state(state, device: torch.device):
    """Move completed video/audio latents to the decoder device."""
    state.values["latents"] = state.values["latents"].to(device)
    state.values["audio_latents"] = state.values["audio_latents"].to(device)
    return state


def build_pipeline(config: Config):
    """Load MiniMax-H3 according to ``config.strategy``."""
    builders = {
        "max_gpu": _build_max_gpu,
        "bf16_single": _build_bf16_single,
        "multi_gpu": _build_multi_gpu,
        "auto_offload": _build_auto_offload,
    }
    return builders[config.strategy](config)


def _build_max_gpu(config: Config):
    """4-stage int8 T2VA: one big model per GPU, no sharding, no CPU offload.

    A single MiniMax-H3 transformer cannot be sharded across GPUs (its internal
    ``index_select`` / rope / ``context_embedder`` mix devices). Instead each big
    model is int8-quantised onto its own GPU and the pipeline is split at the
    official workflow-block boundaries, moving only the required tensors between
    stages (pipeline state follows a single execution device).
    """
    print(f"[MiniMax-H3]   transformer  -> cuda:{config.transformer_gpu} (int8, whole)")
    transformer = _load_int8_transformer(device_map={"": f"cuda:{config.transformer_gpu}"})

    print(f"[MiniMax-H3]   conditioner  -> cuda:{config.conditioner_gpu} (int8, whole)")
    text_encoder = _load_int8_text_encoder(device_map={"": f"cuda:{config.conditioner_gpu}"})

    workflow = ModularPipeline.from_pretrained(MODEL_INDEX).blocks.get_workflow("t2va")

    # Pop every independent stage before init_pipeline(), which mutates the
    # workflow's block collection.
    text_block = workflow.sub_blocks.pop("text_encoder")
    video_decode_block = workflow.sub_blocks.pop("decode.video")
    audio_decode_block = workflow.sub_blocks.pop("decode.audio")

    # Stage 1: prompt → prompt embeddings (conditioner GPU).
    conditioner = text_block.init_pipeline(MODEL_INDEX)
    conditioner.update_components(text_encoder=text_encoder)
    conditioner.load_components(dtype=torch.bfloat16)

    # Stage 2: layout + latent prep + denoising (transformer GPU).
    generator_pipe = workflow.init_pipeline(MODEL_INDEX)
    generator_pipe.update_components(transformer=transformer)
    generator_pipe.load_components(dtype=torch.bfloat16)

    # Stages 3 & 4: video / audio decoding (decoder GPU).
    video_decoder = video_decode_block.init_pipeline(MODEL_INDEX)
    video_decoder.load_components(dtype=torch.bfloat16)
    video_decoder.vae.to(f"cuda:{config.decoder_gpu}")

    audio_decoder = audio_decode_block.init_pipeline(MODEL_INDEX)
    audio_decoder.load_components(dtype=torch.bfloat16)
    audio_decoder.audio_vae.to(f"cuda:{config.decoder_gpu}")
    return conditioner, generator_pipe, video_decoder, audio_decoder


def _build_bf16_single(config: Config):
    """Full bf16 on one 80 GB card with CPU auto-offload."""
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=config.device, memory_reserve_margin="12GB")
    pipe = ModularPipeline.from_pretrained(MODEL_INDEX, components_manager=manager)
    pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
    return pipe


def _build_multi_gpu(config: Config):
    """Conditioner on cuda:1, everything else on cuda:0 (bf16 + auto-offload)."""
    workflow = ModularPipeline.from_pretrained(MODEL_INDEX).blocks.get_workflow("t2va")

    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device="cuda:1")
    conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
        MODEL_INDEX, components_manager=text_manager
    )
    conditioner.load_components(dtype=torch.bfloat16)

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda:0")
    pipe = workflow.init_pipeline(MODEL_INDEX, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    return conditioner, pipe   # caller must chain manually


def _build_auto_offload(config: Config):
    """Single GPU: int8 weights + block-level CPU streaming (24-32 GB VRAM)."""
    from diffusers.hooks.group_offloading import apply_group_offloading

    pipe = ModularPipeline.from_pretrained(MODEL_INDEX)
    pipe.update_components(
        transformer=_load_int8_transformer(),
        text_encoder=_load_int8_text_encoder(),
    )
    pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)

    # Freeze so quantised tensors stay pinnable for streamed offload.
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    pipe.transformer.enable_group_offload(
        onload_device=torch.device(config.device),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
    )
    apply_group_offloading(
        pipe.text_encoder.model,
        onload_device=torch.device(config.device),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True,
    )
    pipe.vae.to(config.device)
    pipe.audio_vae.to(config.device)
    return pipe


def _print_run_header(config: Config, mode: str) -> None:
    print(f"\n[MiniMax-H3] Mode:   {mode}")
    print(f"[MiniMax-H3] Prompt: {config.prompt}")
    print(
        f"[MiniMax-H3] Frames: {config.num_frames} (~{config.duration_s:.1f}s @ {FPS}fps)"
        f"  Size: {config.width}×{config.height}  Steps: {config.steps}  Seed: {config.seed}\n"
    )


def run_generation(pipe, config: Config) -> tuple[dict, str]:
    """Run one generation and return ``(results, mode)``."""
    generator = torch.Generator().manual_seed(config.seed)
    outputs = ["videos", "audio", "sampling_rate"]
    kwargs: dict[str, object] = dict(
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        num_inference_steps=config.steps,
        generator=generator,
    )

    # 4-stage split pipeline (max_gpu): drive each stage, moving tensors across GPUs.
    if isinstance(pipe, tuple):
        if len(pipe) != 4:
            raise RuntimeError("Unsupported split pipeline; use --strategy max_gpu.")
        if config.image is not None:
            raise ValueError("The max_gpu split supports T2VA only; omit --image.")
        conditioner, generator_pipe, video_decoder, audio_decoder = pipe

        print("[MiniMax-H3] Encoding prompt on conditioner GPU …")
        state = conditioner(prompt=config.prompt)
        state = move_conditioning_state(state, torch.device(f"cuda:{config.transformer_gpu}"))

        _print_run_header(config, "t2va")
        state = generator_pipe(state=state, **kwargs)
        state = move_latent_state(state, torch.device(f"cuda:{config.decoder_gpu}"))
        state = video_decoder(state=state)
        state = audio_decoder(state=state)
        return {name: state.values[name] for name in outputs}, "t2va"

    # Single-pipeline strategies (bf16_single / auto_offload).
    kwargs["prompt"] = config.prompt
    if config.image is not None:
        from diffusers.utils.loading_utils import load_image
        kwargs["image"] = load_image(config.image)
        mode = "fl2va"
    else:
        mode = "t2va"

    _print_run_header(config, mode)
    return pipe(**kwargs, output=outputs), mode


def save_output(results: dict, config: Config, mode: str) -> Path:
    """Mux video + audio into an MP4 and return its path."""
    filename = config.output_filename or f"minimax_h3_{mode}.mp4"
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"
    out_path = config.output_dir / filename
    encode_video(
        results["videos"][0],
        fps=FPS,
        output_path=str(out_path),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    print(f"\n[MiniMax-H3] Saved → {out_path.resolve()}")
    return out_path


def parse_args() -> Config:
    """Parse CLI arguments into a :class:`Config`."""
    defaults = Config()
    strategy_help = "; ".join(f"{name} = {desc}" for name, desc in STRATEGY_HELP.items())
    parser = argparse.ArgumentParser(
        description="MiniMax-H3 local video+audio generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default=defaults.prompt, help="Text prompt")
    parser.add_argument("--image", default=defaults.image, help="First-frame image → FL2VA mode")
    parser.add_argument(
        "--frames", type=int, default=defaults.num_frames,
        help=f"Frame count @ {FPS}fps; snapped to 17*n+5 in [{MIN_FRAMES}, {MAX_FRAMES}]",
    )
    parser.add_argument("--height", type=int, default=defaults.height, help="Frame height (mult. of 32)")
    parser.add_argument("--width", type=int, default=defaults.width, help="Frame width (mult. of 32)")
    parser.add_argument("--steps", type=int, default=defaults.steps, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="RNG seed")
    parser.add_argument(
        "--strategy", default=defaults.strategy, choices=STRATEGIES, help=strategy_help,
    )
    parser.add_argument("--output-dir", default=str(defaults.output_dir), help="Output folder")
    parser.add_argument(
        "--output", dest="output_filename", default=defaults.output_filename,
        help="Output MP4 filename inside the output folder",
    )
    args = parser.parse_args()

    return Config(
        prompt=args.prompt,
        image=args.image,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        strategy=args.strategy,
        output_dir=Path(args.output_dir),
        output_filename=args.output_filename,
    )


def main() -> None:
    config = parse_args()

    snapped = snap_num_frames(config.num_frames)
    if snapped != config.num_frames:
        print(
            f"[MiniMax-H3] Adjusted frames {config.num_frames} → {snapped} "
            f"(valid 17*n+5 in [{MIN_FRAMES}, {MAX_FRAMES}], ~{snapped / FPS:.1f}s)"
        )
        config.num_frames = snapped

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MiniMax-H3] Loading model with strategy='{config.strategy}' …")
    pipe = build_pipeline(config)

    results, mode = run_generation(pipe, config)
    save_output(results, config, mode)


if __name__ == "__main__":
    main()

