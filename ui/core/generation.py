"""Effective parameters, request validation, and generator command construction."""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

from . import constants as C
from .models import GenerateStep, GenParams, Job, JobRequest, Project, ResolvedImage
from .settings import Settings

_SECRET_ENV = re.compile(r"(OPENAI|API_KEY|_KEY$|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)


def effective_params(project: Project, step: GenerateStep) -> GenParams:
    data = project.defaults.model_dump()
    data.update(step.overrides)
    return GenParams.model_validate(data)


def validate_params(params: GenParams, mode: str) -> list[str]:
    errors = []
    for name in ("width", "height"):
        value = getattr(params, name)
        if value % C.DIM_MULTIPLE or not C.MIN_DIM <= value <= C.MAX_DIM:
            errors.append(
                f"{name.title()} must be a multiple of {C.DIM_MULTIPLE} "
                f"between {C.MIN_DIM} and {C.MAX_DIM}."
            )
    if params.frames != C.snap_frames(params.frames) or params.frames < C.MIN_FRAMES:
        errors.append(
            f"Frames must be 17*n+5 between {C.MIN_FRAMES} and {C.MAX_FRAMES} "
            f"(nearest valid: {C.snap_frames(params.frames)})."
        )
    if not 1 <= params.steps <= C.MAX_STEPS:
        errors.append(f"Steps must be between 1 and {C.MAX_STEPS}.")
    if params.seed != "random" and not 0 <= params.seed <= C.SEED_MAX:
        errors.append(f"Seed must be between 0 and {C.SEED_MAX}, or random.")
    if params.strategy not in C.STRATEGIES:
        errors.append(f"Unknown strategy: {params.strategy}.")
    if params.strategy == "context_parallel" and not (
        C.MIN_CP_WORLD_SIZE <= params.cp_world_size <= C.MAX_CP_WORLD_SIZE
    ):
        errors.append(
            f"GPU count must be between {C.MIN_CP_WORLD_SIZE} and {C.MAX_CP_WORLD_SIZE}."
        )
    if mode == "ref2va" and params.reference_short_edge is not None and not (
        C.MIN_REFERENCE_SHORT_EDGE <= params.reference_short_edge <= C.MAX_REFERENCE_SHORT_EDGE
    ):
        errors.append(
            f"Reference short edge must be between {C.MIN_REFERENCE_SHORT_EDGE} "
            f"and {C.MAX_REFERENCE_SHORT_EDGE}."
        )
    return errors


def selected_images(project: Project, step: GenerateStep) -> list[ResolvedImage]:
    """Enabled, bound slots in slot order; unselected pool images never appear."""
    resolved = []
    for binding in step.images:
        if binding.enabled and binding.asset_id:
            asset = project.asset(binding.asset_id)
            resolved.append(ResolvedImage(asset_id=asset.id, path=asset.path))
    return resolved


def validate_step(project: Project, step: GenerateStep, prompt_text: str) -> list[str]:
    errors = []
    if not prompt_text.strip():
        errors.append("The prompt is empty; save a prompt first.")
    if len(step.images) > C.IMAGE_SLOTS:
        errors.append(f"At most {C.IMAGE_SLOTS} image slots are allowed.")
    try:
        images = selected_images(project, step)
    except KeyError as exc:
        return [*errors, str(exc)]
    if step.mode == "t2va" and images:
        errors.append("Text-only mode uses no images; disable the image slots.")
    if step.mode == "fl2va" and len(images) != 1:
        errors.append("First-frame mode needs exactly one enabled image.")
    if step.mode == "ref2va" and not images:
        errors.append("Reference mode needs one or two enabled images.")
    errors += validate_params(effective_params(project, step), step.mode)
    return errors


def build_request(project: Project, step: GenerateStep, prompt_text: str) -> JobRequest:
    """Freeze the request; a random seed becomes a concrete, recorded value."""
    params = effective_params(project, step)
    if params.seed == "random":
        params = params.model_copy(update={"seed": secrets.randbelow(C.SEED_MAX + 1)})
    return JobRequest(
        mode=step.mode,
        prompt_text=prompt_text,
        prompt_revision=step.prompt_revision,
        images=selected_images(project, step),
        params=params,
    )


def build_command(
    settings: Settings, job: Job, project_dir: Path, job_dir: Path, take_dir: Path
) -> list[str]:
    req = job.request
    p = req.params
    cmd = [
        str(settings.generator_python), str(settings.generator_script),
        "--strategy", p.strategy,
        "--prompt-file", str(job_dir / "prompt.txt"),
        "--frames", str(p.frames),
        "--width", str(p.width),
        "--height", str(p.height),
        "--steps", str(p.steps),
        "--seed", str(p.seed),
        "--output-dir", str(take_dir),
        "--output", "clip.mp4",
        "--no-auto-download",
        "--run-id", job.id,
        "--progress-jsonl", str(job_dir / "progress.jsonl"),
    ]
    if req.mode == "fl2va":
        cmd += ["--image", str(project_dir / req.images[0].path)]
    elif req.mode == "ref2va":
        for image in req.images:
            cmd += ["--reference-image", str(project_dir / image.path)]
        if p.reference_short_edge is not None:
            cmd += ["--reference-short-edge", str(p.reference_short_edge)]
    return cmd


def build_env(settings: Settings, params: GenParams, master_port: int) -> dict[str, str]:
    """Generator environment without credentials or unrelated secrets."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    env.pop("NICEGUI_STORAGE_PATH", None)
    env.update(
        HF_HOME=str(settings.hf_home),
        MASTER_PORT=str(master_port),
        PYTHONUNBUFFERED="1",
        LC_ALL="C.UTF-8",
        LANG="C.UTF-8",
    )
    if params.strategy == "context_parallel":
        env["CP_WORLD_SIZE"] = str(params.cp_world_size)
    else:
        env.pop("CP_WORLD_SIZE", None)
    return env
