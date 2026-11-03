"""Environment and launch checks that never import torch or LoMMH.py."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from .models import JobRequest
from .settings import Settings

MODEL_DIR = "models--MiniMaxAI--MiniMax-H3"
MIN_FREE_GB = 20


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass
class Gpu:
    index: int
    name: str
    total_mib: int
    used_mib: int


@lru_cache(maxsize=None)
def load_repo_module(repo_root: Path, name: str) -> ModuleType:
    """Import a stdlib-only helper script from the repository root by path."""
    spec = importlib.util.spec_from_file_location(name, repo_root / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name}.py from {repo_root}")
    module = importlib.util.module_from_spec(spec)
    # Must be registered before executing: @dataclass looks its module up in sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def snapshot_dir(settings: Settings) -> Path | None:
    cache = settings.hf_home / "hub" / MODEL_DIR
    try:
        revision = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snapshot = cache / "snapshots" / revision
    return snapshot if (snapshot / "modular_model_index.json").is_file() else None


def reference_partition(settings: Settings) -> tuple[bool, str]:
    snapshot = snapshot_dir(settings)
    if snapshot is None:
        return False, "model snapshot not found"
    module = load_repo_module(settings.repo_root, "download_transformer_ref")
    return module.partition_status(snapshot)


def query_gpus() -> list[Gpu] | None:
    """GPU memory via nvidia-smi; None when it cannot be queried."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        index, name, total, used = (part.strip() for part in line.split(","))
        gpus.append(Gpu(int(index), name, int(total), int(used)))
    return gpus


def gpus_needed(strategy: str, cp_world_size: int) -> list[int]:
    if strategy == "context_parallel":
        return list(range(cp_world_size))
    if strategy == "max_gpu":
        return [0, 1, 2]
    return [0]


def run_preflight(settings: Settings) -> list[Check]:
    checks = [
        Check("Generator Python", os.access(settings.generator_python, os.X_OK),
              str(settings.generator_python)),
        Check("Generator script", settings.generator_script.is_file(), str(settings.generator_script)),
    ]
    snapshot = snapshot_dir(settings)
    checks.append(Check("Model snapshot", snapshot is not None,
                        str(snapshot) if snapshot else f"not found under {settings.hf_home}"))
    if snapshot is not None:
        ok, detail = reference_partition(settings)
        checks.append(Check("Reference model (transformer_ref)", ok,
                            detail if ok else f"{detail} - needed only for reference-image mode",
                            blocking=False))
    if settings.gpu_check:
        gpus = query_gpus()
        if gpus is None:
            checks.append(Check("GPUs", False, "nvidia-smi could not be queried"))
        else:
            busy = [g.index for g in gpus if g.used_mib > settings.gpu_busy_mib]
            summary = ", ".join(f"{g.index}: {g.name} {g.used_mib}/{g.total_mib} MiB" for g in gpus)
            checks.append(Check("GPUs", bool(gpus), summary or "no GPUs found"))
            if busy:
                checks.append(Check("GPUs idle", False,
                                    f"GPU(s) {busy} are in use; jobs wait until they are free",
                                    blocking=False))
    free_gb = shutil.disk_usage(settings.data_root).free / 1024**3
    checks.append(Check("Free disk (data root)", free_gb >= MIN_FREE_GB,
                        f"{free_gb:.0f} GB free at {settings.data_root}", blocking=False))
    return checks


def launch_check(settings: Settings, request: JobRequest, project_dir: Path) -> tuple[str, str]:
    """Return ("ok" | "wait" | "fail", message) for starting a job now."""
    if not os.access(settings.generator_python, os.X_OK):
        return "fail", f"Generator Python not found: {settings.generator_python}"
    if not settings.generator_script.is_file():
        return "fail", f"Generator script not found: {settings.generator_script}"
    missing = [img.path for img in request.images if not (project_dir / img.path).is_file()]
    if missing:
        return "fail", f"Input image(s) missing: {', '.join(missing)}"
    if settings.model_check:
        if snapshot_dir(settings) is None:
            return "fail", f"MiniMax-H3 model snapshot not found under {settings.hf_home}"
        if request.mode == "ref2va":
            ok, detail = reference_partition(settings)
            if not ok:
                return "fail", (
                    f"Reference model is incomplete ({detail}). Run "
                    "download_transformer_ref.py, then retry."
                )
    if settings.gpu_check:
        gpus = query_gpus()
        if gpus is not None:
            needed = gpus_needed(request.params.strategy, request.params.cp_world_size)
            available = {g.index: g for g in gpus}
            if any(i not in available for i in needed):
                return "fail", f"Needs GPU(s) {needed}; found {sorted(available)}"
            busy = [i for i in needed if available[i].used_mib > settings.gpu_busy_mib]
            if busy:
                return "wait", f"Waiting for busy GPU(s) {busy}"
    return "ok", ""
