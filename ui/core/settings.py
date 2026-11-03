"""Application settings: environment variables over ``<data-root>/settings.json``."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGED_CACHE = Path.home() / "hf_models" / "hub" / "models--MiniMaxAI--MiniMax-H3"


def _default_hf_home() -> str:
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    return str(Path.home() / "hf_models") if _STAGED_CACHE.is_dir() else "/mnt/d/hf_models"


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data_root: Path
    repo_root: Path = REPO_ROOT
    generator_python: Path = Path.home() / "minimax-venv" / "bin" / "python"
    generator_script: Path = REPO_ROOT / "LoMMH.py"
    hf_home: Path = Path(_default_hf_home())
    host: str = "127.0.0.1"
    port: int = 8080
    allowed_hosts: list[str] = ["127.0.0.1", "localhost"]
    # GPUs whose used memory exceeds this are treated as busy (WSL hides process lists).
    gpu_busy_mib: int = 2048
    gpu_check: bool = True
    model_check: bool = True
    poll_interval_s: float = 1.0
    cancel_grace_s: float = 20.0


_ENV_MAP = {
    "MINIMAX_UI_DATA": "data_root",
    "MINIMAX_GEN_PYTHON": "generator_python",
    "MINIMAX_GEN_SCRIPT": "generator_script",
    "MINIMAX_UI_HOST": "host",
    "MINIMAX_UI_PORT": "port",
}


def load_settings() -> Settings:
    """Read settings.json from the data root, then apply environment overrides."""
    data_root = Path(os.environ.get("MINIMAX_UI_DATA", str(Path.home() / "minimax-ui-data")))
    values: dict = {}
    settings_file = data_root / "settings.json"
    if settings_file.is_file():
        values.update(json.loads(settings_file.read_text(encoding="utf-8")))
    for env, field in _ENV_MAP.items():
        if os.environ.get(env):
            values[field] = os.environ[env]
    if os.environ.get("HF_HOME"):
        values["hf_home"] = os.environ["HF_HOME"]
    if os.environ.get("MINIMAX_UI_ALLOWED_HOSTS"):
        values["allowed_hosts"] = os.environ["MINIMAX_UI_ALLOWED_HOSTS"].split(",")
    if os.environ.get("MINIMAX_UI_GPU_CHECK") == "0":
        values["gpu_check"] = False
    if os.environ.get("MINIMAX_UI_MODEL_CHECK") == "0":
        values["model_check"] = False
    values["data_root"] = data_root
    return Settings(**values)
