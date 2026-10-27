"""Persistent records (JSON) for projects, jobs, and activity events."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from . import constants as C

SCHEMA_VERSION = 1

Mode = Literal["t2va", "fl2va", "ref2va"]
JobState = Literal["queued", "blocked", "running", "succeeded", "failed", "cancelled", "interrupted"]
TERMINAL_STATES = ("succeeded", "failed", "cancelled", "interrupted")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(5)}"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenParams(_Model):
    width: int = 1056
    height: int = 576
    frames: int = C.MAX_FRAMES
    steps: int = 40
    seed: int | Literal["random"] = "random"
    strategy: str = "context_parallel"
    cp_world_size: int = 4
    reference_short_edge: int | None = 1536


class ImageAsset(_Model):
    id: str
    name: str
    path: str  # project-relative
    thumb: str
    width: int
    height: int
    sha256: str
    added_utc: str


class ImageBinding(_Model):
    source: Literal["asset"] = "asset"
    asset_id: str | None = None
    enabled: bool = True


class PromptRevision(_Model):
    number: int
    created_utc: str
    origin: Literal["manual", "imported", "assistant"] = "manual"


class Take(_Model):
    id: str
    job_id: str
    created_utc: str
    video: str  # project-relative
    run_log: str
    first_frame: str
    last_frame: str
    seed: int
    frames: int
    width: int
    height: int
    runtime_s: float | None = None


class GenerateStep(_Model):
    id: str
    type: Literal["generate"] = "generate"
    name: str
    revision: int = 1
    mode: Mode = "t2va"
    prompt_revision: int = 0  # 0 = no prompt saved yet
    prompt_revisions: list[PromptRevision] = Field(default_factory=list)
    images: list[ImageBinding] = Field(
        default_factory=lambda: [ImageBinding(enabled=False) for _ in range(C.IMAGE_SLOTS)]
    )
    overrides: dict[str, Any] = Field(default_factory=dict)
    takes: list[Take] = Field(default_factory=list)
    selected_take: str | None = None


class ExportInput(_Model):
    step_id: str
    step_name: str
    take_id: str


class Export(_Model):
    id: str
    name: str
    created_utc: str
    video: str  # project-relative
    inputs: list[ExportInput]
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    has_audio: bool
    runtime_s: float | None = None


class Project(_Model):
    schema_version: int = SCHEMA_VERSION
    id: str
    name: str
    created_utc: str
    updated_utc: str
    archived: bool = False
    defaults: GenParams = Field(default_factory=GenParams)
    assets: list[ImageAsset] = Field(default_factory=list)
    steps: list[GenerateStep] = Field(default_factory=list)
    exports: list[Export] = Field(default_factory=list)

    def step(self, step_id: str) -> GenerateStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"Unknown step: {step_id}")

    def export(self, export_id: str) -> Export:
        for export in self.exports:
            if export.id == export_id:
                return export
        raise KeyError(f"Unknown export: {export_id}")

    def asset(self, asset_id: str) -> ImageAsset:
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        raise KeyError(f"Unknown image: {asset_id}")


class ResolvedImage(_Model):
    asset_id: str
    path: str  # project-relative


class JobRequest(_Model):
    mode: Mode
    prompt_text: str
    prompt_revision: int
    images: list[ResolvedImage]
    params: GenParams  # seed is always concrete here


class ProcessInfo(_Model):
    pid: int
    pgid: int
    start_ticks: int
    master_port: int | None = None


class Progress(_Model):
    stage: str | None = None
    step: int | None = None
    steps: int | None = None
    updated: float | None = None


class Job(_Model):
    schema_version: int = SCHEMA_VERSION
    id: str
    type: Literal["generate"] = "generate"
    project_id: str
    step_id: str
    step_name: str
    step_revision: int
    take_id: str
    seq: int = 0  # queue order; 0 for jobs saved before it existed
    state: JobState = "queued"
    wait_reason: str | None = None
    cancel_requested: bool = False
    created_utc: str
    started_utc: str | None = None
    finished_utc: str | None = None
    request: JobRequest
    command: list[str] = Field(default_factory=list)
    process: ProcessInfo | None = None
    progress: Progress = Field(default_factory=Progress)
    exit_code: int | None = None
    error: str | None = None


def job_order(job: Job) -> tuple:
    # Not created_utc alone: it has 1 s resolution and IDs are random, which broke FIFO order.
    return (job.seq, job.created_utc, job.id)


class ActivityEvent(_Model):
    id: str
    ts_utc: str
    type: str
    message: str
    step_id: str | None = None
    job_id: str | None = None
    take_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
