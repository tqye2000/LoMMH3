"""Operations the UI calls; the boundary a future API or frontend would wrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import constants as C
from .generation import build_request, effective_params, validate_step
from .images import ingest_image
from .models import GenerateStep, GenParams, ImageAsset, ImageBinding, Job, Project, PromptRevision, new_id, utc_now
from .preflight import Check, run_preflight
from .scheduler import Scheduler, tail_lines
from .settings import Settings
from .storage import Store, check_id


class ServiceError(ValueError):
    """A user-correctable problem; ``errors`` lists every reason."""

    def __init__(self, errors: list[str] | str) -> None:
        self.errors = [errors] if isinstance(errors, str) else errors
        super().__init__("; ".join(self.errors))


class Service:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.data_root.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.data_root)
        self.scheduler = Scheduler(settings, self.store)

    def open(self) -> None:
        self.store.acquire_instance_lock()
        self.scheduler.load()

    def close(self) -> None:
        self.store.release_instance_lock()

    @property
    def version(self) -> int:
        return self.store.version

    # -- projects --------------------------------------------------------
    def list_projects(self) -> list[Project]:
        return [p for p in self.store.list_projects() if not p.archived]

    def get_project(self, project_id: str) -> Project:
        return self.store.load_project(check_id(project_id))

    def create_project(self, name: str) -> Project:
        name = name.strip()
        if not name:
            raise ServiceError("Project name is required.")
        project = self.store.create_project(name[:100])
        self.store.append_activity(project.id, "project.created", f"Project '{project.name}' created")
        return project

    def rename_project(self, project_id: str, name: str) -> None:
        name = name.strip()
        if not name:
            raise ServiceError("Project name is required.")
        project = self.get_project(project_id)
        old, project.name = project.name, name[:100]
        self.store.save_project(project)
        self.store.append_activity(project_id, "project.renamed", f"Renamed '{old}' to '{project.name}'")

    def update_defaults(self, project_id: str, values: dict[str, Any]) -> None:
        project = self.get_project(project_id)
        project.defaults = GenParams.model_validate({**project.defaults.model_dump(), **values})
        self.store.save_project(project)
        self.store.append_activity(project_id, "project.defaults", "Project defaults updated")

    # -- image pool ------------------------------------------------------
    def prepare_image(self, project_id: str, filename: str, data: bytes) -> ImageAsset:
        """Validate and store image files only; safe to run in a worker thread."""
        return ingest_image(self.store.project_dir(check_id(project_id)), filename, data)

    def register_image(self, project_id: str, asset: ImageAsset) -> tuple[ImageAsset, bool]:
        """Record a prepared image; duplicates reuse the existing asset. Returns (asset, created)."""
        project = self.get_project(project_id)
        for existing in project.assets:
            if existing.sha256 == asset.sha256:
                for leftover in (asset.path, asset.thumb):
                    (self.store.project_dir(project_id) / leftover).unlink(missing_ok=True)
                return existing, False
        project.assets.append(asset)
        self.store.save_project(project)
        self.store.append_activity(project_id, "asset.added", f"Image '{asset.name}' added",
                                   details={"asset_id": asset.id})
        return asset, True

    def add_image(self, project_id: str, filename: str, data: bytes) -> tuple[ImageAsset, bool]:
        return self.register_image(project_id, self.prepare_image(project_id, filename, data))

    # -- steps and prompts -----------------------------------------------
    def add_step(self, project_id: str, name: str = "") -> GenerateStep:
        project = self.get_project(project_id)
        step = GenerateStep(id=new_id("stp"), name=name.strip() or f"Scene {len(project.steps) + 1}")
        project.steps.append(step)
        self.store.save_project(project)
        self.store.append_activity(project_id, "step.created", f"Step '{step.name}' added", step_id=step.id)
        return step

    def update_step(
        self,
        project_id: str,
        step_id: str,
        *,
        name: str | None = None,
        mode: str | None = None,
        images: list[ImageBinding] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Apply edits; generation-relevant changes bump the step revision."""
        project = self.get_project(project_id)
        step = project.step(step_id)
        changes = []
        if name is not None and name.strip() and name.strip() != step.name:
            step.name = name.strip()[:100]
            changes.append("name")
        if mode is not None and mode != step.mode:
            if mode not in C.MODES:
                raise ServiceError(f"Unknown mode: {mode}")
            step.mode = mode  # type: ignore[assignment]
            changes.append("mode")
        if images is not None:
            if len(images) > C.IMAGE_SLOTS:
                raise ServiceError(f"At most {C.IMAGE_SLOTS} image slots are allowed.")
            for binding in images:
                if binding.asset_id:
                    project.asset(binding.asset_id)
            padded = images + [ImageBinding(enabled=False)] * (C.IMAGE_SLOTS - len(images))
            if padded != step.images:
                step.images = padded
                changes.append("images")
        if params is not None:
            merged = GenParams.model_validate({**project.defaults.model_dump(), **params}).model_dump()
            defaults = project.defaults.model_dump()
            overrides = {k: v for k, v in merged.items() if v != defaults[k]}
            if overrides != step.overrides:
                step.overrides = overrides
                changes.append("parameters")
        if not changes:
            return
        if changes != ["name"]:
            step.revision += 1
        self.store.save_project(project)
        self.store.append_activity(project_id, "step.updated",
                                   f"Step '{step.name}' updated ({', '.join(changes)})", step_id=step_id)

    def step_params(self, project_id: str, step_id: str) -> GenParams:
        project = self.get_project(project_id)
        return effective_params(project, project.step(step_id))

    def get_prompt(self, project_id: str, step_id: str) -> str:
        step = self.get_project(project_id).step(step_id)
        return self.store.read_prompt(project_id, step_id, step.prompt_revision)

    def save_prompt(self, project_id: str, step_id: str, text: str, origin: str = "manual") -> bool:
        """Save a new prompt revision if the text changed; returns whether it did."""
        project = self.get_project(project_id)
        step = project.step(step_id)
        if text == self.store.read_prompt(project_id, step_id, step.prompt_revision):
            return False
        number = step.prompt_revision + 1
        self.store.write_prompt(project_id, step_id, number, text)
        step.prompt_revision = number
        step.prompt_revisions.append(PromptRevision(number=number, created_utc=utc_now(), origin=origin))  # type: ignore[arg-type]
        step.revision += 1
        self.store.save_project(project)
        self.store.append_activity(project_id, "prompt.saved",
                                   f"Prompt for '{step.name}' saved (revision {number})", step_id=step_id)
        return True

    def select_take(self, project_id: str, step_id: str, take_id: str) -> None:
        project = self.get_project(project_id)
        step = project.step(step_id)
        if take_id not in {t.id for t in step.takes}:
            raise ServiceError(f"Unknown take: {take_id}")
        step.selected_take = take_id
        self.store.save_project(project)
        self.store.append_activity(project_id, "take.selected", f"'{step.name}': selected take {take_id}",
                                   step_id=step_id, take_id=take_id)

    def validate_step(self, project_id: str, step_id: str) -> list[str]:
        project = self.get_project(project_id)
        step = project.step(step_id)
        return validate_step(project, step, self.get_prompt(project_id, step_id))

    def check_draft(self, project_id: str, step_id: str, prompt: str, params: dict[str, Any]) -> list[str]:
        """Validate unsaved editor values without persisting them."""
        project = self.get_project(project_id)
        step = project.step(step_id)
        try:
            merged = GenParams.model_validate({**project.defaults.model_dump(), **params})
        except ValidationError as exc:
            return [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        step.overrides = merged.model_dump()
        return validate_step(project, step, prompt)

    # -- jobs ------------------------------------------------------------
    def run_step(self, project_id: str, step_id: str) -> Job:
        project = self.get_project(project_id)
        step = project.step(step_id)
        prompt = self.store.read_prompt(project_id, step_id, step.prompt_revision)
        errors = validate_step(project, step, prompt)
        if errors:
            raise ServiceError(errors)
        job = Job(
            id=new_id("job"), project_id=project_id, step_id=step_id, step_name=step.name,
            step_revision=step.revision, take_id=new_id("tk"), created_utc=utc_now(),
            request=build_request(project, step, prompt),
        )
        self.scheduler.enqueue(job)
        return job

    def retry_job(self, job_id: str) -> Job:
        """Queue the same frozen request again as a new job and take."""
        old = self.scheduler.jobs[check_id(job_id)]
        if old.state not in ("failed", "cancelled", "interrupted"):
            raise ServiceError("Only failed, cancelled, or interrupted jobs can be retried.")
        job = old.model_copy(update={
            "id": new_id("job"), "take_id": new_id("tk"), "state": "queued", "created_utc": utc_now(),
            "started_utc": None, "finished_utc": None, "process": None, "exit_code": None,
            "error": None, "wait_reason": None, "cancel_requested": False, "command": [],
        }, deep=True)
        job.progress.stage = job.progress.step = job.progress.steps = None
        self.scheduler.enqueue(job)
        return job

    def cancel_job(self, job_id: str) -> None:
        self.scheduler.cancel(check_id(job_id))

    def set_queue_paused(self, paused: bool) -> None:
        self.scheduler.set_paused(paused)

    def jobs(self, project_id: str | None = None) -> list[Job]:
        return self.scheduler.ordered_jobs(project_id)

    def job_log(self, job: Job, lines: int = 400) -> str:
        return tail_lines(self.store.job_dir(job.project_id, job.id) / "run.log", lines)

    def activity(self, project_id: str, limit: int = 200):
        return self.store.read_activity(check_id(project_id), limit)

    def preflight(self) -> list[Check]:
        return run_preflight(self.settings)

    # -- media -----------------------------------------------------------
    def media_file(self, project_id: str, relative: str) -> Path:
        """Resolve a project-relative media path, refusing anything outside the project."""
        root = self.store.project_dir(project_id).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ServiceError("Media not found.")
        return path

    def asset_file(self, project_id: str, asset_id: str, kind: str) -> Path:
        try:
            asset = self.get_project(project_id).asset(asset_id)
        except KeyError as exc:
            raise ServiceError("Media not found.") from exc
        return self.media_file(project_id, asset.thumb if kind == "thumb" else asset.path)

    def take_file(self, project_id: str, step_id: str, take_id: str, kind: str) -> Path:
        try:
            step = self.get_project(project_id).step(step_id)
        except KeyError as exc:
            raise ServiceError("Media not found.") from exc
        take = next((t for t in step.takes if t.id == take_id), None)
        if take is None or kind not in ("video", "first", "last"):
            raise ServiceError("Media not found.")
        relative = {"video": take.video, "first": take.first_frame, "last": take.last_frame}[kind]
        return self.media_file(project_id, relative)
