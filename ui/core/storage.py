"""File-based persistence: atomic JSON documents and JSONL activity logs."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import SCHEMA_VERSION, ActivityEvent, Job, Project, job_order, new_id, utc_now

log = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
_ID_RE = re.compile(r"^[a-z]{2,4}_[0-9a-f]{10}$")


class StorageError(RuntimeError):
    pass


class InstanceLockError(RuntimeError):
    pass


def check_id(value: str) -> str:
    """Reject anything that is not a generated ID, so IDs are safe path components."""
    if not _ID_RE.match(value):
        raise StorageError(f"Invalid identifier: {value!r}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a flushed temporary sibling, keeping the previous version as ``.bak``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)


def write_model(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def _parse(path: Path, model_cls: type[ModelT]) -> ModelT:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("schema_version", SCHEMA_VERSION) if isinstance(data, dict) else None
    if version is not None and version > SCHEMA_VERSION:
        raise StorageError(f"{path} uses unsupported schema version {version}")
    return model_cls.model_validate(data)


def read_model(path: Path, model_cls: type[ModelT]) -> ModelT:
    """Read and validate a record, falling back to its last-known-good backup."""
    try:
        return _parse(path, model_cls)
    except (OSError, ValueError, ValidationError) as exc:
        backup = path.with_name(path.name + ".bak")
        if backup.is_file():
            try:
                model = _parse(backup, model_cls)
                log.warning("Recovered %s from backup after: %s", path, exc)
                return model
            except (OSError, ValueError, ValidationError):
                pass
        raise StorageError(f"Cannot read {path}: {exc}") from exc


class Store:
    """Owns every metadata write under the data root (single writer)."""

    def __init__(self, data_root: Path) -> None:
        self.root = data_root
        self.projects_dir = data_root / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.version = 0  # bumped on every change; UI pages poll it
        self._cache: dict[str, Project] = {}
        self._lock = threading.RLock()
        self._lock_handle: Any = None

    # -- instance lock ---------------------------------------------------
    def acquire_instance_lock(self) -> None:
        handle = open(self.root / ".instance.lock", "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise InstanceLockError(
                f"Another MiniMax UI instance is already using {self.root}"
            ) from exc
        self._lock_handle = handle

    def release_instance_lock(self) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def touch(self) -> None:
        self.version += 1

    # -- projects --------------------------------------------------------
    def project_dir(self, project_id: str) -> Path:
        return self.projects_dir / check_id(project_id)

    def create_project(self, name: str) -> Project:
        now = utc_now()
        project = Project(id=new_id("prj"), name=name, created_utc=now, updated_utc=now)
        root = self.project_dir(project.id)
        for sub in ("prompts", "assets", "jobs", "outputs", "exports"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self.save_project(project)
        return project

    def list_projects(self) -> list[Project]:
        projects = []
        for path in sorted(self.projects_dir.glob("*/project.json")):
            try:
                projects.append(self.load_project(path.parent.name))
            except StorageError as exc:
                log.error("Skipping unreadable project: %s", exc)
        return sorted(projects, key=lambda p: p.updated_utc, reverse=True)

    def load_project(self, project_id: str) -> Project:
        with self._lock:
            if project_id not in self._cache:
                path = self.project_dir(project_id) / "project.json"
                self._cache[project_id] = read_model(path, Project)
            return self._cache[project_id].model_copy(deep=True)

    def save_project(self, project: Project) -> None:
        with self._lock:
            project.updated_utc = utc_now()
            write_model(self.project_dir(project.id) / "project.json", project)
            self._cache[project.id] = project.model_copy(deep=True)
            self.touch()

    # -- prompts ---------------------------------------------------------
    def prompt_path(self, project_id: str, step_id: str, number: int) -> Path:
        return self.project_dir(project_id) / "prompts" / check_id(step_id) / f"r{number:03d}.txt"

    def read_prompt(self, project_id: str, step_id: str, number: int) -> str:
        if number <= 0:
            return ""
        return self.prompt_path(project_id, step_id, number).read_text(encoding="utf-8")

    def write_prompt(self, project_id: str, step_id: str, number: int, text: str) -> None:
        atomic_write_text(self.prompt_path(project_id, step_id, number), text)

    # -- jobs ------------------------------------------------------------
    def job_dir(self, project_id: str, job_id: str) -> Path:
        return self.project_dir(project_id) / "jobs" / check_id(job_id)

    def save_job(self, job: Job) -> None:
        with self._lock:
            write_model(self.job_dir(job.project_id, job.id) / "job.json", job)
            self.touch()

    def load_all_jobs(self) -> list[Job]:
        jobs = []
        for path in self.projects_dir.glob("*/jobs/*/job.json"):
            try:
                jobs.append(read_model(path, Job))
            except StorageError as exc:
                log.error("Skipping unreadable job: %s", exc)
        return sorted(jobs, key=job_order)

    # -- activity log ----------------------------------------------------
    def activity_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project.log.jsonl"

    def append_activity(self, project_id: str, type_: str, message: str, **fields: Any) -> None:
        event = ActivityEvent(id=new_id("evt"), ts_utc=utc_now(), type=type_, message=message, **fields)
        path = self.activity_path(project_id)
        with self._lock, open(path, "ab+") as handle:
            # Start on a fresh line if a crash left a partial trailing record.
            handle.seek(0, os.SEEK_END)
            if handle.tell() > 0:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write((event.model_dump_json() + "\n").encode("utf-8"))
            handle.flush()
        self.touch()

    def read_activity(self, project_id: str, limit: int = 200) -> list[ActivityEvent]:
        path = self.activity_path(project_id)
        if not path.is_file():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(ActivityEvent.model_validate_json(line))
            except ValidationError:
                log.warning("Skipping malformed activity record in %s", path)
        return list(reversed(events[-limit:]))
