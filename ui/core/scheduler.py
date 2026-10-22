"""Durable single-GPU job queue: launch, monitor, cancel, and reconcile jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import procs
from .generation import build_command, build_env
from .models import TERMINAL_STATES, Job, ProcessInfo, Progress, Take, utc_now
from .preflight import launch_check, load_repo_module
from .settings import Settings
from .storage import Store, atomic_write_text

log = logging.getLogger(__name__)

ORPHAN_GRACE_S = 15.0
ERROR_TAIL_LINES = 15


def extract_frames(repo_root: Path, video: Path) -> tuple[Path, Path]:
    module = load_repo_module(repo_root, "save_video_frames")
    first, last, _count = module.save_first_and_last_frames(video, video.parent)
    return first, last


def tail_lines(path: Path, lines: int) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join([line for line in text.splitlines() if line.strip()][-lines:])


class Scheduler:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.jobs: dict[str, Job] = {}
        self.paused = False
        self._popens: dict[str, subprocess.Popen] = {}
        self._offsets: dict[str, int] = {}
        self._cancel_sent: dict[str, float] = {}
        self._killed: set[str] = set()
        self._orphan_since: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._stopping = False

    # -- state -----------------------------------------------------------
    @property
    def _queue_file(self) -> Path:
        return self.settings.data_root / "queue.json"

    def load(self) -> None:
        """Load all jobs; running ones are reconciled on the next tick."""
        self.jobs = {job.id: job for job in self.store.load_all_jobs()}
        if self._queue_file.is_file():
            self.paused = bool(json.loads(self._queue_file.read_text()).get("paused", False))

    def ordered_jobs(self, project_id: str | None = None) -> list[Job]:
        jobs = [j for j in self.jobs.values() if project_id is None or j.project_id == project_id]
        return sorted(jobs, key=lambda j: (j.created_utc, j.id))

    def running_jobs(self) -> list[Job]:
        return [j for j in self.ordered_jobs() if j.state == "running"]

    def _save(self, job: Job) -> None:
        self.jobs[job.id] = job
        self.store.save_job(job)

    def _activity(self, job: Job, type_: str, message: str, **details: Any) -> None:
        self.store.append_activity(
            job.project_id, type_, message,
            step_id=job.step_id, job_id=job.id, take_id=job.take_id, details=details,
        )

    # -- commands --------------------------------------------------------
    def enqueue(self, job: Job) -> None:
        self._save(job)
        self._activity(job, "job.queued", f"{job.step_name}: queued (seed {job.request.params.seed})")

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        atomic_write_text(self._queue_file, json.dumps({"paused": paused}) + "\n")
        self.store.touch()

    def cancel(self, job_id: str) -> None:
        job = self.jobs[job_id]
        if job.state in ("queued", "blocked"):
            job.state, job.finished_utc, job.wait_reason = "cancelled", utc_now(), None
            self._save(job)
            self._activity(job, "job.cancelled", f"{job.step_name}: cancelled before start")
        elif job.state == "running" and not job.cancel_requested:
            job.cancel_requested = True
            self._save(job)
            if job.process:
                procs.signal_group(job.process.pgid, signal.SIGTERM)
            self._cancel_sent[job.id] = time.monotonic()
            self._activity(job, "job.cancel_requested", f"{job.step_name}: cancelling")

    # -- loop ------------------------------------------------------------
    def start(self) -> None:
        self._stopping = False
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        """Stop scheduling; running generations continue and are reattached later."""
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self.tick()
            except Exception:  # keep the queue alive; the error is logged
                log.exception("Scheduler tick failed")
            await asyncio.sleep(self.settings.poll_interval_s)

    async def tick(self) -> None:
        for job in self.running_jobs():
            await self._monitor(job)
        if self.paused or self.running_jobs():
            return
        for job in self.ordered_jobs():
            if job.state == "queued":
                if self._launch(job):
                    break
                if job.wait_reason:  # waiting for GPUs; keep FIFO order
                    break

    # -- launch ----------------------------------------------------------
    def _launch(self, job: Job) -> bool:
        project_dir = self.store.project_dir(job.project_id)
        status, message = launch_check(self.settings, job.request, project_dir)
        if status == "wait":
            if job.wait_reason != message:
                job.wait_reason = message
                self._save(job)
            return False
        if status == "fail":
            job.state, job.error, job.finished_utc, job.wait_reason = "failed", message, utc_now(), None
            self._save(job)
            self._activity(job, "job.failed", f"{job.step_name}: not started - {message}")
            return False

        job_dir = self.store.job_dir(job.project_id, job.id)
        take_dir = project_dir / "outputs" / job.step_id / job.take_id
        take_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "prompt.txt").write_text(job.request.prompt_text, encoding="utf-8")
        for stale in ("progress.jsonl", "exit.json", "runner.json"):
            (job_dir / stale).unlink(missing_ok=True)

        port = procs.free_port()
        command = build_command(self.settings, job, project_dir, job_dir, take_dir)
        env = build_env(self.settings, job.request.params, port)
        with open(job_dir / "run.log", "ab") as log_handle:
            popen = subprocess.Popen(
                [sys.executable, "-m", "ui.core.runner", "--job-dir", str(job_dir), "--", *command],
                cwd=self.settings.repo_root, env=env, stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
        ticks = procs.start_ticks(popen.pid) or 0
        self._popens[job.id] = popen
        self._offsets[job.id] = 0
        job.process = ProcessInfo(pid=popen.pid, pgid=popen.pid, start_ticks=ticks, master_port=port)
        job.state, job.started_utc, job.wait_reason = "running", utc_now(), None
        job.command = command
        job.progress = Progress(stage="starting", updated=time.time())
        self._save(job)
        p = job.request.params
        self._activity(job, "job.started",
                       f"{job.step_name}: started ({p.strategy}, {p.width}x{p.height}, "
                       f"{p.frames} frames, {p.steps} steps)")
        return True

    # -- monitor ---------------------------------------------------------
    def _read_progress(self, job: Job) -> None:
        path = self.store.job_dir(job.project_id, job.id) / "progress.jsonl"
        offset = self._offsets.get(job.id, 0)
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read()
        except OSError:
            return
        complete = data[: data.rfind(b"\n") + 1]  # ignore a partially written line
        self._offsets[job.id] = offset + len(complete)
        stage_changed = False
        for line in complete.decode("utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            if kind == "stage":
                job.progress.stage = event.get("name")
                job.progress.step = None
                job.progress.steps = event.get("steps")
                stage_changed = True
            elif kind == "step":
                job.progress.step, job.progress.steps = event.get("i"), event.get("n")
            elif kind in ("saved", "done"):
                job.progress.stage = kind
                stage_changed = True
            job.progress.updated = event.get("ts")
        if stage_changed:
            self._save(job)
        elif complete:
            self.store.touch()

    async def _monitor(self, job: Job) -> None:
        if job.process is None:
            await self._finalize(job)
            return
        self._read_progress(job)
        popen = self._popens.get(job.id)
        if popen is not None:
            popen.poll()  # reap our own child so it does not linger as a zombie
        runner_alive = procs.is_same_process(job.process.pid, job.process.start_ticks)
        group_alive = procs.group_alive(job.process.pgid)
        now = time.monotonic()

        sent = self._cancel_sent.get(job.id)
        if job.cancel_requested and sent is None:
            procs.signal_group(job.process.pgid, signal.SIGTERM)  # e.g. after a restart
            self._cancel_sent[job.id] = sent = now
        if (job.cancel_requested and group_alive and job.id not in self._killed
                and now - sent > self.settings.cancel_grace_s):
            procs.signal_group(job.process.pgid, signal.SIGKILL)
            self._killed.add(job.id)

        if runner_alive:
            return
        if group_alive:
            # Runner ended but ranks linger; free the GPUs after a grace period.
            since = self._orphan_since.setdefault(job.id, now)
            if now - since > ORPHAN_GRACE_S and job.id not in self._killed:
                procs.signal_group(job.process.pgid, signal.SIGKILL)
                self._killed.add(job.id)
            return
        await self._finalize(job)

    async def _finalize(self, job: Job) -> None:
        job_dir = self.store.job_dir(job.project_id, job.id)
        exit_file = job_dir / "exit.json"
        exit_info = json.loads(exit_file.read_text()) if exit_file.is_file() else None
        for registry in (self._popens, self._offsets, self._cancel_sent, self._orphan_since):
            registry.pop(job.id, None)
        self._killed.discard(job.id)
        job.finished_utc = utc_now()
        job.exit_code = exit_info["returncode"] if exit_info else None
        runtime = exit_info.get("runtime_s") if exit_info else None

        if job.cancel_requested:
            job.state = "cancelled"
            self._save(job)
            self._activity(job, "job.cancelled", f"{job.step_name}: cancelled")
            return
        if exit_info is None:
            job.state, job.error = "interrupted", "The generation process ended without an exit record."
            self._save(job)
            self._activity(job, "job.interrupted", f"{job.step_name}: interrupted")
            return
        if job.exit_code != 0:
            job.state = "failed"
            job.error = f"Exit code {job.exit_code}.\n" + tail_lines(job_dir / "run.log", ERROR_TAIL_LINES)
            self._save(job)
            self._activity(job, "job.failed", f"{job.step_name}: failed (exit code {job.exit_code})")
            return

        problem = await self._record_take(job, runtime)
        if problem:
            job.state, job.error = "failed", problem
            self._save(job)
            self._activity(job, "job.failed", f"{job.step_name}: output check failed - {problem}")
            return
        job.state, job.error = "succeeded", None
        self._save(job)
        minutes = f" in {runtime / 60:.1f} min" if runtime else ""
        self._activity(job, "job.succeeded", f"{job.step_name}: generated take {job.take_id}{minutes}",
                       runtime_s=runtime)

    async def _record_take(self, job: Job, runtime: float | None) -> str | None:
        """Verify outputs, extract frames, and add the take; return a problem or None."""
        project_dir = self.store.project_dir(job.project_id)
        take_dir = project_dir / "outputs" / job.step_id / job.take_id
        video, run_log = take_dir / "clip.mp4", take_dir / "clip.json"
        if not video.is_file() or video.stat().st_size == 0:
            return "clip.mp4 was not produced"
        try:
            logged = json.loads(run_log.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "clip.json run log is missing or unreadable"
        if logged.get("run_id") != job.id:
            return "run log does not belong to this job"
        try:
            first, last = await asyncio.to_thread(extract_frames, self.settings.repo_root, video)
        except Exception as exc:  # decoding problems mean the clip is unusable
            return f"frame extraction failed: {exc}"

        def rel(path: Path) -> str:
            return path.relative_to(project_dir).as_posix()

        params = job.request.params
        take = Take(
            id=job.take_id, job_id=job.id, created_utc=utc_now(),
            video=rel(video), run_log=rel(run_log), first_frame=rel(first), last_frame=rel(last),
            seed=int(params.seed), frames=int(logged.get("parameters", {}).get("frames", params.frames)),
            width=params.width, height=params.height, runtime_s=runtime,
        )
        project = self.store.load_project(job.project_id)
        try:
            step = project.step(job.step_id)
        except KeyError:
            return None
        step.takes.append(take)
        if step.selected_take is None:
            step.selected_take = take.id
        self.store.save_project(project)
        return None

    def is_terminal(self, job_id: str) -> bool:
        return self.jobs[job_id].state in TERMINAL_STATES
