from __future__ import annotations

import json
import os
from pathlib import Path

from ui.core import procs
from ui.core.models import ImageBinding, ProcessInfo
from ui.core.service import Service

from .conftest import png_bytes, run_until


def _queued_step(service: Service, prompt: str = "A fox in snow.") -> tuple[str, str]:
    project = service.create_project("Sched")
    asset, _ = service.add_image(project.id, "ref.png", png_bytes((200, 30, 30)))
    step = service.add_step(project.id, "Scene 1")
    service.save_prompt(project.id, step.id, prompt)
    service.update_step(project.id, step.id, mode="ref2va", images=[ImageBinding(asset_id=asset.id)])
    return project.id, step.id


async def test_successful_job_records_take_and_frames(service: Service, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    pid, sid = _queued_step(service, "一只狐狸")
    job = service.run_step(pid, sid)
    await run_until(service, lambda: service.scheduler.jobs[job.id].state == "succeeded")

    done = service.scheduler.jobs[job.id]
    assert done.exit_code == 0 and done.progress.stage == "done"
    step = service.get_project(pid).step(sid)
    assert [t.id for t in step.takes] == [job.take_id] and step.selected_take == job.take_id
    for kind in ("video", "first", "last"):
        assert service.take_file(pid, sid, job.take_id, kind).stat().st_size > 0

    log = service.job_log(done)
    assert "OPENAI_API_KEY=unset" in log and "一只狐狸" in log
    types = [e.type for e in service.activity(pid)]
    assert types[:3] == ["job.succeeded", "job.started", "job.queued"]


async def test_failed_job_reports_error(service: Service, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_GEN_MODE", "fail")
    pid, sid = _queued_step(service)
    job = service.run_step(pid, sid)
    await run_until(service, lambda: service.scheduler.jobs[job.id].state == "failed")
    failed = service.scheduler.jobs[job.id]
    assert failed.exit_code == 3 and "simulated failure" in (failed.error or "")
    assert service.get_project(pid).step(sid).takes == []

    monkeypatch.setenv("FAKE_GEN_MODE", "ok")
    retry = service.retry_job(job.id)
    assert retry.request == failed.request and retry.take_id != failed.take_id
    await run_until(service, lambda: service.scheduler.jobs[retry.id].state == "succeeded")


async def test_cancel_kills_whole_process_group(service: Service, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_GEN_MODE", "child")
    pid, sid = _queued_step(service)
    job = service.run_step(pid, sid)
    take_dir = service.store.project_dir(pid) / "outputs" / sid / job.take_id
    await run_until(service, lambda: (take_dir / "child.pid").is_file())
    grandchild = int((take_dir / "child.pid").read_text())
    pgid = service.scheduler.jobs[job.id].process.pgid
    assert procs.group_alive(pgid)

    service.cancel_job(job.id)
    await run_until(service, lambda: service.scheduler.jobs[job.id].state == "cancelled")
    assert not procs.group_alive(pgid)
    assert procs.start_ticks(grandchild) is None


async def test_queue_runs_one_job_at_a_time_in_order(service: Service, monkeypatch, tmp_path: Path) -> None:
    release = tmp_path / "release"
    monkeypatch.setenv("FAKE_GEN_MODE", "release")
    monkeypatch.setenv("FAKE_GEN_RELEASE", str(release))
    pid, sid = _queued_step(service)
    first = service.run_step(pid, sid)
    second = service.run_step(pid, sid)
    await run_until(service, lambda: service.scheduler.jobs[first.id].state == "running")
    for _ in range(5):
        await service.scheduler.tick()
    assert service.scheduler.jobs[second.id].state == "queued"
    release.touch()
    await run_until(service, lambda: service.scheduler.jobs[second.id].state == "succeeded")
    assert service.scheduler.jobs[first.id].finished_utc <= service.scheduler.jobs[second.id].started_utc


async def test_paused_queue_does_not_start_jobs(service: Service) -> None:
    pid, sid = _queued_step(service)
    service.set_queue_paused(True)
    job = service.run_step(pid, sid)
    for _ in range(5):
        await service.scheduler.tick()
    assert service.scheduler.jobs[job.id].state == "queued"
    service.set_queue_paused(False)
    await run_until(service, lambda: service.scheduler.jobs[job.id].state == "succeeded")


async def test_cancel_queued_job(service: Service) -> None:
    pid, sid = _queued_step(service)
    service.set_queue_paused(True)
    job = service.run_step(pid, sid)
    service.cancel_job(job.id)
    assert service.scheduler.jobs[job.id].state == "cancelled"


async def test_restart_reattaches_to_live_runner(settings, monkeypatch, tmp_path: Path) -> None:
    release = tmp_path / "release"
    monkeypatch.setenv("FAKE_GEN_MODE", "release")
    monkeypatch.setenv("FAKE_GEN_RELEASE", str(release))
    first = Service(settings)
    first.open()
    pid, sid = _queued_step(first)
    job = first.run_step(pid, sid)
    await run_until(first, lambda: first.scheduler.jobs[job.id].state == "running")
    first.close()  # app server "restarts"; the runner keeps going

    second = Service(settings)
    second.open()
    try:
        assert second.scheduler.jobs[job.id].state == "running"
        release.touch()
        await run_until(second, lambda: second.scheduler.jobs[job.id].state == "succeeded")
        assert len(second.scheduler.jobs) == 1  # nothing was relaunched
    finally:
        second.close()


async def test_restart_marks_dead_job_interrupted(settings) -> None:
    first = Service(settings)
    first.open()
    pid, sid = _queued_step(first)
    first.set_queue_paused(True)
    job = first.run_step(pid, sid)
    stale = first.scheduler.jobs[job.id]
    stale.state = "running"
    stale.process = ProcessInfo(pid=2**22 - 1, pgid=2**22 - 1, start_ticks=1)
    first.store.save_job(stale)
    first.close()

    second = Service(settings)
    second.open()
    try:
        await second.scheduler.tick()
        assert second.scheduler.jobs[job.id].state == "interrupted"
    finally:
        second.close()


async def test_restart_finalizes_job_that_finished_while_down(settings) -> None:
    first = Service(settings)
    first.open()
    pid, sid = _queued_step(first)
    first.set_queue_paused(True)
    job = first.run_step(pid, sid)
    stale = first.scheduler.jobs[job.id]
    stale.state = "running"
    stale.process = ProcessInfo(pid=2**22 - 1, pgid=2**22 - 1, start_ticks=1)
    first.store.save_job(stale)
    job_dir = first.store.job_dir(pid, job.id)
    (job_dir / "exit.json").write_text(json.dumps({"returncode": 5, "runtime_s": 1.0}))
    first.close()

    second = Service(settings)
    second.open()
    try:
        await second.scheduler.tick()
        assert second.scheduler.jobs[job.id].state == "failed"
        assert second.scheduler.jobs[job.id].exit_code == 5
    finally:
        second.close()


async def test_missing_generator_fails_without_launch(service: Service) -> None:
    service.settings.generator_script = Path("/nonexistent/LoMMH.py")
    pid, sid = _queued_step(service)
    job = service.run_step(pid, sid)
    await service.scheduler.tick()
    failed = service.scheduler.jobs[job.id]
    assert failed.state == "failed" and failed.process is None
    assert "Generator script not found" in (failed.error or "")


def test_process_helpers() -> None:
    assert procs.group_alive(os.getpgid(0))
    assert not procs.group_alive(2**22 - 1)
    assert procs.start_ticks(2**22 - 1) is None
    assert procs.is_same_process(os.getpid(), procs.start_ticks(os.getpid()) or -1)
