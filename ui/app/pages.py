"""NiceGUI pages: project list and project workspace."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from nicegui import run, ui

from ui.core import constants as C
from ui.core.generation import validate_params
from ui.core.images import ImageRejected
from ui.core.models import GenerateStep, GenParams, Job, Project
from ui.core.service import ServiceError
from ui.core.storage import StorageError

from . import state
from .media import asset_url, take_url

STATE_COLORS = {
    "queued": "grey", "blocked": "brown", "running": "primary", "succeeded": "positive",
    "failed": "negative", "cancelled": "orange", "interrupted": "purple",
}
MONO = "font-family: ui-monospace, monospace; white-space: pre-wrap; font-size: 12px"


def _notify_error(exc: Exception) -> None:
    errors = exc.errors if isinstance(exc, ServiceError) else [str(exc)]
    ui.notify("\n".join(errors), type="negative", multi_line=True, close_button=True)


def _elapsed(job: Job) -> str:
    if not job.started_utc:
        return ""
    end = datetime.fromisoformat(job.finished_utc) if job.finished_utc else datetime.now(timezone.utc)
    seconds = int((end - datetime.fromisoformat(job.started_utc)).total_seconds())
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _progress_text(job: Job) -> str:
    if job.state == "queued":
        return job.wait_reason or "waiting in queue"
    if job.state != "running":
        return (job.error or "").splitlines()[0] if job.error else ""
    p = job.progress
    text = p.stage or "starting"
    if p.step is not None and p.steps:
        text += f" - step {p.step}/{p.steps}"
    if job.cancel_requested:
        text += " (cancelling)"
    return text


def queue_controls() -> None:
    """Pause/resume the queue (pause = stop after the current job)."""
    @ui.refreshable
    def controls() -> None:
        paused = state.service.scheduler.paused
        ui.badge("Queue paused" if paused else "Queue running", color="orange" if paused else "positive")
        if paused:
            ui.button("Resume", icon="play_arrow", on_click=lambda: (state.service.set_queue_paused(False),
                                                                   controls.refresh())).props("flat dense color=white")
        else:
            ui.button("Stop after current", icon="pause",
                      on_click=lambda: (state.service.set_queue_paused(True), controls.refresh())
                      ).props("flat dense color=white")
    controls()


def job_rows(jobs: list[Job], *, show_project: bool, on_log: Callable[[Job], None] | None = None) -> None:
    if not jobs:
        ui.label("No jobs yet.").classes("text-grey")
        return
    for job in reversed(jobs):
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.badge(job.state, color=STATE_COLORS[job.state]).classes("w-24 justify-center")
            if show_project:
                ui.link(job.step_name, f"/project/{job.project_id}").classes("w-40 ellipsis")
            else:
                ui.label(job.step_name).classes("w-40 ellipsis")
            p = job.request.params
            ui.label(f"{p.width}x{p.height} {p.frames}f {p.steps} steps seed {p.seed}").classes("text-caption w-64")
            ui.label(_elapsed(job)).classes("text-caption w-16")
            if job.state == "running" and job.progress.step and job.progress.steps:
                ui.linear_progress(job.progress.step / job.progress.steps, show_value=False).classes("w-32")
            ui.label(_progress_text(job)).classes("text-caption ellipsis col")
            if job.state in ("queued", "running") and not job.cancel_requested:
                ui.button(icon="stop", on_click=lambda j=job: _cancel(j)).props("flat dense round").tooltip("Cancel")
            if job.state in ("failed", "cancelled", "interrupted"):
                ui.button(icon="replay", on_click=lambda j=job: _retry(j)).props("flat dense round").tooltip("Retry")
            if on_log is not None:
                ui.button(icon="article", on_click=lambda j=job: on_log(j)).props("flat dense round").tooltip("Log")


def _cancel(job: Job) -> None:
    state.service.cancel_job(job.id)
    ui.notify(f"Cancelling {job.step_name}")


def _retry(job: Job) -> None:
    try:
        state.service.retry_job(job.id)
        ui.notify(f"{job.step_name} queued again")
    except ServiceError as exc:
        _notify_error(exc)


def version_timer(callback: Callable[[], None], interval: float = 1.0) -> None:
    """Call ``callback`` whenever persisted state changes."""
    seen = {"version": state.service.version}

    def check() -> None:
        if state.service.version != seen["version"]:
            seen["version"] = state.service.version
            callback()
    ui.timer(interval, check)


def param_form(params: GenParams, mode: str) -> Callable[[], dict[str, Any]]:
    """Render generation-parameter inputs; returns a getter for their values."""
    frame_options = {n: f"{n} ({n / C.FPS:.1f} s)" for n in C.valid_frame_counts()}
    with ui.grid(columns=4).classes("w-full gap-2"):
        width = ui.number("Width", value=params.width, min=C.MIN_DIM, max=C.MAX_DIM,
                          step=C.DIM_MULTIPLE, format="%d")
        height = ui.number("Height", value=params.height, min=C.MIN_DIM, max=C.MAX_DIM,
                           step=C.DIM_MULTIPLE, format="%d")
        frames = ui.select(frame_options, value=C.snap_frames(params.frames), label="Frames")
        steps = ui.number("Steps", value=params.steps, min=1, max=C.MAX_STEPS, step=1, format="%d")
        seed = ui.input("Seed", value=str(params.seed), placeholder="random or a number")
        strategy = ui.select(dict(C.STRATEGY_LABELS), value=params.strategy, label="Strategy")
        gpus = ui.number("GPUs", value=params.cp_world_size, min=C.MIN_CP_WORLD_SIZE,
                         max=C.MAX_CP_WORLD_SIZE, step=1, format="%d")
        gpus.bind_visibility_from(strategy, "value", backward=lambda v: v == "context_parallel")
        ref_edge = ui.number("Reference short edge (px)", value=params.reference_short_edge,
                             min=C.MIN_REFERENCE_SHORT_EDGE, max=C.MAX_REFERENCE_SHORT_EDGE, step=64,
                             format="%d")
        ref_edge.set_visibility(mode in ("ref2va", "any"))

    def values() -> dict[str, Any]:
        def as_int(value: Any) -> Any:
            return int(value) if isinstance(value, (int, float)) else value
        seed_text = (seed.value or "").strip().lower()
        return {
            "width": as_int(width.value), "height": as_int(height.value), "frames": frames.value,
            "steps": as_int(steps.value),
            "seed": "random" if seed_text in ("", "random") else (int(seed_text) if seed_text.isdigit() else seed_text),
            "strategy": strategy.value, "cp_world_size": as_int(gpus.value),
            "reference_short_edge": as_int(ref_edge.value),
        }
    return values


def install() -> None:
    @ui.page("/")
    def home() -> None:
        service = state.service
        with ui.header().classes("items-center"):
            ui.label("MiniMax Studio").classes("text-h6")
            ui.space()
            queue_controls()

        with ui.row().classes("w-full gap-4 no-wrap items-start"):
            with ui.card().classes("w-1/2"):
                ui.label("Projects").classes("text-h6")
                with ui.row().classes("w-full items-center"):
                    name = ui.input("New project name").classes("col")

                    def create() -> None:
                        try:
                            project = service.create_project(name.value or "")
                        except ServiceError as exc:
                            _notify_error(exc)
                            return
                        ui.navigate.to(f"/project/{project.id}")
                    ui.button("Create", icon="add", on_click=create)
                    name.on("keydown.enter", create)

                @ui.refreshable
                def project_list() -> None:
                    projects = service.list_projects()
                    if not projects:
                        ui.label("No projects yet.").classes("text-grey")
                    for project in projects:
                        with ui.row().classes("w-full items-center"):
                            ui.link(project.name, f"/project/{project.id}").classes("text-subtitle1")
                            ui.label(f"{len(project.steps)} steps, {len(project.assets)} images").classes(
                                "text-caption text-grey")
                            ui.space()
                            ui.label(project.updated_utc.replace("T", " ")[:16]).classes("text-caption")
                project_list()

            with ui.card().classes("w-1/2"):
                with ui.row().classes("w-full items-center"):
                    ui.label("Environment").classes("text-h6")
                    ui.space()
                    refresh_button = ui.button(icon="refresh").props("flat round").tooltip("Run checks")
                checks_box = ui.column().classes("w-full gap-1")

                async def run_checks() -> None:
                    checks_box.clear()
                    with checks_box:
                        ui.spinner()
                    checks = await run.io_bound(service.preflight)
                    checks_box.clear()
                    with checks_box:
                        for check in checks or []:
                            with ui.row().classes("items-start no-wrap gap-2"):
                                icon = "check_circle" if check.ok else ("error" if check.blocking else "warning")
                                color = "positive" if check.ok else ("negative" if check.blocking else "warning")
                                ui.icon(icon, color=color)
                                with ui.column().classes("gap-0"):
                                    ui.label(check.name).classes("text-weight-medium")
                                    ui.label(check.detail).classes("text-caption").style("word-break: break-all")
                refresh_button.on_click(run_checks)
                ui.timer(0.1, run_checks, once=True)

        with ui.card().classes("w-full"):
            ui.label("Job queue").classes("text-h6")

            @ui.refreshable
            def queue() -> None:
                job_rows(service.jobs()[-30:], show_project=True)
            queue()
        version_timer(lambda: (queue.refresh(), project_list.refresh()))

    @ui.page("/project/{project_id}")
    def project_page(project_id: str) -> None:
        try:
            state.service.get_project(project_id)
        except (StorageError, OSError):
            ui.label("Project not found.").classes("text-h6")
            ui.link("Back to projects", "/")
            return
        ProjectWorkspace(project_id).build()


class ProjectWorkspace:
    """One browser tab's view of a project."""

    def __init__(self, project_id: str) -> None:
        self.pid = project_id
        self.service = state.service
        project = self.service.get_project(project_id)
        self.step_id: str | None = project.steps[0].id if project.steps else None
        self.log_job_id: str | None = None
        self.view_take: dict[str, str] = {}

    @property
    def project(self) -> Project:
        return self.service.get_project(self.pid)

    def step(self) -> GenerateStep | None:
        if self.step_id is None:
            return None
        try:
            return self.project.step(self.step_id)
        except KeyError:
            return None

    # -- layout ----------------------------------------------------------
    def build(self) -> None:
        with ui.header().classes("items-center"):
            ui.button(icon="home", on_click=lambda: ui.navigate.to("/")).props("flat round color=white")
            self.title = ui.label(self.project.name).classes("text-h6")
            ui.button(icon="edit", on_click=self.rename_dialog).props("flat round dense color=white").tooltip(
                "Rename project")
            ui.button("Project defaults", icon="tune", on_click=self.defaults_dialog).props("flat color=white")
            ui.space()
            queue_controls()

        with ui.left_drawer(value=True, bordered=True).props("width=240"):
            ui.label("Steps").classes("text-subtitle1 text-weight-medium")
            self.step_list()
            ui.button("Add step", icon="add", on_click=self.add_step).props("flat").classes("w-full")

        with ui.splitter(value=58).classes("w-full") as splitter:
            with splitter.before:
                with ui.column().classes("w-full pr-3"):
                    self.editor()
            with splitter.after:
                with ui.column().classes("w-full pl-3"):
                    with ui.tabs().classes("w-full") as tabs:
                        takes_tab = ui.tab("Takes", icon="movie")
                        pool_tab = ui.tab("Image pool", icon="photo_library")
                    with ui.tab_panels(tabs, value=takes_tab).classes("w-full"):
                        with ui.tab_panel(takes_tab):
                            self.takes()
                        with ui.tab_panel(pool_tab):
                            self.pool_panel()

        ui.separator()
        with ui.tabs().classes("w-full") as bottom_tabs:
            jobs_tab = ui.tab("Jobs", icon="list")
            log_tab = ui.tab("Log", icon="article")
            activity_tab = ui.tab("Activity", icon="history")
        self.bottom_tabs, self.log_tab = bottom_tabs, log_tab
        with ui.tab_panels(bottom_tabs, value=jobs_tab).classes("w-full"):
            with ui.tab_panel(jobs_tab):
                self.jobs()
            with ui.tab_panel(log_tab):
                self.log_view()
            with ui.tab_panel(activity_tab):
                self.activity()

        version_timer(self.on_change)
        ui.timer(2.0, self.refresh_running_log)

    def on_change(self) -> None:
        self.step_list.refresh()
        self.takes.refresh()
        self.pool.refresh()
        self.jobs.refresh()
        self.activity.refresh()

    # -- steps -----------------------------------------------------------
    @ui.refreshable_method
    def step_list(self) -> None:
        project = self.project
        if not project.steps:
            ui.label("Add a step to start.").classes("text-caption text-grey")
        for step in project.steps:
            selected = step.id == self.step_id
            ui.button(step.name, on_click=lambda s=step: self.select_step(s.id)).props(
                f"{'unelevated' if selected else 'flat'} no-caps align=left").classes("w-full")

    def select_step(self, step_id: str) -> None:
        self.step_id = step_id
        self.step_list.refresh()
        self.editor.refresh()
        self.takes.refresh()

    def add_step(self) -> None:
        step = self.service.add_step(self.pid)
        self.select_step(step.id)

    # -- editor ----------------------------------------------------------
    @ui.refreshable_method
    def editor(self) -> None:
        step = self.step()
        if step is None:
            ui.label("Select or add a step.").classes("text-grey")
            return
        project = self.project
        name = ui.input("Step name", value=step.name).classes("w-full")
        name.on("blur", lambda: self._update(name=name.value))

        mode = ui.toggle(dict(C.MODE_LABELS), value=step.mode,
                         on_change=lambda e: (self._update(mode=e.value), self.editor.refresh()))

        prompt_text = self.service.get_prompt(self.pid, step.id)
        ui.label(f"Prompt (revision {step.prompt_revision or '-'})").classes("text-subtitle2 q-mt-sm")
        prompt = ui.textarea(value=prompt_text, placeholder="Describe the scene, action, camera, and sound...") \
            .props("outlined autogrow input-style='min-height: 160px'").classes("w-full")
        self.prompt_input = prompt
        with ui.row().classes("items-center"):
            ui.button("Save prompt", icon="save", on_click=self.save_prompt).props("outline")
            dirty = ui.label("").classes("text-caption text-orange")
            prompt.on_value_change(lambda e: dirty.set_text(
                "Unsaved changes" if e.value != self.service.get_prompt(self.pid, step.id) else ""))

        if step.mode != "t2va":
            self.image_slots(project, step)

        ui.label("Parameters").classes("text-subtitle2 q-mt-sm")
        overridden = ", ".join(sorted(step.overrides)) or "none (using project defaults)"
        ui.label(f"Overrides: {overridden}").classes("text-caption text-grey")
        get_params = param_form(self.service.step_params(self.pid, step.id), step.mode)
        with ui.row():
            ui.button("Apply parameters", icon="check", on_click=lambda: self.apply_params(get_params())).props(
                "outline")
            ui.button("Use project defaults", icon="restart_alt",
                      on_click=lambda: self.apply_params({})).props("flat")

        errors_box = ui.column().classes("gap-0")

        def show_errors() -> None:
            errors = self.service.check_draft(self.pid, step.id, prompt.value or "", get_params())
            errors_box.clear()
            with errors_box:
                for error in errors:
                    ui.label(error).classes("text-negative text-caption")
        ui.button("Generate", icon="movie_creation", on_click=lambda: self.generate(get_params)) \
            .props("unelevated color=primary").classes("q-mt-sm")
        show_errors()
        prompt.on("blur", show_errors)

    def _update(self, **changes: Any) -> None:
        try:
            self.service.update_step(self.pid, self.step_id, **changes)
        except (ServiceError, KeyError) as exc:
            _notify_error(exc)

    def save_prompt(self) -> bool:
        changed = self.service.save_prompt(self.pid, self.step_id, self.prompt_input.value or "")
        ui.notify("Prompt saved" if changed else "No changes to save")
        self.editor.refresh()
        return changed

    def apply_params(self, values: dict[str, Any]) -> None:
        try:
            self.service.update_step(self.pid, self.step_id, params=values)
        except (ServiceError, ValueError) as exc:
            _notify_error(exc)
            return
        self.editor.refresh()

    def generate(self, get_params: Callable[[], dict[str, Any]]) -> None:
        """Save the prompt and parameters as shown, then queue the step."""
        try:
            self.service.save_prompt(self.pid, self.step_id, self.prompt_input.value or "")
            self.service.update_step(self.pid, self.step_id, params=get_params())
            job = self.service.run_step(self.pid, self.step_id)
        except (ServiceError, ValueError) as exc:
            _notify_error(exc)
            return
        ui.notify(f"Queued with seed {job.request.params.seed}", type="positive")
        self.editor.refresh()

    # -- image slots -----------------------------------------------------
    def image_slots(self, project: Project, step: GenerateStep) -> None:
        slots = step.images if step.mode == "ref2va" else step.images[:1]
        enabled = [i for i, b in enumerate(step.images) if b.enabled and b.asset_id]
        ui.label("Reference images" if step.mode == "ref2va" else "First-frame image").classes(
            "text-subtitle2 q-mt-sm")
        with ui.row().classes("gap-3"):
            for index, binding in enumerate(slots):
                with ui.card().classes("w-48 items-center"):
                    if binding.enabled and binding.asset_id and index in enabled and step.mode == "ref2va":
                        ui.label(f"Picture {enabled.index(index) + 1}").classes("text-weight-medium")
                    else:
                        ui.label(f"Slot {index + 1}").classes("text-grey")
                    if binding.asset_id:
                        try:
                            asset = project.asset(binding.asset_id)
                            ui.image(asset_url(self.pid, asset.id)).classes("w-40 h-28").props("fit=contain")
                            ui.label(asset.name).classes("text-caption ellipsis w-40")
                        except KeyError:
                            ui.label("Missing image").classes("text-negative")
                    else:
                        ui.label("Empty").classes("text-grey h-28 flex items-center")
                    with ui.row().classes("gap-1"):
                        ui.checkbox("Use", value=binding.enabled,
                                    on_change=lambda e, i=index: self.set_slot(i, enabled=e.value))
                        ui.button(icon="photo_library", on_click=lambda i=index: self.choose_image(i)).props(
                            "flat dense round").tooltip("Choose from pool")
                        ui.button(icon="clear", on_click=lambda i=index: self.set_slot(i, asset_id=None,
                                                                                         enabled=False)).props(
                            "flat dense round").tooltip("Clear slot")
            if step.mode == "ref2va":
                ui.button(icon="swap_horiz", on_click=self.swap_slots).props("flat round").tooltip("Swap order")

    def set_slot(self, index: int, **changes: Any) -> None:
        step = self.step()
        if step is None:
            return
        images = [b.model_copy() for b in step.images]
        images[index] = images[index].model_copy(update=changes)
        if step.mode == "fl2va" and changes.get("enabled"):
            for i in range(1, len(images)):
                images[i] = images[i].model_copy(update={"enabled": False})
        self._update(images=images)
        self.editor.refresh()

    def swap_slots(self) -> None:
        step = self.step()
        if step is None:
            return
        images = list(reversed([b.model_copy() for b in step.images]))
        self._update(images=images)
        self.editor.refresh()

    def choose_image(self, index: int) -> None:
        project = self.project
        with ui.dialog() as dialog, ui.card().classes("w-[900px] max-w-full"):
            ui.label(f"Choose an image for slot {index + 1}").classes("text-h6")
            if not project.assets:
                ui.label("The image pool is empty; upload images in the Image pool tab.").classes("text-grey")
            search = ui.input("Filter by name").classes("w-full")

            @ui.refreshable
            def grid() -> None:
                term = (search.value or "").lower()
                with ui.grid(columns=5).classes("w-full gap-2"):
                    for asset in project.assets:
                        if term and term not in asset.name.lower():
                            continue
                        with ui.card().classes("p-1 cursor-pointer items-center").on(
                                "click", lambda a=asset: (self.set_slot(index, asset_id=a.id, enabled=True),
                                                          dialog.close())):
                            ui.image(asset_url(self.pid, asset.id)).classes("w-36 h-24").props("fit=contain")
                            ui.label(asset.name).classes("text-caption ellipsis w-36")
            grid()
            search.on_value_change(grid.refresh)
            ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    # -- takes -----------------------------------------------------------
    @ui.refreshable_method
    def takes(self) -> None:
        step = self.step()
        if step is None or not step.takes:
            ui.label("No takes yet. Generate this step to create one.").classes("text-grey")
            return
        current = self.view_take.get(step.id) or step.selected_take or step.takes[-1].id
        take = next((t for t in step.takes if t.id == current), step.takes[-1])
        ui.video(take_url(self.pid, step.id, take.id, "video")).classes("w-full")
        with ui.row().classes("items-center gap-2"):
            ui.label(f"{take.id}: seed {take.seed}, {take.frames} frames, {take.width}x{take.height}").classes(
                "text-caption")
            if take.id == step.selected_take:
                ui.badge("selected", color="positive")
            else:
                ui.button("Select take", on_click=lambda: (self.service.select_take(self.pid, step.id, take.id))
                          ).props("flat dense")
            ui.button("Reuse seed", on_click=lambda: self.apply_params(
                {**self.service.step_params(self.pid, step.id).model_dump(), "seed": take.seed})).props("flat dense")
            ui.button("Delete take", icon="delete", on_click=lambda: self.confirm_delete_take(step.id, take.id)
                      ).props("flat dense color=negative")
        with ui.row().classes("gap-2"):
            for kind in ("first", "last"):
                with ui.column().classes("items-center gap-0"):
                    ui.image(take_url(self.pid, step.id, take.id, kind)).classes("w-40").props("fit=contain")
                    ui.label(f"{kind} frame").classes("text-caption")
                    ui.button("Add to pool", icon="add_photo_alternate",
                              on_click=lambda k=kind: self.add_frame_to_pool(step.id, take.id, k)
                              ).props("flat dense no-caps size=sm")
                    if kind == "last":
                        ui.button("Use in next step", icon="redo",
                                  on_click=lambda: self.use_in_next_step(step.id, take.id)
                                  ).props("flat dense no-caps size=sm").tooltip(
                            "Bind this frame to image slot 1 of the next step (created if missing)")
        ui.label("All takes").classes("text-subtitle2 q-mt-sm")
        for other in reversed(step.takes):
            label = f"{other.id} - seed {other.seed}" + (" (selected)" if other.id == step.selected_take else "")
            ui.button(label, on_click=lambda t=other: self.show_take(step.id, t.id)).props(
                f"{'unelevated' if other.id == take.id else 'flat'} dense no-caps")

    def show_take(self, step_id: str, take_id: str) -> None:
        self.view_take[step_id] = take_id
        self.takes.refresh()

    def confirm_delete_take(self, step_id: str, take_id: str) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete take {take_id}?").classes("text-h6")
            ui.label("Its video, run log, and frame images are removed permanently. "
                     "Images already copied to the pool are kept.").classes("text-caption")

            def delete() -> None:
                try:
                    self.service.delete_take(self.pid, step_id, take_id)
                except (ServiceError, OSError) as exc:
                    ui.notify(str(exc), type="negative")
                    return
                self.view_take.pop(step_id, None)
                dialog.close()
                self.takes.refresh()
                ui.notify(f"Deleted take {take_id}")
            with ui.row():
                ui.button("Delete", icon="delete", on_click=delete).props("color=negative")
                ui.button("Cancel", on_click=dialog.close).props("flat")
        dialog.open()

    def add_frame_to_pool(self, step_id: str, take_id: str, kind: str) -> None:
        try:
            asset, created = self.service.add_take_frame(self.pid, step_id, take_id, kind)
        except (ServiceError, ImageRejected) as exc:
            ui.notify(str(exc), type="negative")
            return
        ui.notify(f"Added '{asset.name}' to the image pool" if created else f"Already in the pool as '{asset.name}'")

    def use_in_next_step(self, step_id: str, take_id: str) -> None:
        try:
            target = self.service.use_last_frame_in_next_step(self.pid, step_id, take_id)
        except (ServiceError, ImageRejected) as exc:
            ui.notify(str(exc), type="negative")
            return
        self.select_step(target.id)
        ui.notify(f"Last frame set as image 1 of '{target.name}'")

    # -- image pool ------------------------------------------------------
    def pool_panel(self) -> None:
        async def handle_upload(event) -> None:
            data = await event.file.read()
            try:
                asset = await run.io_bound(self.service.prepare_image, self.pid, event.file.name, data)
            except ImageRejected as exc:
                ui.notify(str(exc), type="negative")
                return
            if asset is None:
                return
            _asset, created = self.service.register_image(self.pid, asset)
            ui.notify(f"Added {event.file.name}" if created else f"{event.file.name} is already in the pool")

        ui.upload(label="Add images (JPEG, PNG, WebP)", multiple=True, auto_upload=True,
                  max_file_size=C.MAX_UPLOAD_BYTES, on_upload=handle_upload,
                  on_rejected=lambda: ui.notify("File rejected (too large?)", type="warning")) \
            .props("accept='image/jpeg,image/png,image/webp'").classes("w-full")
        self.pool()

    @ui.refreshable_method
    def pool(self) -> None:
        project = self.project
        ui.label(f"{len(project.assets)} image(s). Uploading does not select an image; "
                 "choose images per step in the editor.").classes("text-caption text-grey")
        with ui.grid(columns=3).classes("w-full gap-2"):
            for asset in project.assets:
                with ui.card().classes("p-1 items-center"):
                    ui.image(asset_url(self.pid, asset.id)).classes("w-full h-28").props("fit=contain")
                    ui.label(asset.name).classes("text-caption ellipsis w-full")
                    ui.label(f"{asset.width}x{asset.height}").classes("text-caption text-grey")

    # -- jobs, log, activity ---------------------------------------------
    @ui.refreshable_method
    def jobs(self) -> None:
        job_rows(self.service.jobs(self.pid), show_project=False, on_log=self.show_log)

    def show_log(self, job: Job) -> None:
        self.log_job_id = job.id
        self.log_view.refresh()
        self.bottom_tabs.set_value(self.log_tab)

    def _log_job(self) -> Job | None:
        jobs = self.service.jobs(self.pid)
        if self.log_job_id:
            return next((j for j in jobs if j.id == self.log_job_id), None)
        running = [j for j in jobs if j.state == "running"]
        return running[-1] if running else (jobs[-1] if jobs else None)

    @ui.refreshable_method
    def log_view(self) -> None:
        job = self._log_job()
        if job is None:
            ui.label("No job selected.").classes("text-grey")
            return
        ui.label(f"{job.step_name} - {job.id} ({job.state})").classes("text-caption")
        if job.error:
            ui.label(job.error).classes("text-negative").style(MONO)
        # column-reverse keeps the scroll position pinned to the newest lines across refreshes
        with ui.element("div").classes("w-full bg-grey-2 q-pa-sm").style(
                "max-height: 360px; overflow-y: auto; display: flex; flex-direction: column-reverse"):
            ui.label(self.service.job_log(job) or "(no output yet)").style(MONO)

    def refresh_running_log(self) -> None:
        job = self._log_job()
        if job is not None and job.state == "running":
            self.log_view.refresh()

    @ui.refreshable_method
    def activity(self) -> None:
        events = self.service.activity(self.pid, limit=100)
        if not events:
            ui.label("No activity yet.").classes("text-grey")
        for event in events:
            with ui.row().classes("gap-3 no-wrap"):
                ui.label(event.ts_utc.replace("T", " ")[:19]).classes("text-caption text-grey w-36")
                ui.label(event.message).classes("text-caption")

    # -- dialogs ---------------------------------------------------------
    def rename_dialog(self) -> None:
        with ui.dialog() as dialog, ui.card():
            name = ui.input("Project name", value=self.project.name).classes("w-80")

            def save() -> None:
                try:
                    self.service.rename_project(self.pid, name.value or "")
                except ServiceError as exc:
                    _notify_error(exc)
                    return
                self.title.set_text(self.project.name)
                dialog.close()
            with ui.row():
                ui.button("Save", on_click=save)
                ui.button("Cancel", on_click=dialog.close).props("flat")
        dialog.open()

    def defaults_dialog(self) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-[760px] max-w-full"):
            ui.label("Project defaults").classes("text-h6")
            ui.label("Steps use these unless they override a value.").classes("text-caption text-grey")
            get_values = param_form(self.project.defaults, "any")

            def save() -> None:
                try:
                    values = get_values()
                    params = GenParams.model_validate(values)
                    errors = validate_params(params, "ref2va")
                    if errors:
                        raise ServiceError(errors)
                    self.service.update_defaults(self.pid, values)
                except (ServiceError, ValueError) as exc:
                    _notify_error(exc)
                    return
                dialog.close()
                self.editor.refresh()
            with ui.row():
                ui.button("Save", on_click=save)
                ui.button("Cancel", on_click=dialog.close).props("flat")
        dialog.open()
