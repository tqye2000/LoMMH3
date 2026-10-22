# Video Project UI Implementation Plan

Status: Phase 1 (core vertical slice) implemented in `ui/` (revision 4); phases 2-5 not started.

Date: 2026-10-22

Revision 2 incorporates the plan review: required changes to the existing
scripts, a reduced and reordered first release, concrete specifications (code
layout, service interface, example records, job states), and wording corrections.

Revision 3 replaces the React + TypeScript/FastAPI stack with NiceGUI, a
Python-only UI framework, over framework-independent core modules.

Revision 4 records phase 1 as built. Deviations from revision 3: the app uses its
own WSL venv (`~/minimax-ui-venv`, created by `ui/setup_ui.sh`) instead of adding
packages to the GPU venv; pages refresh by polling a change counter instead of an
event bus; each job has its own folder; and take media routes include the step ID.

## 1. Purpose

Build a local browser application for creating short MiniMax-H3 clips and
assembling them into longer videos. Users should be able to manage projects,
edit prompts, upload any number of reference images into a project-level pool,
explicitly select up to two image inputs per generation, configure generation
parameters, and execute an ordered workflow without editing shell scripts.

An optional AI assistant will help create and edit prompts using OpenAI's GPT
models and an OpenAI API key supplied by the user. Suggestions require explicit
approval before changing a prompt and never start video generation automatically.

The application will reuse the existing generation and media utilities. It will
use JSON files for persistent metadata and an append-only project activity log,
not SQLite.

## 2. Scope and Assumptions

### First version

- One user on one workstation, accessed through a local browser.
  There are no user accounts; a per-launch local session token protects the app.
- The NiceGUI app server and generation runs execute inside WSL2 (required: the Windows `.venv`
  has no NCCL, so `context_parallel` cannot run there); native Linux is also a
  target. The Windows `.venv` remains for manual script use only.
- One GPU generation job at a time, including across multiple projects.
- Durable projects, workflow definitions, queued jobs, and completed results.
- Text-to-video, first-frame-to-video, and reference-image-to-video generation.
- A reusable project-level reference-image pool with no fixed image-count limit
  (add and select only; see later releases).
- Up to two explicitly selected image inputs per generation, including continuity
  frames; uploading images does not automatically select them.
- Ordered generation and combination steps, run unattended with
  stop-after-current and cancel controls.
- Hard-cut assembly of compatible clips (matching resolution and frame rate, each
  with audio), preserving each clip's audio; incompatible inputs are rejected.
- Optional GPT prompt assistant (whole-prompt suggestions) with a user-supplied
  OpenAI API key, delivered after the core workflow works.
- Explicit import of existing prompts, images, and completed videos.

### Later releases (planned)

- Review pauses (approval gates) between workflow steps.
- Image renaming, archiving, and guarded deletion in the pool.
- Assembly normalization: letterboxing mismatched sizes, explicit frame-rate
  conversion, silence for audio-less clips, and resolving nested assemblies to
  their original clips.
- Assistant streaming, selected-text edits, and follow-up conversation.
- Cleanup of unused takes and other disk-management tools.

### Deferred (not planned yet)

- User accounts, multi-user access, remote hosting, and cloud execution.
- Simultaneous generation on separate GPU groups.
- A general node-graph editor or full nonlinear editing timeline.
- Trimming, transitions, subtitles, background music, and audio mixing.
- Reference-video controls, although the existing generator supports them.
- Keeping models resident across jobs or resuming partial denoising.
- Other AI providers, custom API endpoints, AI image/video analysis, autonomous
  workflow execution, and persistent API-key storage.

Existing project folders and launcher scripts remain usable and unchanged by
default. Importing an asset copies it into managed storage rather than modifying
its original location.

## 3. Existing Integration Points

| Existing file | Planned use | Important constraint |
| --- | --- | --- |
| [LoMMH.py](../LoMMH.py) | Launch generation through its CLI | Keep GPU/model loading outside the web server |
| [save_video_frames.py](../save_video_frames.py) | Extract first/last frames for previews and continuity | Returns PNG files; decodes the full video to find its last frame |
| [join_videos_with_audio.py](../join_videos_with_audio.py) | Extend assembly to an ordered input list | Currently takes two videos and requires audio in both |
| [run_ref2va_cp.sh](../run_ref2va_cp.sh) | Reference for WSL configuration and generation parameters | Hard-coded project paths are examples, not application configuration |
| [run_fl2va.sh](../run_fl2va.sh) | Reference for first-frame generation | First-frame mode is distinct from reference-image mode |
| [README.md](../README.md) | Existing installation and model guidance | Add application startup instructions when implementation is available |

The generator already writes an MP4 and a JSON run log. Preserve both, and add
application-owned job metadata for workflow dependencies, reference short-edge
settings, environment selection, and lifecycle state.

First-frame mode (`--image`) cannot be combined with reference-image mode
(`--reference-image`). The generator permits more than two reference images;
the two-slot limit is an intentional per-generation UI requirement, not a model
limitation or a limit on the number of uploaded images in the project pool.

### Required changes to existing scripts

The current code has behaviors the application must account for. All changes are
backward compatible: new flags default to today's behavior, so the existing
launchers keep working.

| Current behavior | Location | Planned change |
| --- | --- | --- |
| A missing `transformer_ref` partition (~62 GB) is downloaded automatically inside a generation run | `_ensure_reference_partition` in [LoMMH.py](../LoMMH.py) | Add `--no-auto-download`; the app always passes it. Preflight uses `download_transformer_ref.py --check`, and downloading becomes a separate, explicitly approved job |
| Progress is only progress-bar console text | Diffusers pipeline output | Add optional `--progress-jsonl PATH` writing stage changes, denoise step i/N, and completion; console parsing is a fallback only |
| The run log omits the reference short edge and context-parallel GPU count | `save_output` in [LoMMH.py](../LoMMH.py) | Record `reference_short_edge` and `cp_world_size`; add optional `--run-id` stored in the log |
| `HF_HOME` defaults to `D:\hf_models`, and the snapshot is resolved when the script is imported | Top of [LoMMH.py](../LoMMH.py) | The app always sets `HF_HOME` explicitly. Preflight uses its own lightweight check and never imports `LoMMH.py` or torch |
| `context_parallel` spawns one child process per GPU | `_cp_spawn` in [LoMMH.py](../LoMMH.py) | Launch the generator in its own session/process group; verify with a deliberate rank failure that a failed rank produces a non-zero exit code |
| Rendezvous defaults to `MASTER_PORT=29500` | `_cp_worker` in [LoMMH.py](../LoMMH.py) | The app sets a free `MASTER_PORT` per job; preflight detects conflicts with manually started runs |
| The join utility accepts exactly two inputs, both with audio | [join_videos_with_audio.py](../join_videos_with_audio.py) | Add an N-input function/CLI while keeping the two-input CLI |

Application dependencies (NiceGUI, which bundles FastAPI/Uvicorn; Pydantic; later
the OpenAI SDK) are listed in `ui/requirements.txt` and installed by
`ui/setup_ui.sh` into a separate WSL venv (`~/minimax-ui-venv`). The UI never
imports torch; it launches [LoMMH.py](../LoMMH.py) with the GPU venv's interpreter.
[wsl_setup.sh](../wsl_setup.sh) and the GPU [requirements.txt](../requirements.txt)
are not changed.

## 4. Recommended Architecture

- **UI:** NiceGUI pages and components written in Python. NiceGUI runs on
  FastAPI/Uvicorn and renders Quasar/Vue components in the browser, pushing
  updates over its own websocket. No Node.js toolchain or separate frontend build.
- **Core:** framework-independent Python modules for storage/schemas, scheduling,
  generator launching, media, and the prompt assistant. The NiceGUI layer is a
  thin presentation layer calling these modules, so a different frontend could be
  added later without rewriting the core.
- **Prompt assistant:** a core module using the official OpenAI Python SDK and
  Responses API, with a configurable supported GPT model.
- **Execution:** the scheduler runs as a background task in the app server; each
  generation or assembly runs as a separate process through a small runner wrapper.
- **Persistence:** versioned JSON documents and JSON Lines activity logs.
- **Deployment:** one Python process started by `ui/run_ui.sh`, bound to localhost.
  It must run as a single Uvicorn worker, because the scheduler and UI state are
  per-process.

```text
Windows browser (localhost, session token)
      |
      v
WSL2 NiceGUI app server ---- project.json, job JSON, project.log.jsonl
 (UI pages + core modules + scheduler)
      |            \
      |             +---- Prompt assistant ---- HTTPS ----> OpenAI API
      v
Runner wrapper (own process group) ---- per-job log, progress JSONL, exit record
      |
      v
LoMMH.py / media utilities ---- MP4, run JSON, extracted frames
```

The app server (the "backend" in this document) owns metadata writes and
scheduling decisions. The runner wrapper launches the generator, writes the
execution log, progress JSONL, and a final exit record (exit code, timestamps),
but never mutates authoritative metadata.

Browser closure or refresh must not stop a running job: jobs belong to the server,
not to a browser session. Because the runner starts in its own session/process
group, a generation can outlive an app-server restart. On startup, reconciliation
either reattaches to a verified live runner (PID, start time, and run ID match),
following its progress and exit record, or marks the job `interrupted`.

Keep blocking work off NiceGUI's event loop: hashing, thumbnails, and OpenAI calls
use NiceGUI's `run.io_bound`/`run.cpu_bound` helpers or async clients. Assembly
runs as a subprocess in a CPU lane that does not block the single GPU queue.

Use configurable interpreter and model-cache paths, with the existing WSL venv
and launchers as setup references. Do not import the GPU pipeline into the app
server or construct commands by concatenating shell strings.

Assistant requests go through the local backend to OpenAI's official HTTPS API;
the browser does not call OpenAI directly. Keep this service separate from the
GPU runner and queue. Ordinary editing and local generation remain usable without
an API key or internet access once model files are present. Only optional AI
assistance requires connectivity to OpenAI.

### Code layout

```text
ui/
  main.py             Entry point: settings, ui.run(...)
  core/               Framework-independent; no NiceGUI imports
    constants.py      Frame rules, dimension step, image-slot limit
    models.py         Pydantic records (project, step, take, job, activity)
    settings.py       Settings from settings.json and environment variables
    storage.py        Atomic JSON writes, backups, instance lock, activity log
    images.py         Image validation, immutable storage, thumbnails
    generation.py     Effective parameters, validation, command and environment
    preflight.py      Environment and launch checks (no torch import)
    procs.py          Process identity and process-group helpers
    runner.py         Wrapper process that launches the generator and records exit
    scheduler.py      Durable queue, monitoring, cancellation, reconciliation
    service.py        Operations the UI calls (see service interface)
  app/                NiceGUI layer: state, auth, media routes, pages
  tests/              pytest suites, NiceGUI user tests, fake generator
  requirements.txt    App dependencies (requirements-dev.txt adds test tools)
  setup_ui.sh         Creates ~/minimax-ui-venv
  run_ui.sh           WSL launcher
```

Existing root scripts stay in place and are invoked by path. Generation limits
(`17*n+5` frames within 124-345, dimensions in multiples of 32, the two-slot image
limit) are defined once in `ui/core/constants.py` and used by both validation and
the UI. The slot limit is configurable; the model accepts up to 9 reference images.
A test checks the constants against [LoMMH.py](../LoMMH.py).

### Environment and paths

- The app runs from its own WSL venv (`~/minimax-ui-venv`) and launches generations
  with the GPU venv created by [wsl_setup.sh](../wsl_setup.sh) (default
  `~/minimax-venv`); the Windows `.venv` is not supported for the app.
- Default data root on WSL-native storage: `~/minimax-ui-data`.
  Settings show the equivalent Windows path (`\\wsl.localhost\<distro>\...`) for
  browsing; exports can also be downloaded through the browser.
- Imports accept browser uploads or a typed path. Windows paths such as `D:\...`
  are translated to `/mnt/d/...`, validated, and copied into managed storage.

### Service interface and routes

NiceGUI event handlers call `ui/core/service.py` directly; there is no public JSON
API in the first version. The service layer is the boundary a future REST API or
alternative frontend would wrap.

| Area | Service operations |
| --- | --- |
| Settings and preflight | Get/update settings, run preflight |
| Projects | List, create, open, rename projects |
| Image pool | Add images (multi-file), list assets, get thumbnail/file path |
| Steps and prompts | Create/update/delete/reorder steps, save prompt, list prompt revisions |
| Jobs and queue | Run step, run remaining, list/get jobs, cancel, retry, pause/resume queue |
| Activity and logs | Read activity page, tail job log from an offset |
| Assistant | Set/clear key, test connection, request suggestion, cancel request |

Only a few plain HTTP routes are needed, registered on NiceGUI's FastAPI `app`:

| Route | Purpose |
| --- | --- |
| `GET /media/{pid}/takes/{sid}/{tid}/video` | Take video with HTTP range support for seeking |
| `GET /media/{pid}/takes/{sid}/{tid}/{first,last}` | Extracted frames |
| `GET /media/{pid}/assets/{aid}/{file,thumb}` | Pool images and thumbnails |
| `GET /media/{pid}/exports/{eid}` | Export download (phase 3) |

Live updates use NiceGUI's websocket. Storage bumps a change counter on every write
and progress update; each page polls it once per second and refreshes only when it
changes, so multiple tabs stay consistent. The editor is not refreshed
automatically, so it never overwrites unsaved edits.

The launcher prints a localhost URL containing a one-time token. Opening it stores
the current launch's session nonce in NiceGUI's signed per-browser storage
(`storage_secret` set per launch). Middleware validates the `Host` header against
localhost to block DNS rebinding, then rejects pages and media routes without the
session. NiceGUI's own assets and websocket (`/_nicegui...`) are not gated, because
websocket clients can only attach to page instances created by authenticated
requests. NiceGUI persists `app.storage` to disk, so it must never hold API keys or
prompt text.

## 5. Project Storage and Data Model

### Directory layout

```text
<application-data>/
  settings.json
  projects/
    <project-id>/
      project.json
      project.log.jsonl
      prompts/
      assets/
      jobs/
        <job-id>/
          job.json        Authoritative job record
          prompt.txt      Frozen prompt passed to the generator
          run.log         Runner and generator output
          progress.jsonl  Structured progress from --progress-jsonl
          runner.json     Runner/process-group identity
          exit.json       Exit code and runtime written by the runner
      outputs/
        <step-id>/
          <take-id>/
            clip.mp4
            clip.json
            clip_first.png
            clip_last.png
      exports/
```

Prefer WSL-native storage for active project data. Use stable IDs and
project-relative artifact paths so a project can be moved or backed up as a
folder. Environment-specific absolute paths belong in application settings.

### Authoritative records

| Record | Contents |
| --- | --- |
| Application settings | Data root, Python interpreter, model cache, GPU defaults, local service configuration, GPT model ID and assistant output-token limit; no API keys |
| Project manifest | Schema version, project ID/name, timestamps, archive flag, defaults, assets, ordered steps, selected takes, review policy |
| Image asset (in the project manifest) | Stable asset ID, display name, project-relative image/thumbnail paths, dimensions, content hash, upload timestamp, archive flag |
| Generation step | Stable ID, name, prompt revision, explicit mode, ordered image-slot bindings to pool asset IDs or continuity frames, enabled flags, parameter overrides, revision |
| Prompt revision | Saved UTF-8 text, revision ID, step ID, timestamp, manual/AI-assisted origin; accepted AI edits record the model and source revision, never credentials |
| Combination step | Stable ID, name, ordered inputs, output profile, revision |
| Job record | Job ID/type, project/step/revision IDs, queue order, frozen request, upstream job/take IDs, resolved inputs, state, timestamps, process identity, outputs, error summary |
| Activity event | Event ID, UTC timestamp, event type, relevant step/job/take IDs, readable message, structured details |

Store editable prompts as UTF-8 files. At queue time, copy the prompt text and
effective settings into the job record so later edits cannot change a pending
or historical request. Ingested assets should be immutable; replacing an image
creates a new asset ID. Record asset identities/hashes for reproducibility.

Accepted assistant edits become normal saved prompt revisions. Keep unaccepted
suggestions and conversation context session-only initially. API keys stay in
backend memory (session-entered or read from the backend environment), outside
all JSON files, project folders, exports, logs, browser persistent storage, and
worker environments.

Keep pool membership separate from per-step selection. Multiple steps can reuse
the same asset without copying it. Snapshot only the enabled, selected image IDs
and their order in each job, not the entire pool. Adding or renaming pool images
must not alter existing selections or mark unrelated results as outdated.

For dependencies that are not yet generated, freeze the upstream job identity
when building the queue. Resolve and record its exact output/frame artifact
before the dependent job starts. Do not resolve against a mutable "latest take"
while a workflow is running.

Project manifests reference jobs and takes; they do not duplicate authoritative
job status. The project activity log is an audit trail, not the recovery database.

### Example records

Pydantic models in `ui/core/storage.py` are the source of truth; JSON Schemas are exported
from them for documentation and tests. Abbreviated examples:

```json
{
  "schema_version": 1,
  "id": "prj_kungfu_cat",
  "name": "Kung Fu Cat",
  "defaults": {
    "width": 1056, "height": 576, "frames": 345, "steps": 40,
    "seed": "random", "strategy": "context_parallel", "reference_short_edge": 1536
  },
  "assets": [
    {"id": "img_01", "name": "male_3.jpg", "path": "assets/img_01.jpg",
     "thumb": "assets/img_01_thumb.jpg", "width": 768, "height": 1024, "sha256": "..."}
  ],
  "steps": [
    {"id": "stp_01", "type": "generate", "name": "Scene 1", "revision": 3,
     "mode": "ref2va", "prompt_revision": "prm_01_r2",
     "images": [{"source": "asset", "asset_id": "img_01", "enabled": true}],
     "overrides": {"steps": 35}, "selected_take": "tk_01"},
    {"id": "stp_02", "type": "generate", "name": "Scene 2", "revision": 1,
     "mode": "ref2va", "prompt_revision": "prm_02_r1",
     "images": [{"source": "last_frame", "step_id": "stp_01", "enabled": true},
                {"source": "asset", "asset_id": "img_01", "enabled": true}]},
    {"id": "stp_03", "type": "combine", "name": "Part 1", "revision": 1,
     "inputs": ["stp_01", "stp_02"]}
  ]
}
```

```json
{
  "schema_version": 1,
  "id": "job_0007",
  "type": "generate",
  "project_id": "prj_kungfu_cat",
  "step_id": "stp_02",
  "step_revision": 1,
  "state": "queued",
  "wait_reason": "dependency",
  "depends_on": ["job_0006"],
  "request": {
    "mode": "ref2va", "prompt_text": "...",
    "images": [{"from_job": "job_0006", "frame": "last"}, {"asset_id": "img_01"}],
    "width": 1056, "height": 576, "frames": 345, "steps": 40, "seed": 1834201,
    "strategy": "context_parallel", "cp_world_size": 4, "reference_short_edge": 1536
  },
  "process": null,
  "outputs": null,
  "error": null
}
```

```json
{"id": "evt_0142", "ts_utc": "2026-10-22T10:15:03Z", "type": "job.succeeded",
 "step_id": "stp_02", "job_id": "job_0007", "take_id": "tk_04",
 "message": "Scene 2 generated (14.4 s clip in 52.3 min)", "details": {"runtime_s": 3138}}
```

### Job states

| State | Meaning | Allowed transitions |
| --- | --- | --- |
| `queued` | Waiting; `wait_reason` is `dependency`, `queue_paused`, or empty | `running`, `blocked`, `cancelled` |
| `blocked` | An upstream job failed or was cancelled | `queued` (after a successful upstream retry), `cancelled` |
| `running` | Generator process launched | `succeeded`, `failed`, `cancelled`, `interrupted` |
| `succeeded` | Exit code 0 and outputs verified | Terminal |
| `failed` | Non-zero exit or output verification failed | Terminal; retry creates a new job |
| `cancelled` | Cancelled by the user; process termination confirmed | Terminal |
| `interrupted` | Found `running` at startup with no live process | Terminal; retry creates a new job |

`cancel_requested` is a transient flag on a running job until termination is
confirmed. A later release adds a `review` wait reason for approval gates.

### Save and recovery rules

- Enforce one managing application instance per data root, plus serialized
  metadata writes inside that instance.
- Write JSON to a temporary sibling file, flush it, and atomically replace the
  destination. Keep a last-known-good backup.
- Include a schema version and validate records before use. Handle unsupported
  versions explicitly rather than silently rewriting them.
- Treat an individual file replacement as atomic, not a multi-file operation.
  Make job creation and manifest updates recoverable and idempotent by ID.
- On startup, reconcile job records, manifest references, process state, and
  artifacts before releasing the queue. Do not infer success from an MP4 alone.
- Never silently rerun an interrupted generation. Present its state and allow an
  explicit retry after confirming that no previous worker is still active.

## 6. Project Activity and Execution Logs

Use `project.log.jsonl` as an append-only sequence of JSON events, one per line.
Record project creation/rename/archive, saved prompt/settings changes, workflow
edits, image uploads/archive actions, per-step image-selection changes, job
queuing/start/completion/failure/cancellation, selected takes, and exports. Do not
emit an activity entry for every editor keystroke.

Record assistant request outcomes and accepted edits with model ID, relevant
step/revision IDs, and token usage when available. Do not log API keys,
authorization headers, full conversations, request/response bodies, or raw
provider errors that might include credentials or submitted content.

The UI presents these events as a readable, filterable project history. Selecting
a job opens its detailed stdout/stderr log. Preserve non-ASCII prompt text through
UTF-8 handling, but avoid duplicating full prompts or sensitive environment
values in activity messages.

Metadata is committed before its corresponding audit event. Recovery may record
a reconciliation event if a crash occurred between those writes; do not claim
transactional consistency across separate files. Detect and recover an incomplete
trailing JSONL record without discarding valid earlier events.

## 7. Project and Clip Editing

### Project lifecycle

Create, open, rename, and archive projects. Define project-level generation
defaults and a target export resolution. Allow explicit import of existing
prompts, images, and completed videos without changing their source files.

### Project image pool

- Allow single or batch uploads and repeated additions, with no fixed limit on
  the total number of reference images. Per-file validation and available disk
  space still apply.
- Display a searchable thumbnail gallery with names, dimensions, and full-image
  previews. Load thumbnails incrementally so large pools remain usable.
- Keep uploaded images unselected until the user assigns them to a generation
  step. Different steps can select different images from the same pool.
- The first version offers no deletion, so referenced files are never removed.
- Later release: image renaming and archiving. Retain files referenced by steps
  or job history; archiving hides an image from new selection without breaking
  existing bindings. Only allow permanent deletion of unreferenced assets.

### Prompt and image controls

- Create, upload, edit, save, and duplicate UTF-8 prompts.
- Provide two ordered image slots with thumbnails and enable/disable controls.
- Explicitly bind each slot to an image chosen from the project pool or an earlier
  clip's last frame. Allow replacing, clearing, and reordering those selections.
- Send only enabled selections to the generator; unselected pool images are
  never included. Clearing a slot does not delete the image from the pool.
- Show the actual enabled reference ordering as Picture 1 and Picture 2.
- Keep mode selection explicit: text-only, first-frame, or reference-image.
- In first-frame mode, require one enabled image and disallow reference inputs.
- In reference mode, require one or two enabled images; do not silently fall
  back to text-only when both are disabled.
- A continuity frame consumes one of the two slots. First-frame mode cannot
  also use a separate character/style reference.

### AI prompt assistant

- Provide a panel beside the prompt editor for drafting a prompt from a scene
  brief and editing selected text or the whole prompt through free-form requests
  and follow-up instructions.
- Support refinement of action, camera, lighting, dialogue/audio, and continuity;
  shortening, expansion, alternatives, and translation. Preserve user-specified
  names, dialogue, language, and Picture 1/Picture 2 roles unless asked to change
  them. Use the generation mode and effective clip duration as constraints.
- Preview the outgoing context: the user's instruction, current prompt or selected
  text, and relevant generation settings. Include earlier prompts or continuity
  notes only when explicitly selected. Do not automatically send the project
  history, image pool, media, or local paths. Initial requests are text-only; use
  user-provided image-role descriptions without claiming to have inspected images.
- Ground the assistant's instructions in MiniMax-H3 prompt conventions drawn from
  [docs/prompts.txt](prompts.txt) and existing project prompts (Picture N roles,
  dialogue and sound descriptions, clip duration). Keep the instructions versioned
  in the repository and record their version on accepted revisions.
- First version: whole-prompt suggestions with an original-versus-proposed
  preview, replace-prompt, discard, and undo. Later release: streaming,
  apply-to-selection edits, and follow-up conversation. Applying creates a saved
  prompt revision and uses the existing downstream-outdated rules; it never starts
  generation or changes already queued requests.
- Bind each suggestion to its source prompt revision. If the prompt changed while
  the request was running, show a fresh comparison and require confirmation
  before replacing, rather than overwriting newer work.
- Provide cancellation, bounded timeouts, and clear invalid-key, unavailable-model,
  quota/rate-limit, and network error states without losing the current prompt.
  Disable automatic SDK retries (`max_retries=0`) so potentially billable
  requests are never silently repeated; the user retries explicitly.
- Show returned token usage and enforce a configurable output-token limit. Explain
  that OpenAI API billing is separate from ChatGPT subscriptions and that cancelling
  a request may not prevent charges already incurred. No automatic requests while
  typing; only show monetary estimates when applicable pricing is known.

### OpenAI configuration and credentials

Use OpenAI's official API endpoint for the first version, not an arbitrary
OpenAI-compatible URL. Let users select a supported GPT model available to their
API project, with a validated model-ID override. Choose and document the tested
default at implementation time rather than hard-coding an assumed current model
here. Handle model-specific parameter support explicitly.

Provide a masked API-key field in application settings, an explicit connection
test, and a clear-key action. Send the key only to the authenticated local backend,
then clear the input; return configured/not-configured status rather than the key.
Keep it in backend session memory, clear it on session expiry, and require re-entry
after a backend restart. Never save it in plaintext JSON or browser local storage.
Any later persistent credential option must use an OS credential store.

Alternatively, if `OPENAI_API_KEY` is set in the backend's environment at startup,
use it to avoid re-entering the key each session. Settings show "key from
environment" and never reveal it; a key entered in the UI overrides it for the
session. The app never writes either key to disk, and both are removed from the
generation worker's environment.

Connection tests must not send project content. Restrict outbound credentials to
the official OpenAI endpoint; do not honor arbitrary base-URL overrides or forward
authorization across redirects. State what text leaves the workstation and that
OpenAI's applicable data-handling policies apply. Request `store=false` where
supported, without describing that option as a guarantee of zero provider retention.

Treat responses as untrusted text, render them safely, and do not execute suggested
commands or expose filesystem/generation tools to the model. The API key must never
appear in model context. Manual editing and other non-assistant features remain
available when no key is configured or OpenAI is unavailable.

### Generation parameters

Expose width, height, frame count/duration, steps, seed, strategy, and reference
short-edge size. Place GPU count and environment choices in advanced settings.
Project defaults can be overridden per step and are frozen for each queued job.

The seed accepts a number or "random". A random seed is resolved to a concrete
value when the job is queued and recorded with the take; a "reuse seed from this
take" action copies it back into the step.

Validate frame counts against `17*n+5` within 124-345 frames, show the effective
duration at 24 fps, and require dimensions that are multiples of 32. Validate
positive steps, reference resizing bounds, seed compatibility, and strategy/mode
constraints on the server as well as in the UI.

Use hardware-aware strategy guidance rather than blindly copying the CLI's
`max_gpu` default. Offer context parallelism for compatible WSL/Linux GPUs and
offloading where appropriate. Reference resizing is independent of output
resolution; include it in preflight memory guidance.

## 8. Workflow and Take Management

Use an ordered step list with dependency references, not a general graph editor.
Generation steps produce clips; combination steps assemble clips or earlier
assemblies in a specified order. Dependencies must point to earlier steps, and
reordering must preserve a valid execution order.

```text
Generate Scene 1
Generate Scene 2 using Scene 1's last frame + a character reference
Combine Scene 1 + Scene 2 into Part 1
Generate Scene 3
Combine Part 1 + Scene 3 into Final Video
```

Support running one step or the remaining workflow, duplicating steps, and
reordering compatible steps. The first version runs unattended with
stop-after-current. A later release adds an optional review pause after each
generation, implemented as a dependency gate rather than a worker holding GPU
memory while waiting for input.

Every regeneration creates a new take. Users preview and explicitly select takes
for later generations or assembly. Changing a prompt, settings, an input binding,
or a selected upstream take marks affected downstream results as outdated;
completed artifacts remain available and are never silently overwritten.

Queued jobs retain their frozen inputs. Applying edits to pending work requires
explicit cancellation/requeueing. Retrying a failed generation creates a new job
record, preserving the failed attempt's diagnostics.

Last-frame conditioning can improve visual continuity but does not guarantee
seamless motion, identity, or audio continuity between clips.

## 9. Job Execution and Safety

### Lifecycle

Use persisted states such as queued, running, succeeded, failed, cancelled, and
interrupted. Queued jobs can also have a dependency/review wait reason. Expose
transient cancellation requests separately until process termination is confirmed.

- Preflight the Python environment, `HF_HOME` and model snapshot, reference
  partition, input files, GPU count and compatibility, GPUs already busy with other
  compute processes (via `nvidia-smi`), rendezvous-port availability, output
  destination, and free storage before launching expensive work.
- Generation jobs always pass `--no-auto-download`. If the reference partition is
  missing, offer a separate, explicitly approved download job using
  [download_transformer_ref.py](../download_transformer_ref.py), showing its size
  and the available disk space.
- Use explicit argument lists, a saved UTF-8 prompt file, and per-job output paths.
  Start the generator in a new session/process group with `HF_HOME`,
  `MASTER_PORT`, and `CP_WORLD_SIZE` set explicitly, and remove API keys and other
  unrelated secrets from its environment.
- Read structured progress from `--progress-jsonl`, capture unbuffered output, and
  show elapsed time, stage, and denoising step. Do not invent a total percentage
  or completion time; instead show past runtimes for similar settings (resolution,
  frames, steps, strategy), labelled as history.
- Stop-after-current pauses scheduling. Cancel-active terminates the entire
  process group, including context-parallel child workers, with bounded escalation.
- Confirm worker termination and release of managed resources before starting
  another GPU job. Do not terminate unrelated processes.
- Verify output artifacts and completion status before releasing dependent jobs.
  Preserve incomplete artifacts for diagnosis without treating them as valid takes.
- On failure, leave dependent jobs blocked and offer retry or explicit workflow
  changes. Retain already successful clips.
- Show per-project disk usage. Each long clip with its frames is large; cleanup
  tools for unused takes arrive in a later release.

Validate uploaded content and size, sanitize displayed filenames, prevent path
traversal, and serve only managed media through the authenticated media routes.
Bind to localhost by default. Require the launch-token session for every page,
websocket connection, and media route, and validate the `Host` header; localhost
binding alone is not an authorization mechanism. Render untrusted text (prompts,
AI suggestions, logs, filenames) through escaping components, never as raw HTML.

The scheduler launches a fresh generation process per job. Model loading may
remain a substantial part of runtime; this plan does not assume cross-job model
caching. "Resume workflow" means continue unfinished steps, not resume denoising.

## 10. Video Assembly

Extend the existing join utility to accept an ordered list while preserving its
current two-input CLI. Encode an assembly in one pass rather than repeatedly
joining pairs. Inputs can be clips or earlier assemblies.

First version: inspect every input before writing the output. Require matching
resolution and frame rate and an audio track in each input, and reject mismatches
with a clear explanation. Normalize audio sample rate/layout, as the existing
utility already does, and align each input's audio to its video duration by
trimming or padding, to prevent cumulative drift.

Later release: dimension normalization with aspect-ratio-preserving letterboxing
(never stretching), explicit frame-rate conversion that does not change playback
speed, silence for audio-less imported clips, and resolving nested assemblies to
their pinned original clips to avoid cumulative recompression.

Store each export's exact ordered input takes and output profile. Write to a
temporary output and publish only after successful completion and validation.
Playback/download must use the completed artifact, not a partially written file.

## 11. Interface Layout

- **Left:** project navigation and the ordered workflow with status indicators.
- **Center:** the selected step's prompt/settings with an optional GPT assistant
  panel and suggestion comparison, or ordered combination inputs.
- **Right:** tabs for video preview/takes and the project image pool, with a
  searchable gallery for assigning images to the selected step's input slots.
  Keep that step's selected references and extracted frames easy to inspect.
- **Bottom:** persistent job queue, project activity, and expandable execution logs.

NiceGUI building blocks: a left drawer for projects and the workflow list, a
splitter for the center and right panels, `ui.tabs` for preview/takes and the
image pool, `ui.textarea` or `ui.codemirror` for prompts, `ui.upload` (multiple)
for images, a paginated `ui.grid` of `ui.image` thumbnails, `ui.video` for
playback, `ui.log` for execution logs, and a footer for the job queue. The first
version reorders steps with move-up/down buttons; drag-and-drop (SortableJS through
a small custom element) can follow. The assistant comparison shows original and
proposed text side by side with escaped content.

Primary actions are save, generate selected, run remaining, stop after current,
cancel active, and export. Use a compact work-focused layout, familiar icons with
accessible labels/tooltips, clear pending/error states, and responsive panels.
The first screen should be the usable project workspace, not a marketing page.

## 12. Implementation Phases

| Phase | Deliverables | Acceptance criteria |
| --- | --- | --- |
| 1. Core vertical slice | Script changes (`--no-auto-download`, `--progress-jsonl`, `--run-id`), JSON storage and schemas, session token, project creation, image pool (add/select), prompt editor, per-step image selection, parameter validation, preflight, durable single-job queue with logs, cancellation, restart reconciliation, video preview, pytest and fake generator | Upload more than two images, select a subset, and generate a clip; closing the browser mid-run does not stop it; cancelling a `context_parallel` run stops every rank; a backend restart never duplicates work; the project reopens with pool, selections, and result intact |
| 2. Workflow | Ordered steps, run remaining, stop-after-current, frozen dependencies, last-frame chaining, blocked/retry handling, take history, outdated tracking, project activity view | Generate two linked clips; retry a failed second step without rerunning the first |
| 3. Assembly | N-input join, compatibility checks, audio alignment, export history | Export an ordered sequence with correct order, duration, and synchronized audio; incompatible inputs are rejected before encoding |
| 4. Prompt assistant | OpenAI settings, session or environment key, whole-prompt draft/edit, comparison, apply/undo, usage and error handling | Draft and revise a prompt using a supplied key; only approved edits persist; credentials never enter project files, logs, or the worker environment |
| 5. Packaging and polish | `ui/run_ui.sh`, README section, disk-usage view, runtime history, drag-and-drop step reordering | Start from documented commands and repeat the complete workflow after a restart |
| Later releases | Review gates, pool rename/archive, assembly normalization and nested resolution, assistant streaming/selection edits, take cleanup | Prioritized after phase 5 |

Preserve the existing CLI and launcher workflows throughout implementation.
Organize code as described in the code layout (section 4): framework-independent
`ui/core` modules and a thin NiceGUI `ui/app` layer. No database, external queue
service, or Node.js toolchain is required for this scope.

## 13. Verification Plan

- **Test infrastructure:** phase 1 adds pytest and a fake generator that mimics
  the [LoMMH.py](../LoMMH.py) CLI: progress JSONL, output files, exit codes,
  delays, and child processes. Core modules are tested without NiceGUI. UI tests
  use NiceGUI's pytest plugin: the fast simulated `user` fixture for most flows
  and the browser-based `screen` fixture for a few smoke tests.
- **Storage:** validate schemas, atomic replacement, backups, duplicate-instance
  prevention, corrupt/incomplete records, and multi-file crash recovery.
- **Image pool:** verify uploads beyond two images, batch uploads, reuse across
  steps, persistent selections, archiving without broken bindings, and rejection
  of permanent deletion for referenced assets. Adding an unselected image must
  not modify queued jobs or invalidate completed results.
- **Requests:** verify mode constraints, per-generation two-slot limits, enabled
  image ordering, exclusion of unselected pool images, effective parameter
  snapshots, supported frame counts, Unicode prompts, and paths with spaces.
- **Prompt assistant:** mock OpenAI responses to verify drafting, selected-text
  editing, streaming/cancellation, apply/discard/undo, revision conflicts, Unicode,
  token limits, model compatibility, disabled retries, environment-key handling,
  and authentication/quota/network failures.
  Verify no automatic generation or unapproved context/media transmission, no
  secrets in files/logs/error responses/worker environments, session-key clearing,
  endpoint/redirect protection, and offline manual editing. Live API tests require
  explicit consent to use a supplied key and incur charges; routine tests use mocks.
- **Scheduling:** use a mock generator to test durable queue order, review gates,
  cancellation of child processes, retries, interrupted jobs, and no duplicate runs.
- **Generator integration:** verify `--no-auto-download` fails fast when the
  partition is missing, progress JSONL parsing, a non-zero exit code when one
  `context_parallel` rank fails, rendezvous-port conflict detection, and that the
  existing launchers behave unchanged.
- **Dependencies:** verify exact take binding, deferred frame resolution, invalid
  reordering, queued-edit behavior, and downstream outdated indicators.
- **Media:** use tiny synthetic clips to check order, duration, frame rate, aspect
  ratio, silent inputs, differing audio layouts/rates, synchronization, and failures
  that must not publish a completed export.
- **UI:** NiceGUI `user` tests for project reopening, uploads, prompt edits,
  generation, queue status, take selection, activity history, assembly, and
  downloads; multiple tabs receiving the same job updates; and rejection of pages,
  websocket connections, and media routes without the launch-token session.
  Inspect desktop and narrow layouts for overflow and usable controls.
- **Reattach:** restart the app server during a fake-generator run and verify it
  reattaches to the live runner, or marks the job `interrupted` when it is gone.
- **Integration:** perform a short real GPU generation, then a two-clip chained
  workflow and final export in WSL2. Keep routine tests independent of model files
  and GPU availability.

## 14. Decisions for Review

Status: **Accepted** means agreed during review; **Proposed** still needs
confirmation.

| Topic | Default | Status |
| --- | --- | --- |
| UI stack | NiceGUI (Python-only, on FastAPI/Uvicorn) over framework-independent core modules. Gradio rejected: its session-bound, function-oriented model fits demos, not multi-project workflows. React + TypeScript remains a later option on top of the same core | Accepted |
| Persistence | JSON metadata, JSONL project activity, separate execution logs; no SQLite | Accepted |
| Project activity log | Per-project `project.log.jsonl` | Accepted |
| Runtime | NiceGUI app server and generation runs in WSL2, browser on Windows, localhost only, launch-token session | Accepted |
| Project location | Configurable data root, default `~/minimax-ui-data` on WSL-native storage | Accepted |
| Code location | New `ui/` folder; existing root scripts unchanged except backward-compatible flags | Accepted |
| Script changes | `--no-auto-download`, `--progress-jsonl`, `--run-id`, extra run-log fields (done); N-input join (phase 3) | Accepted |
| Project image pool | Any number of uploaded images, subject to file validation and available storage; no automatic selection | Accepted |
| Per-generation image limit | Up to two explicitly selected inputs from the pool or continuity frames (configurable); first-frame mode accepts one | Accepted |
| AI prompt assistant | Optional text drafting/editing with review and explicit apply; whole-prompt in the first version | Accepted |
| AI provider | OpenAI GPT models via the official Python SDK and Responses API; model configurable, tested default chosen at implementation | Accepted |
| API-key source | UI-entered key in backend session memory, or `OPENAI_API_KEY` from the backend environment; never written to disk | Accepted |
| AI context sharing | Only explicitly included text/settings; no automatic project/media upload | Accepted |
| Workflow editor | Ordered generation/combination steps, dependencies only on earlier steps | Accepted |
| Automation | Unattended with stop-after-current; review pauses in a later release | Accepted |
| Initial assembly | Hard cuts of compatible clips with audio; normalization in a later release | Accepted |
| Phase order | Core slice with durable queue, then workflow, assembly, assistant, packaging | Accepted |
| Imported media | Copy into managed project storage; leave existing folders untouched; existing project folders are not imported in phase 1 | Accepted |
| Generation defaults | 1056x576, 345 frames, 40 steps, random seed, `context_parallel` on 4 GPUs, reference short edge 1536 | Accepted |
| UI venv | Separate `~/minimax-ui-venv` (WSL Python 3.12); GPU venv untouched | Accepted |

JSON persistence is appropriate for the initial single-user, single-writer scope.
Reconsider a database only if later requirements introduce concurrent users,
multiple writers, or transactional updates that cannot be handled clearly with
these file-based records.