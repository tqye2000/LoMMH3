# Plan: Reference image(s) support for MiniMax-H3 (`ref2va`)

Status legend: [ ] todo · [~] in progress · [x] done

## Background / findings (diffusers 0.41.0.dev0 + local snapshot)

- MiniMax-H3 already exposes a dedicated **`ref2va`** ("omni-reference") workflow
  alongside `t2va` and `fl2va`. It is auto-selected whenever a `references=[...]`
  list is passed to the pipeline (`select_block` keys off `references`).
- Reference inputs are typed dataclasses in
  `diffusers.modular_pipelines.minimax_h3`:
  - `MiniMaxH3ImageReference` — subject/style/scene image, **<= 9 per request**
  - `MiniMaxH3VideoReference` — motion + soundtrack, <= 3
  - `MiniMaxH3AudioReference` — voice/music, <= 3, never alone
  - Total <= 12 references. Each has a `.from_file(path_or_url)` loader
    (image uses `load_image`).
- `ref2va` runs against a **different transformer partition**: the
  `transformer_ref/` subfolder (a separate ~61.7 GB checkpoint), not
  `transformer/`.
- BLOCKER for offline use: the local snapshot `modular_model_index.json` declares
  `transformer_ref` (subfolder `transformer_ref`), but the folder is **not present
  on disk** — only `transformer/` is downloaded. `ref2va` cannot run until
  `transformer_ref/` is fetched. `_localize_model_index()` already rewrites its
  path to `<snapshot>/transformer_ref`, so once the folder exists it loads offline
  with no further index work.
- `--image` / `--last_image` (fl2va keyframes) and `references` (ref2va) are
  **mutually exclusive** modes — different branch, different transformer.

## Phase 0 — Prerequisite (weights)

- [x] Shard-aware guard that checks `<snapshot>/transformer_ref/`. If missing or
      incomplete, `_ensure_reference_partition()` launches the standalone,
      resumable `download_transformer_ref.py` process. The focused downloader
      temporarily enables Hub access, writes into `D:\hf_models`, verifies all
      14 weight shards, and then the generation process continues offline.

## Phase 1 — Reference images on single-pipeline strategies (`auto_offload`, `bf16_single`)

Low-risk path: these strategies branch automatically via the pipeline's
`select_block`.

- [x] `Config`: add `reference_images: list[str]` (default empty). Keep `image`.
- [x] CLI (`parse_args`): add `--reference-image` (repeatable, `action="append"`).
      Validate (in `_validate_config`): <= 9 images; not combined with `--image`;
      files exist; `auto_offload`/`bf16_single` only in Phase 1.
- [x] int8 loader: add `_load_int8_transformer_ref(...)` (same as
      `_load_int8_transformer` but `subfolder="transformer_ref"`); inject as the
      `transformer_ref` component.
- [x] `build_pipeline` / `_build_auto_offload` / `_build_bf16_single`: when
      references are present, `load_components(workflow="ref2va")` (loads
      `transformer_ref`, skips `transformer`) and apply offload / group-offload
      wiring to `transformer_ref` instead of `transformer` (via `_workflow_for`).
- [x] `run_generation`: build
      `references = [MiniMaxH3ImageReference.from_file(p) for p in
      config.reference_images]`, pass `references=references`, set
      `mode = "ref2va"`.
- [x] `save_output`: default filename `minimax_h3_ref2va.mp4` (via `mode`);
      `_print_run_header` shows reference count.

> Phase 1 code complete and validated up to model load. Full end-to-end run
> requires the `transformer_ref/` weights (~61.7 GB), not yet downloaded.

## Phase 2 — Reference images on `max_gpu` (multi-GPU split) — optional/follow-up

- [ ] The hand-rolled 4-stage split is built from `get_workflow("t2va")` and
      hard-codes `transformer`. Supporting `ref2va` means switching to
      `get_workflow("ref2va")`, loading `transformer_ref` on the transformer GPU,
      and adding the reference **setup + VAE reference-encoder** stages (they
      produce `condition_latents` / `audio_condition_latents`) before denoise.
      Until done, keep the explicit guard but point ref2va users at
      `--strategy auto_offload`.

## Phase 3 — Docs & validation

- [ ] Update `README.md` with a "Reference images (ref2va)" section: example
      command, the <= 9 constraint, and the `transformer_ref` download note.
- [ ] Add a fast smoke test
      (`--steps 2 --strategy auto_offload --reference-image images\example_1.jpg`)
      once weights are present.

## Optional extension

- [ ] The same `references` list also accepts video/audio references; the CLI
      could later add `--reference-video` / `--reference-audio` with near-zero
      extra pipeline work. Keep `--reference-image` specific rather than a generic
      `--reference` to leave room.
