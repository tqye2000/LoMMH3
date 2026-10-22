from __future__ import annotations

from pathlib import Path

import pytest

from ui.core.generation import build_command, build_env
from ui.core.images import ImageRejected
from ui.core.models import ImageBinding
from ui.core.service import Service, ServiceError

from .conftest import png_bytes


def _pool(service: Service, count: int) -> tuple[str, list[str]]:
    project = service.create_project("Pool")
    ids = []
    for index in range(count):
        asset, created = service.add_image(project.id, f"image{index}.png", png_bytes((index * 60, 10, 10)))
        assert created
        ids.append(asset.id)
    return project.id, ids


def test_pool_accepts_more_than_two_images_and_dedupes(service: Service) -> None:
    pid, ids = _pool(service, 4)
    assert len(service.get_project(pid).assets) == 4
    again, created = service.add_image(pid, "copy.png", png_bytes((0, 10, 10)))
    assert not created and again.id == ids[0]
    assert len(service.get_project(pid).assets) == 4


def test_invalid_upload_is_rejected(service: Service) -> None:
    project = service.create_project("Bad")
    with pytest.raises(ImageRejected):
        service.add_image(project.id, "notes.png", b"not an image")
    assert service.get_project(project.id).assets == []


def test_mode_rules(service: Service) -> None:
    pid, ids = _pool(service, 2)
    step = service.add_step(pid)
    assert any("prompt is empty" in e for e in service.validate_step(pid, step.id))
    service.save_prompt(pid, step.id, "A cat practices kung fu.")

    service.update_step(pid, step.id, mode="ref2va")
    assert any("one or two enabled images" in e for e in service.validate_step(pid, step.id))

    both = [ImageBinding(asset_id=ids[0]), ImageBinding(asset_id=ids[1])]
    service.update_step(pid, step.id, mode="fl2va", images=both)
    assert any("exactly one" in e for e in service.validate_step(pid, step.id))

    service.update_step(pid, step.id, mode="t2va")
    assert any("Text-only" in e for e in service.validate_step(pid, step.id))

    service.update_step(pid, step.id, mode="ref2va")
    assert service.validate_step(pid, step.id) == []


def test_request_uses_only_selected_images_in_slot_order(service: Service, tmp_path: Path) -> None:
    pid, ids = _pool(service, 3)
    step = service.add_step(pid, "Scene")
    service.save_prompt(pid, step.id, "Picture 1 greets Picture 2.")
    service.update_step(pid, step.id, mode="ref2va",
                        images=[ImageBinding(asset_id=ids[2]), ImageBinding(asset_id=ids[0])])
    job = service.run_step(pid, step.id)
    assert [img.asset_id for img in job.request.images] == [ids[2], ids[0]]

    service.update_step(pid, step.id, images=[ImageBinding(asset_id=ids[2]),
                                              ImageBinding(asset_id=ids[0], enabled=False)])
    job2 = service.run_step(pid, step.id)
    assert [img.asset_id for img in job2.request.images] == [ids[2]]

    project_dir = service.store.project_dir(pid)
    cmd = build_command(service.settings, job, project_dir, tmp_path / "job", tmp_path / "take")
    refs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--reference-image"]
    assert refs == [str(project_dir / f"assets/{ids[2]}.png"), str(project_dir / f"assets/{ids[0]}.png")]
    for flag in ("--no-auto-download", "--run-id", "--progress-jsonl", "--reference-short-edge"):
        assert flag in cmd
    assert cmd[cmd.index("--run-id") + 1] == job.id


def test_random_seed_is_frozen_and_explicit_seed_kept(service: Service) -> None:
    project = service.create_project("Seeds")
    step = service.add_step(project.id)
    service.save_prompt(project.id, step.id, "Rain on a window.")
    job = service.run_step(project.id, step.id)
    assert isinstance(job.request.params.seed, int)
    service.update_step(project.id, step.id, params={"seed": 1234})
    assert service.run_step(project.id, step.id).request.params.seed == 1234
    assert service.get_project(project.id).step(step.id).overrides == {"seed": 1234}


def test_invalid_parameters_are_reported(service: Service) -> None:
    project = service.create_project("Params")
    step = service.add_step(project.id)
    service.save_prompt(project.id, step.id, "Waves.")
    service.update_step(project.id, step.id, params={"frames": 200, "width": 1000})
    with pytest.raises(ServiceError) as info:
        service.run_step(project.id, step.id)
    text = " ".join(info.value.errors)
    assert "17*n+5" in text and "multiple of 32" in text


def test_unicode_prompt_round_trips(service: Service) -> None:
    project = service.create_project("Unicode")
    step = service.add_step(project.id)
    prompt = "一只猫在雨中练功夫。"
    service.save_prompt(project.id, step.id, prompt)
    assert service.run_step(project.id, step.id).request.prompt_text == prompt
    assert not service.save_prompt(project.id, step.id, prompt)


def test_env_strips_secrets(service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "x")
    project = service.create_project("Env")
    params = service.get_project(project.id).defaults
    env = build_env(service.settings, params, 41234)
    assert "OPENAI_API_KEY" not in env and "SOME_SERVICE_TOKEN" not in env
    assert env["MASTER_PORT"] == "41234"
    assert env["CP_WORLD_SIZE"] == str(params.cp_world_size)
    assert env["HF_HOME"] == str(service.settings.hf_home)
