from __future__ import annotations

import pytest
from nicegui import ui
from nicegui.testing import User

from .conftest import UI_TOKEN

# Applied before the user fixture, so main.py sees the test environment.
pytestmark = pytest.mark.usefixtures("ui_env")


async def test_pages_require_login(user: User) -> None:
    await user.open("/")
    await user.should_see("Open the login link")


async def test_invalid_token_is_rejected(user: User) -> None:
    await user.open("/login?token=wrong")
    await user.should_see("not valid for this session")


async def test_login_create_project_and_add_step(user: User) -> None:
    await user.open(f"/login?token={UI_TOKEN}")
    await user.should_see("Projects")
    user.find("New project name").type("Kung Fu Cat")
    user.find("Create").click()
    await user.should_see("Kung Fu Cat")
    await user.should_see("Add a step to start.")
    user.find("Add step").click()
    await user.should_see("Scene 1")
    await user.should_see("Generate")
    user.find("Exports").click()
    await user.should_see("no selected take")
    await user.should_see("No exports yet.")
    user.find("LoRAs").click()
    await user.should_see("0 LoRA(s) in")
    await user.should_see("None (the LoRA library is empty)")


async def test_step_can_pick_a_lora_from_the_library(user: User, ui_env) -> None:
    (ui_env / "loras").mkdir(parents=True)
    (ui_env / "loras" / "ink_wash.safetensors").write_bytes(b"x")
    await user.open(f"/login?token={UI_TOKEN}")
    user.find("New project name").type("Styled")
    user.find("Create").click()
    await user.should_see("Add a step to start.")
    user.find("Add step").click()
    await user.should_see("Scene 1")
    user.find(kind=ui.button, content="Add LoRA").click()
    await user.should_see(kind=ui.select, content="ink_wash.safetensors")
    user.find("Apply parameters").click()
    await user.should_see("Overrides: loras")


async def test_media_requires_session(user: User) -> None:
    response = await user.http_client.get("/media/prj_0123456789/assets/img_0123456789/thumb")
    assert response.status_code == 401


async def test_foreign_host_is_rejected(user: User) -> None:
    response = await user.http_client.get("/login", headers={"host": "evil.example"})
    assert response.status_code == 403
