from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.core.service import Service  # noqa: E402
from ui.core.settings import Settings  # noqa: E402

FAKE_GENERATOR = Path(__file__).with_name("fake_generator.py")
UI_TOKEN = "test-token"


def make_settings(data_root: Path, **overrides) -> Settings:
    values = dict(
        data_root=data_root,
        generator_python=Path(sys.executable),
        generator_script=FAKE_GENERATOR,
        gpu_check=False,
        model_check=False,
        poll_interval_s=0.05,
        cancel_grace_s=2.0,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path / "data")


@pytest.fixture
def service(settings: Settings):
    svc = Service(settings)
    svc.open()
    yield svc
    svc.close()


def png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


async def run_until(service: Service, predicate, timeout: float = 30.0) -> None:
    """Drive scheduler ticks until ``predicate()`` is true."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await service.scheduler.tick()
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for scheduler condition")


@pytest.fixture
def ui_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Environment for NiceGUI user-simulation tests (main.py reads it at startup)."""
    data = tmp_path / "ui-data"
    monkeypatch.setenv("MINIMAX_UI_DATA", str(data))
    monkeypatch.setenv("MINIMAX_UI_TOKEN", UI_TOKEN)
    monkeypatch.setenv("MINIMAX_UI_ALLOWED_HOSTS", "127.0.0.1,localhost,test")
    monkeypatch.setenv("MINIMAX_UI_GPU_CHECK", "0")
    monkeypatch.setenv("MINIMAX_UI_MODEL_CHECK", "0")
    monkeypatch.setenv("MINIMAX_GEN_PYTHON", sys.executable)
    monkeypatch.setenv("MINIMAX_GEN_SCRIPT", str(FAKE_GENERATOR))
    return data
