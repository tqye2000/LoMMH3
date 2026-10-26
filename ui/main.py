"""MiniMax Studio entry point. Run inside WSL via ui/run_ui.sh."""

from __future__ import annotations

import logging
import os
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.core.settings import load_settings  # noqa: E402

settings = load_settings()
# NiceGUI reads its storage path at import time; keep it inside the data root.
os.environ.setdefault("NICEGUI_STORAGE_PATH", str(settings.data_root / ".nicegui"))

from nicegui import app, ui  # noqa: E402

from ui.app import auth, media, pages, state  # noqa: E402
from ui.core.service import Service  # noqa: E402


def setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = Service(settings)
    service.open()
    state.service = service
    state.launch_token = os.environ.get("MINIMAX_UI_TOKEN") or secrets.token_urlsafe(24)
    state.session_nonce = secrets.token_urlsafe(16)
    state.allowed_hosts = settings.allowed_hosts

    auth.install()
    media.install()
    pages.install()

    async def on_startup() -> None:
        service.scheduler.start()
        print(f"\nMiniMax Studio: open http://localhost:{settings.port}/login?token={state.launch_token}\n",
              flush=True)

    async def on_shutdown() -> None:
        await service.scheduler.stop()
        service.close()

    app.on_startup(on_startup)
    app.on_shutdown(on_shutdown)
    try:
        ui.run(
            host=settings.host,
            port=settings.port,
            title="MiniMax Studio",
            storage_secret=secrets.token_urlsafe(32),
            reload=False,
            show=False,
            show_welcome_message=False,
            fastapi_docs=False,
        )
    except KeyboardInterrupt:
        pass


if __name__ in {"__main__", "__mp_main__"}:
    setup()
