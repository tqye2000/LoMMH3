"""Launch-token login, session check, and Host-header validation."""

from __future__ import annotations

import secrets

from fastapi import Request
from fastapi.responses import PlainTextResponse, RedirectResponse
from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware

from . import state

PUBLIC_PATHS = {"/login"}


def is_authenticated() -> bool:
    return bool(state.session_nonce) and app.storage.user.get("session") == state.session_nonce


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if state.allowed_hosts and host not in state.allowed_hosts:
            return PlainTextResponse("Forbidden host", status_code=403)
        path = request.url.path
        # NiceGUI assets and its websocket; clients are bound to authenticated page loads.
        if path.startswith("/_nicegui") or path in PUBLIC_PATHS:
            return await call_next(request)
        if not is_authenticated():
            if path.startswith("/media/"):
                return PlainTextResponse("Unauthorized", status_code=401)
            return RedirectResponse("/login")
        return await call_next(request)


def install() -> None:
    app.add_middleware(AuthMiddleware)

    @ui.page("/login")
    def login(token: str = ""):
        if token and secrets.compare_digest(token, state.launch_token):
            app.storage.user["session"] = state.session_nonce
            return RedirectResponse("/")
        with ui.column().classes("absolute-center items-center gap-2"):
            ui.label("MiniMax Studio").classes("text-h5")
            ui.label("Open the login link printed in the terminal where run_ui.sh is running.")
            if token:
                ui.label("That link is not valid for this session.").classes("text-negative")
        return None
