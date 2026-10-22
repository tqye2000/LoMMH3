"""Authenticated media routes (FileResponse supports HTTP range requests for seeking)."""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import FileResponse
from nicegui import app

from ui.core.service import ServiceError
from ui.core.storage import StorageError

from . import state


def install() -> None:
    @app.get("/media/{project_id}/assets/{asset_id}/{kind}")
    def asset_media(project_id: str, asset_id: str, kind: str) -> FileResponse:
        try:
            path = state.service.asset_file(project_id, asset_id, kind)
        except (ServiceError, StorageError) as exc:
            raise HTTPException(status_code=404) from exc
        return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/media/{project_id}/takes/{step_id}/{take_id}/{kind}")
    def take_media(project_id: str, step_id: str, take_id: str, kind: str) -> FileResponse:
        try:
            path = state.service.take_file(project_id, step_id, take_id, kind)
        except (ServiceError, StorageError) as exc:
            raise HTTPException(status_code=404) from exc
        return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


def asset_url(project_id: str, asset_id: str, kind: str = "thumb") -> str:
    return f"/media/{project_id}/assets/{asset_id}/{kind}"


def take_url(project_id: str, step_id: str, take_id: str, kind: str) -> str:
    return f"/media/{project_id}/takes/{step_id}/{take_id}/{kind}"
