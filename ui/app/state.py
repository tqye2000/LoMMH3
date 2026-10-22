"""Process-wide app state, set once by ui/main.py at startup."""

from __future__ import annotations

from ui.core.service import Service

service: Service
launch_token: str = ""
session_nonce: str = ""
allowed_hosts: list[str] = []
