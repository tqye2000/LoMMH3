"""Linux process helpers: identity checks and process-group signalling."""

from __future__ import annotations

import os
import signal
import socket
from pathlib import Path


def start_ticks(pid: int) -> int | None:
    """Process start time (clock ticks since boot) from /proc, or None if gone/zombie."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # Fields after the parenthesised command name; state is first, starttime is 20th.
    fields = raw[raw.rfind(")") + 2:].split()
    if fields[0] == "Z":
        return None
    return int(fields[19])


def is_same_process(pid: int, ticks: int) -> bool:
    return start_ticks(pid) == ticks


def group_alive(pgid: int) -> bool:
    """Whether any non-zombie process still belongs to process group ``pgid``."""
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            raw = Path(entry.path, "stat").read_text()
        except OSError:
            continue
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] != "Z" and int(fields[2]) == pgid:
            return True
    return False


def signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
