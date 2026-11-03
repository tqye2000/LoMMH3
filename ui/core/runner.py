"""Runner wrapper: launches one generator command and records its exit.

Started by the scheduler in a new session, so this process leads the job's
process group and the generator (plus any context-parallel ranks) inherit it.
It survives app-server restarts and writes ``exit.json`` when the command ends.

Usage: python -m ui.core.runner --job-dir DIR -- COMMAND...
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command")

    terminated = {"signal": None}

    def _on_signal(signum: int, _frame: object) -> None:
        # The whole group receives cancellation signals; keep waiting so the exit is recorded.
        terminated["signal"] = signum

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    started = time.time()
    print(f"[runner] {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {command[0]} ...", flush=True)
    proc = subprocess.Popen(command)
    _write_json(args.job_dir / "runner.json", {
        "runner_pid": os.getpid(), "pgid": os.getpgid(0), "child_pid": proc.pid, "started": started,
    })
    returncode = proc.wait()
    finished = time.time()
    _write_json(args.job_dir / "exit.json", {
        "returncode": returncode,
        "signal": terminated["signal"],
        "started": started,
        "finished": finished,
        "runtime_s": round(finished - started, 1),
    })
    print(f"[runner] exit code {returncode} after {finished - started:.0f}s", flush=True)
    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
