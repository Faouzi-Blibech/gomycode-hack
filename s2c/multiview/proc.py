"""Run an outside program safely: argument list, output to a log file, and a timeout that kills the whole
process tree (a Windows grandchild can otherwise keep a pipe open forever). Spec 2026-09-23-studio section 10."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run(cmd: list[str], timeout_s: float, log_path: Path, cwd: Path | None = None) -> int | None:
    """The exit code, or None when the program ran past `timeout_s` and was killed."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8", errors="replace") as out:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=cwd, creationflags=flags,
                                start_new_session=os.name != "nt")
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            return None


def tail(log_path: Path, n: int = 20) -> str:
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])
