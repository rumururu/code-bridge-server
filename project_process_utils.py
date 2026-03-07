"""Utilities for subprocess lifecycle handling."""

from __future__ import annotations

import subprocess


def is_process_running(process: subprocess.Popen) -> bool:
    """Return whether a process is still running."""
    return process.poll() is None


def terminate_process_safely(
    process: subprocess.Popen,
    *,
    terminate_timeout: float = 5.0,
    kill_timeout: float = 2.0,
) -> None:
    """Attempt graceful terminate first, then force kill if needed."""
    try:
        process.terminate()
        process.wait(timeout=terminate_timeout)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=kill_timeout)
        except Exception:
            pass
