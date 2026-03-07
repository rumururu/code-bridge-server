"""Helpers for dev-server process waiting and error extraction."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any, Callable


async def wait_for_project_server_port(
    project_path: str,
    project_type: Any,
    *,
    detect_port: Callable[[str, Any], int | None],
    process: subprocess.Popen | None = None,
    timeout_seconds: float = 15.0,
    poll_interval_seconds: float = 0.5,
) -> int | None:
    """Wait until a dev server for the project starts listening."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        if process is not None and process.poll() is not None:
            return None

        detected_port = detect_port(project_path, project_type)
        if detected_port is not None:
            return detected_port
        await asyncio.sleep(poll_interval_seconds)
    return None


def extract_process_error(process: subprocess.Popen) -> str | None:
    """Extract a concise stderr hint from a failed process."""
    try:
        if process.poll() is None:
            return None

        if process.stderr is None:
            return None

        raw = process.stderr.read()
        if not raw:
            return None

        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        return lines[-1][:240]
    except Exception:
        return None
