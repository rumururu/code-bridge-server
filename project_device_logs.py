"""Helpers for project device-run log path and tail handling."""

from __future__ import annotations

from pathlib import Path


def sanitize_log_name(value: str) -> str:
    """Normalize an identifier segment for a safe log filename."""
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)
    safe = safe.strip("_")
    return safe or "project"


def device_run_log_path(project_name: str, device_id: str) -> Path:
    """Build canonical log path for a project/device run session."""
    safe_project = sanitize_log_name(project_name)
    safe_device = sanitize_log_name(device_id)
    return Path("/tmp") / f"code_bridge_device_run_{safe_project}_{safe_device}.log"


def read_log_tail(
    log_path: str | Path,
    max_lines: int = 120,
    max_chars: int = 16000,
) -> str:
    """Read a bounded UTF-8-safe tail from a log file."""
    path = Path(log_path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:]) if lines else text
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail
