"""Helpers for detecting running project dev-server ports."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from project_runtime_helpers import (
    extract_port,
    is_candidate_command,
    match_project_path_score,
    select_preferred_port,
)

ListeningProcessMap = dict[int, dict[str, Any]]


def get_bridge_port() -> int | None:
    """Read current bridge server port from config when available."""
    try:
        from config import get_config

        return get_config().port
    except Exception:
        return None


def list_listening_processes(
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ListeningProcessMap:
    """List TCP listening processes from lsof."""
    try:
        result = run_command(
            ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpcn"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    listeners: ListeningProcessMap = {}
    current_pid: int | None = None

    for raw in result.stdout.splitlines():
        if not raw:
            continue

        field = raw[0]
        value = raw[1:]

        if field == "p":
            if not value.isdigit():
                current_pid = None
                continue
            current_pid = int(value)
            listeners.setdefault(current_pid, {"command": "", "ports": set()})
        elif field == "c" and current_pid is not None:
            listeners[current_pid]["command"] = value
        elif field == "n" and current_pid is not None:
            port = extract_port(value)
            if port is not None:
                listeners[current_pid]["ports"].add(port)

    return listeners


ProcessCwdMap = dict[int, Path]


def list_process_cwds(
    pids: list[int] | None = None,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ProcessCwdMap:
    """Get CWDs for all processes in a single lsof call.

    Much faster than calling get_process_cwd() for each PID individually.
    """
    try:
        # Get CWDs for all listening processes at once
        result = run_command(
            ["lsof", "-d", "cwd", "-Fpn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    cwd_map: ProcessCwdMap = {}
    current_pid: int | None = None

    for raw in result.stdout.splitlines():
        if not raw:
            continue

        field = raw[0]
        value = raw[1:]

        if field == "p":
            if value.isdigit():
                current_pid = int(value)
            else:
                current_pid = None
        elif field == "n" and current_pid is not None:
            # Filter to only requested PIDs if specified
            if pids is None or current_pid in pids:
                try:
                    cwd_map[current_pid] = Path(value).resolve()
                except Exception:
                    pass

    return cwd_map


def get_process_cwd(
    pid: int,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path | None:
    """Get process current working directory via lsof.

    Note: For multiple PIDs, use list_process_cwds() for better performance.
    """
    try:
        result = run_command(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    for line in result.stdout.splitlines():
        if line.startswith("n") and len(line) > 1:
            try:
                return Path(line[1:]).resolve()
            except Exception:
                return None
    return None


def detect_port_for_project(
    project_path: str,
    project_type: Any,
    *,
    bridge_port: int | None = None,
    listeners: ListeningProcessMap | None = None,
    cwd_map: ProcessCwdMap | None = None,
    get_cwd: Callable[[int], Path | None] | None = None,
) -> int | None:
    """Detect listening port for a project by matching process CWD.

    Args:
        project_path: Path to the project directory
        project_type: Type of project (flutter, nextjs, etc)
        bridge_port: Port to exclude (Code Bridge server port)
        listeners: Cached listening processes map (from list_listening_processes)
        cwd_map: Cached process CWD map (from list_process_cwds) - preferred for performance
        get_cwd: Function to get CWD for a PID (fallback, slower)
    """
    try:
        target = Path(project_path).resolve()
    except Exception:
        return None

    resolved_bridge_port = get_bridge_port() if bridge_port is None else bridge_port
    resolved_listeners = list_listening_processes() if listeners is None else listeners

    # Use cached CWD map if available, otherwise fall back to individual lookups
    if cwd_map is not None:
        def lookup_cwd(pid: int) -> Path | None:
            return cwd_map.get(pid)
        resolved_get_cwd = lookup_cwd
    else:
        resolved_get_cwd = get_cwd if get_cwd is not None else get_process_cwd

    best_score = -1
    best_port: int | None = None

    for pid, listener in resolved_listeners.items():
        command = str(listener.get("command", "")).lower()
        if not is_candidate_command(command, project_type):
            continue

        cwd_path = resolved_get_cwd(pid)
        if cwd_path is None:
            continue

        match_score = match_project_path_score(target, cwd_path)
        if match_score < 0:
            continue

        ports = [
            port
            for port in listener.get("ports", set())
            if isinstance(port, int) and port >= 1024 and port != resolved_bridge_port
        ]
        if not ports:
            continue

        selected_port = select_preferred_port(ports)
        if match_score > best_score:
            best_score = match_score
            best_port = selected_port

    return best_port
