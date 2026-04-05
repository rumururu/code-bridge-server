"""Helpers for project runtime process and port detection."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any


def match_project_path_score(project_path: Path, cwd_path: Path) -> int:
    """Score CWD matching against project path."""
    if cwd_path == project_path:
        return 3
    if project_path in cwd_path.parents:
        return 2
    if cwd_path in project_path.parents:
        return 1
    return -1


def is_candidate_command(command: str, project_type: Any) -> bool:
    """Check if command likely hosts a dev server."""
    normalized = command.lower()
    project_type_value = str(getattr(project_type, "value", project_type)).lower()

    node_like_tokens = ("node", "npm", "pnpm", "yarn", "bun", "next", "vite", "deno")
    flutter_like_tokens = ("flutter", "dart")
    python_like_tokens = ("python", "uvicorn", "gunicorn")

    if project_type_value == "flutter":
        return any(token in normalized for token in flutter_like_tokens)
    if project_type_value == "nextjs":
        return any(token in normalized for token in node_like_tokens)

    return (
        any(token in normalized for token in node_like_tokens)
        or any(token in normalized for token in flutter_like_tokens)
        or any(token in normalized for token in python_like_tokens)
    )


def select_preferred_port(ports: list[int]) -> int:
    """Select the most likely app port from candidates."""
    preferred_ports = (3000, 5173, 4200, 4173, 8081, 8082, 8000, 5000)
    unique_ports = sorted(set(ports))
    for preferred in preferred_ports:
        if preferred in unique_ports:
            return preferred
    return unique_ports[0]


def extract_port(endpoint: str) -> int | None:
    """Extract local port from lsof endpoint field."""
    if ":" not in endpoint:
        return None

    local = endpoint.split("->", 1)[0]
    candidate = local.rsplit(":", 1)[-1]
    return int(candidate) if candidate.isdigit() else None


def is_local_port_open(port: int) -> bool:
    """Check if localhost TCP port is accepting connections."""
    for host in ("127.0.0.1", "::1"):
        try:
            family = socket.AF_INET6 if host == "::1" else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            continue
    return False
