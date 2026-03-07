"""System status helpers for health/debug routes."""

from __future__ import annotations

import os
from typing import Any, Mapping

from config import get_config


def get_health_status_for_current_server() -> dict[str, str]:
    """Return static server health payload."""
    return {"status": "ok", "service": "claude-bridge"}


def get_debug_port_snapshot_for_current_server(
    *,
    config: Any | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return effective port debug snapshot."""
    resolved_config = config or get_config()
    resolved_env = env or os.environ

    return {
        "dashboard_port": resolved_config.dashboard_port,
        "api_port": resolved_config.api_port,
        "env_dashboard_port": resolved_env.get("CODEBRIDGE_DASHBOARD_PORT"),
        "env_api_port": resolved_env.get("CODEBRIDGE_API_PORT"),
        "runtime_port": getattr(resolved_config, "_runtime_port", None),
    }
