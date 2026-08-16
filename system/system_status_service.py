"""System status helpers for health/debug routes."""

from __future__ import annotations

import os
from typing import Any, Mapping

from core.config import get_config


def get_health_status_for_current_server() -> dict[str, str]:
    """Return static server health payload."""
    return {"status": "ok", "service": "claude-bridge"}


def is_cloudflared_installed() -> bool:
    """Whether the ``cloudflared`` binary is reachable from this process.

    "From this process" is the load-bearing part: the launcher used to start
    the server with a login-shell-free PATH, so a cloudflared installed by
    Homebrew looked missing. With the PATH bootstrap in place this probe can
    be shown to the user, which is what the dashboard banner does.

    Deliberately delegates to :meth:`TunnelService.is_cloudflared_installed`
    rather than calling ``shutil.which`` again — one definition of "installed"
    keeps the banner and the tunnel start path from disagreeing.
    """
    try:
        from remote.tunnel_service import TunnelService
    except ImportError:
        return False
    try:
        return bool(TunnelService.is_cloudflared_installed())
    except Exception:  # pragma: no cover - probe must never break the overview
        return False


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
