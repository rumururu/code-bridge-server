"""Dashboard service for aggregating server state data."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any, Optional

from system.autostart_service import get_autostart_status


def _is_port_listening(host: str, port: int) -> bool:
    """Check if a port is actually listening.

    Args:
        host: Host to check (use '127.0.0.1' for localhost, '0.0.0.0' checks all interfaces)
        port: Port number to check

    Returns:
        True if the port is listening and accepting connections
    """
    try:
        # For 0.0.0.0, check on localhost since we're on the same machine
        check_host = "127.0.0.1" if host == "0.0.0.0" else host
        with socket.create_connection((check_host, port), timeout=1):
            return True
    except (socket.timeout, socket.error, OSError):
        return False
from core.config import VERSION, get_config
from core.database import get_project_db, get_usage_db
from agent.agent_store import get_agent_store
from devices.device_action_service import (
    get_scrcpy_status_for_current_server,
    list_connected_devices_for_current_server,
)
from llm.llm_settings import get_llm_options_snapshot
from system.optional_services import FIREBASE_AVAILABLE, TUNNEL_AVAILABLE, get_firebase_auth, get_tunnel_service
from system.system_status_service import is_cloudflared_installed
from pairing.pairing import get_pairing_service
from projects.project_action_service import list_projects_for_current_server


@dataclass(frozen=True)
class DashboardServerStatus:
    """Server status for dashboard."""

    status: str
    version: str
    local_url: str
    tunnel_url: Optional[str]
    server_name: str
    remote_access_enabled: bool
    prefer_local: bool
    api_host: str
    dashboard_host: str
    api_port: int
    dashboard_port: int
    api_listening: bool = True  # Whether API port is actually listening
    dashboard_listening: bool = True  # Whether dashboard port is actually listening

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "local_url": self.local_url,
            "tunnel_url": self.tunnel_url,
            "server_name": self.server_name,
            "remote_access_enabled": self.remote_access_enabled,
            "prefer_local": self.prefer_local,
            "api_host": self.api_host,
            "dashboard_host": self.dashboard_host,
            "api_port": self.api_port,
            "dashboard_port": self.dashboard_port,
            "api_listening": self.api_listening,
            "dashboard_listening": self.dashboard_listening,
        }


@dataclass(frozen=True)
class DashboardLlmStatus:
    """LLM status for dashboard."""

    selected_company: Optional[str]
    selected_model: Optional[str]
    connected: bool
    companies: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": {
                "company_id": self.selected_company,
                "model": self.selected_model,
            },
            "connected": self.connected,
            "companies": self.companies,
        }


@dataclass(frozen=True)
class DashboardPairingStatus:
    """Pairing status for dashboard."""

    active_clients: int
    pending_tokens: int
    clients: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_clients": self.active_clients,
            "pending_tokens": self.pending_tokens,
            "clients": self.clients,
        }


@dataclass(frozen=True)
class DashboardProjectStatus:
    """Projects status for dashboard."""

    total: int
    running: int
    items: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "running": self.running,
            "items": self.items,
        }


@dataclass(frozen=True)
class DashboardDeviceStatus:
    """Devices status for dashboard."""

    total: int
    scrcpy_running: bool
    scrcpy_url: Optional[str]
    items: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "scrcpy_running": self.scrcpy_running,
            "scrcpy_url": self.scrcpy_url,
            "items": self.items,
        }


@dataclass(frozen=True)
class DashboardTunnelStatus:
    """Tunnel status for dashboard."""

    available: bool
    running: bool
    url: Optional[str]
    # Whether the cloudflared binary is on the server's PATH. Distinct from
    # `available` (the tunnel integration importing) and from `running`:
    # without this the dashboard could only say "the tunnel is not running"
    # for a machine where the tunnel can never start at all.
    cloudflared_installed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "running": self.running,
            "url": self.url,
            "cloudflared_installed": self.cloudflared_installed,
        }


@dataclass(frozen=True)
class DashboardFirebaseStatus:
    """Firebase status for dashboard."""

    available: bool
    authenticated: bool
    user_id: Optional[str]
    email: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "authenticated": self.authenticated,
            "user_id": self.user_id,
            "email": self.email,
        }


@dataclass(frozen=True)
class DashboardAutostartStatus:
    """Autostart status for dashboard."""

    available: bool
    enabled: bool
    platform: str
    method: str
    path: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "enabled": self.enabled,
            "platform": self.platform,
            "method": self.method,
            "path": self.path,
        }


@dataclass(frozen=True)
class DashboardOverview:
    """Complete dashboard overview."""

    server: DashboardServerStatus
    llm: DashboardLlmStatus
    pairing: DashboardPairingStatus
    projects: DashboardProjectStatus
    devices: DashboardDeviceStatus
    tunnel: DashboardTunnelStatus
    firebase: DashboardFirebaseStatus
    autostart: DashboardAutostartStatus
    work: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server.as_dict(),
            "llm": self.llm.as_dict(),
            "pairing": self.pairing.as_dict(),
            "projects": self.projects.as_dict(),
            "devices": self.devices.as_dict(),
            "tunnel": self.tunnel.as_dict(),
            "firebase": self.firebase.as_dict(),
            "autostart": self.autostart.as_dict(),
            "work": self.work,
            "usage": self.usage,
        }


def _build_server_status() -> DashboardServerStatus:
    """Build server status from config.

    local_url uses api_port since external clients connect to API server.
    """
    config = get_config()
    pairing_service = get_pairing_service()
    local_ip = pairing_service.get_local_ip()
    local_url = f"http://{local_ip}:{config.api_port}"

    tunnel_url: Optional[str] = None
    tunnel_service = get_tunnel_service()
    if tunnel_service:
        status = tunnel_service.get_status()
        raw_url = status.get("url") if isinstance(status, dict) else None
        tunnel_url = raw_url if isinstance(raw_url, str) else None

    # Check if ports are actually listening
    api_listening = _is_port_listening(config.api_host, config.api_port)
    dashboard_listening = _is_port_listening(config.dashboard_host, config.dashboard_port)

    return DashboardServerStatus(
        status="ok" if api_listening and dashboard_listening else "degraded",
        version=VERSION,
        local_url=local_url,
        tunnel_url=tunnel_url,
        server_name=config.server_name,
        remote_access_enabled=config.remote_access_enabled,
        prefer_local=True,  # Always recommend local connection for better speed
        api_host=config.api_host,
        dashboard_host=config.dashboard_host,
        api_port=config.api_port,
        dashboard_port=config.dashboard_port,
        api_listening=api_listening,
        dashboard_listening=dashboard_listening,
    )


def _build_llm_status() -> DashboardLlmStatus:
    """Build LLM status from settings."""
    snapshot = get_llm_options_snapshot()
    selected = snapshot.get("selected", {})
    companies = snapshot.get("companies", [])

    # Find if selected company is connected
    connected = False
    selected_company = selected.get("company_id")
    for company in companies:
        if company.get("id") == selected_company:
            connected = bool(company.get("connected", False))
            break

    return DashboardLlmStatus(
        selected_company=selected.get("company_id"),
        selected_model=selected.get("model"),
        connected=connected,
        companies=companies,
    )


def _build_pairing_status() -> DashboardPairingStatus:
    """Build pairing status from pairing service."""
    pairing_service = get_pairing_service()
    status = pairing_service.get_pairing_status()

    clients = [client.as_response_fields() for client in status.clients]

    return DashboardPairingStatus(
        active_clients=status.active_clients,
        pending_tokens=status.pending_tokens,
        clients=clients,
    )


def _build_project_status() -> DashboardProjectStatus:
    """Build project status from database."""
    projects = list_projects_for_current_server()

    running_count = 0
    items = []
    for proj in projects:
        is_running = bool(proj.get("server_running", False))
        if is_running:
            running_count += 1
        items.append({
            "name": proj.get("name", ""),
            "type": proj.get("type", ""),
            "path": proj.get("path", ""),
            "enabled": proj.get("enabled", True),
            "server_running": is_running,
            "server_port": proj.get("server_port"),
        })

    return DashboardProjectStatus(
        total=len(projects),
        running=running_count,
        items=items,
    )


async def _build_device_status() -> DashboardDeviceStatus:
    """Build device status from scrcpy manager."""
    devices = await list_connected_devices_for_current_server()
    scrcpy_status = get_scrcpy_status_for_current_server()

    scrcpy_running = bool(scrcpy_status.get("running", False))
    scrcpy_url = scrcpy_status.get("url") if scrcpy_running else None

    items = []
    for device in devices:
        items.append({
            "id": device.get("id", ""),
            "model": device.get("model", ""),
            "name": device.get("name", device.get("model", "")),
        })

    return DashboardDeviceStatus(
        total=len(devices),
        scrcpy_running=scrcpy_running,
        scrcpy_url=scrcpy_url if isinstance(scrcpy_url, str) else None,
        items=items,
    )


def _build_tunnel_status() -> DashboardTunnelStatus:
    """Build tunnel status from tunnel service."""
    # Probed regardless of whether a tunnel service instance exists: the case
    # worth telling the user about is precisely the one where no tunnel has
    # ever started because the binary is not there.
    cloudflared_installed = is_cloudflared_installed()

    if not TUNNEL_AVAILABLE:
        return DashboardTunnelStatus(
            available=False,
            running=False,
            url=None,
            cloudflared_installed=cloudflared_installed,
        )

    tunnel_service = get_tunnel_service()
    if not tunnel_service:
        return DashboardTunnelStatus(
            available=True,
            running=False,
            url=None,
            cloudflared_installed=cloudflared_installed,
        )

    status = tunnel_service.get_status()
    running = bool(status.get("running", False))
    raw_url = status.get("url")
    url = raw_url if isinstance(raw_url, str) else None
    # The service's own status carries the same probe; prefer it when present
    # so the banner and a tunnel start attempt cannot disagree.
    installed = status.get("installed")
    if isinstance(installed, bool):
        cloudflared_installed = installed

    return DashboardTunnelStatus(
        available=True,
        running=running,
        url=url,
        cloudflared_installed=cloudflared_installed,
    )


def _build_firebase_status() -> DashboardFirebaseStatus:
    """Build Firebase status from firebase auth."""
    if not FIREBASE_AVAILABLE:
        return DashboardFirebaseStatus(
            available=False,
            authenticated=False,
            user_id=None,
            email=None,
        )

    firebase_auth = get_firebase_auth()
    if not firebase_auth:
        return DashboardFirebaseStatus(
            available=True,
            authenticated=False,
            user_id=None,
            email=None,
        )

    status = firebase_auth.get_status()
    authenticated = bool(status.get("authenticated", False))
    user_id = status.get("user_id") if isinstance(status.get("user_id"), str) else None
    email = status.get("email") if isinstance(status.get("email"), str) else None

    return DashboardFirebaseStatus(
        available=True,
        authenticated=authenticated,
        user_id=user_id,
        email=email,
    )


def _build_autostart_status() -> DashboardAutostartStatus:
    """Build autostart status from autostart service."""
    status = get_autostart_status()
    return DashboardAutostartStatus(
        available=status.available,
        enabled=status.enabled,
        platform=status.platform,
        method=status.method,
        path=status.path,
    )


def _build_work_status() -> dict[str, Any]:
    store = get_agent_store()
    tasks = store.list_tasks(limit=200)
    runs = store.list_runs(limit=200)
    pending_tasks = [task for task in tasks if task.get("status") in {"queued", "backlog", "todo", "pending"}]
    active_tasks = [task for task in tasks if task.get("status") in {"running", "in_progress", "review", "blocked"}]
    done_tasks = [task for task in tasks if task.get("status") == "done"]
    active_runs = [run for run in runs if run.get("status") in {"queued", "running", "pending_approval"}]
    return {
        "task_count": len(tasks),
        "pending_task_count": len(pending_tasks),
        "active_task_count": len(active_tasks),
        "done_task_count": len(done_tasks),
        "active_run_count": len(active_runs),
        "recent_tasks": tasks[:10],
        "recent_runs": runs[:10],
    }


def _build_usage_status() -> dict[str, Any]:
    config = get_config()
    usage_db = get_usage_db()
    return {
        "summary": usage_db.get_weekly_summary(
            budget_usd=config.weekly_budget_usd,
            window_days=config.usage_window_days,
        ),
        "by_provider": usage_db.breakdown("provider_id", window_days=config.usage_window_days),
        "recent_turns": usage_db.list_events(window_days=config.usage_window_days, limit=10),
    }


async def build_dashboard_overview() -> DashboardOverview:
    """Build complete dashboard overview aggregating all server state."""
    server = _build_server_status()
    llm = _build_llm_status()
    pairing = _build_pairing_status()
    projects = _build_project_status()
    devices = await _build_device_status()
    tunnel = _build_tunnel_status()
    firebase = _build_firebase_status()
    autostart = _build_autostart_status()
    work = _build_work_status()
    usage = _build_usage_status()

    return DashboardOverview(
        server=server,
        llm=llm,
        pairing=pairing,
        projects=projects,
        devices=devices,
        tunnel=tunnel,
        firebase=firebase,
        autostart=autostart,
        work=work,
        usage=usage,
    )


async def get_dashboard_overview_for_current_server() -> dict[str, Any]:
    """Get dashboard overview as dictionary for API response."""
    overview = await build_dashboard_overview()
    return overview.as_dict()
