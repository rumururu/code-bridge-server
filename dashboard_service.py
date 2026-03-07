"""Dashboard service for aggregating server state data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from config import VERSION, get_config
from database import get_project_db
from device_action_service import (
    get_scrcpy_status_for_current_server,
    list_connected_devices_for_current_server,
)
from llm_settings import get_llm_options_snapshot
from optional_services import FIREBASE_AVAILABLE, TUNNEL_AVAILABLE, get_firebase_auth, get_tunnel_service
from pairing import get_pairing_service
from project_action_service import list_projects_for_current_server


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "local_url": self.local_url,
            "tunnel_url": self.tunnel_url,
            "server_name": self.server_name,
            "remote_access_enabled": self.remote_access_enabled,
            "prefer_local": self.prefer_local,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "running": self.running,
            "url": self.url,
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
class DashboardOverview:
    """Complete dashboard overview."""

    server: DashboardServerStatus
    llm: DashboardLlmStatus
    pairing: DashboardPairingStatus
    projects: DashboardProjectStatus
    devices: DashboardDeviceStatus
    tunnel: DashboardTunnelStatus
    firebase: DashboardFirebaseStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server.as_dict(),
            "llm": self.llm.as_dict(),
            "pairing": self.pairing.as_dict(),
            "projects": self.projects.as_dict(),
            "devices": self.devices.as_dict(),
            "tunnel": self.tunnel.as_dict(),
            "firebase": self.firebase.as_dict(),
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

    return DashboardServerStatus(
        status="ok",
        version=VERSION,
        local_url=local_url,
        tunnel_url=tunnel_url,
        server_name=config.server_name,
        remote_access_enabled=config.remote_access_enabled,
        prefer_local=True,  # Always recommend local connection for better speed
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
    if not TUNNEL_AVAILABLE:
        return DashboardTunnelStatus(
            available=False,
            running=False,
            url=None,
        )

    tunnel_service = get_tunnel_service()
    if not tunnel_service:
        return DashboardTunnelStatus(
            available=True,
            running=False,
            url=None,
        )

    status = tunnel_service.get_status()
    running = bool(status.get("running", False))
    raw_url = status.get("url")
    url = raw_url if isinstance(raw_url, str) else None

    return DashboardTunnelStatus(
        available=True,
        running=running,
        url=url,
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


async def build_dashboard_overview() -> DashboardOverview:
    """Build complete dashboard overview aggregating all server state."""
    server = _build_server_status()
    llm = _build_llm_status()
    pairing = _build_pairing_status()
    projects = _build_project_status()
    devices = await _build_device_status()
    tunnel = _build_tunnel_status()
    firebase = _build_firebase_status()

    return DashboardOverview(
        server=server,
        llm=llm,
        pairing=pairing,
        projects=projects,
        devices=devices,
        tunnel=tunnel,
        firebase=firebase,
    )


async def get_dashboard_overview_for_current_server() -> dict[str, Any]:
    """Get dashboard overview as dictionary for API response."""
    overview = await build_dashboard_overview()
    return overview.as_dict()
