"""Dashboard routes for Code Bridge Server."""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from system.autostart_service import get_autostart_status, set_autostart
from core.config import get_config
from dashboard.dashboard_page import render_agents_html, render_dashboard_html
from dashboard.dashboard_service import get_dashboard_overview_for_current_server
from system.optional_services import FIREBASE_AVAILABLE, get_firebase_auth, get_active_tunnel_url
from pairing.pairing_service import get_pairing_service
from projects.project_device_logs import read_log_tail
from core.server_logging import resolve_server_log_path

from .deps import require_dashboard_auth, require_local_access

router = APIRouter()


# Get the git repository root (search upward from server directory)
def _find_git_root() -> Path | None:
    """Find the git repository root by searching upward."""
    current = Path(__file__).parent.parent
    for _ in range(5):  # Search up to 5 levels
        if (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None

GIT_ROOT = _find_git_root()
SERVER_DIR = Path(__file__).parent.parent
DESKTOP_ARTIFACTS_DIR = GIT_ROOT / "dist" / "desktop_server_app" if GIT_ROOT else None
DESKTOP_INSTALLERS = {
    "macos": {
        "filename": "Code Bridge Server.dmg",
        "media_type": "application/x-apple-diskimage",
    },
    "windows": {
        "filename": "Code Bridge Server.msi",
        "media_type": "application/x-msi",
    },
}


async def _restart_server_delayed(delay: float = 1.0) -> None:
    """Restart the server after a delay to allow response to be sent.

    Used by check_update() when git pull brings new changes.
    """
    import subprocess

    await asyncio.sleep(delay)
    logger.info("Restarting server after update...")

    current_pid = os.getpid()
    server_script = SERVER_DIR / "server_cli.py"
    venv_python = SERVER_DIR / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    # Remove PID file to allow new server to start
    pid_file = SERVER_DIR / ".server.pid"
    if pid_file.exists():
        pid_file.unlink()

    # Use shell to: kill current server (force), wait, then start new server
    restart_cmd = f"sleep 1 && kill -9 {current_pid} && sleep 2 && {python} {server_script} >> /tmp/code_bridge_server.log 2>&1"

    subprocess.Popen(
        restart_cmd,
        shell=True,
        cwd=str(SERVER_DIR),
        start_new_session=True,
    )
    logger.info("Restart command spawned")


@router.get("/", include_in_schema=False, dependencies=[Depends(require_local_access)])
async def root_redirect() -> RedirectResponse:
    """Redirect root to dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    include_in_schema=False,
    dependencies=[Depends(require_local_access)],
)
async def get_dashboard() -> HTMLResponse:
    """Serve the dashboard HTML page.

    This page is only accessible from local network.
    External access via Cloudflare Tunnel is blocked for security.
    """
    return HTMLResponse(content=render_dashboard_html())


@router.get(
    "/settings",
    response_class=HTMLResponse,
    include_in_schema=False,
    dependencies=[Depends(require_local_access)],
)
async def get_settings_page() -> HTMLResponse:
    """Serve the settings view of the console."""
    return HTMLResponse(content=render_dashboard_html("settings"))


@router.get(
    "/agents",
    response_class=HTMLResponse,
    include_in_schema=False,
    dependencies=[Depends(require_local_access)],
)
async def get_agents_page() -> HTMLResponse:
    """Serve the standalone Agents page.

    Same local-only gate as the dashboard — it drives the
    ``/api/dashboard/agent/*`` endpoints, which are never mounted on the
    tunnel-exposed API app.
    """
    return HTMLResponse(content=render_agents_html())


@router.get(
    "/downloads/desktop-server/{platform}",
    include_in_schema=False,
    dependencies=[Depends(require_local_access)],
)
async def download_desktop_server_installer(platform: str) -> FileResponse:
    """Download packaged desktop server installers from local artifacts."""
    installer = DESKTOP_INSTALLERS.get(platform.lower())
    if installer is None:
        raise HTTPException(status_code=404, detail="Unsupported desktop installer platform")
    if DESKTOP_ARTIFACTS_DIR is None:
        raise HTTPException(status_code=404, detail="Desktop installer artifacts are not available")

    filename = installer["filename"]
    artifact_path = DESKTOP_ARTIFACTS_DIR / filename
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Desktop installer artifact not found: {filename}",
        )

    return FileResponse(
        artifact_path,
        media_type=installer["media_type"],
        filename=filename,
    )


@router.get(
    "/api/system/overview",
    dependencies=[Depends(require_dashboard_auth)],
)
async def get_system_overview() -> dict[str, Any]:
    """Get aggregated system overview for dashboard.

    Returns comprehensive server state including:
    - Server status (local/tunnel URLs)
    - LLM configuration
    - Pairing status
    - Projects and their dev server status
    - Connected devices
    - Tunnel and Firebase status

    Only accessible from local network for security.
    Contains sensitive information (user IDs, emails, URLs).
    """
    return await get_dashboard_overview_for_current_server()


@router.post(
    "/api/system/check-update",
    dependencies=[Depends(require_dashboard_auth)],
)
async def check_update() -> dict[str, Any]:
    """Check for server updates by running git pull.

    Only accessible from local network for security.

    Returns:
        - updated: True if new changes were pulled
        - message: Status message
        - output: Git command output
    """
    if GIT_ROOT is None:
        return {
            "updated": False,
            "message": "Git repository not found. Server was not installed via git.",
            "error": True,
        }

    try:
        # Run git fetch first to get latest refs
        fetch_proc = await asyncio.create_subprocess_exec(
            "git", "fetch",
            cwd=str(GIT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await fetch_proc.communicate()

        # Check if we're behind
        status_proc = await asyncio.create_subprocess_exec(
            "git", "status", "-uno",
            cwd=str(GIT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        status_stdout, _ = await status_proc.communicate()
        status_text = status_stdout.decode()

        if "Your branch is up to date" in status_text:
            return {
                "updated": False,
                "message": "Already up to date",
                "output": status_text.strip(),
            }

        # Run git pull
        pull_proc = await asyncio.create_subprocess_exec(
            "git", "pull",
            cwd=str(GIT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        pull_stdout, pull_stderr = await pull_proc.communicate()
        output = pull_stdout.decode() + pull_stderr.decode()

        if pull_proc.returncode == 0:
            updated = "Already up to date" not in output
            if updated:
                # Schedule server restart in background
                asyncio.create_task(_restart_server_delayed(1.5))
            return {
                "updated": updated,
                "message": "Updated successfully. Server restarting..." if updated else "Already up to date",
                "output": output.strip(),
                "restarting": updated,
            }
        else:
            return {
                "updated": False,
                "message": "Update failed",
                "output": output.strip(),
                "error": True,
            }

    except OSError as e:
        return {
            "updated": False,
            "message": f"Update check failed: {str(e)}",
            "error": True,
        }


@router.get(
    "/api/system/server-log",
    dependencies=[Depends(require_dashboard_auth)],
)
async def get_server_log(lines: int = Query(default=200, ge=20, le=1000)) -> dict[str, Any]:
    """Return bounded server log tail for dashboard display."""
    log_path = resolve_server_log_path()
    log_exists = log_path.exists()
    log_text = read_log_tail(log_path, max_lines=lines, max_chars=50000)

    updated_at = None
    if log_exists:
        try:
            updated_at = datetime.fromtimestamp(
                log_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
        except OSError:
            updated_at = None

    return {
        "path": str(log_path),
        "exists": log_exists,
        "updated_at": updated_at,
        "lines": lines,
        "text": log_text,
    }


@router.post(
    "/api/system/update-firebase",
    dependencies=[Depends(require_dashboard_auth)],
)
async def update_firebase_server_info() -> dict[str, Any]:
    """Manually update server info in Firebase.

    Re-registers the device with current local URL and tunnel URL.
    Useful when port or IP has changed but automatic update didn't trigger.

    Only accessible from local network for security.
    """
    if not FIREBASE_AVAILABLE:
        return {
            "success": False,
            "message": "Firebase is not available",
            "error": True,
        }

    firebase_auth = get_firebase_auth()
    if firebase_auth is None:
        return {
            "success": False,
            "message": "Firebase auth service not initialized",
            "error": True,
        }

    if not firebase_auth.is_authenticated:
        return {
            "success": False,
            "message": "Not authenticated with Firebase. QR pairing required.",
            "error": True,
        }

    try:
        config = get_config()
        pairing = get_pairing_service()
        local_ip = pairing.get_local_ip()
        local_url = f"http://{local_ip}:{config.api_port}"
        tunnel_url = get_active_tunnel_url()

        success = await firebase_auth.register_device(tunnel_url, local_url)

        if success:
            return {
                "success": True,
                "message": "Server info updated in Firebase",
                "local_url": local_url,
                "tunnel_url": tunnel_url or "(not active)",
            }
        else:
            return {
                "success": False,
                "message": "Failed to update Firebase. Check server logs.",
                "error": True,
            }
    except Exception as e:
        logger.error("Failed to update Firebase server info: %s", e)
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "error": True,
        }


@router.post(
    "/api/system/restart",
    dependencies=[Depends(require_dashboard_auth)],
)
async def restart_server() -> dict[str, Any]:
    """Restart the server.

    Kills all related processes and restarts the server.
    Only accessible from local network for security.

    Returns:
        - success: True if restart initiated
        - message: Status message
    """
    import subprocess

    current_pid = os.getpid()
    logger.info("Restart requested. Current PID: %d", current_pid)

    # Find the server script
    server_script = SERVER_DIR / "server_cli.py"
    venv_python = SERVER_DIR / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    # Remove PID file to allow new server to start
    pid_file = SERVER_DIR / ".server.pid"
    if pid_file.exists():
        pid_file.unlink()

    # Use shell to: 1) kill current server (force), 2) wait, 3) start new server
    # The 'kill' command will be executed after this response is sent
    restart_cmd = f"sleep 1 && kill -9 {current_pid} && sleep 2 && {python} {server_script} >> /tmp/code_bridge_server.log 2>&1"

    subprocess.Popen(
        restart_cmd,
        shell=True,
        cwd=str(SERVER_DIR),
        start_new_session=True,
    )

    logger.info("Restart command spawned")

    return {
        "success": True,
        "message": "Server restart initiated. Page will reload automatically.",
        "restarting": True,
    }


class AutostartRequest(BaseModel):
    """Request body for autostart setting."""

    enabled: bool


@router.get(
    "/api/system/autostart",
    dependencies=[Depends(require_dashboard_auth)],
)
async def get_autostart() -> dict[str, Any]:
    """Get current autostart status.

    Returns autostart configuration for the current platform:
    - available: Whether autostart can be configured
    - enabled: Whether autostart is currently enabled
    - platform: Current OS (darwin, linux, windows)
    - method: How autostart is implemented (launchagent, systemd, startup_folder)
    - path: Path to the autostart file if enabled

    Only accessible from local network for security.
    """
    status = get_autostart_status()
    return status.as_dict()


@router.post(
    "/api/system/autostart",
    dependencies=[Depends(require_dashboard_auth)],
)
async def set_autostart_setting(request: AutostartRequest) -> dict[str, Any]:
    """Enable or disable autostart on system boot.

    Configures the server to start automatically when the system boots:
    - macOS: Creates/removes LaunchAgent in ~/Library/LaunchAgents/
    - Linux: Creates/removes systemd user service in ~/.config/systemd/user/
    - Windows: Creates/removes shortcut in Startup folder

    Only accessible from local network for security.

    Args:
        enabled: True to enable autostart, False to disable
    """
    success, message = set_autostart(request.enabled)
    status = get_autostart_status()

    return {
        "success": success,
        "message": message,
        "autostart": status.as_dict(),
    }
