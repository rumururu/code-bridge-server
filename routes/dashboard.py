"""Dashboard routes for Code Bridge Server."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from dashboard_page import render_dashboard_html
from dashboard_service import get_dashboard_overview_for_current_server
from project_device_logs import read_log_tail
from server_logging import resolve_server_log_path

from .deps import require_dashboard_auth, require_local_access

router = APIRouter()


# Get the server directory for git operations
SERVER_DIR = Path(__file__).parent.parent


@router.get("/", include_in_schema=False, dependencies=[Depends(require_local_access)])
async def root_redirect():
    """Redirect root to dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    include_in_schema=False,
    dependencies=[Depends(require_local_access)],
)
async def get_dashboard():
    """Serve the dashboard HTML page.

    This page is only accessible from local network.
    External access via Cloudflare Tunnel is blocked for security.
    """
    return HTMLResponse(content=render_dashboard_html())


@router.get(
    "/api/system/overview",
    dependencies=[Depends(require_dashboard_auth)],
)
async def get_system_overview():
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
async def check_update():
    """Check for server updates by running git pull.

    Only accessible from local network for security.

    Returns:
        - updated: True if new changes were pulled
        - message: Status message
        - output: Git command output
    """
    try:
        # Run git fetch first to get latest refs
        fetch_proc = await asyncio.create_subprocess_exec(
            "git", "fetch",
            cwd=str(SERVER_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await fetch_proc.communicate()

        # Check if we're behind
        status_proc = await asyncio.create_subprocess_exec(
            "git", "status", "-uno",
            cwd=str(SERVER_DIR),
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
            cwd=str(SERVER_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        pull_stdout, pull_stderr = await pull_proc.communicate()
        output = pull_stdout.decode() + pull_stderr.decode()

        if pull_proc.returncode == 0:
            updated = "Already up to date" not in output
            return {
                "updated": updated,
                "message": "Updated successfully. Restart server to apply changes." if updated else "Already up to date",
                "output": output.strip(),
            }
        else:
            return {
                "updated": False,
                "message": "Update failed",
                "output": output.strip(),
                "error": True,
            }

    except Exception as e:
        return {
            "updated": False,
            "message": f"Update check failed: {str(e)}",
            "error": True,
        }


@router.get(
    "/api/system/server-log",
    dependencies=[Depends(require_dashboard_auth)],
)
async def get_server_log(lines: int = Query(default=200, ge=20, le=1000)):
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
        except Exception:
            updated_at = None

    return {
        "path": str(log_path),
        "exists": log_exists,
        "updated_at": updated_at,
        "lines": lines,
        "text": log_text,
    }
