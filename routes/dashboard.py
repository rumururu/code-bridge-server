"""Dashboard routes for Code Bridge Server."""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse

logger = logging.getLogger(__name__)

from dashboard_page import render_dashboard_html
from dashboard_service import get_dashboard_overview_for_current_server
from project_device_logs import read_log_tail
from server_logging import resolve_server_log_path

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


async def _restart_server_delayed(delay: float = 1.0) -> None:
    """Restart the server after a delay to allow response to be sent."""
    await asyncio.sleep(delay)
    logger.info("Restarting server after update...")

    # Get the original command line arguments
    python = sys.executable
    args = sys.argv[:]

    # If running as module, reconstruct properly
    if args[0].endswith("server_cli.py") or args[0].endswith("main.py"):
        script = args[0]
        os.chdir(SERVER_DIR)
        os.execv(python, [python, script] + args[1:])
    else:
        # Fallback: restart with same args
        os.execv(python, [python] + args)


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
