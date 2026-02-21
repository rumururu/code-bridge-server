"""Code Bridge Server - FastAPI application."""

import asyncio
import json
import os
import re
import select
import subprocess
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Optional

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from claude_session import ClaudeSession, get_session_manager
from config import get_config
from database import get_project_db, get_usage_db
from files import get_file_manager
from pairing import get_pairing_service, PairingData
from qr_display import display_pairing_qr, QRCODE_AVAILABLE
from llm_settings import (
    get_llm_options_snapshot,
    set_selected_llm,
    add_custom_llm_model,
    remove_custom_llm_model,
)
from models import (
    DeviceRunRequest,
    ProjectCreate,
    ProjectUpdate,
    ProjectImport,
    FileWrite,
    FileCreate,
    LlmSelectionUpdate,
    LlmCustomModelUpdate,
    PairVerifyRequest,
)
from preview import get_preview_proxy
from projects import get_project_manager
from scrcpy_manager import get_scrcpy_manager

# Optional services for network discovery and remote access
try:
    from mdns_service import create_mdns_service, get_mdns_service
    MDNS_AVAILABLE = True
except ImportError:
    MDNS_AVAILABLE = False
    def create_mdns_service(*args, **kwargs): return None
    def get_mdns_service(): return None

try:
    from tunnel_service import create_tunnel_service, get_tunnel_service
    TUNNEL_AVAILABLE = True
except ImportError:
    TUNNEL_AVAILABLE = False
    def create_tunnel_service(*args, **kwargs): return None
    def get_tunnel_service(): return None

try:
    from firebase_auth import get_firebase_auth
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    def get_firebase_auth(): return None


def _sanitize_project_name(value: str) -> str:
    """Sanitize user-provided project name into a safe identifier."""
    sanitized = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "project"


def _build_unique_project_name(base_name: str, existing_names: set[str]) -> str:
    """Create a unique name by adding numeric suffix if needed."""
    if base_name not in existing_names:
        return base_name

    index = 2
    while f"{base_name}_{index}" in existing_names:
        index += 1
    return f"{base_name}_{index}"


def _infer_project_type(project_path: Path) -> str:
    """Infer project type from common marker files."""
    if (project_path / "pubspec.yaml").exists():
        return "flutter"

    package_json = project_path / "package.json"
    if package_json.exists():
        try:
            package_data = json.loads(package_json.read_text(encoding="utf-8"))
            dependencies = package_data.get("dependencies", {}) or {}
            dev_dependencies = package_data.get("devDependencies", {}) or {}
            if "next" in dependencies or "next" in dev_dependencies:
                return "nextjs"
        except Exception:
            pass
        return "web"

    for marker in (
        "next.config.js",
        "next.config.mjs",
        "next.config.ts",
        "next.config.cjs",
    ):
        if (project_path / marker).exists():
            return "nextjs"

    return "other"


def _normalize_project_type(value: Optional[str], project_path: Path) -> str:
    """Normalize provided project type or infer from path."""
    if value is None or not value.strip():
        return _infer_project_type(project_path)

    normalized = value.strip().lower()
    if normalized in {"next", "next.js"}:
        return "nextjs"
    return normalized


DEFAULT_SCAN_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".dart_tool",
    ".next",
    "node_modules",
    "build",
    "dist",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

MAX_SCAN_DEPTH = 5
MAX_SCAN_RESULTS = 300


CLAUDE_USAGE_CACHE_TTL_SECONDS = 300
CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS = 300
_claude_usage_cache: dict[str, Any] = {
    "expires_at": 0.0,
    "payload": None,
}
_claude_usage_probe_lock = asyncio.Lock()
_ansi_escape_pattern = re.compile(
    r"\x1B\[[0-?]*[ -/]*[@-~]|\x1B\][^\x07]*(?:\x07|\x1B\\)|\x1B[@-_]"
)


def _strip_ansi(text: str) -> str:
    """Strip ANSI control sequences for easier text parsing."""
    return _ansi_escape_pattern.sub("", text)


def _extract_usage_percent_from_text(text: str) -> float | None:
    """Extract first likely usage percentage from /usage text output."""
    if not text.strip():
        return None

    normalized = _strip_ansi(text)
    prioritized_patterns = (
        r"(?:used|usage|weekly|limit)[^\n]{0,80}?(\d{1,3}(?:\.\d+)?)\s*%",
        r"(\d{1,3}(?:\.\d+)?)\s*%[^\n]{0,80}?(?:weekly|limit|usage)",
    )
    for pattern in prioritized_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match is not None:
            try:
                return round(float(match.group(1)), 2)
            except (TypeError, ValueError):
                pass

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    focus_lines = [
        line
        for line in lines
        if any(token in line.lower() for token in ("usage", "limit", "week", "weekly"))
    ]
    candidates = focus_lines or lines

    for line in candidates:
        for match in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", line):
            try:
                return round(float(match.group(1)), 2)
            except (TypeError, ValueError):
                continue
    return None


def _probe_claude_usage_percent_via_tui(timeout_seconds: float = 10.0) -> float | None:
    """Launch interactive Claude in a PTY, run /usage, and parse % from screen output."""
    master_fd: int | None = None
    slave_fd: int | None = None
    process: subprocess.Popen[Any] | None = None
    collected_chunks: list[str] = []
    sent_usage = False
    sent_exit = False

    try:
        master_fd, slave_fd = os.openpty()
        process = subprocess.Popen(
            ["claude"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
            env=os.environ.copy(),
        )
        os.close(slave_fd)
        slave_fd = None

        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout_seconds:
            if not sent_usage and (time.monotonic() - start_time) > 1.2:
                os.write(master_fd, b"/usage\n")
                sent_usage = True

            readable, _, _ = select.select([master_fd], [], [], 0.2)
            if readable:
                chunk = os.read(master_fd, 8192)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                collected_chunks.append(text)
                if len(collected_chunks) > 120:
                    collected_chunks = collected_chunks[-120:]

                parsed_percent = _extract_usage_percent_from_text("".join(collected_chunks))
                if parsed_percent is not None:
                    return parsed_percent

            if sent_usage and not sent_exit and (time.monotonic() - start_time) > 4.5:
                os.write(master_fd, b"/exit\n")
                sent_exit = True

        return None
    except Exception:
        return None
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except Exception:
                    pass
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
        if slave_fd is not None:
            try:
                os.close(slave_fd)
            except OSError:
                pass


async def _fetch_claude_usage_snapshot(
    force_refresh: bool = False,
    allow_refresh: bool = True,
) -> dict[str, Any]:
    """Get Claude weekly usage percentage via /usage when available."""
    now = time.monotonic()
    cached_payload = _claude_usage_cache.get("payload")
    expires_at = float(_claude_usage_cache.get("expires_at") or 0.0)
    if not force_refresh and cached_payload is not None and now < expires_at:
        return dict(cached_payload)

    if not allow_refresh:
        if isinstance(cached_payload, dict):
            return dict(cached_payload)
        return {
            "claude_usage_supported": None,
            "claude_usage_percent": None,
            "claude_usage_error": "unavailable",
        }

    payload: dict[str, Any] = {
        "claude_usage_supported": None,
        "claude_usage_percent": None,
        "claude_usage_error": None,
    }
    ttl_seconds = CLAUDE_USAGE_CACHE_TTL_SECONDS

    try:
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "/usage",
            "--output-format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )

        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            payload["claude_usage_error"] = "timeout"
            ttl_seconds = 30
        else:
            stdout_text = stdout_raw.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_raw.decode("utf-8", errors="replace").strip()

            if not stdout_text:
                payload["claude_usage_error"] = "empty_output"
                ttl_seconds = 30
            else:
                try:
                    parsed = json.loads(stdout_text)
                except json.JSONDecodeError:
                    result_text = stdout_text
                else:
                    result_text = ""
                    if isinstance(parsed, dict):
                        raw_result = parsed.get("result")
                        if isinstance(raw_result, str):
                            result_text = raw_result

                lowered = result_text.lower()
                if "unknown skill: usage" in lowered:
                    payload["claude_usage_error"] = "unsupported"
                    # Fallback: probe interactive TUI output and parse percentage.
                    async with _claude_usage_probe_lock:
                        tui_percent = await asyncio.to_thread(
                            _probe_claude_usage_percent_via_tui,
                            10.0,
                        )
                    if tui_percent is not None:
                        payload["claude_usage_supported"] = True
                        payload["claude_usage_percent"] = tui_percent
                        payload["claude_usage_error"] = None
                        ttl_seconds = CLAUDE_USAGE_CACHE_TTL_SECONDS
                    else:
                        payload["claude_usage_supported"] = False
                        payload["claude_usage_error"] = "unsupported"
                        ttl_seconds = CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS
                else:
                    usage_percent = _extract_usage_percent_from_text(result_text)
                    payload["claude_usage_supported"] = True
                    payload["claude_usage_percent"] = usage_percent
                    if usage_percent is None:
                        payload["claude_usage_error"] = "percent_not_found"
                        ttl_seconds = 30

                if stderr_text and payload["claude_usage_error"] is None:
                    payload["claude_usage_error"] = f"stderr: {stderr_text[:120]}"
    except FileNotFoundError:
        payload["claude_usage_supported"] = False
        payload["claude_usage_error"] = "claude_not_found"
        ttl_seconds = CLAUDE_USAGE_UNSUPPORTED_TTL_SECONDS
    except Exception as exc:
        payload["claude_usage_error"] = f"exec_error: {exc}"
        ttl_seconds = 30

    _claude_usage_cache["payload"] = dict(payload)
    _claude_usage_cache["expires_at"] = now + float(ttl_seconds)
    return dict(payload)


def _merge_usage_for_display(
    weekly_summary: dict[str, Any],
    claude_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Attach display percent. Only Claude /usage is used for UI display."""
    merged = {**weekly_summary, **claude_snapshot}

    claude_percent = merged.get("claude_usage_percent")
    if isinstance(claude_percent, (int, float)):
        merged["display_usage_percent"] = round(float(claude_percent), 2)
        merged["display_usage_source"] = "claude_usage"
    else:
        merged["display_usage_percent"] = None
        merged["display_usage_source"] = "unavailable"

    return merged


def _resolve_project_path(path_value: str) -> tuple[Path | None, str | None, int | None]:
    """Resolve and validate a project path."""
    raw_path = path_value.strip()
    if not raw_path:
        return None, "Project path is required", 400

    requested_path = Path(raw_path).expanduser()
    if not requested_path.is_absolute():
        return None, "Project path must be absolute (start with /)", 400

    try:
        resolved_path = requested_path.resolve()
    except Exception:
        return None, "Invalid project path", 400

    if not resolved_path.exists():
        return None, f"Project path not found: {resolved_path}", 404

    if not resolved_path.is_dir():
        return None, f"Not a directory: {resolved_path}", 400

    return resolved_path, None, None


def _collect_existing_project_state(
    projects: list[dict[str, Any]],
) -> tuple[set[str], dict[str, str]]:
    """Build lookup maps for existing project names and paths."""
    existing_names: set[str] = set()
    existing_paths: dict[str, str] = {}

    for project in projects:
        project_name = str(project.get("name", "")).strip()
        if project_name:
            existing_names.add(project_name)

        project_path = project.get("path")
        if not isinstance(project_path, str):
            continue

        try:
            resolved = str(Path(project_path).expanduser().resolve())
            existing_paths[resolved] = project_name or resolved
        except Exception:
            continue

    return existing_names, existing_paths


def _prepare_project_payload(
    path_value: str,
    existing_names: set[str],
    existing_paths: dict[str, str],
    requested_name: Optional[str] = None,
    requested_type: Optional[str] = None,
    dev_server: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """Validate input and build DB payload for a project."""
    resolved_path, error, status_code = _resolve_project_path(path_value)
    if resolved_path is None:
        return None, error, status_code

    resolved_key = str(resolved_path)
    existing_owner = existing_paths.get(resolved_key)
    if existing_owner is not None:
        return None, f"Path already registered as project {existing_owner}", 400

    if requested_name is None or not requested_name.strip():
        base_name = _sanitize_project_name(resolved_path.name)
        project_name = _build_unique_project_name(base_name, existing_names)
    else:
        project_name = _sanitize_project_name(requested_name)
        if project_name in existing_names:
            return None, f"Project {project_name} already exists", 400

    payload: dict[str, Any] = {
        "name": project_name,
        "path": resolved_key,
        "type": _normalize_project_type(requested_type, resolved_path),
    }

    if dev_server:
        payload["dev_server"] = dev_server

    existing_names.add(project_name)
    existing_paths[resolved_key] = project_name
    return payload, None, None


def _parse_excluded_dirs(raw_value: Optional[str]) -> set[str]:
    """Parse comma-separated excluded directory names."""
    excluded = set(DEFAULT_SCAN_EXCLUDE_DIRS)
    if raw_value is None:
        return excluded

    for item in raw_value.split(","):
        name = item.strip()
        if name:
            excluded.add(name)
    return excluded


def _detect_project_candidate(path: Path) -> tuple[str, str] | None:
    """Detect whether a directory looks like a project."""
    if (path / "pubspec.yaml").exists():
        return "flutter", "pubspec.yaml"

    package_json = path / "package.json"
    if package_json.exists():
        project_type = _infer_project_type(path)
        if project_type == "other":
            project_type = "web"
        return project_type, "package.json"

    for marker in (
        "next.config.js",
        "next.config.mjs",
        "next.config.ts",
        "next.config.cjs",
    ):
        if (path / marker).exists():
            return "nextjs", marker

    for marker in ("pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml"):
        if (path / marker).exists():
            return "other", marker

    if (path / ".git").is_dir():
        return "other", ".git"

    return None


def _scan_project_candidates(
    root_path: Path,
    excluded_dirs: set[str],
    max_depth: int,
) -> list[dict[str, str]]:
    """Scan root folder and collect candidate project directories."""
    visited: set[str] = set()
    candidates: dict[str, dict[str, str]] = {}
    queue: deque[tuple[Path, int]] = deque([(root_path, 0)])

    while queue and len(candidates) < MAX_SCAN_RESULTS:
        current, depth = queue.popleft()
        current_key = str(current)
        if current_key in visited:
            continue
        visited.add(current_key)

        detected = _detect_project_candidate(current)
        if detected is not None:
            project_type, marker = detected
            candidates[current_key] = {
                "name": current.name or current_key,
                "path": current_key,
                "type": project_type,
                "marker": marker,
            }

        if depth >= max_depth:
            continue

        try:
            children = sorted(
                (child for child in current.iterdir() if child.is_dir()),
                key=lambda child: child.name.lower(),
            )
        except (PermissionError, OSError):
            continue

        for child in children:
            if child.name in excluded_dirs:
                continue
            queue.append((child, depth + 1))

    return sorted(candidates.values(), key=lambda item: item["path"].lower())


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None, alias="api_key"),
):
    """Verify API key from header or query parameter."""
    config = get_config()
    expected_key = config.api_key

    # Empty key = local development (auth disabled)
    if not expected_key:
        return True

    provided_key = x_api_key or api_key
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Code Bridge Server starting...")

    # Migrate projects from config.yaml to database
    config = get_config()
    config.migrate_projects_to_db()

    # Start mDNS service for local network discovery
    mdns_service = None
    if MDNS_AVAILABLE and config.mdns_enabled:
        try:
            mdns_service = create_mdns_service(
                port=config.port,
                server_name=config.server_name,
                api_key=config.api_key if config.api_key else None,
            )
            await mdns_service.start()
            print(f"mDNS service registered: {config.server_name}")
        except Exception as e:
            print(f"Warning: mDNS service failed to start: {e}")

    # Start Cloudflare Tunnel for remote access (if enabled)
    tunnel_service = None
    firebase_auth = None
    if TUNNEL_AVAILABLE and config.remote_access_enabled:
        try:
            # Initialize Firebase auth if available
            if FIREBASE_AVAILABLE and config.firebase_enabled:
                firebase_auth = get_firebase_auth()
                await firebase_auth.initialize()

            # Define callback to update Firebase when tunnel URL changes
            async def on_tunnel_url_change(url: str):
                if firebase_auth and firebase_auth.get_status().get("authenticated"):
                    await firebase_auth.update_tunnel_url(url)
                    print(f"Updated tunnel URL in Firebase: {url}")

            # Create and start tunnel
            tunnel_service = create_tunnel_service(
                local_port=config.port,
                on_url_change=lambda url: asyncio.create_task(on_tunnel_url_change(url)),
            )
            tunnel_url = await tunnel_service.start()
            if tunnel_url:
                print(f"Cloudflare Tunnel started: {tunnel_url}")

                # Register device in Firebase
                if firebase_auth and firebase_auth.get_status().get("authenticated"):
                    await firebase_auth.register_device(tunnel_url)
        except Exception as e:
            print(f"Warning: Remote access setup failed: {e}")

    yield

    # Shutdown
    print("Code Bridge Server shutting down...")

    # Stop tunnel service
    if tunnel_service:
        try:
            await tunnel_service.stop()
        except Exception as e:
            print(f"Warning: Tunnel shutdown error: {e}")

    # Stop mDNS service
    if mdns_service:
        try:
            await mdns_service.stop()
        except Exception as e:
            print(f"Warning: mDNS shutdown error: {e}")

    await get_session_manager().close_all()
    await get_preview_proxy().close()


app = FastAPI(
    title="Code Bridge",
    description="Remote IDE bridge for Claude Code",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Flutter web
_cors_origins = get_config().cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST API Endpoints
# ============================================================================


@app.get("/api/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "ok", "service": "claude-bridge"}


# ============================================================================
# QR Pairing API
# ============================================================================


@app.get("/api/pair/qr")
async def get_pair_qr():
    """Get QR code pairing data.

    Returns QR payload and image data for pairing.
    No auth required - this is used for initial pairing.
    """
    config = get_config()
    pairing = get_pairing_service()

    # Get tunnel URL if available
    tunnel_url = None
    tunnel_service = get_tunnel_service()
    if tunnel_service:
        tunnel_status = tunnel_service.get_status()
        tunnel_url = tunnel_status.get("url")

    # Create pairing data
    pairing_data = pairing.create_pairing_data(
        port=config.port,
        server_name=config.server_name,
        tunnel_url=tunnel_url,
    )

    return {
        "qr_url": pairing_data.to_qr_url(),
        "payload": pairing_data.to_dict(),
        "local_url": pairing_data.local_url,
        "tunnel_url": pairing_data.tunnel_url,
        "expires_in_seconds": pairing_data.expires - int(__import__("time").time()),
    }


@app.post("/api/pair/verify")
async def verify_pair_token(request: PairVerifyRequest):
    """Verify a pairing token and issue an API key.

    No auth required - this is part of the pairing flow.

    Request body:
        pair_token: The pairing token from QR code
        client_id: Optional client identifier
        device_name: Optional device display name

    Returns:
        API key on success, error on failure
    """
    pairing = get_pairing_service()
    result = pairing.verify_pair_token(
        pair_token=request.pair_token,
        client_id=request.client_id,
        device_name=request.device_name,
    )

    if not result.get("success"):
        return JSONResponse(
            status_code=400,
            content={"error": result.get("error", "Pairing failed")},
        )

    return result


@app.get("/api/pair/status", dependencies=[Depends(verify_api_key)])
async def get_pair_status():
    """Get current pairing status.

    Returns list of paired clients and pending tokens.
    """
    pairing = get_pairing_service()
    return pairing.get_pairing_status()


@app.delete("/api/pair/clients/{client_id}", dependencies=[Depends(verify_api_key)])
async def revoke_paired_client(client_id: str):
    """Revoke a paired client's API key.

    Args:
        client_id: The client ID to revoke
    """
    pairing = get_pairing_service()
    success = pairing.revoke_client(client_id)

    if not success:
        return JSONResponse(
            status_code=404,
            content={"error": f"Client {client_id} not found"},
        )

    return {"success": True, "message": f"Client {client_id} revoked"}


@app.get("/api/system/network-status", dependencies=[Depends(verify_api_key)])
async def get_network_status():
    """Get network discovery and remote access status."""
    config = get_config()

    status = {
        "mdns": {
            "available": MDNS_AVAILABLE,
            "enabled": config.mdns_enabled,
            "registered": False,
            "server_name": config.server_name,
        },
        "tunnel": {
            "available": TUNNEL_AVAILABLE,
            "enabled": config.remote_access_enabled,
            "running": False,
            "url": None,
        },
        "firebase": {
            "available": FIREBASE_AVAILABLE,
            "enabled": config.firebase_enabled,
            "authenticated": False,
            "user_id": None,
        },
    }

    # Get mDNS status
    mdns_service = get_mdns_service()
    if mdns_service:
        mdns_status = mdns_service.get_status()
        status["mdns"]["registered"] = mdns_status.get("registered", False)
        status["mdns"]["local_ip"] = mdns_status.get("local_ip")

    # Get tunnel status
    tunnel_service = get_tunnel_service()
    if tunnel_service:
        tunnel_status = tunnel_service.get_status()
        status["tunnel"]["running"] = tunnel_status.get("running", False)
        status["tunnel"]["url"] = tunnel_status.get("url")
        status["tunnel"]["installed"] = tunnel_status.get("installed", False)

    # Get Firebase auth status
    if FIREBASE_AVAILABLE:
        firebase_auth = get_firebase_auth()
        if firebase_auth:
            auth_status = firebase_auth.get_status()
            status["firebase"]["authenticated"] = auth_status.get("authenticated", False)
            status["firebase"]["user_id"] = auth_status.get("user_id")
            status["firebase"]["device_id"] = auth_status.get("device_id")

    return status


@app.post("/api/system/remote-access/login", dependencies=[Depends(verify_api_key)])
async def remote_access_login(request: Request):
    """Authenticate server with Firebase ID token from app.

    The app handles Google/Apple Sign-In and sends the Firebase ID token
    to the server. The server verifies the token and registers the device.

    Request body:
        id_token: Firebase ID token from the app
        register_device: Whether to register device to Firestore (default: True)
    """
    if not FIREBASE_AVAILABLE:
        return JSONResponse(
            status_code=400,
            content={"error": "Firebase is not available"},
        )

    config = get_config()
    if not config.firebase_enabled:
        return JSONResponse(
            status_code=400,
            content={"error": "Firebase is not enabled in config"},
        )

    firebase_auth = get_firebase_auth()
    if not firebase_auth:
        return JSONResponse(
            status_code=500,
            content={"error": "Firebase auth not initialized"},
        )

    # Parse request body
    try:
        body = await request.json()
        id_token = body.get("id_token")
        register_device = body.get("register_device", True)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request body. Expected JSON with 'id_token' field."},
        )

    if not id_token:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'id_token' in request body"},
        )

    # Verify token and authenticate
    success = await firebase_auth.authenticate_with_token(id_token)
    if not success:
        return JSONResponse(
            status_code=401,
            content={"error": "Token verification failed"},
        )

    status = firebase_auth.get_status()

    # Optionally register device to Firestore
    if register_device:
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.tunnel_url if tunnel_service else None
        await firebase_auth.register_device(tunnel_url)

    return {
        "success": True,
        "user_id": status.get("user_id"),
        "device_id": status.get("device_id"),
        "device_name": status.get("device_name"),
    }


@app.post("/api/system/remote-access/logout", dependencies=[Depends(verify_api_key)])
async def remote_access_logout():
    """Sign out from remote access."""
    if not FIREBASE_AVAILABLE:
        return JSONResponse(
            status_code=400,
            content={"error": "Firebase is not available"},
        )

    firebase_auth = get_firebase_auth()
    if firebase_auth:
        await firebase_auth.sign_out()

    return {"success": True}


@app.post("/api/system/tunnel/start", dependencies=[Depends(verify_api_key)])
async def start_tunnel():
    """Manually start Cloudflare Tunnel."""
    if not TUNNEL_AVAILABLE:
        return JSONResponse(
            status_code=400,
            content={"error": "Tunnel service not available"},
        )

    tunnel_service = get_tunnel_service()
    if not tunnel_service:
        config = get_config()
        tunnel_service = create_tunnel_service(local_port=config.port)

    url = await tunnel_service.start()
    if url:
        return {"success": True, "url": url}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to start tunnel"},
        )


@app.post("/api/system/tunnel/stop", dependencies=[Depends(verify_api_key)])
async def stop_tunnel():
    """Stop Cloudflare Tunnel."""
    tunnel_service = get_tunnel_service()
    if tunnel_service:
        await tunnel_service.stop()
        return {"success": True}
    else:
        return {"success": True, "message": "No tunnel running"}


@app.get("/api/projects", dependencies=[Depends(verify_api_key)])
async def list_projects():
    """List all configured projects."""
    manager = get_project_manager()
    return {"projects": manager.get_all_projects()}


@app.get("/api/projects/{name}", dependencies=[Depends(verify_api_key)])
async def get_project(name: str):
    """Get specific project info."""
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    return project


@app.post("/api/projects", dependencies=[Depends(verify_api_key)])
async def create_project(project: ProjectCreate):
    """Create a new project."""
    db = get_project_db()
    existing_projects = db.get_all()
    existing_names, existing_paths = _collect_existing_project_state(existing_projects)
    dev_server = project.dev_server.model_dump(exclude_none=True) if project.dev_server else None

    payload, error, status_code = _prepare_project_payload(
        path_value=project.path,
        existing_names=existing_names,
        existing_paths=existing_paths,
        requested_name=project.name,
        requested_type=project.type,
        dev_server=dev_server,
    )
    if payload is None:
        return JSONResponse(status_code=status_code or 400, content={"error": error or "Invalid project"})

    result = db.create(payload)
    return result


@app.post("/api/projects/import", dependencies=[Depends(verify_api_key)])
async def import_projects(project_import: ProjectImport):
    """Import multiple projects by absolute paths."""
    db = get_project_db()
    if not project_import.paths:
        return JSONResponse(status_code=400, content={"error": "No project paths provided"})

    existing_projects = db.get_all()
    existing_names, existing_paths = _collect_existing_project_state(existing_projects)

    created: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for raw_path in project_import.paths:
        payload, error, status_code = _prepare_project_payload(
            path_value=raw_path,
            existing_names=existing_names,
            existing_paths=existing_paths,
        )

        if payload is None:
            item = {"path": raw_path, "reason": error or "Invalid project path"}
            if status_code == 400 and "already registered as project" in (error or ""):
                skipped.append(item)
            else:
                failed.append(item)
            continue

        try:
            created.append(db.create(payload))
        except Exception as exc:
            failed.append({"path": raw_path, "reason": f"Failed to create: {exc}"})

    return {
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "summary": {
            "created": len(created),
            "skipped": len(skipped),
            "failed": len(failed),
            "requested": len(project_import.paths),
        },
    }


@app.put("/api/projects/{name}", dependencies=[Depends(verify_api_key)])
async def update_project(name: str, project: ProjectUpdate):
    """Update an existing project."""
    db = get_project_db()

    if not db.exists(name):
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    result = db.update(name, project.model_dump(exclude_unset=True))
    return result


@app.delete("/api/projects/{name}", dependencies=[Depends(verify_api_key)])
async def delete_project(name: str):
    """Delete a project."""
    db = get_project_db()

    if not db.exists(name):
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    # Stop dev server if running
    manager = get_project_manager()
    if manager.is_server_running(name):
        await manager.stop_dev_server(name)

    success = db.delete(name)
    if not success:
        return JSONResponse(status_code=500, content={"error": "Failed to delete project"})

    return {"status": "deleted", "name": name}


@app.post("/api/projects/{name}/start", dependencies=[Depends(verify_api_key)])
async def start_dev_server(name: str):
    """Start dev server for project."""
    manager = get_project_manager()
    result = await manager.start_dev_server(name)

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)

    return result


@app.post("/api/projects/{name}/stop", dependencies=[Depends(verify_api_key)])
async def stop_dev_server(name: str):
    """Stop dev server for project."""
    manager = get_project_manager()
    result = await manager.stop_dev_server(name)

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)

    return result


@app.post("/api/projects/{name}/run-device", dependencies=[Depends(verify_api_key)])
async def run_project_on_device(name: str, request: DeviceRunRequest):
    """Run Flutter project on selected Android device and capture logs."""
    manager = get_project_manager()
    result = await manager.run_project_on_device(name, request.device_id)

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/api/projects/{name}/stop-device-run", dependencies=[Depends(verify_api_key)])
async def stop_project_on_device(name: str):
    """Stop running Flutter device process for project."""
    manager = get_project_manager()
    result = await manager.stop_project_on_device(name)

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/api/projects/{name}/device-run-log", dependencies=[Depends(verify_api_key)])
async def get_project_device_run_log(name: str, lines: int = 120):
    """Get captured log tail for the project's latest Flutter device run."""
    manager = get_project_manager()
    return manager.get_device_run_log(name, lines=lines)


@app.post("/api/projects/{name}/build", dependencies=[Depends(verify_api_key)])
async def build_flutter_web(name: str):
    """Build Flutter web app.

    Executes `flutter build web --release` for the specified project.
    Returns build path on success.
    """
    manager = get_project_manager()
    result = await manager.build_flutter_web(name)

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)

    return result


@app.get("/api/projects/{name}/build-status", dependencies=[Depends(verify_api_key)])
async def get_build_status(name: str):
    """Get Flutter web build status.

    Returns current build status: none, building, ready, or error.
    """
    manager = get_project_manager()
    return manager.get_build_status(name)


# ============================================================================
# System File Browser API
# ============================================================================


@app.get("/api/system/directories", dependencies=[Depends(verify_api_key)])
async def list_system_directories(path: Optional[str] = None):
    """List subdirectories for an absolute server path.

    Args:
        path: Absolute directory path on server (default: current user home)

    Returns:
        Current path, parent path, and child directories
    """
    try:
        if path is None or not path.strip():
            target_path = Path.home()
        else:
            requested_path = Path(path).expanduser()
            if not requested_path.is_absolute():
                return JSONResponse(
                    status_code=400,
                    content={"error": "Path must be absolute (start with /)"},
                )
            target_path = requested_path.resolve()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid path"})

    if not target_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Path not found: {target_path}"})

    if not target_path.is_dir():
        return JSONResponse(status_code=400, content={"error": f"Not a directory: {target_path}"})

    try:
        entries = []
        for item in sorted(target_path.iterdir(), key=lambda p: p.name.lower()):
            try:
                if not item.is_dir():
                    continue
                entries.append(
                    {
                        "name": item.name,
                        "path": str(item),
                    }
                )
            except (OSError, PermissionError):
                # Skip inaccessible entries and continue browsing available directories.
                continue
    except (OSError, PermissionError):
        return JSONResponse(
            status_code=403,
            content={"error": f"Cannot access directory: {target_path}"},
        )

    parent_path = str(target_path.parent) if target_path.parent != target_path else None

    return {
        "current_path": str(target_path),
        "parent_path": parent_path,
        "directories": entries,
    }


@app.get("/api/system/project-candidates", dependencies=[Depends(verify_api_key)])
async def list_project_candidates(
    root_path: str,
    exclude_dirs: Optional[str] = None,
    max_depth: int = 1,
):
    """Scan a root directory and return candidate project folders."""
    resolved_root, error, status_code = _resolve_project_path(root_path)
    if resolved_root is None:
        return JSONResponse(status_code=status_code or 400, content={"error": error or "Invalid root path"})

    if max_depth < 0 or max_depth > MAX_SCAN_DEPTH:
        return JSONResponse(
            status_code=400,
            content={"error": f"max_depth must be between 0 and {MAX_SCAN_DEPTH}"},
        )

    excluded = _parse_excluded_dirs(exclude_dirs)
    candidates = _scan_project_candidates(
        root_path=resolved_root,
        excluded_dirs=excluded,
        max_depth=max_depth,
    )
    existing_projects = get_project_db().get_all()
    _, existing_paths = _collect_existing_project_state(existing_projects)

    enriched_candidates = []
    for candidate in candidates:
        path = candidate.get("path", "")
        registered_name = existing_paths.get(path)
        enriched = {
            **candidate,
            "registered": registered_name is not None,
            "registered_project_name": registered_name,
        }
        enriched_candidates.append(enriched)

    return {
        "root_path": str(resolved_root),
        "excluded_dirs": sorted(excluded),
        "candidates": enriched_candidates,
        "count": len(enriched_candidates),
    }


@app.get("/api/system/usage", dependencies=[Depends(verify_api_key)])
async def get_system_usage():
    """Get rolling weekly usage summary and budget percentage."""
    config = get_config()
    usage_db = get_usage_db()
    summary = usage_db.get_weekly_summary(
        budget_usd=config.weekly_budget_usd,
        window_days=config.usage_window_days,
    )
    claude_snapshot = await _fetch_claude_usage_snapshot()
    return _merge_usage_for_display(summary, claude_snapshot)


@app.get("/api/system/llm/options", dependencies=[Depends(verify_api_key)])
async def get_llm_options():
    """Return connectable LLM company/model options and selected value."""
    return get_llm_options_snapshot()


@app.put("/api/system/llm/selection", dependencies=[Depends(verify_api_key)])
async def update_llm_selection(payload: LlmSelectionUpdate):
    """Select active LLM company/model for chat runtime."""
    try:
        return set_selected_llm(payload.company_id, payload.model)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/system/llm/models", dependencies=[Depends(verify_api_key)])
async def add_llm_model(payload: LlmCustomModelUpdate):
    """Add one custom model under a company."""
    try:
        return add_custom_llm_model(payload.company_id, payload.model)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.delete("/api/system/llm/models", dependencies=[Depends(verify_api_key)])
async def delete_llm_model(company_id: str, model: str):
    """Delete one custom model from a company."""
    try:
        return remove_custom_llm_model(company_id, model)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


# ============================================================================
# Session Management
# ============================================================================


@app.post("/api/sessions/{project_name}/close", dependencies=[Depends(verify_api_key)])
async def close_session(project_name: str):
    """Close Claude session for a project.

    Used when switching projects to clean up server-side session state.
    """
    session_manager = get_session_manager()
    await session_manager.close_session(project_name)
    return {"status": "closed", "project": project_name}


# ============================================================================
# File Browser API
# ============================================================================


@app.get("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)])
async def list_files(name: str, path: str = ""):
    """List directory contents for a project.

    Args:
        name: Project name
        path: Relative path within project (default: root)

    Returns:
        List of file/directory entries
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.list_directory(path)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.get("/api/projects/{name}/files/content", dependencies=[Depends(verify_api_key)])
async def read_file_content(name: str, path: str):
    """Read file content for a project.

    Args:
        name: Project name
        path: Relative file path within project

    Returns:
        File content with language detection
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.read_file(path)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.put("/api/projects/{name}/files/content", dependencies=[Depends(verify_api_key)])
async def write_file_content(name: str, file_data: FileWrite):
    """Write content to a file.

    Args:
        name: Project name
        file_data: File path and content to write

    Returns:
        Success status with file info
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.write_file(file_data.path, file_data.content, file_data.create_dirs)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.post("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)])
async def create_file_or_directory(name: str, file_data: FileCreate):
    """Create a new file or directory.

    Args:
        name: Project name
        file_data: File/directory path and optional content

    Returns:
        Success status
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)

    if file_data.is_directory:
        result = file_manager.create_directory(file_data.path)
    else:
        result = file_manager.write_file(file_data.path, file_data.content or "", create_dirs=True)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.delete("/api/projects/{name}/files", dependencies=[Depends(verify_api_key)])
async def delete_file(name: str, path: str, recursive: bool = False):
    """Delete a file or directory.

    Args:
        name: Project name
        path: Relative file path within project
        recursive: If true, delete directories recursively

    Returns:
        Success status
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)

    # Try file first, then directory if recursive
    result = file_manager.delete_file(path)
    if "error" in result and "Not a file" in result["error"] and recursive:
        result = file_manager.delete_directory(path)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.post("/api/projects/{name}/files/rename", dependencies=[Depends(verify_api_key)])
async def rename_file(name: str, old_path: str, new_path: str):
    """Rename or move a file/directory.

    Args:
        name: Project name
        old_path: Current relative path
        new_path: New relative path

    Returns:
        Success status
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.rename_file(old_path, new_path)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.post("/api/projects/{name}/files/copy", dependencies=[Depends(verify_api_key)])
async def copy_file(name: str, source: str, dest: str):
    """Copy a file or directory.

    Args:
        name: Project name
        source: Source relative path
        dest: Destination relative path

    Returns:
        Success status
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.copy_file(source, dest)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.post("/api/projects/{name}/files/move", dependencies=[Depends(verify_api_key)])
async def move_file(name: str, source: str, dest: str):
    """Move a file or directory (alias for rename).

    Args:
        name: Project name
        source: Source relative path
        dest: Destination relative path

    Returns:
        Success status
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.rename_file(source, dest)

    if "error" in result:
        return JSONResponse(status_code=result.get("code", 400), content={"error": result["error"]})

    return result


@app.get("/api/projects/{name}/files/search", dependencies=[Depends(verify_api_key)])
async def search_files(name: str, q: str, limit: int = 50):
    """Search files in a project.

    Args:
        name: Project name
        q: Search query
        limit: Maximum number of results

    Returns:
        List of matching files
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.search_files(q, limit=limit)

    return result


@app.get("/api/projects/{name}/files/search-content", dependencies=[Depends(verify_api_key)])
async def search_file_content(name: str, q: str, limit: int = 100, case_sensitive: bool = False):
    """Search file contents in a project.

    Args:
        name: Project name
        q: Search query
        limit: Maximum number of results
        case_sensitive: Whether search is case-sensitive

    Returns:
        List of matching lines with context
    """
    manager = get_project_manager()
    project = manager.get_project(name)

    if project is None:
        return JSONResponse(status_code=404, content={"error": f"Project {name} not found"})

    project_path = project.get("path")
    if not project_path:
        return JSONResponse(status_code=400, content={"error": "Project has no path configured"})

    file_manager = get_file_manager(project_path)
    result = file_manager.search_content(q, limit=limit, case_sensitive=case_sensitive)

    return result


# ============================================================================
# Android Device & Scrcpy API
# ============================================================================


@app.get("/api/devices", dependencies=[Depends(verify_api_key)])
async def list_devices():
    """List connected Android devices.

    Returns list of devices with id, model, and connection state.
    """
    scrcpy = get_scrcpy_manager()
    devices = await scrcpy.get_devices()
    return {"devices": devices}


@app.get("/api/scrcpy/status", dependencies=[Depends(verify_api_key)])
async def scrcpy_status():
    """Get ws-scrcpy server status.

    Returns running state, URL, and installation status.
    """
    scrcpy = get_scrcpy_manager()
    return scrcpy.get_status()


@app.post("/api/scrcpy/start", dependencies=[Depends(verify_api_key)])
async def start_scrcpy():
    """Start ws-scrcpy server.

    Returns URL for accessing the mirror interface.
    """
    scrcpy = get_scrcpy_manager()
    result = await scrcpy.start()

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)

    return result


@app.post("/api/scrcpy/stop", dependencies=[Depends(verify_api_key)])
async def stop_scrcpy():
    """Stop ws-scrcpy server."""
    scrcpy = get_scrcpy_manager()
    result = await scrcpy.stop()

    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)

    return result


# ============================================================================
# Preview Proxy
# ============================================================================

# Track last previewed project for _next asset routing
_last_previewed_project: dict[str, str] = {}  # Maps session/origin to project name


@app.api_route("/preview/{name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def preview_proxy(request: Request, name: str, path: str = ""):
    """Proxy requests to dev server."""
    global _last_previewed_project

    manager = get_project_manager()
    port = manager.get_server_port(name)

    if port is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No running dev server found for project {name}"},
        )

    # Track this project as last previewed (for _next routing)
    _last_previewed_project["current"] = name

    proxy = get_preview_proxy()
    return await proxy.proxy_request(request, port, path)


async def _proxy_from_last_preview_project(request: Request, path: str):
    """Proxy root-level asset requests for the most recently previewed project."""
    project_name = _last_previewed_project.get("current")
    if not project_name:
        return JSONResponse(
            status_code=404,
            content={"error": "No active preview session"},
        )

    manager = get_project_manager()
    port = manager.get_server_port(project_name)

    if port is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No dev server running for {project_name}"},
        )

    proxy = get_preview_proxy()
    return await proxy.proxy_request(request, port, path)


@app.api_route("/_next/{path:path}", methods=["GET", "POST"])
async def next_assets_proxy(request: Request, path: str = ""):
    """Proxy Next.js asset requests to the active dev server.

    Next.js apps request assets from /_next/static/... which need to be
    proxied to the correct dev server.
    """
    global _last_previewed_project

    # Get the last previewed project
    project_name = _last_previewed_project.get("current")
    if not project_name:
        return JSONResponse(
            status_code=404,
            content={"error": "No active preview session"},
        )

    manager = get_project_manager()
    port = manager.get_server_port(project_name)

    if port is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No dev server running for {project_name}"},
        )

    proxy = get_preview_proxy()
    return await proxy.proxy_request(request, port, f"_next/{path}")


@app.api_route("/@{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_at_assets_proxy(request: Request, path: str = ""):
    """Proxy root-level @* assets used by Vite and similar dev servers."""
    normalized_path = f"@{path}"
    return await _proxy_from_last_preview_project(request, normalized_path)


@app.api_route("/src/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_src_assets_proxy(request: Request, path: str = ""):
    """Proxy root-level src assets for dev servers."""
    return await _proxy_from_last_preview_project(request, f"src/{path}")


@app.api_route("/node_modules/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_node_modules_proxy(request: Request, path: str = ""):
    """Proxy root-level node_modules assets for dev servers."""
    return await _proxy_from_last_preview_project(request, f"node_modules/{path}")


@app.api_route("/assets/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def root_assets_proxy(request: Request, path: str = ""):
    """Proxy root-level assets folder for dev servers."""
    return await _proxy_from_last_preview_project(request, f"assets/{path}")


@app.api_route("/{filename}", methods=["GET"])
async def root_file_proxy(request: Request, filename: str):
    """Proxy common root-level files requested by dev server HTML."""
    if "/" in filename:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    allowed_root_files = {
        "favicon.ico",
        "vite.svg",
        "manifest.json",
        "robots.txt",
        "sitemap.xml",
    }
    if filename not in allowed_root_files:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return await _proxy_from_last_preview_project(request, filename)


# ============================================================================
# Build Preview (Static Files)
# ============================================================================


# MIME type mapping for common web files
MIME_TYPES = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".eot": "application/vnd.ms-fontobject",
    ".wasm": "application/wasm",
}


@app.get("/build-preview/{name}/{path:path}")
async def build_preview(name: str, path: str = ""):
    """Serve static files from web build (Flutter or Next.js).

    Supports:
    - Flutter: build/web/ with base href rewriting
    - Next.js static export: out/ directory
    - Next.js default: .next/ (limited support, some features may not work)

    Falls back to index.html for SPA routing.
    """
    manager = get_project_manager()
    build_path = manager.get_build_path(name)
    build_status = manager.get_build_status(name)
    project_type = build_status.get("project_type", "flutter")

    if not build_path:
        return JSONResponse(
            status_code=404,
            content={"error": f"No build available for project {name}. Run build first."},
        )

    build_dir = Path(build_path)
    if not build_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Build directory not found: {build_path}"},
        )

    # Determine file path
    file_path = build_dir / (path or "index.html")

    # If path doesn't exist, try index.html (SPA fallback)
    if not file_path.exists() or file_path.is_dir():
        if file_path.is_dir():
            file_path = file_path / "index.html"
        if not file_path.exists():
            file_path = build_dir / "index.html"

    # For Next.js .next directory, handle special cases
    if project_type == "nextjs" and not file_path.exists():
        # Try _next/static paths
        if path.startswith("_next/"):
            file_path = build_dir / path
        # Try static files in public
        elif not path.startswith("_next"):
            public_file = build_dir.parent / "public" / path
            if public_file.exists():
                file_path = public_file

    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {path}"},
        )

    # Security check: ensure file is within allowed directories
    try:
        resolved = file_path.resolve()
        # Allow build directory and parent (for public folder access)
        allowed_base = build_dir.resolve().parent if project_type == "nextjs" else build_dir.resolve()
        resolved.relative_to(allowed_base)
    except ValueError:
        return JSONResponse(
            status_code=403,
            content={"error": "Access denied"},
        )

    # Determine content type
    suffix = file_path.suffix.lower()
    media_type = MIME_TYPES.get(suffix, "application/octet-stream")

    # For index.html, apply project-specific modifications
    if file_path.name == "index.html":
        content = file_path.read_text()
        correct_base = f"/build-preview/{name}/"

        if project_type == "flutter":
            # Flutter: Replace base href
            content = content.replace('<base href="/">', f'<base href="{correct_base}">')
        elif project_type == "nextjs":
            # Next.js static export: Fix asset paths if needed
            # Usually Next.js handles this correctly with assetPrefix
            pass

        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=content)

    return FileResponse(file_path, media_type=media_type)


# ============================================================================
# WebSocket - Claude Code Session
# ============================================================================


def _format_tool_result_content(content: Any) -> str:
    """Format tool_result content from Claude event payload."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue

            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_value = item.get("text", "").strip()
                    if text_value:
                        parts.append(text_value)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
                continue

            parts.append(str(item))
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content.get("text", "").strip()
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _extract_assistant_text(message: dict[str, Any]) -> str:
    """Extract text blocks from an assistant message payload."""
    blocks = message.get("content")
    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


async def _stream_claude_turn(
    websocket: WebSocket,
    session: ClaudeSession,
    project_name: str,
    user_message: str | None = None,
    retry_from_permission: bool = False,
    deny_from_permission_message: str | None = None,
) -> bool:
    """Stream one Claude turn and forward rich events to websocket client."""
    full_response_chunks: list[str] = []
    fallback_response = ""
    seen_tool_use_ids: set[str] = set()
    turn_completed = False

    async def emit_tool_use(
        tool_id: Any,
        tool_name: Any,
        tool_input: Any,
    ) -> None:
        resolved_id = tool_id if isinstance(tool_id, str) else None
        if resolved_id is not None:
            if resolved_id in seen_tool_use_ids:
                return
            seen_tool_use_ids.add(resolved_id)

        await websocket.send_json(
            {
                "type": "tool_use",
                "id": resolved_id,
                "name": tool_name,
                "input": tool_input if isinstance(tool_input, dict) else {},
            }
        )

    if deny_from_permission_message is not None:
        event_stream = session.deny_pending_permissions(message=deny_from_permission_message)
    elif retry_from_permission:
        event_stream = session.approve_pending_permissions_and_retry()
    else:
        if user_message is None:
            raise ValueError("user_message is required when retry_from_permission is False")
        event_stream = session.send_message(user_message)

    async for event in event_stream:
        event_type = event.get("type")

        if event_type == "stream_event":
            stream_event = event.get("event", {})
            if not isinstance(stream_event, dict):
                continue

            stream_type = stream_event.get("type")
            if stream_type == "content_block_start":
                content_block = stream_event.get("content_block", {})
                if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                    await emit_tool_use(
                        content_block.get("id"),
                        content_block.get("name"),
                        content_block.get("input"),
                    )
                continue

            if stream_type == "content_block_delta":
                delta = stream_event.get("delta", {})
                if not isinstance(delta, dict):
                    continue

                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if isinstance(text, str) and text:
                        full_response_chunks.append(text)
                        await websocket.send_json({"type": "stream", "content": text})
                elif delta_type == "input_json_delta":
                    partial_json = delta.get("partial_json", "")
                    if isinstance(partial_json, str) and partial_json:
                        await websocket.send_json(
                            {
                                "type": "tool_input_delta",
                                "content": partial_json,
                                "index": stream_event.get("index"),
                            }
                        )
                continue

            continue

        if event_type == "assistant":
            message_payload = event.get("message", {})
            if isinstance(message_payload, dict):
                blocks = message_payload.get("content", [])
                if isinstance(blocks, list):
                    for block in blocks:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        await emit_tool_use(
                            block.get("id"),
                            block.get("name"),
                            block.get("input"),
                        )

                if not full_response_chunks and not fallback_response:
                    fallback_response = _extract_assistant_text(message_payload)
            continue

        if event_type == "user":
            message_payload = event.get("message", {})
            if not isinstance(message_payload, dict):
                continue

            blocks = message_payload.get("content", [])
            if not isinstance(blocks, list):
                continue

            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue

                await websocket.send_json(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id"),
                        "is_error": bool(block.get("is_error", False)),
                        "content": _format_tool_result_content(block.get("content")),
                    }
                )
            continue

        if event_type == "result":
            turn_completed = True
            if not fallback_response:
                result_text = event.get("result")
                if isinstance(result_text, str):
                    fallback_response = result_text

            usage = event.get("usage")
            usage_dict = usage if isinstance(usage, dict) else {}
            model_usage = event.get("modelUsage")
            if not isinstance(model_usage, dict):
                model_usage = event.get("model_usage")
            model_usage_dict = model_usage if isinstance(model_usage, dict) else {}
            total_cost = event.get("total_cost_usd")
            if isinstance(total_cost, (int, float)):
                total_cost_usd = float(total_cost)
            else:
                try:
                    total_cost_usd = float(str(total_cost))
                except (TypeError, ValueError):
                    total_cost_usd = 0.0
            input_tokens = usage_dict.get("input_tokens")
            output_tokens = usage_dict.get("output_tokens")
            try:
                input_tokens_int = int(input_tokens) if input_tokens is not None else 0
            except (TypeError, ValueError):
                input_tokens_int = 0
            try:
                output_tokens_int = int(output_tokens) if output_tokens is not None else 0
            except (TypeError, ValueError):
                output_tokens_int = 0

            await websocket.send_json(
                {
                    "type": "turn_metrics",
                    "duration_ms": event.get("duration_ms"),
                    "duration_api_ms": event.get("duration_api_ms"),
                    "num_turns": event.get("num_turns"),
                    "total_cost_usd": total_cost_usd,
                    "usage": usage_dict,
                    "model_usage": model_usage_dict,
                }
            )

            try:
                usage_db = get_usage_db()
                usage_db.record_turn(
                    project_name=project_name,
                    cost_usd=total_cost_usd,
                    input_tokens=input_tokens_int,
                    output_tokens=output_tokens_int,
                )
                config = get_config()
                weekly_summary = usage_db.get_weekly_summary(
                    budget_usd=config.weekly_budget_usd,
                    window_days=config.usage_window_days,
                )
                claude_snapshot = await _fetch_claude_usage_snapshot()
                merged_usage = _merge_usage_for_display(weekly_summary, claude_snapshot)
                await websocket.send_json({"type": "weekly_usage", **merged_usage})
            except Exception as exc:
                await websocket.send_json(
                    {
                        "type": "claude_event",
                        "event": {
                            "type": "system",
                            "subtype": "status",
                            "status": f"Usage summary update failed: {exc}",
                        },
                    }
                )
            continue

        if event_type == "control_request":
            request = event.get("request", {})
            if not isinstance(request, dict):
                await websocket.send_json({"type": "claude_event", "event": event})
                continue

            if request.get("subtype") == "can_use_tool":
                tool_name = request.get("tool_name")
                tool_input = request.get("input")
                request_id = event.get("request_id")
                tool_use_id = request.get("tool_use_id")

                denials = [
                    {
                        "request_id": request_id,
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    }
                ]
                await websocket.send_json(
                    {
                        "type": "permission_required",
                        "denials": denials,
                        "request_id": request_id,
                        "message": (
                            f"Tool '{tool_name}' requires approval to continue."
                            if isinstance(tool_name, str) and tool_name
                            else "A tool requires approval to continue."
                        ),
                    }
                )
                return False

            # Other control requests are forwarded for debug visibility.
            await websocket.send_json({"type": "claude_event", "event": event})
            continue

        if event_type == "error":
            error_payload = event.get("error")
            if isinstance(error_payload, dict):
                error_message = str(error_payload.get("message", "Unknown error"))
            else:
                error_message = str(error_payload or "Unknown error")
            await websocket.send_json({"type": "error", "message": error_message})
            continue

        if event_type == "output":
            text = event.get("text")
            if isinstance(text, str) and text:
                await websocket.send_json({"type": "status", "message": text})
            continue

        # Forward unknown Claude events for client-side visibility/debugging.
        await websocket.send_json({"type": "claude_event", "event": event})

    if not turn_completed:
        return False

    final_response = "".join(full_response_chunks).strip()
    if not final_response:
        final_response = fallback_response.strip()

    await websocket.send_json({"type": "complete", "content": final_response})
    return True


@app.websocket("/ws/claude/{project_name}")
async def claude_websocket(
    websocket: WebSocket,
    project_name: str,
    api_key: Optional[str] = Query(None),
):
    """WebSocket endpoint for Claude Code communication."""
    # Verify API key
    config = get_config()
    expected_key = config.api_key

    if expected_key and api_key != expected_key:
        await websocket.close(code=4001, reason="Invalid API Key")
        return

    await websocket.accept()

    project = config.get_project(project_name)

    if project is None:
        await websocket.send_json({"type": "error", "message": f"Project {project_name} not found"})
        await websocket.close()
        return

    project_path = project.get("path")
    session_manager = get_session_manager()

    try:
        # Send connection status
        await websocket.send_json({"type": "status", "message": "Connecting to Claude Code..."})

        llm_snapshot = get_llm_options_snapshot()
        selected = llm_snapshot.get("selected") if isinstance(llm_snapshot, dict) else {}
        selected_company = selected.get("company_id") if isinstance(selected, dict) else None
        selected_model = selected.get("model") if isinstance(selected, dict) else None

        if selected_company and selected_company != "anthropic":
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        f"Selected LLM company '{selected_company}' is not supported by this chat backend yet. "
                        "Select Anthropic in Settings > LLM Configuration."
                    ),
                }
            )
            await websocket.close()
            return

        # Get or create Claude session
        session = await session_manager.get_or_create_session(
            project_name,
            project_path,
            model=selected_model if isinstance(selected_model, str) and selected_model.strip() else None,
        )

        await websocket.send_json({"type": "status", "message": "Connected to Claude Code"})

        # Message handling loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "message":
                    user_message_raw = message.get("content", "")
                    user_message = str(user_message_raw).strip()
                    if not user_message:
                        await websocket.send_json({"type": "error", "message": "Message content is empty"})
                        continue

                    # Send user message acknowledgment
                    await websocket.send_json({"type": "user_message", "content": user_message})

                    # Stream Claude's response
                    await _stream_claude_turn(
                        websocket,
                        session,
                        project_name=project_name,
                        user_message=user_message,
                    )

                elif message.get("type") == "approve_permissions":
                    if not session.has_pending_permission_denials:
                        await websocket.send_json(
                            {"type": "error", "message": "No pending permission request to approve"}
                        )
                        continue

                    await websocket.send_json(
                        {
                            "type": "permission_retry_started",
                            "message": "Permission approved. Continuing current turn...",
                        }
                    )
                    await _stream_claude_turn(
                        websocket,
                        session,
                        project_name=project_name,
                        retry_from_permission=True,
                    )

                elif message.get("type") == "deny_permissions":
                    if not session.has_pending_permission_denials:
                        await websocket.send_json(
                            {"type": "error", "message": "No pending permission request to deny"}
                        )
                        continue

                    deny_message_raw = message.get("message", "Permission denied by user.")
                    deny_message = (
                        str(deny_message_raw).strip()
                        if isinstance(deny_message_raw, str)
                        else "Permission denied by user."
                    )
                    if not deny_message:
                        deny_message = "Permission denied by user."

                    await websocket.send_json(
                        {
                            "type": "permission_retry_started",
                            "message": "Permission denied. Continuing current turn...",
                        }
                    )
                    await _stream_claude_turn(
                        websocket,
                        session,
                        project_name=project_name,
                        deny_from_permission_message=deny_message,
                    )

                elif message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown message type: {message.get('type')}"}
                    )

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})

    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Optionally keep session alive for reconnection
        # await session_manager.close_session(project_name)
        pass


# ============================================================================
# Main
# ============================================================================


# ============================================================================
# Flutter Web App Serving (for iPad/Safari access)
# ============================================================================

# Path to Flutter web build
FLUTTER_WEB_BUILD = Path(__file__).parent.parent / "build" / "web"


@app.get("/app/{path:path}")
async def serve_flutter_app(path: str = ""):
    """Serve Flutter web app for iPad/Safari access.

    Access the app at: http://<server-ip>:8080/app/
    """
    if not FLUTTER_WEB_BUILD.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "Flutter web build not found",
                "hint": "Run 'flutter build web' first",
            },
        )

    # Determine file path
    file_path = FLUTTER_WEB_BUILD / (path or "index.html")

    # SPA fallback: if path doesn't exist, serve index.html
    if not file_path.exists() or file_path.is_dir():
        if file_path.is_dir():
            file_path = file_path / "index.html"
        if not file_path.exists():
            file_path = FLUTTER_WEB_BUILD / "index.html"

    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {path}"},
        )

    # Security check
    try:
        resolved = file_path.resolve()
        resolved.relative_to(FLUTTER_WEB_BUILD.resolve())
    except ValueError:
        return JSONResponse(
            status_code=403,
            content={"error": "Access denied"},
        )

    # Serve index.html with correct base href
    if file_path.name == "index.html":
        content = file_path.read_text()
        content = content.replace('<base href="/">', '<base href="/app/">')
        return HTMLResponse(content=content)

    # Determine content type
    suffix = file_path.suffix.lower()
    media_type = MIME_TYPES.get(suffix, "application/octet-stream")

    return FileResponse(file_path, media_type=media_type)


def show_pairing_qr() -> None:
    """Display QR code for pairing in terminal."""
    if not QRCODE_AVAILABLE:
        print("\n[Error] QR code library not installed.")
        print("Run: pip install qrcode[pil]\n")
        return

    config = get_config()
    pairing = get_pairing_service()

    # Get tunnel URL if available (tunnel may not be started yet in CLI mode)
    tunnel_url = None
    tunnel_service = get_tunnel_service()
    if tunnel_service:
        tunnel_status = tunnel_service.get_status()
        tunnel_url = tunnel_status.get("url")

    # Create pairing data
    pairing_data = pairing.create_pairing_data(
        port=config.port,
        server_name=config.server_name,
        tunnel_url=tunnel_url,
    )

    # Display QR code
    display_pairing_qr(
        qr_url=pairing_data.to_qr_url(),
        local_url=pairing_data.local_url,
        tunnel_url=pairing_data.tunnel_url,
        server_name=config.server_name,
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Code Bridge Server")
    parser.add_argument(
        "--show-qr",
        action="store_true",
        help="Display QR code for pairing before starting server",
    )
    parser.add_argument(
        "--qr-only",
        action="store_true",
        help="Only display QR code (don't start server)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to",
    )
    args = parser.parse_args()

    config = get_config()

    # Override config with CLI args
    host = args.host or config.host
    port = args.port or config.port

    if args.qr_only:
        # Just show QR and exit
        show_pairing_qr()
    else:
        if args.show_qr:
            # Show QR code before starting
            show_pairing_qr()

        # Start server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=config.debug,
            log_level=config.log_level,
        )
