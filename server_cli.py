"""CLI bootstrap for the Code Bridge server."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import webbrowser
from pathlib import Path

import uvicorn

from core.config import get_config
from core.env_bootstrap import bootstrap_path
from core.runtime_paths import SERVER_DIR, runtime_path

logger = logging.getLogger(__name__)

PID_FILE = runtime_path(".server.pid", SERVER_DIR / ".server.pid")


def _get_running_server_pid() -> int | None:
    """Check if a server is already running and return its PID."""
    if not PID_FILE.exists():
        return None

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process is still running
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file is stale or process not running
        PID_FILE.unlink(missing_ok=True)
        return None


def _write_pid_file() -> None:
    """Write current process PID to file."""
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid_file() -> None:
    """Remove PID file on shutdown."""
    PID_FILE.unlink(missing_ok=True)


def _stop_existing_server(pid: int) -> bool:
    """Stop existing server process."""
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to terminate
        import time

        for _ in range(30):  # Wait up to 3 seconds
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                PID_FILE.unlink(missing_ok=True)
                return True
        # Force kill if still running
        os.kill(pid, signal.SIGKILL)
        PID_FILE.unlink(missing_ok=True)
        return True
    except (ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return True


def _check_and_handle_duplicate(auto_restart: bool = False) -> bool:
    """Check for duplicate server and handle it.

    Args:
        auto_restart: If True, automatically stop existing server without prompting.

    Returns True if we should continue starting, False to abort.
    """
    existing_pid = _get_running_server_pid()
    if existing_pid is None:
        return True

    if auto_restart:
        print(f"⚠️  기존 서버(PID: {existing_pid})를 종료하고 재시작합니다...")
        if _stop_existing_server(existing_pid):
            print("✅ 기존 서버가 종료되었습니다.\n")
            return True
        else:
            print("❌ 기존 서버 종료에 실패했습니다.\n")
            return False

    print(f"\n⚠️  서버가 이미 실행 중입니다 (PID: {existing_pid})")
    print("   이전 서버를 종료하고 새로 시작할까요?")

    try:
        response = input("   [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            print(f"   기존 서버(PID: {existing_pid})를 종료합니다...")
            if _stop_existing_server(existing_pid):
                print("   ✅ 기존 서버가 종료되었습니다.\n")
                return True
            else:
                print("   ❌ 기존 서버 종료에 실패했습니다.\n")
                return False
        else:
            print("   서버 시작을 취소합니다.\n")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n   서버 시작을 취소합니다.\n")
        return False
from pairing.mdns_service import start_mdns_service, stop_mdns_service
from system.optional_services import get_active_tunnel_url
from pairing.pairing_qr_service import (
    build_pairing_qr_payload_for_current_server,
    display_pairing_qr_payload,
    open_pairing_page,
)
from pairing.qr_display import QRCODE_AVAILABLE
from remote.remote_access_service import (
    start_tunnel_for_current_server,
    stop_tunnel_for_current_server,
)
from remote.tunnel_service import TunnelService, get_tunnel_service


def show_pairing_qr(open_browser: bool = True) -> None:
    """Display pairing QR in terminal and optionally open pairing page."""
    if not QRCODE_AVAILABLE:
        logger.error("QR code library not installed. Run: pip install qrcode[pil]")
        return

    payload = build_pairing_qr_payload_for_current_server(
        tunnel_url=get_active_tunnel_url(),
    )
    display_pairing_qr_payload(payload)

    if open_browser:
        print(f"\n  Opening browser: {payload.pair_url}\n")
        open_pairing_page(payload.pair_url, opener=webbrowser.open)


async def open_dashboard_after_delay(port: int, delay: float = 2.0) -> None:
    """Open dashboard in browser after a short delay."""
    await asyncio.sleep(delay)
    dashboard_url = f"http://localhost:{port}/dashboard"
    print(f"\n  Opening dashboard: {dashboard_url}\n")
    webbrowser.open(dashboard_url)


def _check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_available_port(host: str, start_port: int, max_tries: int = 20) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_tries):
        port = start_port + offset
        if _check_port_available(host, port):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_tries - 1}")


async def _sync_server_url_to_firebase(api_port: int) -> None:
    """Sync server URL to Firebase if authenticated.

    This ensures that when the server starts (possibly on a new port),
    the app can discover the correct URL via Firebase.
    """
    from system.optional_services import FIREBASE_AVAILABLE, get_firebase_auth, get_tunnel_service
    from pairing.pairing import get_pairing_service

    if not FIREBASE_AVAILABLE:
        return

    firebase_auth = get_firebase_auth()
    if not firebase_auth:
        return

    status = firebase_auth.get_status()
    if not status.get("authenticated"):
        logger.debug("Firebase not authenticated, skipping URL sync")
        return

    # Build URLs
    pairing = get_pairing_service()
    local_url = f"http://{pairing.get_local_ip()}:{api_port}"

    tunnel_url = None
    tunnel_service = get_tunnel_service()
    if tunnel_service and tunnel_service.is_running:
        tunnel_url = tunnel_service.tunnel_url

    # Register to Firebase
    try:
        registered = await firebase_auth.register_device(tunnel_url, local_url)
        if registered:
            logger.info("Server URL synced to Firebase: local=%s, tunnel=%s", local_url, tunnel_url)
        else:
            logger.warning("Failed to sync server URL to Firebase")
    except Exception as e:
        logger.error("Error syncing server URL to Firebase: %s", e)


async def run_dual_servers() -> None:
    """Run Dashboard and API servers concurrently.

    - Dashboard server: 127.0.0.1:dashboard_port (localhost only)
    - API server: 0.0.0.0:api_port (tunnel-exposed)
    """
    from main import api_app, dashboard_app
    from pairing.pairing import get_pairing_service

    config = get_config()

    # Auto-find available ports if default ports are in use
    dashboard_port = _find_available_port("127.0.0.1", config.dashboard_port)
    if dashboard_port != config.dashboard_port:
        logger.info("Dashboard port %d in use, using %d", config.dashboard_port, dashboard_port)
        config.set_runtime_port(dashboard_port)

    api_port = _find_available_port("0.0.0.0", config.api_port)
    if api_port != config.api_port:
        logger.info("API port %d in use, using %d", config.api_port, api_port)
        config.set_runtime_api_port(api_port)

    # Start mDNS service for local network discovery
    pairing_service = get_pairing_service()
    await start_mdns_service(
        server_id=pairing_service.server_id,
        server_name=config.server_name,
        api_port=api_port,
        dashboard_port=dashboard_port,
    )

    # Sync server URL to Firebase if authenticated
    await _sync_server_url_to_firebase(api_port)

    dashboard_config = uvicorn.Config(
        app=dashboard_app,
        host=config.dashboard_host,  # localhost only, not tunnel-exposed
        port=dashboard_port,
        log_level=config.log_level,
        access_log=False,
        # No reload in dual-server mode (not supported with programmatic run)
    )

    api_config = uvicorn.Config(
        app=api_app,
        host=config.api_host,  # External access via tunnel
        port=api_port,
        log_level=config.log_level,
        access_log=False,
    )

    dashboard_server = uvicorn.Server(dashboard_config)
    api_server = uvicorn.Server(api_config)

    print(f"Starting Dashboard server on http://127.0.0.1:{dashboard_port}")
    print(f"Starting API server on http://0.0.0.0:{api_port}")

    # Write PID file for duplicate detection
    _write_pid_file()

    try:
        # Run both servers concurrently, open dashboard after startup
        await asyncio.gather(
            dashboard_server.serve(),
            api_server.serve(),
            open_dashboard_after_delay(dashboard_port),
        )
    finally:
        _remove_pid_file()
        await stop_mdns_service()


async def tunnel_start() -> None:
    """Start Cloudflare tunnel for API server.

    Uses shared service to ensure Firebase gets updated with new tunnel URL.
    Port is determined by config.api_port.
    """
    if not TunnelService.is_cloudflared_installed():
        logger.error("cloudflared not installed. Install with: brew install cloudflared")
        return

    config = get_config()
    logger.info("Starting tunnel for localhost:%d...", config.api_port)
    # Use shared service (updates Firebase automatically)
    result = await start_tunnel_for_current_server()
    if result.success and result.url:
        print(f"\n✅ Tunnel started successfully!")
        print(f"   URL: {result.url}")
        print(f"   Firebase: URL registered for remote discovery\n")
    else:
        print(f"\n❌ Failed to start tunnel")
        if result.error:
            print(f"   Error: {result.error}\n")
        else:
            print()


async def tunnel_stop() -> None:
    """Stop the running Cloudflare tunnel.

    Uses shared service for consistency with dashboard/API.
    """
    result = await stop_tunnel_for_current_server()
    if result.success:
        print("\n✅ Tunnel stopped\n")
    else:
        print(f"\n[Info] {result.error or 'No tunnel is running'}\n")


def tunnel_status() -> None:
    """Show tunnel status."""
    if not TunnelService.is_cloudflared_installed():
        logger.warning("cloudflared not installed. Install with: brew install cloudflared")
        return

    tunnel = get_tunnel_service()
    if tunnel:
        status = tunnel.get_status()
        print(f"\n[Tunnel Status]")
        print(f"  Running: {status['running']}")
        print(f"  URL: {status['url'] or 'N/A'}")
        print(f"  Local Port: {status['local_port']}\n")
    else:
        print("\n[Status] Tunnel service not initialized")
        print("Use 'tunnel start <port>' to start\n")


def main() -> None:
    """Parse CLI flags and run the uvicorn server."""
    # Must run before anything that calls shutil.which() (CLI detection,
    # cloudflared, adb, mmdc): launchd/login-item processes start with a
    # bare PATH that hides every Homebrew/npm/Android SDK install.
    bootstrap_path()

    parser = argparse.ArgumentParser(description="Code Bridge Server")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Tunnel subcommand
    tunnel_parser = subparsers.add_parser("tunnel", help="Manage Cloudflare tunnel")
    tunnel_subparsers = tunnel_parser.add_subparsers(dest="tunnel_action")

    # tunnel start (uses config.api_port automatically)
    tunnel_subparsers.add_parser("start", help="Start tunnel (uses API port from config)")

    # tunnel stop
    tunnel_subparsers.add_parser("stop", help="Stop tunnel")

    # tunnel status
    tunnel_subparsers.add_parser("status", help="Show tunnel status")

    # Legacy arguments for server mode
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
        help="Host to bind to (legacy single-server mode)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard port to bind to",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run in legacy single-server mode (all routes on one port)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Automatically restart if server is already running",
    )
    args = parser.parse_args()

    # Check for duplicate server when starting (not for tunnel subcommands)
    if args.command is None:  # Server start mode
        if not _check_and_handle_duplicate(auto_restart=args.restart):
            sys.exit(0)

    # Handle tunnel subcommand
    if args.command == "tunnel":
        if args.tunnel_action == "start":
            asyncio.run(tunnel_start())
        elif args.tunnel_action == "stop":
            asyncio.run(tunnel_stop())
        elif args.tunnel_action == "status":
            tunnel_status()
        else:
            tunnel_parser.print_help()
        return

    config = get_config()

    if args.port:
        config.set_runtime_port(args.port)

    if args.qr_only:
        show_pairing_qr()
        return

    if args.show_qr:
        show_pairing_qr()

    if args.single:
        # Legacy single-server mode
        host = args.host or config.host
        port = config.dashboard_port
        _write_pid_file()
        try:
            uvicorn.run(
                "main:app",
                host=host,
                port=port,
                reload=config.debug,
                log_level=config.log_level,
                access_log=False,
            )
        finally:
            _remove_pid_file()
    else:
        # Dual-server mode (default)
        asyncio.run(run_dual_servers())


if __name__ == "__main__":
    main()
