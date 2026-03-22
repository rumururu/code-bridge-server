"""CLI bootstrap for the Code Bridge server."""

import argparse
import asyncio
import logging
import signal
import webbrowser

import uvicorn

from config import get_config

logger = logging.getLogger(__name__)
from mdns_service import start_mdns_service, stop_mdns_service
from optional_services import get_active_tunnel_url
from pairing_qr_service import (
    build_pairing_qr_payload_for_current_server,
    display_pairing_qr_payload,
    open_pairing_page,
)
from qr_display import QRCODE_AVAILABLE
from remote_access_service import (
    start_tunnel_for_current_server,
    stop_tunnel_for_current_server,
)
from tunnel_service import TunnelService, get_tunnel_service


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


async def run_dual_servers() -> None:
    """Run Dashboard and API servers concurrently.

    - Dashboard server: 127.0.0.1:dashboard_port (localhost only)
    - API server: 0.0.0.0:api_port (tunnel-exposed)
    """
    from main import api_app, dashboard_app
    from pairing import get_pairing_service

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

    dashboard_config = uvicorn.Config(
        app=dashboard_app,
        host="127.0.0.1",  # localhost only, not tunnel-exposed
        port=dashboard_port,
        log_level=config.log_level,
        # No reload in dual-server mode (not supported with programmatic run)
    )

    api_config = uvicorn.Config(
        app=api_app,
        host="0.0.0.0",  # External access via tunnel
        port=api_port,
        log_level=config.log_level,
    )

    dashboard_server = uvicorn.Server(dashboard_config)
    api_server = uvicorn.Server(api_config)

    print(f"Starting Dashboard server on http://127.0.0.1:{dashboard_port}")
    print(f"Starting API server on http://0.0.0.0:{api_port}")

    try:
        # Run both servers concurrently, open dashboard after startup
        await asyncio.gather(
            dashboard_server.serve(),
            api_server.serve(),
            open_dashboard_after_delay(dashboard_port),
        )
    finally:
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
    args = parser.parse_args()

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
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=config.debug,
            log_level=config.log_level,
        )
    else:
        # Dual-server mode (default)
        asyncio.run(run_dual_servers())


if __name__ == "__main__":
    main()
