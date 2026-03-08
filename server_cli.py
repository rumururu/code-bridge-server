"""CLI bootstrap for the Code Bridge server."""

import argparse
import asyncio
import webbrowser

import uvicorn

from config import get_config
from optional_services import get_active_tunnel_url
from pairing_qr_service import (
    build_pairing_qr_payload_for_current_server,
    display_pairing_qr_payload,
    open_pairing_page,
)
from qr_display import QRCODE_AVAILABLE


def show_pairing_qr(open_browser: bool = True) -> None:
    """Display pairing QR in terminal and optionally open pairing page."""
    if not QRCODE_AVAILABLE:
        print("\n[Error] QR code library not installed.")
        print("Run: pip install qrcode[pil]\n")
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

    config = get_config()

    # Auto-find available ports if default ports are in use
    dashboard_port = _find_available_port("127.0.0.1", config.dashboard_port)
    if dashboard_port != config.dashboard_port:
        print(f"[Port] Dashboard port {config.dashboard_port} in use, using {dashboard_port}")
        config.set_runtime_port(dashboard_port)

    api_port = _find_available_port("0.0.0.0", config.api_port)
    if api_port != config.api_port:
        print(f"[Port] API port {config.api_port} in use, using {api_port}")

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

    # Run both servers concurrently, open dashboard after startup
    await asyncio.gather(
        dashboard_server.serve(),
        api_server.serve(),
        open_dashboard_after_delay(dashboard_port),
    )


def main() -> None:
    """Parse CLI flags and run the uvicorn server."""
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
