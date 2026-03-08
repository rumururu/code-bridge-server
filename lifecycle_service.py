"""Lifecycle flow helpers for server startup and shutdown."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from claude_session import get_session_manager
from heartbeat_settings import get_heartbeat_interval, set_heartbeat_interval
from optional_services import (
    FIREBASE_AVAILABLE,
    TUNNEL_AVAILABLE,
    create_tunnel_service,
    get_firebase_auth,
)
from pairing import get_pairing_service
from pairing_qr_service import (
    build_pairing_qr_payload_for_current_server,
    display_pairing_qr_payload,
)
from preview import get_preview_proxy
from qr_display import QRCODE_AVAILABLE


async def register_device_with_local_url_for_current_server(
    firebase_auth: Any,
    config: Any,
    *,
    pairing_service: Any | None = None,
) -> str:
    """Register device in Firebase using local URL for current server.

    Uses api_port since external clients connect to the API server.
    """
    resolved_pairing = pairing_service or get_pairing_service()
    local_url = f"http://{resolved_pairing.get_local_ip()}:{config.api_port}"
    await firebase_auth.register_device(None, local_url)
    print(f"Device registered to Firebase with local URL: {local_url}")
    return local_url


async def initialize_firebase_for_current_server(
    config: Any,
    *,
    firebase_available: bool = FIREBASE_AVAILABLE,
    firebase_auth_factory: Callable[[], Any | None] = get_firebase_auth,
    pairing_service_factory: Callable[[], Any] = get_pairing_service,
) -> tuple[Any | None, bool]:
    """Initialize firebase auth and determine whether QR pairing is needed."""
    needs_pairing = False

    if firebase_available and config.firebase_enabled:
        try:
            firebase_auth = firebase_auth_factory()
            if firebase_auth is None:
                return None, True

            await firebase_auth.initialize()

            if firebase_auth.is_authenticated:
                await register_device_with_local_url_for_current_server(
                    firebase_auth,
                    config,
                    pairing_service=pairing_service_factory(),
                )
                return firebase_auth, False

            print("Firebase auth expired or invalid. QR pairing required.")
            return firebase_auth, True
        except Exception as exc:
            print(f"Warning: Firebase initialization failed: {exc}")
            needs_pairing = True
    else:
        needs_pairing = True

    return None, needs_pairing


def _is_first_run() -> bool:
    """Check if this is the first server run (no .initialized marker file)."""
    from pathlib import Path
    marker_file = Path(__file__).parent / ".initialized"
    return not marker_file.exists()


def _mark_initialized() -> None:
    """Create .initialized marker file to indicate server has been run before."""
    from pathlib import Path
    marker_file = Path(__file__).parent / ".initialized"
    marker_file.touch()


def display_pairing_qr_for_current_server(
    config: Any,
    *,
    needs_pairing: bool,
    qrcode_available: bool = QRCODE_AVAILABLE,
    payload_builder: Callable[..., Any] = build_pairing_qr_payload_for_current_server,
    payload_display: Callable[..., None] = display_pairing_qr_payload,
    browser_opener: Callable[[str], Any] | None = None,
) -> None:
    """Display QR pairing block when pairing is required and QR is available."""
    if not needs_pairing:
        return

    pair_url = f"http://localhost:{config.dashboard_port}/pair"
    is_first_run = _is_first_run()

    print("\n" + "=" * 50)
    print("QR PAIRING REQUIRED")
    print("=" * 50)
    print("Scan QR code with Code Bridge app to connect.")
    print(f"Or visit: {pair_url}")
    print("=" * 50 + "\n")

    # Only open browser on first run after installation
    if is_first_run:
        if browser_opener is None:
            import webbrowser
            browser_opener = webbrowser.open
        try:
            browser_opener(pair_url)
            print(f"Opening browser: {pair_url}")
        except Exception as exc:
            print(f"Could not open browser: {exc}")
        _mark_initialized()

    if not qrcode_available:
        return

    try:
        payload = payload_builder(tunnel_url=None, config=config)
        payload_display(payload)
    except Exception as exc:
        print(f"Could not display QR: {exc}")


async def start_remote_tunnel_for_current_server(
    config: Any,
    firebase_auth: Any | None,
    *,
    tunnel_available: bool = TUNNEL_AVAILABLE,
    tunnel_service_factory: Callable[..., Any] = create_tunnel_service,
    pairing_service_factory: Callable[[], Any] = get_pairing_service,
    create_task: Callable[..., Any] = asyncio.create_task,
) -> Any | None:
    """Start remote tunnel if enabled and keep Firebase updated with tunnel URL."""
    if not tunnel_available or not config.remote_access_enabled:
        return None

    try:

        async def on_tunnel_url_change(url: str):
            if firebase_auth and firebase_auth.is_authenticated:
                # Ensure token is valid before updating Firebase
                if hasattr(firebase_auth, 'ensure_valid_token'):
                    if not await firebase_auth.ensure_valid_token():
                        print("Warning: Token validation failed, cannot update tunnel URL in Firebase")
                        return

                pairing = pairing_service_factory()
                local_url = f"http://{pairing.get_local_ip()}:{config.api_port}"
                success = await firebase_auth.register_device(url, local_url)
                if success:
                    print(f"Updated tunnel URL in Firebase: {url}")
                else:
                    print(f"Warning: Failed to update tunnel URL in Firebase")

        tunnel_service = tunnel_service_factory(
            local_port=config.api_port,  # Tunnel exposes API server, not Dashboard
            on_url_change=lambda url: create_task(on_tunnel_url_change(url)),
        )
        tunnel_url = await tunnel_service.start()
        if tunnel_url:
            print(f"Cloudflare Tunnel started: {tunnel_url}")

            if firebase_auth and firebase_auth.is_authenticated:
                pairing = pairing_service_factory()
                local_url = f"http://{pairing.get_local_ip()}:{config.api_port}"
                await firebase_auth.register_device(tunnel_url, local_url)
                print("Device registration updated with tunnel URL")

        return tunnel_service
    except Exception as exc:
        print(f"Warning: Remote access setup failed: {exc}")
        return None


def start_heartbeat_for_current_server(
    config: Any,
    firebase_auth: Any | None,
    *,
    sleep_fn: Callable[[float], Any] = asyncio.sleep,
    create_task: Callable[..., Any] = asyncio.create_task,
) -> Any | None:
    """Start heartbeat loop task when firebase is authenticated."""
    if not (firebase_auth and firebase_auth.is_authenticated):
        return None

    set_heartbeat_interval(config.heartbeat_interval_minutes)
    print(f"Starting Firebase heartbeat (interval: {get_heartbeat_interval()} min)")

    async def heartbeat_loop():
        consecutive_failures = 0
        max_failures = 3

        while True:
            interval_seconds = get_heartbeat_interval() * 60
            await sleep_fn(interval_seconds)

            if not (firebase_auth and firebase_auth.is_authenticated):
                print("Heartbeat: Firebase auth no longer valid, stopping heartbeat")
                break

            success = await firebase_auth.heartbeat()
            if success:
                consecutive_failures = 0
                print("Heartbeat sent successfully")
            else:
                consecutive_failures += 1
                print(f"Heartbeat failed (attempt {consecutive_failures}/{max_failures})")

                if consecutive_failures >= max_failures:
                    print("Heartbeat: Too many consecutive failures, server may need re-pairing")
                    # Don't break - keep trying in case network recovers

    return create_task(heartbeat_loop())


async def shutdown_runtime_for_current_server(
    *,
    heartbeat_task: Any | None,
    tunnel_service: Any | None,
    session_manager: Any | None = None,
    preview_proxy: Any | None = None,
    session_manager_factory: Callable[[], Any] = get_session_manager,
    preview_proxy_factory: Callable[[], Any] = get_preview_proxy,
) -> None:
    """Shutdown runtime tasks/services and close global managers."""
    print("Code Bridge Server shutting down...")

    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    if tunnel_service:
        try:
            await tunnel_service.stop()
        except Exception as exc:
            print(f"Warning: Tunnel shutdown error: {exc}")

    resolved_session_manager = session_manager or session_manager_factory()
    await resolved_session_manager.close_all()

    resolved_preview_proxy = preview_proxy or preview_proxy_factory()
    await resolved_preview_proxy.close()
