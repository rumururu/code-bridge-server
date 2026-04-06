"""Lifecycle flow helpers for server startup and shutdown."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

from llm.claude_session import get_session_manager
from remote.heartbeat_settings import get_heartbeat_interval, set_heartbeat_interval
from .optional_services import (
    FIREBASE_AVAILABLE,
    TUNNEL_AVAILABLE,
    create_tunnel_service,
    get_firebase_auth,
)
from pairing.pairing import get_pairing_service
from pairing.pairing_qr_service import (
    build_pairing_qr_payload_for_current_server,
    display_pairing_qr_payload,
)
from preview.preview import get_preview_proxy
from pairing.qr_display import QRCODE_AVAILABLE


async def register_device_with_local_url_for_current_server(
    firebase_auth: Any,
    config: Any,
    *,
    pairing_service: Any | None = None,
) -> str:
    """Register device in Firebase using local URL for current server.

    Uses api_port since external clients connect to the API server.
    Also updates all paired accounts with the local URL.
    """
    resolved_pairing = pairing_service or get_pairing_service()
    local_url = f"http://{resolved_pairing.get_local_ip()}:{config.api_port}"

    # Update primary owner
    await firebase_auth.register_device(None, local_url)
    logger.info("Device registered to Firebase with local URL: %s", local_url)

    # Update all paired accounts (multi-account support)
    await _update_all_paired_accounts_url(firebase_auth, None, local_url)

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

    if firebase_available:
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

            logger.info("Firebase auth expired or invalid. QR pairing required.")
            return firebase_auth, True
        except Exception as exc:
            logger.warning("Firebase initialization failed: %s", exc)
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
            logger.info("Opening browser: %s", pair_url)
        except OSError as exc:
            logger.warning("Could not open browser: %s", exc)
        _mark_initialized()

    if not qrcode_available:
        return

    try:
        payload = payload_builder(tunnel_url=None, config=config)
        payload_display(payload)
    except OSError as exc:
        logger.warning("Could not display QR: %s", exc)


async def _update_all_paired_accounts_url(
    firebase_auth: Any,
    tunnel_url: str | None,
    local_url: str,
) -> None:
    """Update URL for all paired accounts in Firestore.

    This ensures that when the server URL changes (tunnel or local IP),
    all Firebase accounts that have paired with this server get updated.
    """
    try:
        from firebase.paired_accounts import get_paired_accounts_manager

        paired_accounts = get_paired_accounts_manager()
        if paired_accounts.account_count == 0:
            logger.debug("No paired accounts to update")
            return

        project_id = firebase_auth.project_id
        server_id = firebase_auth.server_id
        api_key = firebase_auth._config.get("apiKey") if firebase_auth._config else None

        if not project_id or not server_id or not api_key:
            logger.warning("Cannot update paired accounts: missing Firebase config")
            return

        results = await paired_accounts.update_url_for_all_accounts(
            project_id=project_id,
            server_id=server_id,
            api_key=api_key,
            tunnel_url=tunnel_url,
            local_url=local_url,
        )

        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        logger.info(
            "Updated URL for %d/%d paired accounts (tunnel=%s)",
            success_count,
            total_count,
            tunnel_url or "none",
        )

    except Exception as exc:
        logger.warning("Failed to update paired accounts: %s", exc)


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
            pairing = pairing_service_factory()
            local_url = f"http://{pairing.get_local_ip()}:{config.api_port}"

            # Update primary owner
            if firebase_auth and firebase_auth.is_authenticated:
                # Ensure token is valid before updating Firebase
                if hasattr(firebase_auth, 'ensure_valid_token'):
                    if not await firebase_auth.ensure_valid_token():
                        logger.warning("Token validation failed, cannot update tunnel URL in Firebase")
                        return

                success = await firebase_auth.register_device(url, local_url)
                if success:
                    logger.info("Updated tunnel URL in Firebase: %s", url)
                else:
                    logger.warning("Failed to update tunnel URL in Firebase")

            # Update all paired accounts (multi-account support)
            if firebase_auth:
                await _update_all_paired_accounts_url(firebase_auth, url, local_url)

        tunnel_service = tunnel_service_factory(
            local_port=config.api_port,  # Tunnel exposes API server, not Dashboard
            on_url_change=lambda url: create_task(on_tunnel_url_change(url)),
        )
        tunnel_url = await tunnel_service.start()
        if tunnel_url:
            logger.info("Cloudflare Tunnel started: %s", tunnel_url)

            pairing = pairing_service_factory()
            local_url = f"http://{pairing.get_local_ip()}:{config.api_port}"

            # Update primary owner
            if firebase_auth and firebase_auth.is_authenticated:
                await firebase_auth.register_device(tunnel_url, local_url)
                logger.info("Device registration updated with tunnel URL")

            # Update all paired accounts (multi-account support)
            if firebase_auth:
                await _update_all_paired_accounts_url(firebase_auth, tunnel_url, local_url)

        return tunnel_service
    except OSError as exc:
        logger.warning("Remote access setup failed: %s", exc)
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
    logger.info("Starting Firebase heartbeat (interval: %d min)", get_heartbeat_interval())

    async def heartbeat_loop():
        consecutive_failures = 0
        max_failures = 3

        while True:
            interval_seconds = get_heartbeat_interval() * 60
            await sleep_fn(interval_seconds)

            if not (firebase_auth and firebase_auth.is_authenticated):
                logger.warning("Heartbeat: Firebase auth no longer valid, stopping heartbeat")
                break

            success = await firebase_auth.heartbeat()
            if success:
                consecutive_failures = 0
                logger.debug("Heartbeat sent successfully")
            else:
                consecutive_failures += 1
                logger.warning("Heartbeat failed (attempt %d/%d)", consecutive_failures, max_failures)

                if consecutive_failures >= max_failures:
                    logger.error("Heartbeat: Too many consecutive failures, server may need re-pairing")
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
    logger.info("Code Bridge Server shutting down...")

    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    if tunnel_service:
        try:
            await tunnel_service.stop()
        except OSError as exc:
            logger.warning("Tunnel shutdown error: %s", exc)

    resolved_session_manager = session_manager or session_manager_factory()
    await resolved_session_manager.close_all()

    resolved_preview_proxy = preview_proxy or preview_proxy_factory()
    await resolved_preview_proxy.close()
