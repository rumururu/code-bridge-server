"""Helpers for building and displaying pairing QR metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from config import get_config
from pairing import get_pairing_service
from qr_display import display_pairing_qr


@dataclass(frozen=True)
class PairingQrPayload:
    """Resolved payload for pairing QR output and browser launch."""

    qr_url: str
    local_url: str
    tunnel_url: str | None
    server_name: str
    pair_url: str


def build_pairing_qr_payload_for_current_server(
    *,
    tunnel_url: str | None = None,
    config: Any | None = None,
    pairing_service: Any | None = None,
) -> PairingQrPayload:
    """Build pairing QR payload for current config/server."""
    resolved_config = config or get_config()
    resolved_pairing = pairing_service or get_pairing_service()

    pairing_data = resolved_pairing.create_pairing_data(
        port=resolved_config.api_port,  # App connects to API server
        server_name=resolved_config.server_name,
        tunnel_url=tunnel_url,
    )

    return PairingQrPayload(
        qr_url=pairing_data.to_qr_url(),
        local_url=pairing_data.local_url,
        tunnel_url=pairing_data.tunnel_url,
        server_name=resolved_config.server_name,
        pair_url=f"http://localhost:{resolved_config.dashboard_port}/pair",  # Dashboard page
    )


def display_pairing_qr_payload(
    payload: PairingQrPayload,
    *,
    display_fn: Callable[..., None] = display_pairing_qr,
) -> None:
    """Render pairing QR payload via terminal display function."""
    display_fn(
        qr_url=payload.qr_url,
        local_url=payload.local_url,
        tunnel_url=payload.tunnel_url,
        server_name=payload.server_name,
    )


def open_pairing_page(url: str, *, opener: Callable[[str], Any]) -> None:
    """Open pairing page URL with injected opener."""
    opener(url)
