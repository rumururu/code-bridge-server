"""Helpers for building the web-based pairing page."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import qrcode

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_PAIRING_TEMPLATE: str | None = None


def _get_template() -> str:
    """Load and cache the pairing template."""
    global _PAIRING_TEMPLATE
    if _PAIRING_TEMPLATE is None:
        _PAIRING_TEMPLATE = (_TEMPLATE_DIR / "pairing.html").read_text(encoding="utf-8")
    return _PAIRING_TEMPLATE


def make_qr_png_base64(qr_url: str) -> str:
    """Create a base64-encoded PNG QR image from a pairing URL."""
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def render_pairing_page_html(
    *,
    qr_base64: str,
    local_url: str,
    pair_token: str,
    expires_in_seconds: int,
    pairing_code: str = "",
    embed: bool = False,
) -> str:
    """Render pairing page HTML.

    Args:
        qr_base64: Base64-encoded QR code image
        local_url: Local server URL to display
        pair_token: Pairing token for status checks
        expires_in_seconds: Expiration time in seconds
        pairing_code: 6-digit pairing code
        embed: If True, hides title and dashboard link (for iframe embed in dashboard modal)
    """
    expires_minutes = max(1, expires_in_seconds // 60)
    # Format pairing code with dash for readability (e.g., "123-456")
    formatted_code = (
        f"{pairing_code[:3]}-{pairing_code[3:]}" if len(pairing_code) == 6 else pairing_code
    )

    # Conditional styles for embed mode
    title_display = "display: none;" if embed else ""
    dashboard_btn_display = "display: none;" if embed else ""
    container_padding = "padding: 20px 40px;" if embed else "padding: 40px;"

    # Load template and replace placeholders
    template = _get_template()

    # Use simple string replacement for {{placeholders}}
    replacements = {
        "{{container_padding}}": container_padding,
        "{{title_display}}": title_display,
        "{{formatted_code}}": formatted_code,
        "{{local_url}}": local_url,
        "{{qr_base64}}": qr_base64,
        "{{expires_minutes}}": str(expires_minutes),
        "{{dashboard_btn_display}}": dashboard_btn_display,
        "{{pair_token}}": pair_token,
        "{{expires_in_seconds}}": str(expires_in_seconds),
    }

    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    return result
