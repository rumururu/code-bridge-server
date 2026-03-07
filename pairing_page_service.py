"""Service helpers for rendering pairing web page responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pairing import PairingPageContextResult, build_current_pairing_page_context_result
from pairing_page import make_qr_png_base64, render_pairing_page_html


@dataclass(frozen=True)
class PairingPageRenderResult:
    """Rendered pairing page result."""

    success: bool
    status_code: int
    content: str


def build_pairing_page_html_for_current_server(
    *,
    context_result: PairingPageContextResult | None = None,
    context_builder: Callable[[], PairingPageContextResult] = build_current_pairing_page_context_result,
    qr_encoder: Callable[[str], str] = make_qr_png_base64,
    html_renderer: Callable[..., str] = render_pairing_page_html,
) -> PairingPageRenderResult:
    """Build HTML content for /pair page from current pairing context."""
    resolved_context = context_result or context_builder()
    render_context = resolved_context.to_render_context()

    if render_context is None:
        error_content, error_status = resolved_context.to_html_error()
        return PairingPageRenderResult(
            success=False,
            status_code=error_status,
            content=error_content,
        )

    qr_url, local_url, pair_token, expires_in_seconds, pairing_code = render_context
    qr_base64 = qr_encoder(qr_url)
    html = html_renderer(
        qr_base64=qr_base64,
        local_url=local_url,
        pair_token=pair_token,
        expires_in_seconds=expires_in_seconds,
        pairing_code=pairing_code,
    )
    return PairingPageRenderResult(success=True, status_code=200, content=html)
