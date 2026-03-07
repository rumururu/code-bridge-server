"""Pairing API routes - QR code pairing for mobile app."""

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, Response

from models import PairCodeVerifyRequest, PairVerifyRequest
from pairing import (
    build_current_pairing_qr_result,
    get_pair_token_status_for_current_server,
    get_pairing_status_for_current_server,
    revoke_paired_client_for_current_server,
    verify_pairing_code_for_current_server,
    RateLimiter,
)
from pairing_page import make_qr_png_base64
from pairing_page_service import build_pairing_page_html_for_current_server
from remote_access_service import verify_pair_token_for_current_server
from .result_response import as_route_response
from .deps import require_local_access, verify_api_key
import base64

router = APIRouter(tags=["pairing"])

# Rate limiters for unauthenticated endpoints
_token_status_limiter = RateLimiter(max_attempts=20, window_seconds=60, lockout_seconds=300)
_verify_limiter = RateLimiter(max_attempts=10, window_seconds=60, lockout_seconds=600)


@router.get("/api/pair/qr", dependencies=[Depends(require_local_access)])
async def get_pair_qr():
    """Get QR code pairing data.

    Only accessible from local network for security.
    """
    return as_route_response(build_current_pairing_qr_result())


@router.get("/api/pair/qr-image", dependencies=[Depends(require_local_access)])
async def get_pair_qr_image():
    """Get QR code as PNG image for dashboard display.

    Only accessible from local network for security.
    """
    result = build_current_pairing_qr_result()
    if not result.success or not result.qr_url:
        return Response(content=b"", media_type="image/png", status_code=500)

    qr_base64 = make_qr_png_base64(result.qr_url)
    qr_bytes = base64.b64decode(qr_base64)
    return Response(content=qr_bytes, media_type="image/png")


@router.get(
    "/pair",
    response_class=HTMLResponse,
    dependencies=[Depends(require_local_access)],
)
async def get_pair_page():
    """Show QR code pairing page in browser.

    This page is only accessible from local network.
    External access via Cloudflare Tunnel is blocked for security.
    """
    result = build_pairing_page_html_for_current_server()
    return HTMLResponse(content=result.content, status_code=result.status_code)


@router.get("/api/pair/token-status/{token}")
async def get_token_status(token: str, request: Request):
    """Check if a pairing token has been used.

    Rate limited to prevent token enumeration attacks.
    """
    client_ip = _get_client_ip(request)
    is_allowed, remaining = _token_status_limiter.check_rate_limit(client_ip)

    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Try again in {remaining} seconds."
        )

    result = get_pair_token_status_for_current_server(token)

    # Record attempt (non-existent tokens count as failed attempts)
    if not result.exists:
        _token_status_limiter.record_attempt(client_ip, success=False)

    return result.as_response_fields()


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, with secure proxy handling.

    Security: Only trust specific headers from known proxies.
    - CF-Connecting-IP: Trusted from Cloudflare Tunnel
    - Direct connection: Use request.client.host
    - Untrusted headers (X-Forwarded-For, X-Real-IP): Only as fallback
    """
    # Priority 1: Cloudflare Tunnel header (trusted)
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # Priority 2: Direct client connection (most reliable)
    if request.client and request.client.host:
        # If this is a local request, trust it
        client_host = request.client.host
        if client_host in ("127.0.0.1", "::1", "localhost"):
            return client_host
        # For non-local direct connections, use the actual client IP
        # Don't trust X-Forwarded-For from unknown sources
        return client_host

    # Priority 3: Fallback to forwarded headers (less secure, for compatibility)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    return "unknown"


@router.post("/api/pair/code", dependencies=[Depends(require_local_access)])
async def verify_pairing_code(request: Request, body: PairCodeVerifyRequest):
    """Verify a 6-digit pairing code and return a pair_token.

    Only accessible from local network for security (prevents external brute-force).
    """
    client_ip = _get_client_ip(request)
    result = verify_pairing_code_for_current_server(body.code, client_ip=client_ip)
    return as_route_response(result)


@router.post("/api/pair/verify")
async def verify_pair_token(request: Request, body: PairVerifyRequest):
    """Verify a pairing token and issue an API key.

    Rate limited to prevent brute-force attacks on pairing tokens.
    """
    client_ip = _get_client_ip(request)
    is_allowed, remaining = _verify_limiter.check_rate_limit(client_ip)

    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many verification attempts. Try again in {remaining} seconds."
        )

    result = await verify_pair_token_for_current_server(
        pair_token=body.pair_token,
        client_id=body.client_id,
        device_name=body.device_name,
        firebase_id_token=body.firebase_id_token,
        firebase_refresh_token=body.firebase_refresh_token,
        auth_mode=body.auth_mode,
    )

    # Record attempt
    _verify_limiter.record_attempt(client_ip, success=result.success)

    return as_route_response(result)


@router.get("/api/pair/status", dependencies=[Depends(verify_api_key)])
async def get_pair_status():
    """Get current pairing status."""
    return get_pairing_status_for_current_server().as_response_fields()


@router.delete(
    "/api/pair/clients/{client_id}",
    dependencies=[Depends(require_local_access)],
)
async def revoke_paired_client(client_id: str):
    """Revoke a paired client's API key.

    This is a dashboard management endpoint - only accessible locally.
    """
    result = revoke_paired_client_for_current_server(client_id)
    return as_route_response(result)
