"""Shared wrappers for optional tunnel/firebase integrations."""

from __future__ import annotations

from typing import Any, Callable


def _none_callable(*args, **kwargs):
    return None


def _load_tunnel_bindings() -> tuple[bool, Callable[..., Any], Callable[[], Any]]:
    try:
        from tunnel_service import create_tunnel_service, get_tunnel_service

        return True, create_tunnel_service, get_tunnel_service
    except ImportError:
        return False, _none_callable, _none_callable


def _load_firebase_binding() -> tuple[bool, Callable[[], Any]]:
    try:
        from firebase_auth import get_firebase_auth

        return True, get_firebase_auth
    except ImportError:
        return False, _none_callable


TUNNEL_AVAILABLE, create_tunnel_service, get_tunnel_service = _load_tunnel_bindings()
FIREBASE_AVAILABLE, get_firebase_auth = _load_firebase_binding()


def get_active_tunnel_url() -> str | None:
    """Return current tunnel URL when tunnel service is running."""
    tunnel_service = get_tunnel_service()
    if not tunnel_service:
        return None

    tunnel_status = tunnel_service.get_status()
    if not isinstance(tunnel_status, dict):
        return None

    tunnel_url = tunnel_status.get("url")
    return tunnel_url if isinstance(tunnel_url, str) and tunnel_url else None
