"""Firebase Firestore device registration and heartbeat.

This module handles:
- Registering server devices to Firestore
- Updating tunnel URLs
- Heartbeat for device presence
"""

from __future__ import annotations

import logging
import platform
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


def get_device_name() -> str:
    """Get a human-readable device name."""
    hostname = platform.node()
    system = platform.system()
    return f"{hostname} ({system})"


async def register_device(
    *,
    project_id: str,
    user_id: str,
    server_id: str,
    id_token: str,
    tunnel_url: Optional[str] = None,
    local_url: Optional[str] = None,
) -> bool:
    """Register this server device to Firestore.

    Args:
        project_id: Firebase project ID
        user_id: Firebase user ID (owner)
        server_id: Unique server identifier
        id_token: Firebase ID token for authentication
        tunnel_url: Cloudflare Tunnel URL (if available)
        local_url: Local network URL (http://ip:port)

    Returns:
        True if registration successful, False otherwise
    """
    now = datetime.now(timezone.utc).isoformat()

    device_data = {
        "fields": {
            "type": {"stringValue": "server"},
            "name": {"stringValue": get_device_name()},
            "tunnelUrl": {"stringValue": tunnel_url or ""},
            "localUrl": {"stringValue": local_url or ""},
            "lastSeen": {"timestampValue": now},
            "createdAt": {"timestampValue": now},
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            url = (
                f"https://firestore.googleapis.com/v1/"
                f"projects/{project_id}/databases/(default)/documents/"
                f"users/{user_id}/devices/{server_id}"
            )

            response = await client.patch(
                url,
                json=device_data,
                headers={"Authorization": f"Bearer {id_token}"},
                params={"updateMask.fieldPaths": ["type", "name", "tunnelUrl", "localUrl", "lastSeen"]},
            )

            if response.status_code in (200, 201):
                logger.info("Server registered: %s", server_id)
                return True
            else:
                logger.error("Device registration failed: %s", response.text)
                return False

    except httpx.HTTPError as e:
        logger.error("Device registration error: %s", e)
        return False


async def update_tunnel_url(
    *,
    project_id: str,
    user_id: str,
    server_id: str,
    id_token: str,
    tunnel_url: str,
    local_url: Optional[str] = None,
) -> bool:
    """Update the Tunnel URL for this device in Firestore.

    Args:
        project_id: Firebase project ID
        user_id: Firebase user ID (owner)
        server_id: Unique server identifier
        id_token: Firebase ID token for authentication
        tunnel_url: New Cloudflare Tunnel URL
        local_url: Local network URL (optional)

    Returns:
        True if update successful, False otherwise
    """
    return await register_device(
        project_id=project_id,
        user_id=user_id,
        server_id=server_id,
        id_token=id_token,
        tunnel_url=tunnel_url,
        local_url=local_url,
    )


async def heartbeat(
    *,
    project_id: str,
    user_id: str,
    server_id: str,
    id_token: str,
) -> tuple[bool, bool]:
    """Update lastSeen timestamp for this device.

    Checks if the device document still exists before updating.
    If the document was deleted (e.g., by user from app), returns device_deleted=True.

    Args:
        project_id: Firebase project ID
        user_id: Firebase user ID (owner)
        server_id: Unique server identifier
        id_token: Firebase ID token for authentication

    Returns:
        Tuple of (success, device_deleted):
        - success: True if heartbeat sent successfully
        - device_deleted: True if device was removed from Firebase
    """
    try:
        async with httpx.AsyncClient() as client:
            url = (
                f"https://firestore.googleapis.com/v1/"
                f"projects/{project_id}/databases/(default)/documents/"
                f"users/{user_id}/devices/{server_id}"
            )
            headers = {"Authorization": f"Bearer {id_token}"}

            # First, check if document still exists (GET request)
            check_response = await client.get(url, headers=headers)

            if check_response.status_code == 404:
                # Document was deleted - device removed from Firebase by user
                logger.warning("Device was removed from Firebase. Re-pairing required.")
                return False, True  # Not successful, device deleted

            if check_response.status_code == 403:
                # Permission denied - token expired or revoked
                logger.warning("Firebase auth expired or revoked.")
                return False, False  # Not successful, device not deleted

            # Document exists, update lastSeen
            response = await client.patch(
                url,
                json={
                    "fields": {
                        "lastSeen": {"timestampValue": datetime.now(timezone.utc).isoformat()},
                    }
                },
                headers=headers,
                params={"updateMask.fieldPaths": ["lastSeen"]},
            )

            success = response.status_code in (200, 201)
            return success, False

    except httpx.HTTPError as e:
        logger.error("Heartbeat error: %s", e)
        return False, False


async def delete_device(
    *,
    project_id: str,
    user_id: str,
    server_id: str,
    id_token: str,
) -> bool:
    """Remove this server device from Firestore.

    Args:
        project_id: Firebase project ID
        user_id: Firebase user ID (owner)
        server_id: Unique server identifier
        id_token: Firebase ID token for authentication

    Returns:
        True if deletion successful, False otherwise
    """
    try:
        async with httpx.AsyncClient() as client:
            url = (
                f"https://firestore.googleapis.com/v1/"
                f"projects/{project_id}/databases/(default)/documents/"
                f"users/{user_id}/devices/{server_id}"
            )

            response = await client.delete(
                url,
                headers={"Authorization": f"Bearer {id_token}"},
            )

            if response.status_code in (200, 204):
                logger.info("Server removed from Firebase: %s", server_id)
                return True
            else:
                logger.warning("Failed to remove device from Firebase: %s", response.text)
                return False

    except httpx.HTTPError as e:
        logger.warning("Failed to remove device from Firebase: %s", e)
        return False
