"""Dashboard authentication service.

Provides optional password protection for the dashboard.
When no password is set, dashboard is accessible without authentication.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel

from database import get_settings_db

# Settings keys
DASHBOARD_PASSWORD_HASH_KEY = "dashboard_password_hash"
DASHBOARD_SESSIONS_KEY = "dashboard_sessions"

# Session configuration
SESSION_EXPIRY_HOURS = 24
SESSION_TOKEN_LENGTH = 32


class DashboardAuthStatus(BaseModel):
    """Dashboard authentication status."""

    password_enabled: bool
    session_valid: bool = False


class DashboardLoginResult(BaseModel):
    """Result of dashboard login attempt."""

    success: bool
    session_token: Optional[str] = None
    error: Optional[str] = None


class DashboardPasswordResult(BaseModel):
    """Result of password operations."""

    success: bool
    message: str
    error: Optional[str] = None


def _hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hash_value = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hash_value}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, hash_value = stored_hash.split(":")
        computed_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return secrets.compare_digest(computed_hash, hash_value)
    except (ValueError, AttributeError):
        return False


def _get_sessions() -> dict:
    """Get active sessions from database."""
    db = get_settings_db()
    sessions = db.get_json(DASHBOARD_SESSIONS_KEY, {})
    return sessions if isinstance(sessions, dict) else {}


def _save_sessions(sessions: dict) -> None:
    """Save sessions to database."""
    db = get_settings_db()
    db.set_json(DASHBOARD_SESSIONS_KEY, sessions)


def _cleanup_expired_sessions(sessions: dict) -> dict:
    """Remove expired sessions."""
    now = datetime.now(timezone.utc)
    valid_sessions = {}
    for token, expiry_str in sessions.items():
        try:
            expiry = datetime.fromisoformat(expiry_str)
            if expiry > now:
                valid_sessions[token] = expiry_str
        except (ValueError, TypeError):
            continue
    return valid_sessions


def is_dashboard_password_enabled() -> bool:
    """Check if dashboard password is set."""
    db = get_settings_db()
    password_hash = db.get(DASHBOARD_PASSWORD_HASH_KEY)
    return password_hash is not None and len(password_hash) > 0


def get_dashboard_auth_status(session_token: Optional[str] = None) -> DashboardAuthStatus:
    """Get current dashboard authentication status."""
    password_enabled = is_dashboard_password_enabled()

    session_valid = False
    if password_enabled and session_token:
        session_valid = verify_session_token(session_token)

    return DashboardAuthStatus(
        password_enabled=password_enabled,
        session_valid=session_valid,
    )


def verify_session_token(token: str) -> bool:
    """Verify if session token is valid and not expired."""
    if not token:
        return False

    sessions = _get_sessions()
    sessions = _cleanup_expired_sessions(sessions)
    _save_sessions(sessions)

    if token not in sessions:
        return False

    return True


def create_session() -> str:
    """Create a new session and return the token."""
    token = secrets.token_urlsafe(SESSION_TOKEN_LENGTH)
    expiry = datetime.now(timezone.utc) + timedelta(hours=SESSION_EXPIRY_HOURS)

    sessions = _get_sessions()
    sessions = _cleanup_expired_sessions(sessions)
    sessions[token] = expiry.isoformat()
    _save_sessions(sessions)

    return token


def invalidate_session(token: str) -> bool:
    """Invalidate a session token."""
    sessions = _get_sessions()
    if token in sessions:
        del sessions[token]
        _save_sessions(sessions)
        return True
    return False


def invalidate_all_sessions() -> int:
    """Invalidate all sessions. Returns count of invalidated sessions."""
    sessions = _get_sessions()
    count = len(sessions)
    _save_sessions({})
    return count


def dashboard_login(password: str) -> DashboardLoginResult:
    """Attempt to login to dashboard with password."""
    db = get_settings_db()
    stored_hash = db.get(DASHBOARD_PASSWORD_HASH_KEY)

    if not stored_hash:
        return DashboardLoginResult(
            success=False,
            error="Password not set",
        )

    if not _verify_password(password, stored_hash):
        return DashboardLoginResult(
            success=False,
            error="Invalid password",
        )

    session_token = create_session()
    return DashboardLoginResult(
        success=True,
        session_token=session_token,
    )


def set_dashboard_password(
    new_password: str,
    current_password: Optional[str] = None,
) -> DashboardPasswordResult:
    """Set or change dashboard password."""
    db = get_settings_db()
    stored_hash = db.get(DASHBOARD_PASSWORD_HASH_KEY)

    # If password already set, require current password
    if stored_hash:
        if not current_password:
            return DashboardPasswordResult(
                success=False,
                message="Current password required",
                error="current_password_required",
            )
        if not _verify_password(current_password, stored_hash):
            return DashboardPasswordResult(
                success=False,
                message="Current password is incorrect",
                error="invalid_current_password",
            )

    # Validate new password
    if len(new_password) < 4:
        return DashboardPasswordResult(
            success=False,
            message="Password must be at least 4 characters",
            error="password_too_short",
        )

    # Set new password
    new_hash = _hash_password(new_password)
    db.set(DASHBOARD_PASSWORD_HASH_KEY, new_hash)

    # Invalidate all existing sessions
    invalidate_all_sessions()

    return DashboardPasswordResult(
        success=True,
        message="Password set successfully",
    )


def remove_dashboard_password(current_password: str) -> DashboardPasswordResult:
    """Remove dashboard password (disable protection)."""
    db = get_settings_db()
    stored_hash = db.get(DASHBOARD_PASSWORD_HASH_KEY)

    if not stored_hash:
        return DashboardPasswordResult(
            success=False,
            message="Password not set",
            error="no_password",
        )

    if not _verify_password(current_password, stored_hash):
        return DashboardPasswordResult(
            success=False,
            message="Password is incorrect",
            error="invalid_password",
        )

    # Remove password
    db.set(DASHBOARD_PASSWORD_HASH_KEY, None)

    # Invalidate all sessions
    invalidate_all_sessions()

    return DashboardPasswordResult(
        success=True,
        message="Password removed. Dashboard is now accessible without authentication.",
    )
