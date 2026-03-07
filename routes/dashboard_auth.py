"""Dashboard authentication routes."""

from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Response
from pydantic import BaseModel

from dashboard_auth_service import (
    dashboard_login,
    get_dashboard_auth_status,
    invalidate_session,
    remove_dashboard_password,
    set_dashboard_password,
)

from .deps import require_local_access

router = APIRouter(prefix="/api/dashboard", tags=["dashboard-auth"])

# Cookie settings
SESSION_COOKIE_NAME = "dashboard_session"
SESSION_COOKIE_MAX_AGE = 24 * 60 * 60  # 24 hours


class LoginRequest(BaseModel):
    """Login request body."""

    password: str


class SetPasswordRequest(BaseModel):
    """Set password request body."""

    new_password: str
    current_password: Optional[str] = None


class RemovePasswordRequest(BaseModel):
    """Remove password request body."""

    current_password: str


@router.get("/auth/status", dependencies=[Depends(require_local_access)])
async def get_auth_status(
    dashboard_session: Optional[str] = Cookie(None),
):
    """Get dashboard authentication status.

    Returns whether password is enabled and if current session is valid.
    """
    status = get_dashboard_auth_status(dashboard_session)
    return {
        "password_enabled": status.password_enabled,
        "session_valid": status.session_valid,
        "authenticated": not status.password_enabled or status.session_valid,
    }


@router.post("/auth/login", dependencies=[Depends(require_local_access)])
async def login(
    body: LoginRequest,
    response: Response,
):
    """Login to dashboard with password.

    Sets session cookie on success.
    """
    result = dashboard_login(body.password)

    if result.success and result.session_token:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=result.session_token,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
        return {"success": True, "message": "Login successful"}

    return {"success": False, "error": result.error or "Login failed"}


@router.post("/auth/logout", dependencies=[Depends(require_local_access)])
async def logout(
    response: Response,
    dashboard_session: Optional[str] = Cookie(None),
):
    """Logout from dashboard.

    Clears session cookie.
    """
    if dashboard_session:
        invalidate_session(dashboard_session)

    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"success": True, "message": "Logged out"}


@router.put("/auth/password", dependencies=[Depends(require_local_access)])
async def update_password(
    body: SetPasswordRequest,
    response: Response,
    dashboard_session: Optional[str] = Cookie(None),
):
    """Set or change dashboard password.

    If password is already set, current_password is required.
    Clears all sessions after password change.
    """
    result = set_dashboard_password(
        new_password=body.new_password,
        current_password=body.current_password,
    )

    if result.success:
        # Clear cookie since all sessions are invalidated
        response.delete_cookie(SESSION_COOKIE_NAME)

    return {
        "success": result.success,
        "message": result.message,
        "error": result.error,
    }


@router.delete("/auth/password", dependencies=[Depends(require_local_access)])
async def delete_password(
    body: RemovePasswordRequest,
    response: Response,
):
    """Remove dashboard password (disable protection).

    Requires current password for confirmation.
    """
    result = remove_dashboard_password(body.current_password)

    if result.success:
        response.delete_cookie(SESSION_COOKIE_NAME)

    return {
        "success": result.success,
        "message": result.message,
        "error": result.error,
    }
