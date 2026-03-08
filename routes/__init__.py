"""API route modules for Code Bridge Server."""

from .health import router as health_router
from .debug import router as debug_router
from .system_settings import router as system_settings_router
from .system_remote import router as system_remote_router
from .system_inspect import router as system_inspect_router
from .preview import router as preview_router
from .chat_ws import router as chat_ws_router
from .app_web import router as app_web_router
from .dashboard import router as dashboard_router
from .dashboard_auth import router as dashboard_auth_router
from .filesystem import router as filesystem_router

# Migrated feature routers
from .pairing import router as pairing_router
from .projects import router as projects_router
from .files import router as files_router
from .devices import router as devices_router
from .scrcpy_proxy import router as scrcpy_proxy_router


def register_routers(app) -> None:
    """Register all API routers with the FastAPI app.

    Most HTTP/WebSocket routes are registered through modular routers.
    The preview router is intentionally registered last because it has
    a root-level catch-all (``/{filename}``).
    """
    app.include_router(health_router)
    app.include_router(debug_router)
    app.include_router(system_settings_router)
    app.include_router(system_remote_router)
    app.include_router(system_inspect_router)
    app.include_router(chat_ws_router)
    app.include_router(app_web_router)
    # Incremental migration
    app.include_router(pairing_router)
    app.include_router(projects_router)
    app.include_router(files_router)
    app.include_router(devices_router)
    app.include_router(scrcpy_proxy_router)
    app.include_router(filesystem_router)
    # Dashboard auth router for login/password management
    app.include_router(dashboard_auth_router)
    # Dashboard router has root / redirect and /dashboard page
    app.include_router(dashboard_router)
    # Keep preview router last because it includes a root-level /{filename} catch-all route.
    app.include_router(preview_router)


def register_dashboard_routers(app) -> None:
    """Register Dashboard-specific routers (localhost only, not tunnel-exposed).

    This includes all routers for the full admin dashboard experience.
    The dashboard app binds to 127.0.0.1 only, so it's not accessible
    externally even if tunnel is running.
    """
    # Include all routers - dashboard needs full access
    register_routers(app)


def register_api_routers(app) -> None:
    """Register API-only routers (tunnel-exposed for external access).

    Excludes:
    - Dashboard HTML pages (/, /dashboard, /pair)
    - Dashboard auth endpoints (/api/dashboard/auth/*)
    - Debug endpoints (/api/debug/*)

    These are excluded because:
    - Dashboard pages should only be accessed locally
    - Dashboard auth/debug are local management features
    """
    app.include_router(health_router)
    app.include_router(system_settings_router)
    app.include_router(system_remote_router)
    app.include_router(system_inspect_router)
    app.include_router(chat_ws_router)
    app.include_router(app_web_router)
    # Feature routers
    app.include_router(pairing_router)
    app.include_router(projects_router)
    app.include_router(files_router)
    app.include_router(devices_router)
    app.include_router(scrcpy_proxy_router)
    app.include_router(filesystem_router)
    # Preview router last (has root-level catch-all)
    app.include_router(preview_router)
