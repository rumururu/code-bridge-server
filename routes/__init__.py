"""API route modules for Code Bridge Server."""

from .health import router as health_router
from .debug import router as debug_router
from .system_settings import router as system_settings_router
from .system_remote import router as system_remote_router
from .system_inspect import router as system_inspect_router
from .preview import router as preview_router
from .chat_ws import router as chat_ws_router
from .chat_sessions import router as chat_sessions_router
from .agents import router as agents_router
from .subagents import router as subagents_router
from .agent_ws import router as agent_ws_router
from .agent_browser_rtc import router as agent_browser_rtc_router
from .app_builder import router as app_builder_router
from .workspaces import router as workspaces_router
from .approvals import router as approvals_router
from .audit import router as audit_router
from .policies import router as policies_router
from .usage import router as usage_router
from .git import router as git_router
from .terminal import router as terminal_router
from .dashboard import router as dashboard_router
from .dashboard_agents import router as dashboard_agents_router
from .scripts import router as scripts_router
from .script_proposals import router as script_proposals_router
from .dashboard_auth import router as dashboard_auth_router
from .filesystem import router as filesystem_router
from .mermaid import router as mermaid_router

# Migrated feature routers
from .pairing import router as pairing_router
from .projects import router as projects_router
from .files import router as files_router
from .devices import router as devices_router
from .scrcpy_proxy import router as scrcpy_proxy_router
from .secrets import router as secrets_router


# Routers shared by both apps (Dashboard and API). Ordered so that the
# preview router (root-level ``/{filename}`` catch-all) stays last when
# this list is registered.
_SHARED_ROUTERS = (
    health_router,
    system_settings_router,
    system_remote_router,
    system_inspect_router,
    chat_ws_router,
    chat_sessions_router,
    workspaces_router,
    agents_router,
    subagents_router,
    agent_ws_router,
    agent_browser_rtc_router,
    app_builder_router,
    approvals_router,
    audit_router,
    policies_router,
    usage_router,
    mermaid_router,
    pairing_router,
    projects_router,
    files_router,
    git_router,
    terminal_router,
    devices_router,
    scrcpy_proxy_router,
    filesystem_router,
    secrets_router,
    scripts_router,
    # Read-only: polling a script the builder conversation asked for. The
    # approval that registers one is dashboard-only, like every other write on
    # the script registry (see routes/dashboard_agents.py).
    script_proposals_router,
)

# Routers that must NEVER be reachable on the tunnel-exposed API listener,
# regardless of bind address. They expose local-management surfaces
# (dashboard HTML, dashboard-auth, debug endpoints, debug-level filesystem
# browsing) and are mounted only by ``register_dashboard_routers``.
_DASHBOARD_ONLY_ROUTERS = (
    debug_router,
    dashboard_auth_router,
    dashboard_agents_router,
    dashboard_router,
)


def register_routers(app) -> None:
    """Register every available router (legacy single-app mode).

    Kept for backward compatibility with ``create_code_bridge_app``. New
    code should use :func:`register_dashboard_routers` /
    :func:`register_api_routers` so the dashboard vs. tunnel boundary is
    explicit at the router-registration layer, not just at the bind
    address.
    """
    register_dashboard_routers(app)


def register_dashboard_routers(app) -> None:
    """Register routers for the Dashboard app (localhost-only listener).

    Includes both shared API routers and dashboard-exclusive routers
    (HTML pages, auth, debug). Even though the dashboard listener binds
    to 127.0.0.1 only, declaring the dashboard-exclusive set here keeps
    a defense-in-depth boundary: changing the bind address alone is not
    enough to leak debug/dashboard-auth endpoints, because they are not
    registered on the API app at all.
    """
    for router in _SHARED_ROUTERS:
        app.include_router(router)
    for router in _DASHBOARD_ONLY_ROUTERS:
        app.include_router(router)
    # Root-level ``/{filename}`` catch-all must come after everything else.
    # Preview is not in ``_SHARED_ROUTERS`` and is gated by its own
    # preview-token mechanism.
    app.include_router(preview_router)


def register_api_routers(app) -> None:
    """Register routers for the tunnel-exposed API app.

    Includes only the shared routers and the preview catch-all. The
    dashboard HTML, dashboard-auth and debug routers are deliberately
    omitted so they cannot be reached even if the API binds to
    ``0.0.0.0`` or is fronted by a tunnel.
    """
    for router in _SHARED_ROUTERS:
        app.include_router(router)
    # Preview router last (root-level ``/{filename}`` catch-all).
    app.include_router(preview_router)
