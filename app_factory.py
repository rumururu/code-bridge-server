"""FastAPI app factory for Code Bridge server."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_lifecycle import lifespan
from core.config import get_config
from routes import register_api_routers, register_dashboard_routers, register_routers
from core.server_logging import configure_server_logging

logger = logging.getLogger(__name__)


def create_code_bridge_app(
    *,
    config: Any | None = None,
    router_registrar: Callable[[FastAPI], None] = register_routers,
) -> FastAPI:
    """Create the legacy single-app variant.

    .. deprecated::
        Use :func:`create_dashboard_app` (localhost-only, 127.0.0.1) and
        :func:`create_api_app` (tunnel-exposed) instead. The single-app
        variant exposes dashboard HTML, dashboard-auth and debug routes
        on the same port as the API, so binding it to ``0.0.0.0`` or
        routing it through a tunnel leaks local-management surfaces.

    Retained for the ``--single`` CLI flag and existing tests. New code
    must use the dual-app entry points.
    """
    warnings.warn(
        "create_code_bridge_app() is deprecated; use create_dashboard_app() "
        "and create_api_app() so the dashboard/API boundary is enforced at "
        "the routing layer instead of relying on the bind address.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "Starting in legacy single-app mode. Dashboard HTML, auth and "
        "debug routes share the API port — do not bind to 0.0.0.0 or "
        "expose this port through a tunnel."
    )
    resolved_config = config or get_config()
    configure_server_logging(getattr(resolved_config, "log_level", "info"))

    app = FastAPI(
        title="Code Bridge",
        description="Remote IDE bridge for Claude Code",
        version="1.2.2",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router_registrar(app)
    return app


def create_dashboard_app(
    *,
    config: Any | None = None,
) -> FastAPI:
    """Create Dashboard FastAPI application (localhost only, not tunnel-exposed).

    This app includes all routers and binds to 127.0.0.1 only.
    No lifespan - lifecycle management is handled separately.
    """
    resolved_config = config or get_config()
    configure_server_logging(getattr(resolved_config, "log_level", "info"))

    app = FastAPI(
        title="Code Bridge Dashboard",
        description="Code Bridge Dashboard (localhost only)",
        version="1.2.2",
        # No lifespan - managed in server_cli.py
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_dashboard_routers(app)
    return app


def create_api_app(
    *,
    config: Any | None = None,
) -> FastAPI:
    """Create API FastAPI application (tunnel-exposed for external access).

    This app excludes dashboard HTML pages and local-only endpoints.
    Includes lifespan for startup/shutdown management (tunnel, heartbeat, etc).
    """
    resolved_config = config or get_config()
    configure_server_logging(getattr(resolved_config, "log_level", "info"))

    app = FastAPI(
        title="Code Bridge API",
        description="Code Bridge API (tunnel-exposed)",
        version="1.2.2",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_api_routers(app)
    return app
