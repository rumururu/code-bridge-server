"""FastAPI app factory for Code Bridge server."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_lifecycle import lifespan
from config import get_config
from routes import register_api_routers, register_dashboard_routers, register_routers
from server_logging import configure_server_logging


def create_code_bridge_app(
    *,
    config: Any | None = None,
    router_registrar: Callable[[FastAPI], None] = register_routers,
) -> FastAPI:
    """Create and configure Code Bridge FastAPI application (legacy single-app mode)."""
    resolved_config = config or get_config()
    configure_server_logging(getattr(resolved_config, "log_level", "info"))

    app = FastAPI(
        title="Code Bridge",
        description="Remote IDE bridge for Claude Code",
        version="1.0.0",
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
        version="1.0.0",
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
        version="1.0.0",
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
