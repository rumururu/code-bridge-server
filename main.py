"""Code Bridge Server - FastAPI application.

This module exposes two FastAPI applications for production use:
- ``dashboard_app``: Full dashboard (localhost only, port 8766).
- ``api_app``: API endpoints (tunnel-exposed, port 8767).

A legacy ``app`` attribute is still resolvable for ``--single`` mode and
``uvicorn main:app``, but it is constructed lazily via :pep:`562` so the
deprecation warning only fires when single-app mode is actually used.
"""

from app_factory import create_api_app, create_code_bridge_app, create_dashboard_app

# Dual-app mode for port separation.
dashboard_app = create_dashboard_app()
api_app = create_api_app()


def __getattr__(name: str):
    if name == "app":
        # Legacy single-app variant — only built on demand so importing
        # ``api_app`` / ``dashboard_app`` does not pull in the deprecated
        # path. Triggers ``DeprecationWarning`` from the factory.
        return create_code_bridge_app()
    raise AttributeError(f"module 'main' has no attribute {name!r}")


if __name__ == "__main__":
    from server_cli import main as run_server_cli

    run_server_cli()
