"""Code Bridge Server - FastAPI application.

This module creates two FastAPI applications:
- dashboard_app: Full dashboard (localhost only, port 8766)
- api_app: API endpoints (tunnel-exposed, port 8767)

The legacy `app` is kept for backward compatibility with single-server mode.
"""

from app_factory import create_api_app, create_code_bridge_app, create_dashboard_app

# Legacy single-app mode (for backward compatibility)
app = create_code_bridge_app()

# Dual-app mode for port separation
dashboard_app = create_dashboard_app()
api_app = create_api_app()


if __name__ == "__main__":
    from server_cli import main as run_server_cli

    run_server_cli()
