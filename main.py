"""Code Bridge Server - FastAPI application.

This module exposes two FastAPI applications for production use:
- ``dashboard_app``: Full dashboard (localhost only, port 8766).
- ``api_app``: API endpoints (tunnel-exposed, port 8767).

A legacy ``app`` attribute is still resolvable for ``--single`` mode and
``uvicorn main:app``, but it is constructed lazily via :pep:`562` so the
deprecation warning only fires when single-app mode is actually used.
"""

import os
from pathlib import Path


def _load_env_file_once() -> None:
    """Load ``~/.code-bridge/.env`` into ``os.environ`` if present.

    Used for per-host secrets (e.g. ``REVENUECAT_SECRET_API_KEY``) that
    we don't want sitting in ``config.yaml`` next to the rest of the
    server config. Lines that already exist in ``os.environ`` are left
    untouched so the env can always override the file.

    Format is the standard ``KEY=VALUE`` shell-style; lines starting
    with ``#`` and blank lines are skipped. Quoting is intentionally
    not supported so we don't accidentally swallow a literal ``$`` or
    backslash in a secret.
    """
    env_path = Path.home() / ".code-bridge" / ".env"
    try:
        if not env_path.is_file():
            return
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if key in os.environ:
                continue
            os.environ[key] = value
    except OSError:
        # Best-effort. If the file is unreadable for any reason, we
        # still want the server to start so the user can inspect logs.
        pass


_load_env_file_once()

from app_factory import create_api_app, create_code_bridge_app, create_dashboard_app  # noqa: E402

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
