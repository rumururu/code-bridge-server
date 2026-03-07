"""FastAPI lifespan hooks for startup and shutdown."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import get_config
from database import migrate_accessible_folders_from_projects
from lifecycle_service import (
    display_pairing_qr_for_current_server,
    initialize_firebase_for_current_server,
    shutdown_runtime_for_current_server,
    start_heartbeat_for_current_server,
    start_remote_tunnel_for_current_server,
)
from llm_settings import warmup_cli_cache

# Thread pool for background sync operations
_warmup_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("Code Bridge Server starting...")

    config = get_config()
    config.migrate_projects_to_db()

    # Seed accessible_folders from existing projects (one-time migration)
    migrate_accessible_folders_from_projects()

    # Warm up CLI cache in background (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_warmup_executor, warmup_cli_cache)

    firebase_auth, needs_pairing = await initialize_firebase_for_current_server(config)
    display_pairing_qr_for_current_server(config, needs_pairing=needs_pairing)

    tunnel_service = await start_remote_tunnel_for_current_server(config, firebase_auth)
    heartbeat_task = start_heartbeat_for_current_server(config, firebase_auth)

    yield

    await shutdown_runtime_for_current_server(
        heartbeat_task=heartbeat_task,
        tunnel_service=tunnel_service,
    )
