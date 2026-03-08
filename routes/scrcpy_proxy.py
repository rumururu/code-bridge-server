"""WebSocket proxy for scrcpy streaming through the main API port.

This allows device mirroring to work over Cloudflare tunnel by proxying
the scrcpy WebSocket connection through the main API server.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import websockets
from websockets.exceptions import ConnectionClosed

from scrcpy_manager import get_scrcpy_manager
from auth_service import validate_api_key_for_current_server

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scrcpy-proxy"])


@router.websocket("/ws/scrcpy/stream")
async def scrcpy_stream_proxy(
    websocket: WebSocket,
    udid: str = Query(..., description="Device UDID"),
    displayId: int = Query(0, description="Display ID"),
    maxSize: int = Query(720, description="Max size"),
    maxFps: int = Query(30, description="Max FPS"),
    api_key: Optional[str] = Query(None, description="API key for authentication"),
):
    """Proxy WebSocket connection to local scrcpy server.

    This endpoint allows remote clients (via tunnel) to connect to the
    local scrcpy server running on a different port.
    """
    # Accept connection first (required before sending close codes)
    await websocket.accept()

    # Verify API key
    validation = validate_api_key_for_current_server(api_key)
    if not validation.success:
        logger.warning(f"[ScrcpyProxy] Invalid API key: {api_key[:20] if api_key else 'None'}...")
        await websocket.close(code=4001, reason="Invalid API key")
        return

    scrcpy_manager = get_scrcpy_manager()

    if not scrcpy_manager.is_running:
        await websocket.close(code=4002, reason="Scrcpy server not running")
        return

    # Build local scrcpy WebSocket URL
    local_ws_url = (
        f"ws://127.0.0.1:{scrcpy_manager.port}/stream"
        f"?udid={udid}"
        f"&displayId={displayId}"
        f"&maxSize={maxSize}"
        f"&maxFps={maxFps}"
    )

    logger.info(f"[ScrcpyProxy] Proxying to {local_ws_url}")

    try:
        async with websockets.connect(
            local_ws_url,
            max_size=None,  # No message size limit for video streams
            ping_interval=None,  # Disable ping to avoid interference
        ) as scrcpy_ws:
            # Create bidirectional proxy tasks
            async def client_to_scrcpy():
                """Forward messages from client to scrcpy server."""
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await scrcpy_ws.send(data)
                except WebSocketDisconnect:
                    logger.info("[ScrcpyProxy] Client disconnected")
                except Exception as e:
                    logger.debug(f"[ScrcpyProxy] Client->Scrcpy error: {e}")

            async def scrcpy_to_client():
                """Forward messages from scrcpy server to client."""
                try:
                    async for message in scrcpy_ws:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except ConnectionClosed:
                    logger.info("[ScrcpyProxy] Scrcpy server closed connection")
                except Exception as e:
                    logger.debug(f"[ScrcpyProxy] Scrcpy->Client error: {e}")

            # Run both directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_scrcpy()),
                    asyncio.create_task(scrcpy_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except ConnectionRefusedError:
        logger.error("[ScrcpyProxy] Cannot connect to local scrcpy server")
        await websocket.close(code=4003, reason="Scrcpy server unavailable")
    except Exception as e:
        logger.error(f"[ScrcpyProxy] Proxy error: {e}")
        try:
            await websocket.close(code=4000, reason=str(e))
        except Exception:
            pass
