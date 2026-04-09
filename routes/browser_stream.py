"""WebSocket endpoint for browser streaming.

Provides H264 video stream of a headless browser rendering a web page,
allowing mobile apps to preview web content without using WebView.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.responses import JSONResponse

from auth.auth_service import validate_api_key_for_current_server
from core.browser_stream_service import get_browser_stream_manager
from projects.project_manager import get_project_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["browser-stream"])


def _build_metadata_packet(width: int, height: int, stream_id: str) -> bytes:
    """Build initial metadata packet similar to scrcpy format.

    Format:
    - bytes 0-15: magic "browser_stream\0\0" (16 bytes)
    - bytes 16-79: stream_id (64 bytes, null-padded)
    - bytes 80-83: width (4 bytes, big endian)
    - bytes 84-87: height (4 bytes, big endian)
    """
    magic = b"browser_stream\x00\x00"  # 16 bytes
    stream_id_bytes = stream_id.encode("utf-8")[:64].ljust(64, b"\x00")
    dimensions = struct.pack(">II", width, height)  # 8 bytes

    return magic + stream_id_bytes + dimensions


@router.websocket("/ws/browser/stream")
async def browser_stream_endpoint(
    websocket: WebSocket,
    project: str = Query(..., description="Project name"),
    width: int = Query(1280, description="Browser width"),
    height: int = Query(720, description="Browser height"),
    fps: int = Query(30, description="Target FPS"),
    api_key: Optional[str] = Query(None, description="API key for authentication"),
):
    """WebSocket endpoint for browser streaming.

    Streams H264 video of a headless browser rendering the project's dev server.
    Touch/mouse events from the client are injected into the browser.

    Protocol:
    - Server sends initial metadata packet (88 bytes)
    - Server continuously sends H264 NAL units
    - Client sends JSON-encoded events for interaction
    """
    await websocket.accept()

    # Verify API key
    validation = validate_api_key_for_current_server(api_key)
    if not validation.success:
        logger.warning(f"[BrowserStream] Invalid API key")
        await websocket.close(code=4001, reason="Invalid API key")
        return

    # Get project manager to find dev server URL
    project_manager = get_project_manager()
    port = project_manager.get_server_port(project)

    if port is None:
        logger.error(f"[BrowserStream] No dev server running for project: {project}")
        await websocket.close(code=4002, reason="Dev server not running")
        return

    # Build dev server URL
    url = f"http://localhost:{port}"
    logger.info(f"[BrowserStream] Starting stream for {project} at {url}")

    # Get browser stream manager
    manager = get_browser_stream_manager()

    # Start the stream
    stream_id, error = await manager.start_stream(
        url=url,
        width=width,
        height=height,
        fps=fps,
    )

    if error:
        logger.error(f"[BrowserStream] Failed to start stream: {error}")
        await websocket.close(code=4003, reason=f"Stream error: {error}")
        return

    try:
        # Send metadata packet
        metadata = _build_metadata_packet(width, height, stream_id)
        await websocket.send_bytes(metadata)
        logger.info(f"[BrowserStream] Sent metadata for stream {stream_id}")

        # Create bidirectional communication tasks
        async def send_frames():
            """Send H264 frames to client."""
            logger.info(f"[BrowserStream] Starting frame send loop for {stream_id}")
            frame_count = 0
            try:
                async for frame in manager.get_frames(stream_id):
                    frame_count += 1
                    logger.debug(f"[BrowserStream] Sending frame {frame_count}: {len(frame)} bytes")
                    await websocket.send_bytes(frame)
                    if frame_count == 1:
                        logger.info(f"[BrowserStream] First frame sent for {stream_id}")
            except WebSocketDisconnect:
                logger.info(f"[BrowserStream] Client disconnected from stream {stream_id} after {frame_count} frames")
            except Exception as e:
                logger.error(f"[BrowserStream] Send error after {frame_count} frames: {e}")
            logger.info(f"[BrowserStream] Frame send loop ended for {stream_id}")

        async def receive_events():
            """Receive and process input events from client."""
            try:
                while True:
                    message = await websocket.receive()

                    if "text" in message:
                        # JSON event from client
                        try:
                            event = json.loads(message["text"])
                            await manager.inject_event(stream_id, event)
                        except json.JSONDecodeError:
                            logger.warning(f"[BrowserStream] Invalid JSON event")
                    elif "bytes" in message:
                        # Binary event (touch coordinates, etc.)
                        data = message["bytes"]
                        event = _parse_binary_event(data)
                        if event:
                            await manager.inject_event(stream_id, event)

            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"[BrowserStream] Receive error: {e}")

        # Run both tasks concurrently
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(send_frames()),
                asyncio.create_task(receive_events()),
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

    finally:
        # Cleanup stream
        await manager.stop_stream(stream_id)
        logger.info(f"[BrowserStream] Stream {stream_id} ended")


def _parse_binary_event(data: bytes) -> dict | None:
    """Parse binary touch event from client.

    Simple format for touch events (similar to scrcpy but simplified):
    - byte 0: event type (0=down, 1=up, 2=move)
    - bytes 1-4: x coordinate (big endian float as percentage * 10000)
    - bytes 5-8: y coordinate (big endian float as percentage * 10000)
    """
    if len(data) < 9:
        return None

    event_type = data[0]
    x_percent = struct.unpack(">I", data[1:5])[0] / 10000.0
    y_percent = struct.unpack(">I", data[5:9])[0] / 10000.0

    type_map = {0: "mousedown", 1: "mouseup", 2: "mousemove"}
    event_name = type_map.get(event_type)

    if event_name is None:
        return None

    return {
        "type": event_name,
        "x_percent": x_percent,
        "y_percent": y_percent,
    }


@router.get("/api/browser-stream/status")
async def get_stream_status(
    api_key: Optional[str] = Query(None, description="API key"),
):
    """Get status of all active browser streams."""
    validation = validate_api_key_for_current_server(api_key)
    if not validation.success:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid API key"},
        )

    manager = get_browser_stream_manager()
    streams = manager.get_all_streams()

    return {
        "success": True,
        "streams": streams,
    }


@router.post("/api/browser-stream/stop/{stream_id}")
async def stop_stream(
    stream_id: str,
    api_key: Optional[str] = Query(None, description="API key"),
):
    """Stop a specific browser stream."""
    validation = validate_api_key_for_current_server(api_key)
    if not validation.success:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid API key"},
        )

    manager = get_browser_stream_manager()
    success = await manager.stop_stream(stream_id)

    return {
        "success": success,
        "message": f"Stream {stream_id} stopped" if success else f"Stream {stream_id} not found",
    }
