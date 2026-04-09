"""Browser streaming service using Playwright and FFmpeg.

Captures a headless browser rendering a web page and streams it as H264
video for native playback on mobile devices (avoiding WebView/JavaScript).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BrowserStream:
    """Represents an active browser streaming session."""

    stream_id: str
    url: str
    width: int
    height: int
    fps: int

    # Playwright browser and page
    browser: Any = None
    context: Any = None
    page: Any = None

    # FFmpeg process for encoding
    ffmpeg_process: subprocess.Popen | None = None

    # Screenshot capture task
    capture_task: asyncio.Task | None = None

    # Frame queue for sending to WebSocket
    frame_queue: asyncio.Queue | None = None

    # State flags
    is_running: bool = False
    error: str | None = None


@dataclass
class BrowserStreamManager:
    """Manages browser streaming sessions."""

    _streams: dict[str, BrowserStream] = field(default_factory=dict)
    _playwright: Any = None
    _playwright_initialized: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def _ensure_playwright(self) -> bool:
        """Ensure Playwright is initialized."""
        if self._playwright_initialized:
            return True

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._playwright_initialized = True
            logger.info("[BrowserStream] Playwright initialized")
            return True
        except ImportError:
            logger.error("[BrowserStream] Playwright not installed. Run: pip install playwright && playwright install chromium")
            return False
        except Exception as e:
            logger.error(f"[BrowserStream] Failed to initialize Playwright: {e}")
            return False

    async def start_stream(
        self,
        url: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> tuple[str | None, str | None]:
        """Start a new browser streaming session.

        Returns:
            Tuple of (stream_id, error_message). On success, error is None.
        """
        logger.info(f"[BrowserStream] Starting stream: url={url}, {width}x{height}@{fps}fps")
        async with self._lock:
            if not await self._ensure_playwright():
                return None, "Playwright not available"

            stream_id = str(uuid.uuid4())[:8]

            try:
                # Launch headless browser
                browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        f"--window-size={width},{height}",
                    ],
                )

                context = await browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )

                page = await context.new_page()

                # Navigate to the URL
                logger.info(f"[BrowserStream] Loading URL: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Create frame queue
                frame_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

                # Create stream object
                stream = BrowserStream(
                    stream_id=stream_id,
                    url=url,
                    width=width,
                    height=height,
                    fps=fps,
                    browser=browser,
                    context=context,
                    page=page,
                    frame_queue=frame_queue,
                    is_running=True,
                )

                # Start FFmpeg encoder process
                ffmpeg_success = await self._start_ffmpeg(stream)
                if not ffmpeg_success:
                    await self._cleanup_stream(stream)
                    return None, "Failed to start FFmpeg encoder"

                # Start capture task
                stream.capture_task = asyncio.create_task(
                    self._capture_loop(stream)
                )

                self._streams[stream_id] = stream
                logger.info(f"[BrowserStream] Started stream {stream_id} for {url}")
                return stream_id, None

            except Exception as e:
                logger.error(f"[BrowserStream] Failed to start stream: {e}")
                return None, str(e)

    async def _start_ffmpeg(self, stream: BrowserStream) -> bool:
        """Start FFmpeg process for H264 encoding."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.error("[BrowserStream] FFmpeg not found in PATH")
            stream.error = "FFmpeg not installed"
            return False

        try:
            # FFmpeg command: read raw RGB frames from stdin, output H264 NAL units to stdout
            # Using rawvideo input instead of image2pipe to avoid PNG parsing buffering
            cmd = [
                ffmpeg_path,
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-video_size", f"{stream.width}x{stream.height}",
                "-framerate", str(stream.fps),
                "-i", "-",  # Read from stdin
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",  # Better compatibility
                "-level", "3.1",
                "-g", "1",  # Keyframe every frame for lowest latency
                "-bf", "0",  # No B-frames for lower latency
                "-f", "h264",
                "-",  # Output to stdout
            ]

            stream.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            logger.info(f"[BrowserStream] FFmpeg started for stream {stream.stream_id}, pid={stream.ffmpeg_process.pid}")
            return True

        except Exception as e:
            logger.error(f"[BrowserStream] Failed to start FFmpeg: {e}")
            stream.error = str(e)
            return False

    async def _capture_loop(self, stream: BrowserStream) -> None:
        """Capture screenshots and feed to FFmpeg."""
        frame_interval = 1.0 / stream.fps
        logger.info(f"[BrowserStream] Capture loop started for {stream.stream_id}, fps={stream.fps}")
        frames_captured = 0

        while stream.is_running:
            try:
                start_time = asyncio.get_event_loop().time()

                # Capture screenshot as PNG and convert to raw RGB
                screenshot_png = await stream.page.screenshot(type="png")

                # Convert PNG to raw RGB using PIL
                img = Image.open(io.BytesIO(screenshot_png))
                # Ensure RGB mode (remove alpha if present)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                raw_rgb = img.tobytes()

                if frames_captured == 0:
                    logger.debug(f"[BrowserStream] First screenshot: PNG={len(screenshot_png)}, RGB={len(raw_rgb)} bytes")

                # Send raw RGB to FFmpeg stdin
                if stream.ffmpeg_process and stream.ffmpeg_process.stdin:
                    try:
                        stream.ffmpeg_process.stdin.write(raw_rgb)
                        stream.ffmpeg_process.stdin.flush()
                    except BrokenPipeError:
                        logger.warning("[BrowserStream] FFmpeg stdin broken")
                        break
                else:
                    if frames_captured == 0:
                        logger.warning(f"[BrowserStream] No FFmpeg process available")

                # Check if FFmpeg is still running
                if stream.ffmpeg_process:
                    returncode = stream.ffmpeg_process.poll()
                    if returncode is not None:
                        stderr = stream.ffmpeg_process.stderr.read() if stream.ffmpeg_process.stderr else b''
                        logger.error(f"[BrowserStream] FFmpeg exited unexpectedly with code {returncode}, stderr={stderr[:500]}")
                        break

                # Read encoded H264 data from FFmpeg stdout
                if stream.ffmpeg_process and stream.ffmpeg_process.stdout:
                    # Non-blocking read
                    h264_data = self._read_available(stream.ffmpeg_process.stdout)
                    if h264_data and stream.frame_queue:
                        frames_captured += 1
                        if frames_captured == 1:
                            logger.info(f"[BrowserStream] First H264 frame: {len(h264_data)} bytes")
                        try:
                            stream.frame_queue.put_nowait(h264_data)
                        except asyncio.QueueFull:
                            # Drop oldest frame if queue is full
                            try:
                                stream.frame_queue.get_nowait()
                                stream.frame_queue.put_nowait(h264_data)
                            except asyncio.QueueEmpty:
                                pass

                # Maintain frame rate
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[BrowserStream] Capture error: {e}")
                await asyncio.sleep(0.1)

    def _read_available(self, pipe, max_bytes: int = 65536, timeout: float = 0.1) -> bytes | None:
        """Read available data from pipe with short timeout."""
        import select

        if not pipe:
            return None

        try:
            readable, _, _ = select.select([pipe], [], [], timeout)
            if readable:
                return pipe.read(max_bytes)
        except (ValueError, OSError):
            pass
        return None

    async def get_frames(self, stream_id: str) -> AsyncIterator[bytes]:
        """Async generator yielding H264 frames for the given stream."""
        stream = self._streams.get(stream_id)
        if not stream or not stream.frame_queue:
            return

        while stream.is_running:
            try:
                frame = await asyncio.wait_for(
                    stream.frame_queue.get(),
                    timeout=1.0,
                )
                yield frame
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def inject_event(self, stream_id: str, event: dict) -> bool:
        """Inject mouse/keyboard event into the browser.

        Event types:
        - mousedown: {"type": "mousedown", "x": int, "y": int, "button": "left"|"right"|"middle"}
        - mouseup: {"type": "mouseup", "x": int, "y": int, "button": "left"|"right"|"middle"}
        - mousemove: {"type": "mousemove", "x": int, "y": int}
        - click: {"type": "click", "x": int, "y": int, "button": "left"|"right"|"middle"}
        - scroll: {"type": "scroll", "x": int, "y": int, "deltaX": int, "deltaY": int}
        - keydown: {"type": "keydown", "key": str}
        - keyup: {"type": "keyup", "key": str}
        - keypress: {"type": "keypress", "text": str}
        """
        stream = self._streams.get(stream_id)
        if not stream or not stream.page or not stream.is_running:
            return False

        try:
            event_type = event.get("type", "")
            x = event.get("x", 0)
            y = event.get("y", 0)
            button = event.get("button", "left")

            if event_type == "click":
                await stream.page.mouse.click(x, y, button=button)
            elif event_type == "mousedown":
                await stream.page.mouse.move(x, y)
                await stream.page.mouse.down(button=button)
            elif event_type == "mouseup":
                await stream.page.mouse.move(x, y)
                await stream.page.mouse.up(button=button)
            elif event_type == "mousemove":
                await stream.page.mouse.move(x, y)
            elif event_type == "scroll":
                delta_x = event.get("deltaX", 0)
                delta_y = event.get("deltaY", 0)
                await stream.page.mouse.wheel(delta_x, delta_y)
            elif event_type == "keydown":
                key = event.get("key", "")
                await stream.page.keyboard.down(key)
            elif event_type == "keyup":
                key = event.get("key", "")
                await stream.page.keyboard.up(key)
            elif event_type == "keypress":
                text = event.get("text", "")
                await stream.page.keyboard.type(text)
            else:
                logger.warning(f"[BrowserStream] Unknown event type: {event_type}")
                return False

            return True

        except Exception as e:
            logger.error(f"[BrowserStream] Event injection failed: {e}")
            return False

    async def stop_stream(self, stream_id: str) -> bool:
        """Stop and cleanup a streaming session."""
        async with self._lock:
            stream = self._streams.pop(stream_id, None)
            if not stream:
                return False

            await self._cleanup_stream(stream)
            logger.info(f"[BrowserStream] Stopped stream {stream_id}")
            return True

    async def _cleanup_stream(self, stream: BrowserStream) -> None:
        """Cleanup stream resources."""
        stream.is_running = False

        # Cancel capture task
        if stream.capture_task:
            stream.capture_task.cancel()
            try:
                await stream.capture_task
            except asyncio.CancelledError:
                pass

        # Terminate FFmpeg
        if stream.ffmpeg_process:
            try:
                stream.ffmpeg_process.stdin.close()
                stream.ffmpeg_process.terminate()
                stream.ffmpeg_process.wait(timeout=5)
            except Exception:
                try:
                    stream.ffmpeg_process.kill()
                except Exception:
                    pass

        # Close browser
        if stream.context:
            try:
                await stream.context.close()
            except Exception:
                pass

        if stream.browser:
            try:
                await stream.browser.close()
            except Exception:
                pass

    async def get_stream_info(self, stream_id: str) -> dict | None:
        """Get information about a stream."""
        stream = self._streams.get(stream_id)
        if not stream:
            return None

        return {
            "stream_id": stream.stream_id,
            "url": stream.url,
            "width": stream.width,
            "height": stream.height,
            "fps": stream.fps,
            "is_running": stream.is_running,
            "error": stream.error,
        }

    def get_all_streams(self) -> list[dict]:
        """Get info about all active streams."""
        return [
            {
                "stream_id": s.stream_id,
                "url": s.url,
                "width": s.width,
                "height": s.height,
                "is_running": s.is_running,
            }
            for s in self._streams.values()
        ]

    async def shutdown(self) -> None:
        """Shutdown all streams and cleanup."""
        async with self._lock:
            for stream_id in list(self._streams.keys()):
                stream = self._streams.pop(stream_id, None)
                if stream:
                    await self._cleanup_stream(stream)

            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
                self._playwright_initialized = False

            logger.info("[BrowserStream] Manager shutdown complete")


# Global singleton instance
_browser_stream_manager: BrowserStreamManager | None = None


def get_browser_stream_manager() -> BrowserStreamManager:
    """Get the global browser stream manager instance."""
    global _browser_stream_manager
    if _browser_stream_manager is None:
        _browser_stream_manager = BrowserStreamManager()
    return _browser_stream_manager
