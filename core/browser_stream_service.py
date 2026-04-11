"""Browser streaming service using Playwright and FFmpeg.

Captures a headless browser rendering a web page and streams it as H264
video for native playback on mobile devices (avoiding WebView/JavaScript).

Uses CDP (Chrome DevTools Protocol) Page.screencastFrame for high-performance
event-driven frame capture instead of polling screenshots.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

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

    # CDP session for screencast
    cdp_session: Any = None

    # FFmpeg process for encoding
    ffmpeg_process: subprocess.Popen | None = None

    # FFmpeg reader task (reads H264 output)
    ffmpeg_reader_task: asyncio.Task | None = None

    # Screenshot fallback task (when CDP fails)
    screenshot_task: asyncio.Task | None = None

    # Frame queue for sending to WebSocket
    frame_queue: asyncio.Queue | None = None

    # State flags
    is_running: bool = False
    error: str | None = None

    # Frame counter for logging
    frame_count: int = 0
    last_fps_log_time: float = 0.0


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
                # Note: Avoid --disable-gpu as CDP screencast may need GPU rendering
                browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
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
                    last_fps_log_time=time.time(),
                )

                # Start FFmpeg encoder process (now using MJPEG input)
                ffmpeg_success = await self._start_ffmpeg(stream)
                if not ffmpeg_success:
                    await self._cleanup_stream(stream)
                    return None, "Failed to start FFmpeg encoder"

                # Start frame capture FIRST (before FFmpeg reader)
                # TODO: CDP screencast hangs indefinitely in current Playwright version
                # Using screenshot fallback for now (achieves ~15-20fps)
                # CDP would achieve 30-60fps when working
                logger.info("[BrowserStream] Starting screenshot capture (CDP disabled due to hanging)")
                capture_success = await self._start_screenshot_fallback(stream)

                if not capture_success:
                    await self._cleanup_stream(stream)
                    return None, "Failed to start frame capture"

                # Start FFmpeg stdout reader task AFTER screenshot capture is running
                stream.ffmpeg_reader_task = asyncio.create_task(
                    self._ffmpeg_reader_loop(stream),
                    name=f"ffmpeg-reader-{stream.stream_id}"
                )
                logger.info(f"[BrowserStream] FFmpeg reader task started")

                self._streams[stream_id] = stream
                logger.info(f"[BrowserStream] Started stream {stream_id} for {url} (CDP screencast)")
                return stream_id, None

            except Exception as e:
                logger.error(f"[BrowserStream] Failed to start stream: {e}")
                return None, str(e)

    async def _start_ffmpeg(self, stream: BrowserStream) -> bool:
        """Start FFmpeg process for H264 encoding.

        Uses MJPEG input (from CDP screencast JPEG frames) for better performance
        compared to rawvideo input which requires PNG→RGB conversion.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.error("[BrowserStream] FFmpeg not found in PATH")
            stream.error = "FFmpeg not installed"
            return False

        try:
            # FFmpeg command: read JPEG frames from stdin (image2pipe), output H264 NAL units to stdout
            # CDP screencast provides JPEG frames directly - no PNG→RGB conversion needed
            # Using image2pipe demuxer which properly handles concatenated JPEG images
            cmd = [
                ffmpeg_path,
                "-y",  # Overwrite without asking
                "-fflags", "nobuffer",  # Reduce input buffering
                "-flags", "low_delay",  # Low latency mode
                "-f", "image2pipe",  # For concatenated image input
                "-c:v", "mjpeg",  # Input codec is MJPEG (JPEG frames)
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
                "-x264-params", "rc-lookahead=0:sync-lookahead=0:sliced-threads=1",
                "-flags2", "fast",  # Fast encoding flags
                "-f", "h264",
                "-",  # Output to stdout
            ]

            stream.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=65536,  # Larger buffer for better throughput
            )

            # Start stderr logger task
            asyncio.create_task(self._ffmpeg_stderr_logger(stream))

            logger.info(f"[BrowserStream] FFmpeg started for stream {stream.stream_id}, pid={stream.ffmpeg_process.pid}")
            return True

        except Exception as e:
            logger.error(f"[BrowserStream] Failed to start FFmpeg: {e}")
            stream.error = str(e)
            return False

    def _blocking_read_stderr_line(self, process) -> bytes | None:
        """Blocking read one line from stderr - runs in thread pool."""
        try:
            if process and process.stderr:
                return process.stderr.readline()
        except (ValueError, OSError):
            pass
        return None

    async def _ffmpeg_stderr_logger(self, stream: BrowserStream) -> None:
        """Log FFmpeg stderr output for debugging (non-blocking)."""
        if not stream.ffmpeg_process or not stream.ffmpeg_process.stderr:
            return

        try:
            while stream.is_running:
                # Use thread pool to avoid blocking the event loop
                line = await asyncio.to_thread(
                    self._blocking_read_stderr_line,
                    stream.ffmpeg_process
                )
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str:
                    # Only log important messages, skip progress
                    if "error" in line_str.lower() or "warning" in line_str.lower():
                        logger.warning(f"[FFmpeg] {stream.stream_id}: {line_str}")
                    elif "Stream" in line_str or "Output" in line_str or "frame=" in line_str:
                        logger.debug(f"[FFmpeg] {stream.stream_id}: {line_str}")
        except Exception as e:
            logger.debug(f"[FFmpeg] stderr logger ended: {e}")

    async def _start_cdp_screencast(self, stream: BrowserStream) -> bool:
        """Start CDP screencast for high-performance frame capture.

        CDP screencast pushes JPEG frames as events instead of polling screenshots.
        This eliminates:
        - Screenshot polling latency (15-20ms per frame)
        - PNG→RGB conversion overhead (5-10ms per frame)
        - PIL dependency

        Result: 30-60fps achievable vs ~15fps with screenshot polling.

        Falls back to screenshot-based capture if CDP fails.
        """
        CDP_TIMEOUT = 10.0  # Timeout for CDP operations

        try:
            logger.info(f"[BrowserStream] Creating CDP session for {stream.stream_id}...")

            # Create CDP session with timeout (can hang in some Playwright versions)
            try:
                cdp_session = await asyncio.wait_for(
                    stream.page.context.new_cdp_session(stream.page),
                    timeout=CDP_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(f"[BrowserStream] CDP session creation timed out after {CDP_TIMEOUT}s, using screenshot fallback")
                return await self._start_screenshot_fallback(stream)

            stream.cdp_session = cdp_session
            logger.info(f"[BrowserStream] CDP session created for {stream.stream_id}")

            # Set up frame event handler
            def on_screencast_frame(params: dict) -> None:
                logger.debug(f"[BrowserStream] Frame event received for {stream.stream_id}")
                asyncio.create_task(self._handle_screencast_frame(stream, params))

            cdp_session.on("Page.screencastFrame", on_screencast_frame)
            logger.info(f"[BrowserStream] Frame handler registered for {stream.stream_id}")

            # Start screencast with JPEG format
            # Quality 80 provides good balance between size and quality
            logger.info(f"[BrowserStream] Starting Page.startScreencast for {stream.stream_id}...")
            try:
                await asyncio.wait_for(
                    cdp_session.send("Page.startScreencast", {
                        "format": "jpeg",
                        "quality": 80,
                        "maxWidth": stream.width,
                        "maxHeight": stream.height,
                        "everyNthFrame": 1,  # Capture every frame
                    }),
                    timeout=CDP_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(f"[BrowserStream] startScreencast timed out, using screenshot fallback")
                stream.cdp_session = None
                return await self._start_screenshot_fallback(stream)

            logger.info(f"[BrowserStream] CDP screencast started for {stream.stream_id}")
            return True

        except Exception as e:
            logger.warning(f"[BrowserStream] CDP screencast failed: {e}, using screenshot fallback")
            return await self._start_screenshot_fallback(stream)

    async def _start_screenshot_fallback(self, stream: BrowserStream) -> bool:
        """Start screenshot-based capture as fallback when CDP fails.

        This is slower (~15fps) but more reliable across Playwright versions.
        """
        logger.info(f"[BrowserStream] Starting screenshot fallback for {stream.stream_id}")

        try:
            # Start screenshot capture loop task
            logger.info(f"[BrowserStream] Creating screenshot task...")
            stream.screenshot_task = asyncio.create_task(
                self._screenshot_capture_loop(stream),
                name=f"screenshot-{stream.stream_id}"
            )
            logger.info(f"[BrowserStream] Screenshot task created, forcing start...")

            # Force task to start by yielding with sleep(0)
            await asyncio.sleep(0)
            logger.info(f"[BrowserStream] After yield, task: {stream.screenshot_task}")

            # Wait a bit more for task to actually process
            await asyncio.sleep(0.2)

            if stream.screenshot_task.done():
                exc = stream.screenshot_task.exception()
                if exc:
                    logger.error(f"[BrowserStream] Screenshot task failed: {exc}")
                    return False

            logger.info(f"[BrowserStream] Screenshot fallback started for {stream.stream_id}")
        except Exception as e:
            logger.error(f"[BrowserStream] Failed to create screenshot task: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        return True

    async def _screenshot_capture_loop(self, stream: BrowserStream) -> None:
        """Capture screenshots and feed JPEG frames to FFmpeg."""
        logger.info(f"[BrowserStream] Screenshot capture loop starting for {stream.stream_id}")

        try:
            frame_interval = 1.0 / stream.fps
            last_frame_time = time.time()
            logger.info(f"[BrowserStream] Screenshot loop initialized, fps={stream.fps}, interval={frame_interval:.3f}s")

            while stream.is_running:
                try:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time

                    # Maintain target frame rate
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)

                    last_frame_time = time.time()

                    # Take screenshot as JPEG with timeout
                    try:
                        jpeg_bytes = await asyncio.wait_for(
                            stream.page.screenshot(
                                type="jpeg",
                                quality=80,
                                full_page=False,
                            ),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"[BrowserStream] Screenshot timeout for {stream.stream_id}")
                        continue

                    # Send to FFmpeg stdin (non-blocking to prevent deadlock)
                    if stream.ffmpeg_process and stream.ffmpeg_process.stdin:
                        # Use thread pool to avoid blocking on pipe write
                        write_success = await asyncio.to_thread(
                            self._blocking_write_ffmpeg_stdin,
                            stream.ffmpeg_process,
                            jpeg_bytes
                        )

                        if not write_success:
                            logger.warning(f"[BrowserStream] FFmpeg stdin write failed")
                            stream.is_running = False
                            break

                        # Update frame counter
                        stream.frame_count += 1
                        if stream.frame_count == 1:
                            logger.info(f"[BrowserStream] First screenshot frame: {len(jpeg_bytes)} bytes")

                        # Log FPS every 5 seconds
                        if current_time - stream.last_fps_log_time >= 5.0:
                            fps = stream.frame_count / (current_time - stream.last_fps_log_time)
                            logger.debug(f"[BrowserStream] Screenshot mode {stream.stream_id}: {fps:.1f} fps")
                            stream.frame_count = 0
                            stream.last_fps_log_time = current_time

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[BrowserStream] Screenshot loop error: {e}")
                    await asyncio.sleep(0.1)

            logger.info(f"[BrowserStream] Screenshot capture stopped for {stream.stream_id}")
        except Exception as e:
            logger.error(f"[BrowserStream] Screenshot capture fatal error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _handle_screencast_frame(self, stream: BrowserStream, params: dict) -> None:
        """Handle incoming CDP screencast frame.

        params contains:
        - data: base64-encoded JPEG image
        - metadata: frame timing info
        - sessionId: screencast session ID (for ack)
        """
        if not stream.is_running:
            return

        try:
            # Acknowledge the frame to request next one
            session_id = params.get("sessionId", 0)
            if stream.cdp_session:
                try:
                    await stream.cdp_session.send("Page.screencastFrameAck", {
                        "sessionId": session_id
                    })
                except Exception:
                    pass  # Best effort ack

            # Decode base64 JPEG data
            jpeg_base64 = params.get("data", "")
            if not jpeg_base64:
                return

            jpeg_bytes = base64.b64decode(jpeg_base64)

            # Send JPEG directly to FFmpeg stdin (no PNG→RGB conversion!)
            if stream.ffmpeg_process and stream.ffmpeg_process.stdin:
                try:
                    stream.ffmpeg_process.stdin.write(jpeg_bytes)
                    stream.ffmpeg_process.stdin.flush()

                    # Update frame counter and log FPS periodically
                    stream.frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - stream.last_fps_log_time

                    if stream.frame_count == 1:
                        logger.info(f"[BrowserStream] First CDP frame received: {len(jpeg_bytes)} bytes")

                    if elapsed >= 5.0:  # Log FPS every 5 seconds
                        fps = stream.frame_count / elapsed
                        logger.debug(f"[BrowserStream] {stream.stream_id}: {fps:.1f} fps ({stream.frame_count} frames in {elapsed:.1f}s)")
                        stream.frame_count = 0
                        stream.last_fps_log_time = current_time

                except BrokenPipeError:
                    logger.warning(f"[BrowserStream] FFmpeg stdin broken for {stream.stream_id}")
                    stream.is_running = False
                except Exception as e:
                    logger.error(f"[BrowserStream] Error writing to FFmpeg: {e}")

        except Exception as e:
            logger.error(f"[BrowserStream] Error handling screencast frame: {e}")

    def _blocking_read_ffmpeg(self, process) -> bytes | None:
        """Blocking read from FFmpeg stdout - runs in thread pool."""
        import select
        try:
            if process and process.stdout:
                readable, _, _ = select.select([process.stdout], [], [], 0.03)
                if readable:
                    return process.stdout.read(65536)
        except (ValueError, OSError):
            pass
        return None

    def _blocking_write_ffmpeg_stdin(self, process, data: bytes) -> bool:
        """Blocking write to FFmpeg stdin - runs in thread pool."""
        try:
            if process and process.stdin:
                process.stdin.write(data)
                process.stdin.flush()
                return True
        except (ValueError, OSError, BrokenPipeError) as e:
            return False
        return False

    async def _ffmpeg_reader_loop(self, stream: BrowserStream) -> None:
        """Read H264 data from FFmpeg stdout and queue for WebSocket delivery."""
        logger.info(f"[BrowserStream] FFmpeg reader started for {stream.stream_id}")
        h264_frames_read = 0

        while stream.is_running:
            try:
                # Check if FFmpeg is still running
                if stream.ffmpeg_process:
                    returncode = stream.ffmpeg_process.poll()
                    if returncode is not None:
                        stderr = b""
                        if stream.ffmpeg_process.stderr:
                            stderr = stream.ffmpeg_process.stderr.read()
                        logger.error(f"[BrowserStream] FFmpeg exited with code {returncode}, stderr={stderr[:500]}")
                        break

                # Non-blocking read from FFmpeg stdout using thread pool
                if stream.ffmpeg_process:
                    h264_data = await asyncio.to_thread(
                        self._blocking_read_ffmpeg, stream.ffmpeg_process
                    )
                    if h264_data and stream.frame_queue:
                        h264_frames_read += 1
                        if h264_frames_read == 1:
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

                # Small yield to prevent busy loop
                await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[BrowserStream] FFmpeg reader error: {e}")
                await asyncio.sleep(0.1)

        logger.info(f"[BrowserStream] FFmpeg reader stopped for {stream.stream_id}, total H264 chunks: {h264_frames_read}")

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

            # Validate and clamp coordinates to stream dimensions
            x = max(0, min(int(x), stream.width - 1))
            y = max(0, min(int(y), stream.height - 1))

            logger.debug(f"[BrowserStream] Injecting {event_type} at ({x}, {y}) for stream {stream_id}")

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

        # Stop CDP screencast
        if stream.cdp_session:
            try:
                await stream.cdp_session.send("Page.stopScreencast", {})
                await stream.cdp_session.detach()
            except Exception:
                pass
            stream.cdp_session = None

        # Cancel screenshot fallback task
        if stream.screenshot_task:
            stream.screenshot_task.cancel()
            try:
                await stream.screenshot_task
            except asyncio.CancelledError:
                pass

        # Cancel FFmpeg reader task
        if stream.ffmpeg_reader_task:
            stream.ffmpeg_reader_task.cancel()
            try:
                await stream.ffmpeg_reader_task
            except asyncio.CancelledError:
                pass

        # Terminate FFmpeg
        if stream.ffmpeg_process:
            try:
                if stream.ffmpeg_process.stdin:
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
