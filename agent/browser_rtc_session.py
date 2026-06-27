"""WebRTC view over an existing live browser handoff page."""

from __future__ import annotations

import asyncio
import fractions
import io
import json
import time
from contextlib import suppress
from typing import Any, Awaitable, Callable

from agent.browser_runtime_manager import (
    LiveBrowserRuntime,
    get_browser_runtime_manager,
)


class BrowserRtcUnavailable(RuntimeError):
    """Raised when a live browser RTC stream cannot be attached."""


InputResultCallback = Callable[
    [dict[str, Any], dict[str, Any]],
    Awaitable[dict[str, Any]],
]


class BrowserRtcPeerSession:
    """Attach a WebRTC peer to the live Playwright page for browser handoff."""

    def __init__(
        self,
        browser_session: dict[str, Any],
        *,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        fps: float = 12.0,
        quality: int = 70,
        capture_mode: str = "page",
        input_result_callback: InputResultCallback | None = None,
    ) -> None:
        self.browser_session = browser_session
        self.viewport_width = max(320, min(int(viewport_width), 3840))
        self.viewport_height = max(240, min(int(viewport_height), 2160))
        self.fps = max(1.0, min(float(fps), 30.0))
        self.quality = max(20, min(int(quality), 95))
        self.capture_mode = "page"
        self.input_result_callback = input_result_callback
        self._runtime: LiveBrowserRuntime | None = None
        self._pc: Any | None = None
        self._data_channel: Any | None = None

    async def start(self) -> None:
        if self._pc is not None:
            return
        try:
            from aiortc import RTCPeerConnection
        except Exception as exc:  # pragma: no cover - optional dependency
            raise BrowserRtcUnavailable(f"WebRTC is unavailable: {exc}") from exc

        session_id = str(self.browser_session["id"])
        try:
            manager = get_browser_runtime_manager()
            runtime = manager.get(session_id)
            if runtime is None:
                runtime = await manager.open_session(
                    self.browser_session,
                    viewport_width=self.viewport_width,
                    viewport_height=self.viewport_height,
                )
            else:
                await runtime.set_viewport(self.viewport_width, self.viewport_height)
        except Exception as exc:  # noqa: BLE001
            raise BrowserRtcUnavailable(str(exc)) from exc

        await runtime.reset_scroll_to_top()
        await runtime.sync_state(save_storage=False)
        self._runtime = runtime
        self.browser_session = runtime.browser_session

        pc = RTCPeerConnection()
        self._pc = pc
        pc.addTrack(
            _create_browser_video_track(
                runtime,
                fps=self.fps,
                quality=self.quality,
                capture_mode=self.capture_mode,
            )
        )

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            self._data_channel = channel

            @channel.on("message")
            def on_message(raw: str | bytes) -> None:
                asyncio.create_task(self._handle_data_channel_message(raw))

    async def accept_offer(
        self,
        *,
        sdp: str,
        offer_type: str = "offer",
    ) -> dict[str, Any]:
        pc = self._require_pc()
        try:
            from aiortc import RTCSessionDescription
        except Exception as exc:  # pragma: no cover - optional dependency
            raise BrowserRtcUnavailable(f"WebRTC is unavailable: {exc}") from exc

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=offer_type or "offer")
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering_complete(pc, timeout=3.0)
        local_description = pc.localDescription
        return {
            "type": local_description.type,
            "sdp": local_description.sdp,
        }

    async def add_ice_candidate(self, candidate: dict[str, Any]) -> None:
        pc = self._require_pc()
        raw_candidate = candidate.get("candidate")
        if not raw_candidate:
            await pc.addIceCandidate(None)
            return
        try:
            from aiortc.sdp import candidate_from_sdp
        except Exception as exc:  # pragma: no cover - optional dependency
            raise BrowserRtcUnavailable(f"WebRTC is unavailable: {exc}") from exc

        candidate_text = str(raw_candidate)
        if candidate_text.startswith("candidate:"):
            candidate_text = candidate_text[len("candidate:") :]
        ice_candidate = candidate_from_sdp(candidate_text)
        sdp_mid = candidate.get("sdpMid")
        sdp_mline_index = candidate.get("sdpMLineIndex")
        if sdp_mid is not None:
            ice_candidate.sdpMid = str(sdp_mid)
        if sdp_mline_index is not None:
            ice_candidate.sdpMLineIndex = int(sdp_mline_index)
        await pc.addIceCandidate(ice_candidate)

    async def close(self) -> None:
        pc = self._pc
        self._pc = None
        self._data_channel = None
        if pc is not None:
            with suppress(Exception):
                await pc.close()
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            with suppress(Exception):
                await runtime.sync_state(save_storage=True)

    async def _handle_data_channel_message(self, raw: str | bytes) -> None:
        runtime = self._runtime
        channel = self._data_channel
        if runtime is None or channel is None:
            return
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            message = json.loads(text)
            if not isinstance(message, dict):
                raise ValueError("Message must be a JSON object.")
        except Exception as exc:  # noqa: BLE001
            _send_data_channel_json(
                channel,
                {
                    "type": "browser_stream.input_result",
                    "ok": False,
                    "error": str(exc),
                },
            )
            return

        if str(message.get("type") or "") in {"ping", "snapshot", "frame"}:
            _send_data_channel_json(channel, {"type": "pong"})
            return
        result = await runtime.handle_client_message(message)
        self.browser_session = runtime.browser_session
        if self.input_result_callback is not None:
            result = await self.input_result_callback(message, result)
        _send_data_channel_json(channel, result)

    def _require_pc(self) -> Any:
        if self._pc is None:
            raise BrowserRtcUnavailable("Browser RTC session has not started.")
        return self._pc


def _create_browser_video_track(
    runtime: LiveBrowserRuntime,
    *,
    fps: float,
    quality: int,
    capture_mode: str,
) -> Any:
    try:
        import numpy as np
        from av import VideoFrame
        from aiortc import VideoStreamTrack
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise BrowserRtcUnavailable(
            f"WebRTC video dependencies are unavailable: {exc}"
        ) from exc

    class BrowserVideoTrack(VideoStreamTrack):
        def __init__(self) -> None:
            super().__init__()
            self._frame_interval = 1.0 / max(1.0, min(float(fps), 30.0))
            self._quality = max(20, min(int(quality), 95))
            self._last_frame_at = 0.0
            self._pts = 0
            self._time_base = fractions.Fraction(1, 90000)

        async def recv(self) -> Any:
            elapsed = time.monotonic() - self._last_frame_at
            if elapsed < self._frame_interval:
                await asyncio.sleep(self._frame_interval - elapsed)
            self._last_frame_at = time.monotonic()
            raw = await runtime.screenshot_bytes(
                image_type="jpeg",
                quality=self._quality,
            )
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            frame = VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            self._pts += int(90000 * self._frame_interval)
            frame.pts = self._pts
            frame.time_base = self._time_base
            return frame

    return BrowserVideoTrack()


async def _wait_for_ice_gathering_complete(pc: Any, *, timeout: float) -> None:
    if getattr(pc, "iceGatheringState", None) == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(done.wait(), timeout=timeout)


def _send_data_channel_json(channel: Any, payload: dict[str, Any]) -> None:
    with suppress(Exception):
        channel.send(json.dumps(payload))
