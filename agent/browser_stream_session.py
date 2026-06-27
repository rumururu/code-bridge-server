"""Streaming view over a browser handoff page."""

from __future__ import annotations

from typing import Any

from agent.browser_runtime_manager import LiveBrowserRuntime, get_browser_runtime_manager


class BrowserStreamUnavailable(RuntimeError):
    """Raised when a live browser stream cannot be attached."""


class PlaywrightBrowserStreamSession:
    """Attach to or restore the Playwright page for a browser handoff."""

    def __init__(
        self,
        browser_session: dict[str, Any],
        *,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ) -> None:
        self.browser_session = browser_session
        self.viewport_width = max(320, min(int(viewport_width), 3840))
        self.viewport_height = max(240, min(int(viewport_height), 2160))
        self._runtime: LiveBrowserRuntime | None = None

    async def start(self) -> None:
        if self._runtime is not None:
            return
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
            raise BrowserStreamUnavailable(str(exc)) from exc
        await runtime.sync_state(save_storage=False)
        self._runtime = runtime
        self.browser_session = runtime.browser_session

    async def snapshot(
        self,
        *,
        image_type: str = "jpeg",
        quality: int = 70,
    ) -> dict[str, Any]:
        runtime = self._require_runtime()
        payload = await runtime.snapshot(image_type=image_type, quality=quality)
        self.browser_session = runtime.browser_session
        return payload

    async def handle_client_message(self, message: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime()
        payload = await runtime.handle_client_message(message)
        self.browser_session = runtime.browser_session
        return payload

    async def close(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            await runtime.sync_state(save_storage=True)
        self._runtime = None

    def _require_runtime(self) -> LiveBrowserRuntime:
        if self._runtime is None:
            raise BrowserStreamUnavailable("Browser stream has not started.")
        return self._runtime
