"""Live browser runtime registry for agent browser handoff.

This module owns Playwright browser/page objects for workflow browser handoff.
Live objects are process-local, but persisted browser session records can be
restored after a server restart using their storage state and last known URL.
"""

from __future__ import annotations

import base64
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent.browser_session_store import get_browser_session_store
from system.browser_preferences import (
    persistent_profile_launch_overrides,
    resolve_browser_launch_plan,
)

logger = logging.getLogger(__name__)


class LiveBrowserSessionMissing(RuntimeError):
    """Raised when a browser handoff has no live page in this process."""


class LiveBrowserRuntime:
    """One live Playwright browser/context/page for a browser session."""

    def __init__(
        self,
        browser_session: dict[str, Any],
        *,
        playwright: Any,
        browser: Any,
        context: Any,
        page: Any,
        viewport_width: int,
        viewport_height: int,
        headless: bool,
        browser_owner_pid: int | None = None,
    ) -> None:
        self.browser_session = browser_session
        self.playwright = playwright
        self.browser = browser
        self.context = context
        self.page = page
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self.browser_owner_pid = browser_owner_pid
        self._closed = False

    async def set_viewport(self, width: int, height: int) -> None:
        self.viewport_width = max(320, min(int(width), 3840))
        self.viewport_height = max(240, min(int(height), 2160))
        await self.page.set_viewport_size(
            {"width": self.viewport_width, "height": self.viewport_height}
        )

    async def reset_scroll_to_top(self) -> None:
        await self.page.evaluate(
            "() => { if (window && window.scrollTo) window.scrollTo(0, 0); }"
        )

    async def clear_active_input(self) -> None:
        cleared = await self.page.evaluate(
            """
            () => {
              const el = document.activeElement;
              if (!el) return false;
              const tag = (el.tagName || "").toLowerCase();
              const editable = el.isContentEditable ||
                tag === "input" ||
                tag === "textarea";
              if (!editable) return false;
              if (el.isContentEditable) {
                el.textContent = "";
              } else {
                el.value = "";
              }
              el.dispatchEvent(new InputEvent("input", {
                bubbles: true,
                inputType: "deleteContentBackward",
                data: null,
              }));
              el.dispatchEvent(new Event("change", { bubbles: true }));
              return true;
            }
            """
        )
        if not cleared:
            await self.page.keyboard.press("Meta+A")
            await self.page.keyboard.press("Backspace")

    async def scroll_page_by(self, delta_y: float) -> None:
        await self.page.evaluate(
            """
            (deltaY) => {
              const scrollElement = (el) => {
                if (!el) return false;
                const before = el.scrollTop;
                el.scrollTop = before + deltaY;
                return Math.abs(el.scrollTop - before) > 0.5;
              };

              const root = document.scrollingElement || document.documentElement || document.body;
              const beforeRoot = root ? root.scrollTop : window.scrollY;
              window.scrollBy(0, deltaY);
              if (root && Math.abs(root.scrollTop - beforeRoot) > 0.5) return;
              if (!root && Math.abs(window.scrollY - beforeRoot) > 0.5) return;

              let target = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2);
              while (target) {
                const style = window.getComputedStyle(target);
                if (
                  target.scrollHeight > target.clientHeight &&
                  /(auto|scroll|overlay)/.test(style.overflowY)
                ) {
                  if (scrollElement(target)) return;
                }
                target = target.parentElement;
              }

              const candidates = Array.from(document.querySelectorAll("body *"));
              for (const el of candidates) {
                const style = window.getComputedStyle(el);
                if (
                  el.scrollHeight > el.clientHeight &&
                  /(auto|scroll|overlay)/.test(style.overflowY)
                ) {
                  if (scrollElement(el)) return;
                }
              }
            }
            """,
            float(delta_y),
        )

    async def snapshot(
        self,
        *,
        image_type: str = "jpeg",
        quality: int = 70,
    ) -> dict[str, Any]:
        safe_type = "png" if image_type == "png" else "jpeg"
        options: dict[str, Any] = {"type": safe_type, "full_page": False}
        if safe_type == "jpeg":
            options["quality"] = max(20, min(int(quality), 95))
        raw = await self._screenshot_bytes(
            image_type=safe_type,
            quality=max(20, min(int(quality), 95)),
        )
        await self.sync_state(save_storage=False)
        return {
            "mime_type": f"image/{safe_type}",
            "encoding": "base64",
            "data": base64.b64encode(raw).decode("ascii"),
            "width": self.viewport_width,
            "height": self.viewport_height,
            "url": self.page.url,
            "title": await _safe_title(self.page),
        }

    async def _screenshot_bytes(self, *, image_type: str, quality: int) -> bytes:
        options: dict[str, Any] = {
            "type": image_type,
            "full_page": False,
            "timeout": 5000,
        }
        if image_type == "jpeg":
            options["quality"] = quality
        try:
            return await self.page.screenshot(**options)
        except Exception:
            # Playwright screenshots can block waiting for web fonts on pages
            # such as Naver login. CDP capture does not wait on that path.
            if image_type not in {"jpeg", "png"}:
                image_type = "jpeg"
            session = await self.context.new_cdp_session(self.page)
            try:
                params: dict[str, Any] = {
                    "format": image_type,
                    "fromSurface": True,
                    "captureBeyondViewport": False,
                }
                if image_type == "jpeg":
                    params["quality"] = quality
                result = await session.send("Page.captureScreenshot", params)
            finally:
                await session.detach()
            return base64.b64decode(str(result.get("data") or ""))

    async def screenshot_bytes(self, *, image_type: str = "jpeg", quality: int = 70) -> bytes:
        safe_type = "png" if image_type == "png" else "jpeg"
        return await self._screenshot_bytes(
            image_type=safe_type,
            quality=max(20, min(int(quality), 95)),
        )

    async def handle_client_message(self, message: dict[str, Any]) -> dict[str, Any]:
        message_type = str(message.get("type") or "").strip()
        focus_result: dict[str, Any] | None = None
        try:
            if message_type in {"tap", "click"}:
                x, y = self._resolve_point(message)
                await self.page.mouse.click(x, y)
                focus_result = await self._focus_editable_at(x, y)
                logger.info(
                    "browser handoff input %s session=%s x=%.1f y=%.1f focus=%s url=%s",
                    message_type,
                    self.browser_session.get("id"),
                    x,
                    y,
                    focus_result,
                    self.page.url,
                )
            elif message_type == "pointer":
                await self._handle_pointer(message)
            elif message_type == "scroll":
                await self.page.mouse.wheel(
                    float(message.get("delta_x") or message.get("dx") or 0),
                    float(message.get("delta_y") or message.get("dy") or 0),
                )
            elif message_type == "scroll_page":
                await self.scroll_page_by(
                    float(message.get("delta_y") or message.get("dy") or 0)
                )
            elif message_type in {"type_text", "text"}:
                await self.page.keyboard.type(str(message.get("text") or ""))
            elif message_type == "key":
                key = str(message.get("key") or "").strip()
                if not key:
                    return _input_result(message_type, False, "key is required")
                await self.page.keyboard.press(key)
            elif message_type == "clear_active_input":
                await self.clear_active_input()
            elif message_type == "navigate":
                url = str(message.get("url") or "").strip()
                if not url:
                    return _input_result(message_type, False, "url is required")
                await self.page.goto(url, wait_until="domcontentloaded")
            elif message_type == "refresh":
                await self.page.reload(wait_until="domcontentloaded")
            elif message_type == "back":
                await self.page.go_back(wait_until="domcontentloaded")
            elif message_type == "forward":
                await self.page.go_forward(wait_until="domcontentloaded")
            elif message_type == "set_viewport":
                await self.set_viewport(
                    int(message.get("width") or self.viewport_width),
                    int(message.get("height") or self.viewport_height),
                )
            elif message_type == "reset_scroll":
                await self.reset_scroll_to_top()
            else:
                return _input_result(message_type, False, "unsupported input type")
        except Exception as exc:  # noqa: BLE001
            return _input_result(message_type, False, str(exc))

        await self.sync_state(save_storage=True)
        result = {
            **_input_result(message_type, True, None),
            "url": self.page.url,
            "title": await _safe_title(self.page),
        }
        if focus_result is not None:
            result["editable_focused"] = focus_result.get("focused") is True
            result["focus_target"] = focus_result
        return result

    async def save_storage_state(self) -> str:
        storage_path = _storage_state_path(self.browser_session)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        await self.context.storage_state(path=str(storage_path))
        updated = get_browser_session_store().update(
            str(self.browser_session["id"]),
            {"storage_state_path": str(storage_path)},
        )
        if updated is not None:
            self.browser_session = updated
        return str(storage_path)

    async def sync_state(self, *, save_storage: bool) -> None:
        updates: dict[str, Any] = {
            "current_url": self.page.url,
            "title": await _safe_title(self.page),
        }
        if save_storage:
            updates["storage_state_path"] = await self.save_storage_state()
        updated = get_browser_session_store().update(
            str(self.browser_session["id"]),
            updates,
        )
        if updated is not None:
            self.browser_session = updated

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            await self.sync_state(save_storage=True)
        with suppress(Exception):
            await self.context.close()
        if self.browser is not None:
            # None for a persistent profile: `launch_persistent_context` owns
            # the process and exposes no Browser, so closing the context above
            # already ended it.
            with suppress(Exception):
                await self.browser.close()
        with suppress(Exception):
            await self.playwright.stop()

    def _resolve_point(self, message: dict[str, Any]) -> tuple[float, float]:
        raw_x = message.get("x")
        raw_y = message.get("y")
        if raw_x is None or raw_y is None:
            raise ValueError("x and y are required")
        x = float(raw_x)
        y = float(raw_y)
        normalized = message.get("normalized")
        if normalized is True or (normalized is None and 0 <= x <= 1 and 0 <= y <= 1):
            x *= self.viewport_width
            y *= self.viewport_height
        return (
            max(0.0, min(x, float(self.viewport_width - 1))),
            max(0.0, min(y, float(self.viewport_height - 1))),
        )

    async def _focus_editable_at(self, x: float, y: float) -> dict[str, Any]:
        with suppress(Exception):
            result = await self.page.evaluate(
                """
                ({ x, y }) => {
                  const describe = (el, source, distance = 0) => {
                    if (!el) return { focused: false, source, distance };
                    const tag = (el.tagName || "").toLowerCase();
                    const rect = el.getBoundingClientRect();
                    return {
                      focused: true,
                      source,
                      distance,
                      tag,
                      id: el.id || "",
                      name: el.getAttribute("name") || "",
                      type: el.getAttribute("type") || "",
                      placeholder: el.getAttribute("placeholder") || "",
                      rect: {
                        left: rect.left,
                        top: rect.top,
                        width: rect.width,
                        height: rect.height,
                      },
                    };
                  };
                  const isEditable = (el) => {
                    if (!el) return false;
                    const tag = (el.tagName || "").toLowerCase();
                    return el.isContentEditable ||
                      tag === "input" ||
                      tag === "textarea";
                  };
                  const focus = (el, source, distance = 0) => {
                    el.focus({ preventScroll: true });
                    if (typeof el.click === "function") el.click();
                    return describe(el, source, distance);
                  };

                  let el = document.elementFromPoint(x, y);
                  let current = el;
                  while (current) {
                    if (isEditable(current)) return focus(current, "elementFromPoint");
                    current = current.parentElement;
                  }

                  const candidates = Array.from(
                    document.querySelectorAll("input, textarea, [contenteditable=''], [contenteditable='true']")
                  ).filter((candidate) => {
                    const style = window.getComputedStyle(candidate);
                    const rect = candidate.getBoundingClientRect();
                    return style.visibility !== "hidden" &&
                      style.display !== "none" &&
                      rect.width > 1 &&
                      rect.height > 1;
                  });
                  let best = null;
                  let bestDistance = Infinity;
                  for (const candidate of candidates) {
                    const rect = candidate.getBoundingClientRect();
                    const dx = x < rect.left ? rect.left - x :
                      x > rect.right ? x - rect.right : 0;
                    const dy = y < rect.top ? rect.top - y :
                      y > rect.bottom ? y - rect.bottom : 0;
                    const distance = Math.hypot(dx, dy);
                    if (distance < bestDistance) {
                      best = candidate;
                      bestDistance = distance;
                    }
                  }
                  if (best && bestDistance <= 90) {
                    return focus(best, "nearestEditable", bestDistance);
                  }
                  return {
                    focused: false,
                    source: "none",
                    hitTag: el ? (el.tagName || "").toLowerCase() : "",
                    hitId: el && el.id ? el.id : "",
                  };
                }
                """,
                {"x": x, "y": y},
            )
            if isinstance(result, dict):
                return result
        return {"focused": False, "source": "error"}

    async def _handle_pointer(self, message: dict[str, Any]) -> None:
        phase = str(message.get("phase") or "").strip().lower()
        x, y = self._resolve_point(message)
        if phase in {"down", "start"}:
            await self.page.mouse.move(x, y)
            await self.page.mouse.down()
        elif phase in {"move", "update"}:
            await self.page.mouse.move(x, y)
        elif phase in {"up", "end", "cancel"}:
            await self.page.mouse.move(x, y)
            await self.page.mouse.up()
        else:
            raise ValueError("pointer phase must be down, move, or up")


class BrowserRuntimeManager:
    """Process-local registry of live browser sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, LiveBrowserRuntime] = {}

    def get(self, session_id: str) -> LiveBrowserRuntime | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> LiveBrowserRuntime:
        runtime = self.get(session_id)
        if runtime is None:
            raise LiveBrowserSessionMissing(
                f"Live browser session is missing: {session_id}"
            )
        return runtime

    async def open_session(
        self,
        browser_session: dict[str, Any],
        *,
        input_storage_state_path: str | Path | None = None,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ) -> LiveBrowserRuntime:
        session_id = str(browser_session["id"])
        existing = self._sessions.get(session_id)
        if existing is not None:
            await existing.set_viewport(viewport_width, viewport_height)
            return existing

        try:
            from playwright.async_api import async_playwright
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"Playwright is not available: {exc}") from exc

        # Same three answers the workflow adapter resolves — which browser,
        # headed or headless, whose profile. The handoff must use them too:
        # this is the browser a person signs into, and a login only carries
        # into tomorrow's scheduled run if both open the same profile.
        plan = resolve_browser_launch_plan()
        if not plan.usable:
            raise RuntimeError(str(plan.blocked_message or plan.blocked_reason))

        playwright = await async_playwright().start()
        try:
            headless = plan.headless if _headless_override() is None else bool(_headless_override())
            launch_options: dict[str, Any] = {
                "headless": headless,
                "args": _browser_launch_args(
                    headless=headless,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height,
                ),
            }
            if plan.channel:
                launch_options["channel"] = plan.channel
            context_options: dict[str, Any] = {
                "viewport": {
                    "width": max(320, min(int(viewport_width), 3840)),
                    "height": max(240, min(int(viewport_height), 2160)),
                }
            }
            browser: Any = None
            if plan.persistent and plan.user_data_dir:
                Path(plan.user_data_dir).expanduser().mkdir(parents=True, exist_ok=True)
                launch_options.update(persistent_profile_launch_overrides(plan))
                context = await playwright.chromium.launch_persistent_context(
                    plan.user_data_dir,
                    **launch_options,
                    **context_options,
                )
                pages = list(getattr(context, "pages", None) or [])
                page = pages[0] if pages else await context.new_page()
            else:
                # Only this branch consults the storage state, and that is the
                # whole relationship between the two mechanisms: a persistent
                # profile already holds the cookies in `user_data_dir` (and
                # Playwright refuses `storage_state` alongside it), so the file
                # is the carrier only when there is no profile to carry them.
                # They never both apply, so they cannot disagree.
                browser = await playwright.chromium.launch(**launch_options)
                storage_state = _existing_file_path(input_storage_state_path)
                if storage_state is None:
                    storage_state = _existing_file_path(
                        browser_session.get("storage_state_path")
                    )
                if storage_state is not None:
                    context_options["storage_state"] = str(storage_state)
                context = await browser.new_context(**context_options)
                page = await context.new_page()
            initial_url = _restorable_url(browser_session.get("current_url"))
            if initial_url is not None:
                await page.goto(
                    initial_url,
                    wait_until="domcontentloaded",
                    timeout=15000,
                )
        except Exception:
            with suppress(Exception):
                await playwright.stop()
            raise

        runtime = LiveBrowserRuntime(
            browser_session,
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
            viewport_width=context_options["viewport"]["width"],
            viewport_height=context_options["viewport"]["height"],
            headless=headless,
            browser_owner_pid=_browser_owner_pid(browser if browser is not None else context),
        )
        self._sessions[session_id] = runtime
        await runtime.sync_state(save_storage=False)
        return runtime

    async def close_session(self, session_id: str) -> None:
        runtime = self._sessions.pop(session_id, None)
        if runtime is not None:
            await runtime.close()

    async def close_all(self) -> None:
        session_ids = list(self._sessions)
        for session_id in session_ids:
            await self.close_session(session_id)


_browser_runtime_manager: BrowserRuntimeManager | None = None
def get_browser_runtime_manager() -> BrowserRuntimeManager:
    global _browser_runtime_manager
    if _browser_runtime_manager is None:
        _browser_runtime_manager = BrowserRuntimeManager()
    return _browser_runtime_manager


def _headless_override() -> bool | None:
    """`CODEBRIDGE_BROWSER_HEADLESS` still wins, when it is set at all.

    It predates the stored setting and is what a developer reaches for to watch
    one session. Unset (the normal case) means "no override" so the operator's
    stored choice decides — the old default of 1 would have overridden that
    choice on every machine.
    """
    raw = os.environ.get("CODEBRIDGE_BROWSER_HEADLESS")
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    return value not in {"0", "false", "no", "off"}


def offscreen_window_args(
    *,
    viewport_width: int = 1280,
    viewport_height: int = 720,
) -> list[str]:
    """Window args for a headed launch that must not steal the operator's screen."""
    return _browser_launch_args(
        headless=False,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )


def _browser_launch_args(
    *,
    headless: bool,
    viewport_width: int,
    viewport_height: int,
) -> list[str]:
    if headless:
        return []
    mode = os.environ.get("CODEBRIDGE_BROWSER_WINDOW", "offscreen").strip().lower()
    if mode in {"visible", "foreground", "debug"}:
        return []
    width = max(320, min(int(viewport_width), 3840))
    height = max(240, min(int(viewport_height), 2160))
    return [
        f"--window-size={width},{height}",
        "--window-position=-32000,-32000",
    ]


def _browser_window_mode() -> str:
    return os.environ.get("CODEBRIDGE_BROWSER_WINDOW", "offscreen").strip().lower()


def _restorable_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url or url == "about:blank":
        return None
    if url.startswith(("http://", "https://", "file://")):
        return url
    return None


def _browser_owner_pid(browser: Any) -> int | None:
    candidates = [
        ("process",),
        ("_impl_obj", "_connection", "_transport", "_proc"),
        ("_connection", "_transport", "_proc"),
    ]
    for path in candidates:
        value: Any = browser
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is None:
            continue
        pid = getattr(value, "pid", None)
        if callable(pid):
            with suppress(Exception):
                pid = pid()
        with suppress(TypeError, ValueError):
            parsed = int(pid)
            if parsed > 0:
                return parsed
    return None


def _existing_file_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value if value.is_file() else None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _storage_state_path(browser_session: dict[str, Any]) -> Path:
    explicit = browser_session.get("storage_state_path")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit).expanduser()
    context_dir = Path(str(browser_session["context_dir"])).expanduser()
    return context_dir / "storage_state.json"


async def _safe_title(page: Any) -> str:
    with suppress(Exception):
        title = await page.title()
        if isinstance(title, str):
            return title
    return ""


def _input_result(input_type: str, ok: bool, error: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "browser_stream.input_result",
        "input_type": input_type,
        "ok": ok,
    }
    if error:
        payload["error"] = error
    return payload
