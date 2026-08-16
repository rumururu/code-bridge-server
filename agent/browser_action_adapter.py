"""Browser action adapter boundary for workflow runtime.

The workflow runtime depends on this module instead of importing Playwright
directly. Tests can inject a fake adapter, while production can use the
Playwright-backed adapter when the runtime has a browser available.
"""

from __future__ import annotations

import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from agent.tool_artifacts import ARTIFACT_ROOT
from .browser_runtime_manager import get_browser_runtime_manager

# Derived from the interpreter that is actually running this server, not from a
# guessed repo-relative path. The old literal ("server/.venv/bin/python …")
# named a directory that exists in no install: the dev tree uses `server/venv`
# and a real install uses `~/.code-bridge/venv`, so the diagnostic told the
# operator to run a command that could not work.
PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND = f"{sys.executable} -m playwright install chromium"

# `_probe_browser_runtime_readiness` starts the Playwright driver process on
# every call, so anything that asks per request (commit gate, dashboard rail,
# capability catalog) would pay for a subprocess launch each time. The answer
# only changes when someone installs or removes Playwright, so a short TTL is
# enough to stop the repeat cost without going stale in a way that matters.
BROWSER_READINESS_CACHE_TTL_SECONDS = 60.0

_readiness_cache: dict[str, Any] | None = None
_readiness_cached_at: float = 0.0


@dataclass(frozen=True)
class BrowserActionAdapterResult:
    """Normalized result returned by a browser action adapter."""

    status: str
    message: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    extracted: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    wait_reason: str | None = None
    prompt: str | None = None
    storage_state_path: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def waiting_for_user(self) -> bool:
        return self.status == "waiting_for_user"

    def to_output(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "status": self.status,
            "message": self.message,
            "observations": self.observations,
            "screenshots": self.screenshots,
            "extracted": self.extracted,
        }
        if self.error is not None:
            output["error"] = self.error
        if self.wait_reason:
            output["wait_reason"] = self.wait_reason
        if self.prompt:
            output["prompt"] = self.prompt
        if self.storage_state_path:
            output["storage_state_path"] = self.storage_state_path
        return output


class BrowserActionAdapter(Protocol):
    async def run_actions(
        self,
        actions: list[dict[str, Any]],
        *,
        context: dict[str, Any],
    ) -> BrowserActionAdapterResult:
        """Run browser actions and return a normalized result."""


class PlaywrightBrowserActionAdapter:
    """Conservative Playwright adapter.

    Placeholder targets such as ``configured_url`` intentionally stop with a
    user checkpoint. The adapter only executes actions with concrete URL,
    selector, or assertion values.
    """

    async def run_actions(
        self,
        actions: list[dict[str, Any]],
        *,
        context: dict[str, Any],
    ) -> BrowserActionAdapterResult:
        if not actions:
            return BrowserActionAdapterResult(
                status="waiting_for_user",
                wait_reason="browser_actions_missing",
                prompt="이 브라우저 단계에 실행할 action이 없습니다. Builder에서 action을 추가한 뒤 다시 실행하세요.",
            )

        try:
            from playwright.async_api import async_playwright
        except Exception as exc:  # pragma: no cover - depends on environment
            return BrowserActionAdapterResult(
                status="waiting_for_user",
                wait_reason="browser_adapter_unavailable",
                prompt="서버에서 Playwright를 사용할 수 없습니다. 브라우저 작업을 직접 완료한 뒤 재개하세요.",
                error={"message": str(exc)},
            )

        observations: list[dict[str, Any]] = []
        screenshots: list[str] = []
        extracted: list[dict[str, Any]] = []
        run_id = str(context.get("run_id") or "run")
        step_id = str(context.get("step_id") or "step")
        input_storage_state_path = _existing_file_path(
            context.get("browser_input_storage_state_path")
            or context.get("browser_storage_state_path")
        )
        output_storage_state_path = _storage_state_path(
            context.get("browser_storage_state_path"),
            context.get("browser_context_dir"),
        )
        saved_storage_state_path: str | None = None

        browser_session_id = _text_or_none(context.get("browser_session_id"))
        runtime = None
        playwright = None
        browser = None
        browser_context = None
        page = None
        try:
            if browser_session_id:
                context_dir = _text_or_none(context.get("browser_context_dir"))
                if context_dir is None:
                    return BrowserActionAdapterResult(
                        status="waiting_for_user",
                        wait_reason="browser_session_missing_context_dir",
                        prompt="브라우저 handoff 세션 경로가 없어 자동화를 시작할 수 없습니다.",
                    )
                runtime = await get_browser_runtime_manager().open_session(
                    {
                        "id": browser_session_id,
                        "context_dir": context_dir,
                        "storage_state_path": str(output_storage_state_path)
                        if output_storage_state_path is not None
                        else None,
                    },
                    input_storage_state_path=input_storage_state_path,
                )
                browser_context = runtime.context
                page = runtime.page
            else:
                playwright = await async_playwright().start()
                try:
                    browser = await playwright.chromium.launch(headless=True)
                except Exception as exc:  # pragma: no cover - environment-dependent
                    return BrowserActionAdapterResult(
                        status="waiting_for_user",
                        wait_reason="browser_adapter_unavailable",
                        prompt=(
                            "서버에서 Playwright 브라우저를 실행할 수 없습니다. "
                            "브라우저 작업을 직접 완료한 뒤 재개하세요."
                        ),
                        error={"message": str(exc)},
                    )
                context_options: dict[str, Any] = {}
                if input_storage_state_path is not None:
                    context_options["storage_state"] = str(input_storage_state_path)
                browser_context = await browser.new_context(**context_options)
                page = await browser_context.new_page()

            async def with_storage_state(
                result: BrowserActionAdapterResult,
            ) -> BrowserActionAdapterResult:
                nonlocal saved_storage_state_path
                if runtime is not None:
                    saved_path = await runtime.save_storage_state()
                else:
                    saved_path = await _save_storage_state(
                        browser_context,
                        output_storage_state_path,
                    )
                saved_storage_state_path = saved_path
                return replace(result, storage_state_path=saved_path)

            try:
                for index, action in enumerate(actions, start=1):
                    action_type = str(action.get("type") or "").strip().lower()
                    if _requires_user_target(action, action_type):
                        return await with_storage_state(BrowserActionAdapterResult(
                            status="waiting_for_user",
                            wait_reason="browser_action_needs_concrete_target",
                            prompt=(
                                "브라우저 action에 실제 URL/selector/검증값이 필요합니다. "
                                "Builder에서 target을 구체화하거나 사용자가 직접 처리한 뒤 재개하세요."
                            ),
                            observations=observations,
                            screenshots=screenshots,
                            extracted=extracted,
                        ))

                    if action_type == "navigate":
                        url = str(action.get("url") or action.get("target") or "").strip()
                        await page.goto(url, wait_until="domcontentloaded")
                    elif action_type in {"click", "check", "uncheck"}:
                        selector = _selector(action)
                        if action_type == "click":
                            await page.click(selector)
                        elif action_type == "check":
                            await page.check(selector)
                        else:
                            await page.uncheck(selector)
                    elif action_type in {"type", "fill"}:
                        selector = _selector(action)
                        text = str(action.get("text") or action.get("value") or action.get("source") or "")
                        await page.fill(selector, text)
                    elif action_type == "press":
                        await page.press(_selector(action), str(action.get("key") or "Enter"))
                    elif action_type == "wait":
                        await _wait(page, action)
                    elif action_type == "assert":
                        await _assert_page(page, action)
                    elif action_type == "extract":
                        extracted.append(await _extract(page, action, index=index))
                    elif action_type == "screenshot":
                        path = _screenshot_path(run_id=run_id, step_id=step_id, index=index)
                        await page.screenshot(path=str(path), full_page=True)
                        screenshots.append(str(path))
                    elif action_type in {"select", "evaluate"}:
                        return await with_storage_state(BrowserActionAdapterResult(
                            status="waiting_for_user",
                            wait_reason=f"browser_action_{action_type}_requires_review",
                            prompt=f"{action_type} action은 현재 자동 실행하지 않습니다. 직접 확인/처리한 뒤 재개하세요.",
                            observations=observations,
                            screenshots=screenshots,
                            extracted=extracted,
                        ))
                    else:
                        return await with_storage_state(BrowserActionAdapterResult(
                            status="waiting_for_user",
                            wait_reason="browser_action_unsupported",
                            prompt=f"지원하지 않는 브라우저 action입니다: {action_type}",
                            observations=observations,
                            screenshots=screenshots,
                            extracted=extracted,
                        ))

                    observations.append(await _observe(page, action=action, index=index))
                    blocked = await _blocked_state(page)
                    if blocked is not None:
                        return await with_storage_state(BrowserActionAdapterResult(
                            status="waiting_for_user",
                            wait_reason=blocked,
                            prompt="로그인, 캡차, 권한 확인 등 수동 처리를 완료한 뒤 재개하세요.",
                            observations=observations,
                            screenshots=screenshots,
                            extracted=extracted,
                        ))
                if runtime is not None:
                    saved_storage_state_path = await runtime.save_storage_state()
                else:
                    saved_storage_state_path = await _save_storage_state(
                        browser_context,
                        output_storage_state_path,
                    )
            finally:
                if runtime is not None:
                    await runtime.sync_state(save_storage=True)
                else:
                    if browser_context is not None:
                        await browser_context.close()
                    if browser is not None:
                        await browser.close()
                    if playwright is not None:
                        await playwright.stop()
        except Exception as exc:
            return BrowserActionAdapterResult(
                status="failed",
                message="Browser action failed.",
                observations=observations,
                screenshots=screenshots,
                extracted=extracted,
                error={"message": str(exc)},
            )

        return BrowserActionAdapterResult(
            status="completed",
            message="Browser actions completed.",
            observations=observations,
            screenshots=screenshots,
            extracted=extracted,
            storage_state_path=saved_storage_state_path,
        )


def get_browser_action_adapter() -> BrowserActionAdapter:
    return PlaywrightBrowserActionAdapter()


async def get_browser_runtime_readiness(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return explicit Playwright/Chromium readiness diagnostics.

    Cached for ``BROWSER_READINESS_CACHE_TTL_SECONDS``. Pass
    ``force_refresh=True`` right after an install to re-probe immediately.
    The cache never invents readiness: an unavailable runtime stays reported
    as unavailable until a probe says otherwise.
    """
    global _readiness_cache, _readiness_cached_at

    if not force_refresh:
        cached = get_cached_browser_readiness_sync()
        if cached is not None:
            return cached

    readiness = await _probe_browser_runtime_readiness()
    _readiness_cache = deepcopy(readiness)
    _readiness_cached_at = time.monotonic()
    return readiness


def get_cached_browser_readiness_sync() -> dict[str, Any] | None:
    """Return the cached readiness snapshot, or None when there is no fresh one.

    Synchronous on purpose: callers that must not start a driver process (or
    cannot await) get either a real recent answer or ``None``. ``None`` means
    "unknown" — never treat it as ready.
    """
    if _readiness_cache is None:
        return None
    if (time.monotonic() - _readiness_cached_at) >= BROWSER_READINESS_CACHE_TTL_SECONDS:
        return None
    return deepcopy(_readiness_cache)


def reset_browser_readiness_cache() -> None:
    """Drop the cached snapshot (tests, and after an install completes)."""
    global _readiness_cache, _readiness_cached_at
    _readiness_cache = None
    _readiness_cached_at = 0.0


async def _probe_browser_runtime_readiness() -> dict[str, Any]:
    """Actually start Playwright and look for the Chromium executable."""
    readiness: dict[str, Any] = {
        "ready": False,
        "playwright_python": False,
        "chromium_executable": False,
        "chromium_executable_path": None,
        "install_command": PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
        "message": "",
        "diagnostics": [],
    }
    diagnostics: list[dict[str, Any]] = []

    try:
        from playwright.async_api import async_playwright
    except Exception as exc:
        diagnostics.append(
            {
                "code": "playwright_python_missing",
                "message": str(exc),
            }
        )
        readiness["message"] = "Python Playwright package is not available."
        readiness["diagnostics"] = diagnostics
        return readiness

    readiness["playwright_python"] = True
    try:
        async with async_playwright() as playwright:
            executable_path = str(playwright.chromium.executable_path)
    except Exception as exc:  # pragma: no cover - driver availability is host-specific
        diagnostics.append(
            {
                "code": "playwright_driver_unavailable",
                "message": str(exc),
            }
        )
        readiness["message"] = "Playwright driver is not available."
        readiness["diagnostics"] = diagnostics
        return readiness

    readiness["chromium_executable_path"] = executable_path
    chromium_exists = Path(executable_path).is_file()
    readiness["chromium_executable"] = chromium_exists
    readiness["ready"] = bool(readiness["playwright_python"] and chromium_exists)
    if chromium_exists:
        readiness["message"] = "Playwright Chromium is ready."
    else:
        diagnostics.append(
            {
                "code": "chromium_executable_missing",
                "message": (
                    "Playwright Chromium executable is missing. Run "
                    f"`{PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND}`."
                ),
            }
        )
        readiness["message"] = "Playwright Chromium executable is missing."
    readiness["diagnostics"] = diagnostics
    return readiness


def _requires_user_target(action: dict[str, Any], action_type: str) -> bool:
    if action_type == "screenshot":
        return False
    if action_type == "assert" and str(action.get("kind") or "") == "page_state_readable":
        return False
    values = [
        action.get("url"),
        action.get("selector"),
        action.get("target"),
        action.get("text"),
        action.get("value"),
    ]
    concrete = [str(value).strip() for value in values if value is not None]
    if not concrete:
        return action_type not in {"wait", "extract"}
    return any(_is_placeholder(value) for value in concrete)


def _is_placeholder(value: str) -> bool:
    text = value.strip()
    lowered = text.lower()
    return bool(
        re.fullmatch(r"configured_[a-z0-9_]+", lowered)
        or re.fullmatch(r"\{\{[a-zA-Z0-9_.-]+\}\}", text)
        or lowered.endswith("_required")
    )


def _selector(action: dict[str, Any]) -> str:
    return str(action.get("selector") or action.get("target") or "").strip()


def _existing_file_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _text_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _storage_state_path(raw_path: Any, raw_context_dir: Any) -> Path | None:
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path).expanduser()
    elif isinstance(raw_context_dir, str) and raw_context_dir.strip():
        path = Path(raw_context_dir).expanduser() / "storage_state.json"
    else:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def _save_storage_state(browser_context: Any, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        await browser_context.storage_state(path=str(path))
    except Exception:
        return None
    return str(path)


async def _wait(page: Any, action: dict[str, Any]) -> None:
    selector = str(action.get("selector") or action.get("target") or "").strip()
    if selector:
        await page.wait_for_selector(selector, timeout=int(action.get("timeout_ms") or 5000))
        return
    await page.wait_for_timeout(int(action.get("timeout_ms") or action.get("ms") or 1000))


async def _assert_page(page: Any, action: dict[str, Any]) -> None:
    kind = str(action.get("kind") or "text_visible").strip()
    if kind == "page_state_readable":
        await page.title()
        return
    if kind == "text_visible":
        value = str(action.get("value") or action.get("text") or "").strip()
        if not value:
            raise ValueError("assert text_visible requires value")
        await page.get_by_text(value).first.wait_for(timeout=int(action.get("timeout_ms") or 5000))
        return
    selector = _selector(action)
    if selector:
        await page.wait_for_selector(selector, timeout=int(action.get("timeout_ms") or 5000))
        return
    raise ValueError(f"unsupported assert kind: {kind}")


async def _extract(page: Any, action: dict[str, Any], *, index: int) -> dict[str, Any]:
    selector = _selector(action)
    if selector:
        value = await page.locator(selector).first.inner_text(timeout=int(action.get("timeout_ms") or 5000))
    else:
        value = await page.locator("body").inner_text(timeout=int(action.get("timeout_ms") or 5000))
    return {"action_index": index, "selector": selector or "body", "text": value[:10000]}


async def _observe(page: Any, *, action: dict[str, Any], index: int) -> dict[str, Any]:
    title = await page.title()
    return {
        "action_index": index,
        "action": action,
        "url": page.url,
        "title": title,
    }


async def _blocked_state(page: Any) -> str | None:
    url = ""
    try:
        url = str(page.url or "").casefold()
    except Exception:
        url = ""
    if "nid.naver.com/nidlogin" in url:
        return "login_required"

    try:
        text = await page.locator("body").inner_text(timeout=1000)
    except Exception:
        return None
    normalized = text.casefold()
    if any(marker in normalized for marker in ("captcha", "보안문자", "자동입력", "봇이 아닙니다")):
        return "captcha_or_bot_challenge"
    return None


def _screenshot_path(*, run_id: str, step_id: str, index: int) -> Path:
    directory = ARTIFACT_ROOT / _safe_name(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"browser_{_safe_name(step_id)}_{index:03d}.png"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "item"
