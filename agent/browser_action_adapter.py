"""Browser action adapter boundary for workflow runtime.

The workflow runtime depends on this module instead of importing Playwright
directly. Tests can inject a fake adapter, while production can use the
Playwright-backed adapter when the runtime has a browser available.
"""

from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from agent.tool_artifacts import ARTIFACT_ROOT
from system.browser_preferences import (
    BrowserLaunchPlan,
    get_browser_preferences,
    resolve_browser_launch_plan,
)
from system.browser_runtime_setup import (
    CHROMIUM_DISK_MB,
    CHROMIUM_DOWNLOAD_MB,
    chromium_install_command,
    detect_installed_chrome,
)
from .browser_runtime_manager import get_browser_runtime_manager, offscreen_window_args

# Derived from the interpreter that is actually running this server, not from a
# guessed repo-relative path. The old literal ("server/.venv/bin/python …")
# named a directory that exists in no install: the dev tree uses `server/venv`
# and a real install uses `~/.code-bridge/venv`, so the diagnostic told the
# operator to run a command that could not work.
#
# Defined in `system/browser_runtime_setup.py` so the command this server tells
# you to run, the command the installers run, and the command the dashboard's
# one-click install runs are literally the same string. Three copies in three
# scripts is what let the installer grow a Chromium step that the deploy path
# in daily use never called.
PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND = chromium_install_command()

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
        # Tried in order, each candidate checked for itself. This used to be
        # `_existing_file_path(a or b)`, which is not the same thing: a truthy
        # `a` naming a file that does not exist yet consumed the expression, so
        # `b` was never reached and neither was anything after it.
        #
        # The orchestrator resolves the handoff itself and normally supplies
        # the first candidate (`task_orchestrator._previous_browser_session_for_execution`).
        # The task-scoped lookup stays here as the answer for any caller that
        # does not — it asks the same store method, so the two cannot disagree.
        input_storage_state_path = _first_existing_file(
            context.get("browser_input_storage_state_path"),
            context.get("browser_storage_state_path"),
            lambda: _previous_run_storage_state(
                task_id=context.get("task_id"),
                run_id=context.get("run_id"),
            ),
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
                # The operator's three answers — which browser, headed or
                # headless, whose profile — resolved once, here. Nothing below
                # decides any of them on its own.
                plan = resolve_browser_launch_plan()
                if not plan.usable:
                    return BrowserActionAdapterResult(
                        status="waiting_for_user",
                        wait_reason=str(plan.blocked_reason),
                        prompt=str(plan.blocked_message or ""),
                    )
                playwright = await async_playwright().start()
                try:
                    browser, browser_context = await _launch_from_plan(
                        playwright,
                        plan,
                        storage_state_path=input_storage_state_path,
                    )
                except Exception as exc:  # pragma: no cover - environment-dependent
                    return BrowserActionAdapterResult(
                        status="waiting_for_user",
                        wait_reason="browser_adapter_unavailable",
                        prompt=(
                            f"서버에서 브라우저({plan.label})를 실행할 수 없습니다. "
                            "브라우저 작업을 직접 완료한 뒤 재개하세요."
                        ),
                        error={"message": str(exc)},
                    )
                pages = list(getattr(browser_context, "pages", None) or [])
                page = pages[0] if pages else await browser_context.new_page()

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

                    http_status: int | None = None
                    if action_type == "navigate":
                        url = str(action.get("url") or action.get("target") or "").strip()
                        response = await page.goto(url, wait_until="domcontentloaded")
                        http_status = _response_status(response)
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

                    observations.append(
                        await _observe(
                            page, action=action, index=index, http_status=http_status
                        )
                    )
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


async def _launch_from_plan(
    playwright: Any,
    plan: BrowserLaunchPlan,
    *,
    storage_state_path: Path | None,
) -> tuple[Any, Any]:
    """Start the browser the plan describes and return ``(browser, context)``.

    ``browser`` is ``None`` for a persistent profile: Playwright's
    ``launch_persistent_context`` owns the process itself and exposes no
    ``Browser``. Closing the context closes everything, which is what the
    caller's ``finally`` already does.
    """
    launch_options: dict[str, Any] = {"headless": plan.headless}
    if plan.channel:
        launch_options["channel"] = plan.channel
    if not plan.headless:
        launch_options["args"] = offscreen_window_args()

    if plan.persistent and plan.user_data_dir:
        Path(plan.user_data_dir).expanduser().mkdir(parents=True, exist_ok=True)
        context = await playwright.chromium.launch_persistent_context(
            plan.user_data_dir,
            **launch_options,
        )
        return None, context

    browser = await playwright.chromium.launch(**launch_options)
    context_options: dict[str, Any] = {}
    if storage_state_path is not None:
        context_options["storage_state"] = str(storage_state_path)
    context = await browser.new_context(**context_options)
    return browser, context


def _previous_run_storage_state(*, task_id: Any, run_id: Any) -> str | None:
    """The storage state the previous run of this task left behind, if any."""
    task = _text_or_none(task_id)
    if task is None:
        return None
    try:
        from agent.browser_session_store import get_browser_session_store

        return get_browser_session_store().latest_storage_state_for_task(
            task,
            exclude_run_id=_text_or_none(run_id),
        )
    except Exception:  # noqa: BLE001 - a missing carry-over is not a run failure
        return None


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
    """Actually start Playwright, look for an installed Chrome, and for Chromium.

    Both browsers are decided by a file on disk. Neither is assumed from the
    platform: "this is a Mac, so Chrome is probably there" would let this
    surface report that no download is needed and then park every browser step
    at run time, which is the failure it exists to prevent.
    """
    preferences = get_browser_preferences()
    chrome = detect_installed_chrome()
    readiness: dict[str, Any] = {
        "ready": False,
        "playwright_python": False,
        "chromium_executable": False,
        "chromium_executable_path": None,
        "install_command": PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND,
        # Stated wherever the install is offered. A step that goes quiet for
        # four minutes with no size given reads as hung, and users kill it.
        "install_download_mb": CHROMIUM_DOWNLOAD_MB,
        "install_disk_mb": CHROMIUM_DISK_MB,
        # False whenever a browser step can run without fetching anything —
        # which is the case on every machine that already has Chrome.
        "install_required": True,
        "installed_chrome": chrome.as_dict() if chrome is not None else None,
        "preferences": preferences.as_dict(),
        "plan": None,
        "message": "",
        "diagnostics": [],
    }
    diagnostics: list[dict[str, Any]] = []

    def finish(*, chromium_present: bool | None, chromium_path: str | None) -> dict[str, Any]:
        plan = resolve_browser_launch_plan(
            preferences,
            chrome=chrome,
            detect_chrome=False,
            chromium_executable_path=chromium_path,
            chromium_present=chromium_present,
        )
        readiness["plan"] = plan.as_dict()
        readiness["install_required"] = bool(plan.install_required)
        if not plan.usable:
            readiness["ready"] = False
            readiness["message"] = str(plan.blocked_message or "")
            diagnostics.append(
                {"code": str(plan.blocked_reason), "message": readiness["message"]}
            )
        elif plan.browser == "chrome":
            readiness["ready"] = True
            readiness["message"] = (
                f"{plan.label} will run browser steps — no download needed."
            )
        elif chromium_present:
            readiness["ready"] = True
            readiness["message"] = "Playwright Chromium is ready."
        else:
            readiness["ready"] = False
            readiness["message"] = "Playwright Chromium executable is missing."
            diagnostics.append(
                {
                    "code": "chromium_executable_missing",
                    "message": (
                        "Playwright Chromium executable is missing. Install "
                        "Google Chrome, or run "
                        f"`{PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND}`."
                    ),
                }
            )
        readiness["diagnostics"] = diagnostics
        return readiness

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
        # Without the package no browser runs, installed Chrome or not.
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
    return finish(chromium_present=chromium_exists, chromium_path=executable_path)


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


def _first_existing_file(*candidates: Any) -> Path | None:
    """First candidate that names a file on disk. Callables are asked lazily.

    Lazy so the cross-run store lookup is not paid for on the common path
    where the orchestrator already supplied the state.
    """
    for candidate in candidates:
        value = candidate() if callable(candidate) else candidate
        path = _existing_file_path(value)
        if path is not None:
            return path
    return None


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


def _response_status(response: Any) -> int | None:
    """HTTP status of a navigation, when there was one.

    ``page.goto`` answers ``None`` for a same-document navigation, and some
    fakes have no ``status`` at all, so this never raises — an unknown status
    is reported as unknown.
    """
    if response is None:
        return None
    status = getattr(response, "status", None)
    if callable(status):
        try:
            status = status()
        except Exception:  # noqa: BLE001 - a diagnostic must not fail the step
            return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


async def _observe(
    page: Any,
    *,
    action: dict[str, Any],
    index: int,
    http_status: int | None = None,
) -> dict[str, Any]:
    title = await page.title()
    observation: dict[str, Any] = {
        "action_index": index,
        "action": action,
        "url": page.url,
        "title": title,
    }
    # Recorded, not acted on. The run record used to contain no trace of the
    # server having answered 4xx/5xx: a navigate onto a "503 Service
    # Temporarily Unavailable" page completed silently, and whatever failed
    # next got the blame — an assertion timeout, or an action reported as
    # having no concrete target. The status belongs in the observation so the
    # real reason is on the record.
    #
    # Deliberately not a `waiting_for_user`: "is the site up?" is one of the
    # things these agents are for, and parking on a 503 would break the very
    # check that wants to see it.
    if http_status is not None:
        observation["http_status"] = http_status
        observation["http_ok"] = http_status < 400
    return observation


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
