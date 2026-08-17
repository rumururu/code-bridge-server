"""Browser action adapter boundary for workflow runtime.

The workflow runtime depends on this module instead of importing Playwright
directly. Tests can inject a fake adapter, while production can use the
Playwright-backed adapter when the runtime has a browser available.
"""

from __future__ import annotations

import logging
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
    persistent_profile_launch_overrides,
    resolve_browser_launch_plan,
)
from system.browser_runtime_setup import (
    CHROMIUM_DISK_MB,
    CHROMIUM_DOWNLOAD_MB,
    chromium_install_command,
    detect_installed_chrome,
)
from .browser_runtime_manager import get_browser_runtime_manager, offscreen_window_args

logger = logging.getLogger(__name__)

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
        # What the page told this run. A workflow is written before it runs, so
        # runtime-only values (a cafe's numeric id, a row's link) can only get
        # into a later action through here. Seeded from earlier steps of the
        # same run: a flow naturally finds an id in one step and uses it in the
        # next, and dropping the value at the step boundary parked every such
        # flow on `{{...}}` it had already answered.
        bindings: dict[str, str] = {
            str(k): str(v)
            for k, v in (context.get("bindings") or {}).items()
            if isinstance(k, str) and v is not None
        }
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

            # Playwright dismisses dialogs by default, so a site rejecting the
            # step — "내용을 입력해주세요" on an empty editor — used to vanish
            # and the run reported `completed` while nothing was submitted.
            # Recording them is what turns that into a visible refusal.
            dialogs: list[str] = []
            _attach_dialog_recorder(page, dialogs)

            #: Where the step last pointed the browser. `None` until it points
            #: somewhere, which is why a run resumed onto an already-open page
            #: is judged on what the page is rather than on where it came from.
            requested_url: str | None = None

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
                    action = bind_action(action, bindings)
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
                        if action_type == "type":
                            # Real keypresses. A rich-text editor keeps its own
                            # document model and only listens to input events,
                            # so setting the DOM value leaves it empty: the
                            # title (a textarea) filled, the body did not, and
                            # the post was rejected for having no content.
                            await page.click(selector)
                            await page.keyboard.type(text)
                        else:
                            await page.fill(selector, text)
                    elif action_type == "press":
                        await page.press(_selector(action), str(action.get("key") or "Enter"))
                    elif action_type == "wait":
                        await _wait(page, action)
                    elif action_type == "assert":
                        await _assert_page(page, action)
                    elif action_type == "extract":
                        found = await _extract(page, action, index=index)
                        extracted.append(found)
                        if found.get("name"):
                            bindings[str(found["name"])] = str(found.get("value") or "")
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

                    if dialogs:
                        return await with_storage_state(BrowserActionAdapterResult(
                            status="waiting_for_user",
                            wait_reason="browser_dialog_appeared",
                            prompt=(
                                "사이트가 경고창으로 요청을 거절했습니다: "
                                + " / ".join(dialogs)
                            ),
                            observations=observations,
                            screenshots=screenshots,
                            extracted=extracted,
                            error={"dialogs": list(dialogs)},
                        ))

                    observations.append(
                        await _observe(
                            page, action=action, index=index, http_status=http_status
                        )
                    )
                    if action_type == "navigate":
                        # Carried forward, not read per action: the check runs
                        # after *every* action, and a `wait` right after a
                        # deliberate trip to a login page would otherwise look
                        # like a page nobody asked for and park the run.
                        requested_url = str(
                            action.get("url") or action.get("target") or ""
                        ).strip()
                    blocked = await _blocked_state(page, requested_url=requested_url)
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
        launch_options.update(persistent_profile_launch_overrides(plan))
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


#: Default and ceiling for how much of a page one `extract` puts on the record.
#: The ceiling exists because this text is stored with the run. Defined above
#: the vocabulary because that block quotes both numbers.
_EXTRACT_DEFAULT_CHARS = 10_000
_EXTRACT_MAX_CHARS = 200_000

#: What a `browser_action` step can actually do, in the words the author of a
#: workflow needs. It lives here, next to the code that dispatches these types,
#: because the Configurator writes these actions from a description and cannot
#: write what it was never told exists: the schema it saw said only
#: "navigate, click, type, ...", so every browser step it produced was a
#: `navigate` to a placeholder URL plus a screenshot — a step that parks.
#:
#: `test_browser_action_vocabulary.py` fails if this list and the dispatch below
#: disagree, which is the only thing keeping documentation and behaviour from
#: drifting apart.
BROWSER_ACTION_VOCABULARY: tuple[tuple[str, str], ...] = (
    ("navigate", 'go to a URL — {"type":"navigate","url":"https://..."}'),
    ("click", 'click an element — {"type":"click","selector":"..."} (also "check"/"uncheck"). '
              'Prefer an exact selector: :has-text("등록") also matches "임시등록"'),
    ("type", 'real keystrokes — {"type":"type","selector":"...","text":"..."}. '
             "Required for rich-text editors, which ignore a set value"),
    ("fill", 'set a field value directly — {"type":"fill","selector":"input,textarea","text":"..."}. '
             "Faster, but only works on plain inputs"),
    ("press", 'send a key — {"type":"press","selector":"...","key":"Enter"}'),
    ("wait", 'pause — {"type":"wait","timeout_ms":3000}'),
    (
        "assert",
        'check the page — {"type":"assert","kind":"text_visible","value":"..."} | '
        '{"kind":"url_contains","value":"..."} | {"kind":"url_not_contains","value":"..."}. '
        "url_not_contains is how a step proves it was not bounced to a login page",
    ),
    (
        "extract",
        'read a value and name it for later actions — '
        '{"type":"extract","name":"cafe_id","source":"html","pattern":"clubid=(\\\\d+)"}. '
        'Also {"selector":"...","attribute":"href"} for a link. '
        'Any later action may then use {{cafe_id}} inside url/selector/text — '
        "including in a later *step* of the same run. "
        "This is how a workflow reaches a page whose id only exists at runtime, "
        "instead of hardcoding one account's id. "
        f'Reading a whole page keeps {_EXTRACT_DEFAULT_CHARS} chars unless you '
        f'raise "max_chars" (up to {_EXTRACT_MAX_CHARS})',
    ),
    ("screenshot", 'capture evidence — {"type":"screenshot"}'),
)

#: Named separately because the answer to "can I use it?" is no, and a model
#: that is not told so will keep emitting them and keep parking the run.
BROWSER_ACTIONS_NOT_EXECUTED: tuple[str, ...] = ("select", "evaluate")


def browser_action_vocabulary_block() -> str:
    """The vocabulary as prompt text, for whoever authors these actions."""
    lines = ["browser_action `actions` — the full set:"]
    lines += [f"  {name:<11} {note}" for name, note in BROWSER_ACTION_VOCABULARY]
    lines.append(
        "  " + "/".join(BROWSER_ACTIONS_NOT_EXECUTED)
        + "   NOT executed — a step using these stops and asks the user"
    )
    lines.append(
        "  A URL, selector or value left as a placeholder (configured_… ) or as an "
        "unfilled {{name}} stops the step and asks, so name a real target or an "
        "{{name}} some earlier extract fills."
    )
    return "\n".join(lines)


def _attach_dialog_recorder(page: Any, sink: list[str]) -> None:
    """Record `alert`/`confirm` text so a refusal cannot pass as success.

    Playwright auto-dismisses dialogs, which is what keeps an unattended run
    from hanging — but it also means the site's own words about why it refused
    are thrown away. Measured: submitting a cafe post with an empty body raised
    an alert, Playwright dismissed it, and the step reported `completed` with
    nothing posted.

    A fake page without `.on` is common in tests, so a missing hook is not an
    error here.
    """
    on = getattr(page, "on", None)
    if not callable(on):
        return
    try:
        on("dialog", lambda dialog: sink.append(str(getattr(dialog, "message", "") or "")))
    except Exception:  # noqa: BLE001 - diagnostics must never fail the step
        logger.debug("dialog recorder could not be attached", exc_info=True)


#: Fields an action carries values in. Substitution touches these and nothing
#: else, so a `{{...}}` appearing anywhere else stays untouched and still parks
#: the run rather than being silently swallowed.
_BINDABLE_FIELDS: tuple[str, ...] = ("url", "selector", "target", "text", "value", "key")

_BINDING_REF = re.compile(r"\{\{([a-zA-Z0-9_.-]+)\}\}")


def bind_action(action: dict[str, Any], bindings: dict[str, str]) -> dict[str, Any]:
    """Fill `{{name}}` references from values earlier actions extracted.

    A workflow is authored before it is ever run, so it cannot contain the ids
    a site only reveals at runtime — a Naver cafe's numeric id lives in the
    page, not in the URL a person types. Without this the only way to reach
    such a page is to hardcode the id, which makes the workflow work for one
    cafe and no other.

    Names that were never bound are left as they are, so
    `_requires_user_target` still parks the run and asks. Guessing an empty
    string there would navigate somewhere arbitrary and call it success.
    """
    if not bindings:
        return action
    bound = dict(action)
    for field in _BINDABLE_FIELDS:
        value = bound.get(field)
        if not isinstance(value, str) or "{{" not in value:
            continue
        bound[field] = _BINDING_REF.sub(
            lambda m: bindings.get(m.group(1), m.group(0)), value
        )
    return bound


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
        # Anywhere in the string, not only as the whole of it. A reference
        # embedded in a longer URL — `.../cafes/{{cafe_id}}/menus/0` — used to
        # slip through, and the run navigated to the literal braces
        # percent-encoded and reported success.
        or _BINDING_REF.search(text) is not None
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
    if kind in {"url_contains", "url_not_contains"}:
        # Where a site *sent* you is often the only answer it gives. A page that
        # quietly redirects to a login form still returns 200 and still has a
        # title, so a step verifying "we are still signed in" has nothing else
        # to assert on.
        value = str(action.get("value") or action.get("text") or "").strip()
        if not value:
            raise ValueError(f"assert {kind} requires value")
        current = str(getattr(page, "url", "") or "")
        present = value in current
        if present is (kind == "url_contains"):
            return
        raise AssertionError(
            f"assert {kind} failed: value={value!r} url={current!r}"
        )
    selector = _selector(action)
    if selector:
        await page.wait_for_selector(selector, timeout=int(action.get("timeout_ms") or 5000))
        return
    raise ValueError(f"unsupported assert kind: {kind}")


def _extract_char_limit(action: dict[str, Any]) -> int:
    raw = action.get("max_chars")
    if raw is None:
        return _EXTRACT_DEFAULT_CHARS
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        return _EXTRACT_DEFAULT_CHARS
    return max(1, min(requested, _EXTRACT_MAX_CHARS))


async def _extract(page: Any, action: dict[str, Any], *, index: int) -> dict[str, Any]:
    selector = _selector(action)
    attribute = str(action.get("attribute") or "").strip()
    timeout = int(action.get("timeout_ms") or 5000)
    source = str(action.get("source") or "").strip().lower()
    if attribute:
        # Reading an attribute is how a run follows a link it could not have
        # known at authoring time — the href of "the cafe I am a member of".
        target = page.locator(selector).first if selector else page.locator("body")
        value = await target.get_attribute(attribute, timeout=timeout) or ""
    elif source == "html":
        # Visible text is not where a site keeps its identifiers. A cafe's id
        # sits in link hrefs and script tags, so a run that can only read what
        # a person can see cannot find it at all.
        value = await page.content()
    elif selector:
        value = await page.locator(selector).first.inner_text(timeout=timeout)
    else:
        value = await page.locator("body").inner_text(timeout=timeout)

    # A cap so a page cannot put megabytes into the run record, but one the
    # step can raise: the default silently cut a cafe's board listing in half,
    # and a step whose job is "read the page and summarise it" got a truncated
    # page with nothing saying so. Pattern matching below still runs against
    # the whole document, so a named value is never lost to this.
    limit = _extract_char_limit(action)
    result: dict[str, Any] = {
        "action_index": index,
        "selector": selector or "body",
        "text": value[:limit],
    }
    if len(value) > limit:
        result["truncated"] = True
        result["full_length"] = len(value)
    if attribute:
        result["attribute"] = attribute
    if source:
        result["source"] = source

    captured = value
    pattern = str(action.get("pattern") or "").strip()
    if pattern:
        # The page rarely holds the value alone: a cafe id arrives inside a URL
        # or a script tag. Without a pattern the run can read the page but not
        # name anything in it, which is the same as not reading it.
        match = re.search(pattern, value)
        if match is None:
            result["pattern"] = pattern
            result["matched"] = False
            return result
        captured = match.group(1) if match.groups() else match.group(0)
        result["pattern"] = pattern
        result["matched"] = True

    name = str(action.get("name") or "").strip()
    if name:
        result["name"] = name
        result["value"] = captured[:2000]
    return result


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


def _same_destination(requested: str, landed: str) -> bool:
    """Whether the browser ended up where the step pointed it.

    Compared without the fragment and trailing slash, because a site adding
    `#` or normalising `/` is not a redirect. A query string *is* significant:
    `…/nidlogin.login?url=<where you were going>` is precisely the shape a
    bounce takes.
    """
    def _norm(value: str) -> str:
        text = str(value or "").split("#", 1)[0].casefold()
        return text[:-1] if text.endswith("/") else text

    return _norm(requested) == _norm(landed)


#: Path words a sign-in page is reached through, in the languages these URLs
#: are actually written in. Only consulted for a navigation that was *diverted*
#: — on its own the word proves nothing, since a workflow may be visiting a
#: login page on purpose.
_LOGIN_URL_WORDS: tuple[str, ...] = (
    "login", "signin", "sign_in", "sign-in", "auth", "sso", "account/login",
)


async def _looks_like_a_sign_in_page(page: Any) -> bool:
    """Whether the page in front of us is asking somebody to sign in.

    Judged by what every sign-in page has rather than by whose it is. The
    check used to be the literal string `nid.naver.com/nidlogin`, which meant
    exactly one site's expiry was noticed and every other site's looked like a
    normal page: the step would read the login form, find its assertions
    satisfied or not, and finish. A silently signed-out run reporting success
    is the failure this exists to prevent, so it must not depend on which site
    it happens to.

    A visible password field is the strong signal. The URL wording is the weak
    one, kept for the sites that ask for an identifier first and show no
    password field yet — and the caller only offers it a redirected URL.
    """
    try:
        password = page.locator("input[type='password']")
        if await password.count() and await password.first.is_visible():
            return True
    except Exception:  # noqa: BLE001 - a probe must not fail the step
        pass
    return False


def _url_reads_as_sign_in(url: str) -> bool:
    lowered = str(url or "").casefold()
    return any(word in lowered for word in _LOGIN_URL_WORDS)


async def _blocked_state(page: Any, *, requested_url: str | None = None) -> str | None:
    """Why this run cannot go on without a person, or None.

    ``requested_url`` is the address the step asked for, when the last action
    was a navigation. Landing somewhere else is what distinguishes a session
    that expired from a workflow that meant to open a login page — parking on
    the latter would break any flow whose job is to sign in.
    """
    url = ""
    try:
        url = str(page.url or "")
    except Exception:
        url = ""

    diverted = bool(requested_url) and not _same_destination(requested_url, url)
    asked_for_a_sign_in_page = _url_reads_as_sign_in(requested_url or "")
    if not asked_for_a_sign_in_page and (requested_url is None or diverted):
        if await _looks_like_a_sign_in_page(page):
            return "login_required"
        if diverted and _url_reads_as_sign_in(url):
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
