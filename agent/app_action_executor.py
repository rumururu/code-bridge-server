"""Workflow-facing Android app action execution helper.

Two callers, one judgement. The adapter below runs the actions; the commit-time
gate (``workflow_contract``) asks this module, through ``app_action_gap``,
whether an action *would* park before the workflow is ever saved. Both read the
same table, so the gate cannot pass a payload the runtime refuses, and cannot
refuse one the runtime is happy to run.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

from agent.tool_artifacts import ARTIFACT_ROOT

from .app_action_adapter import AppActionAdapter, AppActionAdapterResult

_PLACEHOLDER_RE = re.compile(
    r"(configured|placeholder|todo|from_current_screen|current_screen|"
    r"selected_request|target_app|app_from_request|request_card|"
    r"join_or_apply|auto_detect|detected|"
    # The authoring layer's own placeholder vocabulary. Without these the
    # detector only knew the words the browser side happened to use, so
    # `{"type": "open_play_store", "source": "user_provided_store_or_package"}`
    # -- written by the server itself -- passed as a real package name and
    # launched `market://details?id=user_provided_store_or_package` on the
    # device. A wrong action that runs is worse than one that stops.
    r"user_provided|previous_step|"
    # `{{x}}` interpolation is not implemented anywhere in the runtime, and
    # `_required` is the browser adapter's suffix for the same idea
    # (`browser_action_adapter._is_placeholder`).
    r"\{\{[^}]*\}\}|_required\b)",
    re.IGNORECASE,
)
_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

# Every action type this adapter executes. `workflow_v2.ALLOWED_ACTION_TYPES`
# is a single set shared with browser steps, so it also admits `navigate`,
# `click`, `fill` and friends into an app step; those are browser verbs with no
# device meaning and are refused, at commit time now rather than at 3am.
SUPPORTED_APP_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "read_screen",
        "read_ui",
        "dump_ui",
        "screenshot",
        "wait",
        "wait_text",
        "tap",
        "tap_text",
        "input_text",
        "type_text",
        "press_key",
        "open_play_store",
        "install_app",
        "launch_app",
        "open_app",
        "close_app",
        "verify_launch",
    }
)

# `wait_reason` values. The runtime already published these; the gate reuses
# them as finding codes so a commit refusal and a parked run name the same
# problem the same way.
REASON_UNSUPPORTED_ACTION = "app_action_unsupported"
REASON_NEEDS_CONCRETE_TARGET = "app_action_needs_concrete_target"
REASON_NEEDS_PACKAGE = "app_action_needs_package_name"
REASON_NEEDS_APK = "app_install_needs_package_or_apk"
REASON_NEEDS_TEXT = "app_action_needs_text"
REASON_TEXT_NOT_TYPEABLE = "app_action_text_not_typeable"
REASON_NEEDS_KEY = "app_action_needs_key"
REASON_ACTIONS_MISSING = "app_actions_missing"

_PACKAGE_FIELDS: tuple[str, ...] = ("package", "package_name", "app_id", "app", "source")
_TEXT_TARGET_FIELDS: tuple[str, ...] = ("text", "target", "label")
_APK_FIELDS: tuple[str, ...] = ("apk_path", "path")
_TYPED_TEXT_FIELDS: tuple[str, ...] = ("text", "value")
_KEY_FIELDS: tuple[str, ...] = ("key", "keycode", "key_code", "text", "target")

_NEEDS_CONCRETE_TARGET_PROMPT = (
    "앱 action에 실제 화면 텍스트, 패키지명, APK 경로 같은 구체 target이 필요합니다. "
    "Builder에서 target을 구체화하거나 사용자가 직접 처리한 뒤 재개하세요."
)
_NEEDS_PACKAGE_PROMPT = (
    "이 앱 action에는 실제 Android 패키지명이 필요합니다. "
    "Builder에서 package 값을 지정한 뒤 다시 실행하세요."
)
_NEEDS_APK_PROMPT = (
    "앱 설치 action에 실제 APK 경로가 없습니다. adb는 Play Store 설치를 대신할 수 없으므로 "
    "서버에서 읽을 수 있는 apk_path를 지정한 뒤 다시 실행하세요."
)
_NEEDS_TEXT_PROMPT = (
    "입력 action에 실제로 입력할 text가 없습니다. Builder에서 입력값을 지정한 뒤 다시 실행하세요."
)
_TEXT_NOT_TYPEABLE_PROMPT = (
    "adb `input text`는 ASCII 문자만 입력할 수 있어서 이 값은 기기에 입력되지 않습니다. "
    "한글 등은 사용자가 직접 입력하거나 다른 방법을 쓰세요."
)

# `adb shell input keyevent` names. An allowlist, not a guess: an unrecognised
# key stops the run instead of pressing something else.
_KEY_ALIASES: dict[str, str] = {
    "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME",
    "enter": "KEYCODE_ENTER",
    "done": "KEYCODE_ENTER",
    "tab": "KEYCODE_TAB",
    "space": "KEYCODE_SPACE",
    "delete": "KEYCODE_DEL",
    "backspace": "KEYCODE_DEL",
    "escape": "KEYCODE_ESCAPE",
    "esc": "KEYCODE_ESCAPE",
    "search": "KEYCODE_SEARCH",
    "menu": "KEYCODE_MENU",
    "power": "KEYCODE_POWER",
    "recent": "KEYCODE_APP_SWITCH",
    "recents": "KEYCODE_APP_SWITCH",
    "app_switch": "KEYCODE_APP_SWITCH",
    "up": "KEYCODE_DPAD_UP",
    "down": "KEYCODE_DPAD_DOWN",
    "left": "KEYCODE_DPAD_LEFT",
    "right": "KEYCODE_DPAD_RIGHT",
}
_KEYCODE_RE = re.compile(r"^KEYCODE_[A-Z0-9_]+$")


@dataclass(frozen=True)
class AppActionGap:
    """Why the runtime would stop on this action, judged from its payload alone.

    ``field`` is the action key the author has to fill — the one they actually
    wrote when there is one, so a UI can patch in place rather than adding a
    second spelling of the same thing. ``message`` is the sentence the runtime
    would show, reused verbatim by the commit gate.
    """

    reason: str
    field: str
    value: str
    message: str


async def execute_app_actions(
    actions: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    adapter: AppActionAdapter | None = None,
) -> AppActionAdapterResult:
    """Execute normalized Android app actions through an injectable adapter."""
    if not actions:
        gap = app_actions_missing_gap()
        return AppActionAdapterResult(
            status="waiting_for_user",
            wait_reason=gap.reason,
            prompt=gap.message,
        )
    runner = adapter or AdbAppActionAdapter()
    return await runner.run_actions(actions, context=context)


class AdbAppActionAdapter:
    """Conservative ADB adapter for Android app workflow steps."""

    async def run_actions(
        self,
        actions: list[dict[str, Any]],
        *,
        context: dict[str, Any],
    ) -> AppActionAdapterResult:
        adb_path = _resolve_adb_path()
        if adb_path is None:
            return AppActionAdapterResult(
                status="waiting_for_user",
                wait_reason="adb_unavailable",
                prompt="서버에서 adb를 찾을 수 없습니다. Android 기기 작업을 직접 완료한 뒤 재개하세요.",
            )

        device_id, device_error = await _resolve_device_id(adb_path, context)
        if device_error:
            return AppActionAdapterResult(
                status="waiting_for_user",
                wait_reason=device_error["reason"],
                prompt=device_error["prompt"],
                observations=device_error.get("observations", []),
            )

        observations: list[dict[str, Any]] = []
        screenshots: list[str] = []
        run_id = str(context.get("run_id") or "run")
        step_id = str(context.get("step_id") or "step")
        assert device_id is not None

        for index, action in enumerate(actions, start=1):
            action_type = str(action.get("type") or "").strip().lower()
            # One judgement, shared with the commit gate. Anything this
            # returns is a stop the author could have been told about before
            # the workflow was saved.
            gap = app_action_gap(action)
            if gap is not None:
                return _gap_result(gap, action, observations, screenshots)

            if action_type in {"read_screen", "read_ui", "dump_ui"}:
                xml = await _dump_ui(adb_path, device_id)
                observations.append(
                    {
                        "type": "ui_dump",
                        "device_id": device_id,
                        "text": _summarize_ui_xml(xml),
                        "xml": _truncate(xml, 12000),
                    }
                )
                continue

            if action_type == "screenshot":
                screenshot = await _screenshot(
                    adb_path,
                    device_id,
                    run_id=run_id,
                    step_id=step_id,
                    index=index,
                )
                screenshots.append(str(screenshot))
                observations.append(
                    {
                        "type": "screenshot",
                        "device_id": device_id,
                        "path": str(screenshot),
                    }
                )
                continue

            if action_type == "wait":
                seconds = _safe_wait_seconds(action)
                await asyncio.sleep(seconds)
                observations.append({"type": "wait", "seconds": seconds})
                continue

            if action_type == "wait_text":
                target = _action_text(action)
                timeout = _wait_text_timeout(action)
                found, xml = await _poll_for_text(
                    adb_path,
                    device_id,
                    target,
                    timeout=timeout,
                )
                if not found:
                    return _target_not_found(
                        target,
                        device_id,
                        xml,
                        observations,
                        screenshots,
                        waited_seconds=timeout,
                    )
                observations.append(
                    {"type": "wait_text", "target": target, "timeout": timeout}
                )
                continue

            if action_type == "tap":
                coordinates = _tap_coordinates(action)
                if coordinates is not None:
                    x, y = coordinates
                    await _adb(
                        adb_path, "-s", device_id, "shell", "input", "tap", str(x), str(y)
                    )
                    observations.append({"type": "tap", "x": x, "y": y})
                    continue
                # No coordinates: the gap check already established there is a
                # concrete on-screen target, so resolve it the same way
                # `tap_text` does rather than inventing a position.
                action_type = "tap_text"

            if action_type == "tap_text":
                target = _action_text(action)
                xml = await _dump_ui(adb_path, device_id)
                bounds = _find_text_bounds(xml, target)
                if bounds is None:
                    return _target_not_found(
                        target, device_id, xml, observations, screenshots
                    )
                x1, y1, x2, y2 = bounds
                await _adb(adb_path, "-s", device_id, "shell", "input", "tap", str((x1 + x2) // 2), str((y1 + y2) // 2))
                observations.append({"type": "tap_text", "target": target, "bounds": bounds})
                continue

            if action_type in {"input_text", "type_text"}:
                text = _typed_text(action)
                # `input text` reads a space as an argument separator; `%s` is
                # its own escape for one. Quoting is for the device shell adb
                # hands the command to.
                await _adb(
                    adb_path,
                    "-s",
                    device_id,
                    "shell",
                    "input",
                    "text",
                    shlex.quote(text.replace(" ", "%s")),
                )
                observations.append({"type": action_type, "text": text})
                continue

            if action_type == "press_key":
                keyevent = _resolve_keyevent(action)
                assert keyevent is not None  # the gap check refused None above
                await _adb(
                    adb_path, "-s", device_id, "shell", "input", "keyevent", keyevent
                )
                observations.append({"type": "press_key", "keyevent": keyevent})
                continue

            if action_type == "open_play_store":
                package_name = _action_package(action)
                await _adb(
                    adb_path,
                    "-s",
                    device_id,
                    "shell",
                    "am",
                    "start",
                    "-a",
                    "android.intent.action.VIEW",
                    "-d",
                    f"market://details?id={package_name}",
                )
                observations.append({"type": "open_play_store", "package": package_name})
                continue

            if action_type == "install_app":
                apk_path = _action_apk_path(action)
                if apk_path is None:
                    # The payload named a concrete path (the gap check saw to
                    # that) and the server cannot read it. That is an
                    # environment problem, not an authoring one, so it can only
                    # be found here.
                    return AppActionAdapterResult(
                        status="waiting_for_user",
                        wait_reason=REASON_NEEDS_APK,
                        prompt=(
                            f"설치할 APK 파일을 서버에서 찾지 못했습니다: "
                            f"{_first_value(action, _APK_FIELDS) or '(경로 없음)'}. "
                            "파일을 올려둔 뒤 재개하세요."
                        ),
                        observations=observations,
                        screenshots=screenshots,
                    )
                await _adb(adb_path, "-s", device_id, "install", "-r", str(apk_path), timeout=120)
                observations.append({"type": "install_app", "apk_path": str(apk_path)})
                continue

            if action_type in {"verify_launch", "launch_app", "open_app"}:
                package_name = _action_package(action)
                component = await _resolve_launch_component(
                    adb_path,
                    device_id,
                    package_name,
                )
                await _adb(adb_path, "-s", device_id, "shell", "am", "start", "-n", component)
                observations.append({"type": action_type, "package": package_name})
                continue

            if action_type == "close_app":
                package_name = _action_package(action)
                await _adb(
                    adb_path,
                    "-s",
                    device_id,
                    "shell",
                    "am",
                    "force-stop",
                    package_name,
                )
                observations.append({"type": "close_app", "package": package_name})
                continue

            return _unsupported_action(action, observations, screenshots)

        return AppActionAdapterResult(
            status="completed",
            message="Android app actions completed.",
            observations=observations,
            screenshots=screenshots,
        )


def _resolve_adb_path() -> str | None:
    return (
        os.environ.get("CODEBRIDGE_ADB_PATH")
        or shutil.which("adb")
        or _existing_path(Path.home() / "Library/Android/sdk/platform-tools/adb")
        or _existing_path(Path.home() / "Android/Sdk/platform-tools/adb")
    )


async def _resolve_device_id(
    adb_path: str,
    context: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    candidates = [
        context.get("android_device_id"),
        context.get("device_id"),
        context.get("adb_device_id"),
        os.environ.get("CODEBRIDGE_AGENT_ANDROID_DEVICE_ID"),
        os.environ.get("ANDROID_SERIAL"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip(), None

    output = await _adb(adb_path, "devices", timeout=10)
    devices = _parse_adb_devices(output)
    if len(devices) == 1:
        return devices[0], None
    if not devices:
        return None, {
            "reason": "android_device_not_connected",
            "prompt": "서버에서 연결된 Android 기기를 찾지 못했습니다. 기기를 연결하거나 에뮬레이터를 실행한 뒤 재개하세요.",
        }
    return None, {
        "reason": "android_device_not_selected",
        "prompt": (
            "연결된 Android 기기가 여러 개라 자동 실행 대상을 고를 수 없습니다. "
            "Agent 실행 설정에 device_id를 지정한 뒤 다시 실행하세요."
        ),
        "observations": [{"type": "adb_devices", "devices": devices}],
    }


async def _dump_ui(adb_path: str, device_id: str) -> str:
    await _adb(adb_path, "-s", device_id, "shell", "uiautomator", "dump", "/sdcard/_cb_agent_ui.xml")
    return await _adb(adb_path, "-s", device_id, "shell", "cat", "/sdcard/_cb_agent_ui.xml")


async def _resolve_launch_component(
    adb_path: str,
    device_id: str,
    package_name: str,
) -> str:
    output = await _adb(
        adb_path,
        "-s",
        device_id,
        "shell",
        "cmd",
        "package",
        "resolve-activity",
        "--brief",
        "-c",
        "android.intent.category.LAUNCHER",
        package_name,
    )
    for line in reversed(output.splitlines()):
        text = line.strip()
        if "/" in text and not text.startswith(("priority=", "No activity")):
            return text
    raise RuntimeError(f"No launchable activity found for package {package_name}")


async def _screenshot(
    adb_path: str,
    device_id: str,
    *,
    run_id: str,
    step_id: str,
    index: int,
) -> Path:
    directory = ARTIFACT_ROOT / "app_action" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{step_id}_{index}.png"
    data = await _adb_bytes(adb_path, "-s", device_id, "exec-out", "screencap", "-p", timeout=20)
    path.write_bytes(data)
    return path


async def _adb(adb_path: str, *args: str, timeout: float = 30) -> str:
    def run() -> str:
        try:
            completed = subprocess.run(
                [adb_path, *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            command = " ".join([adb_path, *args])
            raise RuntimeError(f"adb command timed out after {timeout:g}s: {command}") from exc
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(stderr.strip() or stdout.strip() or "adb command failed")
        return stdout

    return await asyncio.to_thread(run)


async def _adb_bytes(adb_path: str, *args: str, timeout: float = 30) -> bytes:
    def run() -> bytes:
        try:
            completed = subprocess.run(
                [adb_path, *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            command = " ".join([adb_path, *args])
            raise RuntimeError(f"adb command timed out after {timeout:g}s: {command}") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(stderr.strip() or "adb command failed")
        return completed.stdout

    return await asyncio.to_thread(run)


def _parse_adb_devices(output: str) -> list[str]:
    devices: list[str] = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def _find_text_bounds(xml: str, target: str) -> tuple[int, int, int, int] | None:
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return None
    needle = target.casefold()
    for node in root.iter("node"):
        values = [
            node.attrib.get("text", ""),
            node.attrib.get("content-desc", ""),
            node.attrib.get("resource-id", ""),
        ]
        if any(needle in value.casefold() for value in values if value):
            match = _BOUNDS_RE.fullmatch(node.attrib.get("bounds", ""))
            if match:
                return tuple(int(value) for value in match.groups())  # type: ignore[return-value]
    return None


def _summarize_ui_xml(xml: str) -> str:
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return _truncate(xml, 2000)
    snippets: list[str] = []
    for node in root.iter("node"):
        text = node.attrib.get("text") or node.attrib.get("content-desc")
        if text:
            snippets.append(text)
        if len(snippets) >= 80:
            break
    return "\n".join(snippets)


def _unsupported_action(
    action: dict[str, Any],
    observations: list[dict[str, Any]],
    screenshots: list[str],
) -> AppActionAdapterResult:
    return AppActionAdapterResult(
        status="waiting_for_user",
        wait_reason=REASON_UNSUPPORTED_ACTION,
        prompt=_unsupported_action_message(action),
        observations=observations,
        screenshots=screenshots,
    )


def _target_not_found(
    target: str,
    device_id: str,
    xml: str,
    observations: list[dict[str, Any]],
    screenshots: list[str],
    *,
    waited_seconds: float | None = None,
) -> AppActionAdapterResult:
    waited = f" {waited_seconds:g}초 기다렸습니다." if waited_seconds else ""
    return AppActionAdapterResult(
        status="waiting_for_user",
        wait_reason="app_action_target_not_found",
        prompt=(
            f"현재 Android 화면에서 '{target}' 텍스트를 찾지 못했습니다.{waited} "
            "사용자가 화면을 맞춘 뒤 재개하거나 Builder에서 더 구체적인 target을 지정하세요."
        ),
        observations=[
            *observations,
            {
                "type": "ui_dump",
                "device_id": device_id,
                "target": target,
                "text": _summarize_ui_xml(xml),
            },
        ],
        screenshots=screenshots,
    )


async def _poll_for_text(
    adb_path: str,
    device_id: str,
    target: str,
    *,
    timeout: float,
) -> tuple[bool, str]:
    """Dump the screen until ``target`` shows up or the budget runs out."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        xml = await _dump_ui(adb_path, device_id)
        if _find_text_bounds(xml, target) is not None:
            return True, xml
        if asyncio.get_running_loop().time() >= deadline:
            return False, xml
        await asyncio.sleep(min(1.0, timeout))


def _safe_wait_seconds(action: dict[str, Any]) -> float:
    value = action.get("seconds") or action.get("timeout") or 1
    try:
        return max(0.0, min(float(value), 10.0))
    except (TypeError, ValueError):
        return 1.0


def _wait_text_timeout(action: dict[str, Any]) -> float:
    value = action.get("timeout") or action.get("seconds") or 10
    try:
        return max(1.0, min(float(value), 30.0))
    except (TypeError, ValueError):
        return 10.0


def _tap_coordinates(action: dict[str, Any]) -> tuple[int, int] | None:
    x = action.get("x")
    y = action.get("y")
    if x is None or y is None:
        pair = action.get("coordinates") or action.get("position")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            x, y = pair
    if x is None or y is None:
        return None
    try:
        px, py = int(x), int(y)
    except (TypeError, ValueError):
        return None
    return (px, py) if px >= 0 and py >= 0 else None


def _action_text(action: dict[str, Any]) -> str:
    return _first_value(action, _TEXT_TARGET_FIELDS)


def _typed_text(action: dict[str, Any]) -> str:
    return _first_value(action, _TYPED_TEXT_FIELDS)


def _action_package(action: dict[str, Any]) -> str:
    # `app` is what the builder's own `_verify_launch_actions` writes
    # (`configurator.py:1866-1868`); it used to read as an empty package and
    # park every launch step the server itself authored.
    return _first_value(action, _PACKAGE_FIELDS)


def _first_value(action: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field_name in fields:
        value = action.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_present_field(
    action: dict[str, Any],
    fields: tuple[str, ...],
) -> str | None:
    for field_name in fields:
        value = action.get(field_name)
        if isinstance(value, str) and value.strip():
            return field_name
    return None


def _resolve_keyevent(action: dict[str, Any]) -> str | None:
    raw = _first_value(action, _KEY_FIELDS)
    if not raw:
        return None
    if raw.isdigit():
        return raw
    upper = raw.upper()
    if _KEYCODE_RE.fullmatch(upper):
        return upper
    return _KEY_ALIASES.get(raw.strip().lower())


def _action_apk_path(action: dict[str, Any]) -> Path | None:
    value = action.get("apk_path") or action.get("path")
    if not isinstance(value, str) or not value.strip() or _is_placeholder(value):
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return bool(_PLACEHOLDER_RE.search(value))


def _unsupported_action_message(action: dict[str, Any]) -> str:
    return (
        f"지원되지 않는 앱 action입니다: {action.get('type')}. "
        "Builder에서 지원 action으로 바꾼 뒤 재개하세요."
    )


def app_action_gap(action: Any) -> AppActionGap | None:
    """What would stop this action, or ``None`` if only the device can tell.

    Payload-only judgement, so the commit gate can ask the same question the
    runtime asks without a device attached. It deliberately says nothing about
    the environment — whether an APK file exists, whether the text is on screen
    — because those are not knowable at authoring time and a commit refused for
    them would be a false refusal.
    """
    if not isinstance(action, dict):
        return AppActionGap(
            REASON_UNSUPPORTED_ACTION,
            "type",
            "",
            "앱 action이 객체가 아닙니다. Builder에서 action을 다시 작성하세요.",
        )
    action_type = str(action.get("type") or "").strip().lower()
    if action_type not in SUPPORTED_APP_ACTION_TYPES:
        return AppActionGap(
            REASON_UNSUPPORTED_ACTION,
            "type",
            action_type,
            _unsupported_action_message(action),
        )
    check = _ACTION_GAP_CHECKS.get(action_type)
    return check(action) if check is not None else None


def app_actions_missing_gap() -> AppActionGap:
    """The gap for an app step carrying no actions at all.

    ``execute_app_actions`` parks on it before any adapter is reached, so it is
    the same class of guaranteed stall as an unresolved target.
    """
    return AppActionGap(
        REASON_ACTIONS_MISSING,
        "actions",
        "",
        "이 앱 조작 단계에 실행할 action이 없습니다. Builder에서 action을 추가한 뒤 다시 실행하세요.",
    )


def _package_gap(action: dict[str, Any]) -> AppActionGap | None:
    value = _action_package(action)
    if not _is_placeholder(value):
        return None
    return AppActionGap(
        REASON_NEEDS_PACKAGE,
        _first_present_field(action, _PACKAGE_FIELDS) or "package",
        value,
        _NEEDS_PACKAGE_PROMPT,
    )


def _text_target_gap(action: dict[str, Any]) -> AppActionGap | None:
    value = _action_text(action)
    if not _is_placeholder(value):
        return None
    return AppActionGap(
        REASON_NEEDS_CONCRETE_TARGET,
        _first_present_field(action, _TEXT_TARGET_FIELDS) or "text",
        value,
        _NEEDS_CONCRETE_TARGET_PROMPT,
    )


def _tap_gap(action: dict[str, Any]) -> AppActionGap | None:
    if _tap_coordinates(action) is not None:
        return None
    return _text_target_gap(action)


def _apk_gap(action: dict[str, Any]) -> AppActionGap | None:
    value = _first_value(action, _APK_FIELDS)
    if not _is_placeholder(value):
        return None
    return AppActionGap(
        REASON_NEEDS_APK,
        _first_present_field(action, _APK_FIELDS) or "apk_path",
        value,
        _NEEDS_APK_PROMPT,
    )


def _typed_text_gap(action: dict[str, Any]) -> AppActionGap | None:
    value = _typed_text(action)
    field_name = _first_present_field(action, _TYPED_TEXT_FIELDS) or "text"
    if _is_placeholder(value):
        return AppActionGap(REASON_NEEDS_TEXT, field_name, value, _NEEDS_TEXT_PROMPT)
    if not value.isascii() or not value.isprintable():
        # `adb shell input text` drops anything outside ASCII silently — the
        # command succeeds and nothing is typed. Reporting that as a completed
        # step would be the fake success this gate exists to prevent.
        return AppActionGap(
            REASON_TEXT_NOT_TYPEABLE, field_name, value, _TEXT_NOT_TYPEABLE_PROMPT
        )
    return None


def _key_gap(action: dict[str, Any]) -> AppActionGap | None:
    if _resolve_keyevent(action) is not None:
        return None
    return AppActionGap(
        REASON_NEEDS_KEY,
        _first_present_field(action, _KEY_FIELDS) or "key",
        _first_value(action, _KEY_FIELDS),
        (
            "press_key action의 키를 알 수 없습니다. "
            f"지원 키: {', '.join(sorted(_KEY_ALIASES))} 또는 KEYCODE_* 이름."
        ),
    )


# Types absent from this map need no target: `read_screen`/`read_ui`/`dump_ui`
# read whatever is on screen, `screenshot` captures it, `wait` sleeps. Their
# `target`/`label` values are never read by the adapter, so flagging a
# placeholder in one would block a commit the runtime runs happily.
_ACTION_GAP_CHECKS: dict[str, Callable[[dict[str, Any]], AppActionGap | None]] = {
    "tap": _tap_gap,
    "tap_text": _text_target_gap,
    "wait_text": _text_target_gap,
    "input_text": _typed_text_gap,
    "type_text": _typed_text_gap,
    "press_key": _key_gap,
    "install_app": _apk_gap,
    "open_play_store": _package_gap,
    "launch_app": _package_gap,
    "open_app": _package_gap,
    "close_app": _package_gap,
    "verify_launch": _package_gap,
}

_GAP_OBSERVATION_TYPES: dict[str, str] = {
    REASON_NEEDS_CONCRETE_TARGET: "action_needs_target",
    REASON_NEEDS_PACKAGE: "action_needs_package",
    REASON_NEEDS_TEXT: "action_needs_text",
    REASON_TEXT_NOT_TYPEABLE: "action_needs_text",
    REASON_NEEDS_KEY: "action_needs_key",
}


def _gap_result(
    gap: AppActionGap,
    action: dict[str, Any],
    observations: list[dict[str, Any]],
    screenshots: list[str],
) -> AppActionAdapterResult:
    observation_type = _GAP_OBSERVATION_TYPES.get(gap.reason)
    return AppActionAdapterResult(
        status="waiting_for_user",
        wait_reason=gap.reason,
        prompt=gap.message,
        observations=(
            [*observations, {"type": observation_type, "action": action}]
            if observation_type
            else observations
        ),
        screenshots=screenshots,
    )


def _existing_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
