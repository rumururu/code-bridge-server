"""Workflow-facing Android app action execution helper."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from agent.tool_artifacts import ARTIFACT_ROOT

from .app_action_adapter import AppActionAdapter, AppActionAdapterResult

_PLACEHOLDER_RE = re.compile(
    r"(configured|placeholder|todo|from_current_screen|current_screen|"
    r"selected_request|target_app|app_from_request|request_card|"
    r"join_or_apply|auto_detect|detected)",
    re.IGNORECASE,
)
_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


async def execute_app_actions(
    actions: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    adapter: AppActionAdapter | None = None,
) -> AppActionAdapterResult:
    """Execute normalized Android app actions through an injectable adapter."""
    if not actions:
        return AppActionAdapterResult(
            status="waiting_for_user",
            wait_reason="app_actions_missing",
            prompt="이 앱 조작 단계에 실행할 action이 없습니다. Builder에서 action을 추가한 뒤 다시 실행하세요.",
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
            if not action_type:
                return _unsupported_action(action, observations, screenshots)

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

            if action_type == "tap_text":
                target = _action_text(action)
                if _is_placeholder(target):
                    return _needs_concrete_target(action, observations, screenshots)
                xml = await _dump_ui(adb_path, device_id)
                bounds = _find_text_bounds(xml, target)
                if bounds is None:
                    return AppActionAdapterResult(
                        status="waiting_for_user",
                        wait_reason="app_action_target_not_found",
                        prompt=(
                            f"현재 Android 화면에서 '{target}' 텍스트를 찾지 못했습니다. "
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
                x1, y1, x2, y2 = bounds
                await _adb(adb_path, "-s", device_id, "shell", "input", "tap", str((x1 + x2) // 2), str((y1 + y2) // 2))
                observations.append({"type": "tap_text", "target": target, "bounds": bounds})
                continue

            if action_type == "open_play_store":
                package_name = _action_package(action)
                if _is_placeholder(package_name):
                    return _needs_package(action, observations, screenshots)
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
                    return AppActionAdapterResult(
                        status="waiting_for_user",
                        wait_reason="app_install_needs_package_or_apk",
                        prompt=(
                            "앱 설치 action에 실제 APK 경로 또는 Play Store 패키지명이 없습니다. "
                            "Builder에서 설치 대상을 구체화한 뒤 다시 실행하세요."
                        ),
                        observations=observations,
                        screenshots=screenshots,
                    )
                await _adb(adb_path, "-s", device_id, "install", "-r", str(apk_path), timeout=120)
                observations.append({"type": "install_app", "apk_path": str(apk_path)})
                continue

            if action_type == "verify_launch":
                package_name = _action_package(action)
                if _is_placeholder(package_name):
                    return _needs_package(action, observations, screenshots)
                component = await _resolve_launch_component(
                    adb_path,
                    device_id,
                    package_name,
                )
                await _adb(adb_path, "-s", device_id, "shell", "am", "start", "-n", component)
                observations.append({"type": "verify_launch", "package": package_name})
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
        wait_reason="app_action_unsupported",
        prompt=f"지원되지 않는 앱 action입니다: {action.get('type')}. Builder에서 지원 action으로 바꾼 뒤 재개하세요.",
        observations=observations,
        screenshots=screenshots,
    )


def _needs_concrete_target(
    action: dict[str, Any],
    observations: list[dict[str, Any]],
    screenshots: list[str],
) -> AppActionAdapterResult:
    return AppActionAdapterResult(
        status="waiting_for_user",
        wait_reason="app_action_needs_concrete_target",
        prompt=(
            "앱 action에 실제 화면 텍스트, 패키지명, APK 경로 같은 구체 target이 필요합니다. "
            "Builder에서 target을 구체화하거나 사용자가 직접 처리한 뒤 재개하세요."
        ),
        observations=[*observations, {"type": "action_needs_target", "action": action}],
        screenshots=screenshots,
    )


def _needs_package(
    action: dict[str, Any],
    observations: list[dict[str, Any]],
    screenshots: list[str],
) -> AppActionAdapterResult:
    return AppActionAdapterResult(
        status="waiting_for_user",
        wait_reason="app_action_needs_package_name",
        prompt="이 앱 action에는 실제 Android 패키지명이 필요합니다. Builder에서 package 값을 지정한 뒤 다시 실행하세요.",
        observations=[*observations, {"type": "action_needs_package", "action": action}],
        screenshots=screenshots,
    )


def _safe_wait_seconds(action: dict[str, Any]) -> float:
    value = action.get("seconds") or action.get("timeout") or 1
    try:
        return max(0.0, min(float(value), 10.0))
    except (TypeError, ValueError):
        return 1.0


def _action_text(action: dict[str, Any]) -> str:
    return str(action.get("text") or action.get("target") or action.get("label") or "").strip()


def _action_package(action: dict[str, Any]) -> str:
    return str(action.get("package") or action.get("package_name") or action.get("app_id") or action.get("source") or "").strip()


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


def _existing_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
