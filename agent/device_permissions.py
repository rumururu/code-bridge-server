"""Device permissions a person grants from the phone, the server applies over adb.

A run that needs a permission on a connected device — the device cycle
stuck on devfeedbackhub's "battery optimisation" onboarding screen — used
to end with a sentence in a notification: go to the device and tap it.
The permission itself is one adb command; what was missing was the
*asking*. This module turns the need into an approval request
(``operation=device.control``), which reaches every paired phone through
the pending-approvals list, and applies the command when a person approves.

The server never applies one on its own: with no standing rule the policy
engine asks (``device.control`` is a tiered operation), and the decision
path is the same the phone uses for every other approval.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

OPERATION = "device.control"
DETAILS_KEY = "device_permission"

#: What the server knows how to grant, and how it checks it took.
#: ``macro`` kinds drive the device's screen (uiautomator dump + taps) instead
#: of a single command; they run after the decision in a background thread
#: and report through a notification, because a screen walk takes seconds.
KINDS: dict[str, dict[str, Any]] = {
    "app_google_sign_in": {
        "label": "앱 Google 로그인",
        "macro": True,
    },
    "battery_optimization_exempt": {
        "label": "배터리 최적화 제외",
        "grant": lambda pkg: ["shell", "dumpsys", "deviceidle", "whitelist", f"+{pkg}"],
        "verify": lambda pkg: (["shell", "dumpsys", "deviceidle", "whitelist"], pkg),
    },
    "notification_post": {
        "label": "알림 권한",
        "grant": lambda pkg: ["shell", "pm", "grant", pkg, "android.permission.POST_NOTIFICATIONS"],
        "verify": lambda pkg: (["shell", "dumpsys", "package", pkg], "android.permission.POST_NOTIFICATIONS: granted=true"),
    },
}


def _adb_path() -> str | None:
    from agent.app_action_executor import _resolve_adb_path

    return _resolve_adb_path() or shutil.which("adb")


def _run(adb: str, serial: str, args: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run([adb, "-s", serial, *args], capture_output=True, text=True, timeout=timeout, check=False)


def apply_device_permission(details: dict[str, Any]) -> dict[str, Any]:
    """Grant one permission over adb and verify it took. Never raises."""
    kind = str(details.get(DETAILS_KEY) or "")
    serial = str(details.get("serial") or "")
    package = str(details.get("package") or "")
    spec = KINDS.get(kind)
    if not spec or not serial or not package:
        return {"ok": False, "error": f"unsupported request: kind={kind!r} serial={serial!r} package={package!r}"}
    adb = _adb_path()
    if not adb:
        return {"ok": False, "error": "adb is not available on this server"}
    if spec.get("macro"):
        try:
            return _run_macro(kind, adb, serial, package, details)
        except Exception as exc:  # a screen walk must report, never raise
            return {"ok": False, "kind": kind, "serial": serial, "package": package, "error": f"{type(exc).__name__}: {exc}"}
    try:
        granted = _run(adb, serial, spec["grant"](package))
        verify_args, needle = spec["verify"](package)
        check = _run(adb, serial, verify_args)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"adb failed: {exc}"}
    ok = check.returncode == 0 and needle in (check.stdout or "")
    return {
        "ok": ok,
        "kind": kind, "serial": serial, "package": package,
        "output": (granted.stdout or granted.stderr or "").strip()[:500],
        "verified": ok,
        "error": None if ok else (granted.stderr or check.stderr or "not verified after grant").strip()[:300],
    }


# --- screen macros ---------------------------------------------------------

import re as _re
import time as _time
from pathlib import Path as _Path

_MACRO_STEP_WAIT = 3.0
_MACRO_MAX_ROUNDS = 8
_ONBOARDING_NEXT_LABELS = ("다음", "시작하기", "설정 완료", "확인", "Next", "Continue", "Done")


def _dump(adb: str, serial: str) -> str:
    """The current screen as uiautomator XML ('' when the dump fails)."""
    _run(adb, serial, ["shell", "uiautomator", "dump", "/sdcard/cb_dump.xml"], timeout=20)
    got = _run(adb, serial, ["shell", "cat", "/sdcard/cb_dump.xml"], timeout=20)
    return got.stdout or ""


def _nodes(xml: str) -> list[str]:
    return _re.findall(r"<node [^>]*>", xml)


def _bounds(node: str) -> tuple[int, int] | None:
    m = _re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', node)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return (x1 + x2) // 2, (y1 + y2) // 2


def _box(node: str) -> tuple[int, int, int, int] | None:
    m = _re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', node)
    return tuple(map(int, m.groups())) if m else None  # type: ignore[return-value]


def _tap(adb: str, serial: str, node: str, xml: str | None = None) -> bool:
    """Tap a node — or the smallest clickable node enclosing it.

    A picker's account row is clickable; the email inside it is a bare text
    node. Tapping the text's centre lands on the row in practice, but the
    Google picker ignored it on 2026-09-04 while the row itself took the
    tap, so the enclosing clickable is preferred when the dump is known.
    """
    target = node
    if xml is not None and 'clickable="true"' not in node:
        inner = _box(node)
        if inner:
            rows = [
                n for n in _nodes(xml)
                if 'clickable="true"' in n and (b := _box(n)) and b[0] <= inner[0] and b[1] <= inner[1] and b[2] >= inner[2] and b[3] >= inner[3]
            ]
            if rows:
                rows.sort(key=lambda n: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(_box(n)))
                target = rows[0]
    center = _bounds(target)
    if not center:
        return False
    _run(adb, serial, ["shell", "input", "tap", str(center[0]), str(center[1])], timeout=10)
    return True


def _labelled(xml: str, label: str) -> str | None:
    pat = _re.compile(r'(?:text|content-desc)="' + _re.escape(label) + r'(?:&#10;[^"]*)?"')
    for node in _nodes(xml):
        if pat.search(node):
            return node
    return None


def _screenshot(adb: str, serial: str, name: str) -> str | None:
    try:
        out = _Path.home() / ".code-bridge" / "device_permissions"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.png"
        _run(adb, serial, ["shell", "screencap", "-p", "/sdcard/cb_shot.png"], timeout=20)
        _run(adb, serial, ["pull", "/sdcard/cb_shot.png", str(path)], timeout=30)
        return str(path) if path.exists() else None
    except Exception:
        return None


def _run_macro(kind: str, adb: str, serial: str, package: str, details: dict[str, Any]) -> dict[str, Any]:
    if kind == "app_google_sign_in":
        return _macro_google_sign_in(adb, serial, package, details)
    return {"ok": False, "error": f"no macro for {kind}"}


def _macro_google_sign_in(adb: str, serial: str, package: str, details: dict[str, Any]) -> dict[str, Any]:
    """Accept the app's terms, press its Google sign-in button, pick the account.

    Never types anything: it taps checkboxes that are unchecked on the sign-in
    screen, the button the request names, and the account chip whose text
    contains the requested address. A screen it does not recognise — a
    password prompt, a second factor, a picker without that account — ends
    the walk with a screenshot in the report, so the person can finish it.
    ``success_anchor`` (a substring of the home screen's dump) is the proof
    that the app is signed in; without it the walk is reported as not done.
    """
    account = str(details.get("account_email") or "").strip()
    button = str(details.get("sign_in_label") or "Sign in with Google")
    anchor = str(details.get("success_anchor") or "")
    if not account or not anchor:
        return {"ok": False, "error": "account_email and success_anchor are required"}
    steps: list[str] = []
    picked = False
    _run(adb, serial, ["shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"], timeout=20)
    _time.sleep(_MACRO_STEP_WAIT + 2)
    for _round in range(_MACRO_MAX_ROUNDS):
        xml = _dump(adb, serial)
        if anchor in xml:
            return {"ok": True, "kind": "app_google_sign_in", "serial": serial, "package": package, "verified": True, "steps": steps, "error": None}
        if picked and _labelled(xml, button):
            # The account was chosen and the app came back to its own sign-in
            # screen: the app's sign-in failed on its side (devfeedbackhub
            # threw AuthSessionMissingException here). Pressing the button
            # again would only repeat it.
            shot = _screenshot(adb, serial, f"sign_in_{serial}_{int(_time.time())}")
            return {"ok": False, "kind": "app_google_sign_in", "serial": serial, "package": package, "verified": False,
                    "steps": steps, "screenshot": shot,
                    "error": "the account was picked but the app returned to its sign-in screen — the app's own sign-in failed; check the app's log on the device"}
        # 1. unchecked consent boxes on this screen
        unchecked = [n for n in _nodes(xml) if 'checkable="true"' in n and 'checked="false"' in n]
        if unchecked:
            for node in unchecked:
                _tap(adb, serial, node)
            steps.append(f"checked {len(unchecked)} consent box(es)")
            _time.sleep(1.5)
            continue
        # 2. the sign-in button
        node = _labelled(xml, button)
        if node:
            _tap(adb, serial, node)
            steps.append(f"tapped '{button}'")
            _time.sleep(_MACRO_STEP_WAIT + 2)
            continue
        # 3. the account picker: a node whose text carries the address
        chip = next((n for n in _nodes(xml) if account.lower() in n.lower()), None)
        if chip:
            _tap(adb, serial, chip, xml)
            picked = True
            steps.append(f"picked account {account}")
            _time.sleep(_MACRO_STEP_WAIT + 5)
            continue
        # 4. onboarding pages that follow a sign-in
        moved = False
        for label in _ONBOARDING_NEXT_LABELS:
            node = _labelled(xml, label)
            if node:
                _tap(adb, serial, node)
                steps.append(f"tapped '{label}'")
                moved = True
                _time.sleep(_MACRO_STEP_WAIT)
                break
        if moved:
            continue
        shot = _screenshot(adb, serial, f"sign_in_{serial}_{int(_time.time())}")
        return {"ok": False, "kind": "app_google_sign_in", "serial": serial, "package": package, "verified": False,
                "steps": steps, "screenshot": shot,
                "error": "reached a screen this walk does not recognise (no consent box, sign-in button, account chip or next button); finish it on the device"}
    shot = _screenshot(adb, serial, f"sign_in_{serial}_{int(_time.time())}")
    return {"ok": False, "kind": "app_google_sign_in", "serial": serial, "package": package, "verified": False,
            "steps": steps, "screenshot": shot, "error": "home screen anchor not reached after the walk"}


def request_device_permission(
    *,
    serial: str,
    package: str,
    kind: str,
    reason: str = "",
    agent_id: str | None = None,
    run_id: str | None = None,
    actor: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask for a permission on a device. Applied now if policy allows, else pending.

    Returns ``{"status": "applied"|"pending"|"refused", ...}``.
    """
    from approvals.approval_service import request_approval_for_operation

    if kind not in KINDS:
        return {"status": "refused", "error": f"unknown permission kind {kind!r}; known: {sorted(KINDS)}"}
    label = KINDS[kind]["label"]
    details = {
        DETAILS_KEY: kind, "serial": serial, "package": package, "reason": reason,
        "agent_id": agent_id, "label": label,
        **{k: v for k, v in (extra or {}).items() if k in ("account_email", "sign_in_label", "success_anchor")},
    }
    # What the phone draws: a sentence it knows (`display.action`, see the
    # app's approval card) plus the target, and a summary in words. Without
    # these the card showed the details as JSON, which nobody could decide on.
    is_macro = bool(KINDS[kind].get("macro"))
    account = str(details.get("account_email") or "")
    target = f"{serial} · {package}" + (f" — {label}" if not is_macro else f" — {label}" + (f" ({account})" if account else ""))
    details["display"] = {"action": "sign_in_app" if is_macro else "grant_device_permission", "target": target}
    details["summary"] = (
        (f"{serial} 기기의 {package} 앱에 {account} 계정으로 로그인합니다 (약관 동의 → Google 로그인 버튼 → 계정 선택; 비밀번호는 입력하지 않음)."
         if is_macro else f"{serial} 기기의 {package} 앱에 '{label}' 권한을 adb로 부여합니다.")
        + (f" 이유: {reason}" if reason else "")
    )
    result = request_approval_for_operation(
        operation=OPERATION, run_id=run_id, actor=actor or {"type": "server", "surface": "device_permission"},
        details=details, risk_level="medium",
    )
    if result.get("error") and not result.get("allowed"):
        return {"status": "refused", "error": result["error"], "policy": result.get("policy")}
    if result.get("allowed") and not result.get("approval_required"):
        applied = apply_device_permission(details)
        _notify_result(details, applied, agent_id=agent_id, run_id=run_id)
        return {"status": "applied" if applied["ok"] else "failed", "result": applied}
    approval = result.get("approval") or {}
    _notify_pending(details, approval, agent_id=agent_id, run_id=run_id)
    return {"status": "pending", "approval": approval}


def on_approval_decided(request: dict[str, Any] | None, decision: str) -> dict[str, Any] | None:
    """Hook for ``decide_approval``: apply a device permission the person just approved."""
    if not request or request.get("operation") != OPERATION:
        return None
    details = request.get("details") if isinstance(request.get("details"), dict) else {}
    if not details.get(DETAILS_KEY):
        return None
    if not str(decision).startswith("approve"):
        return {"ok": False, "skipped": "denied"}
    kind = str(details.get(DETAILS_KEY) or "")
    if KINDS.get(kind, {}).get("macro"):
        # A screen walk takes tens of seconds; the decision must answer now.
        import threading

        def work() -> None:
            applied = apply_device_permission(details)
            _notify_result(details, applied, agent_id=details.get("agent_id"), run_id=request.get("run_id"))

        threading.Thread(target=work, name=f"device-macro-{kind}", daemon=True).start()
        return {"ok": None, "started": True, "kind": kind}
    applied = apply_device_permission(details)
    _notify_result(details, applied, agent_id=details.get("agent_id"), run_id=request.get("run_id"))
    return applied


def _notify_pending(details: dict[str, Any], approval: dict[str, Any], *, agent_id: str | None, run_id: str | None) -> None:
    try:
        from agent.notification_store import get_notification_store

        get_notification_store().create(
            title=f"기기 권한 요청 — {details.get('label')} ({details.get('serial')})",
            body=(f"{details.get('package')}에 '{details.get('label')}'이 필요합니다. "
                  f"{details.get('reason') or ''}\n앱의 승인 목록에서 허용하면 서버가 adb로 적용합니다.\napproval:{approval.get('id')}"),
            level="warning", run_id=run_id, agent_id=agent_id, reason="device_permission",
        )
    except Exception:
        logger.exception("device permission: pending notification failed")


def _notify_result(details: dict[str, Any], applied: dict[str, Any], *, agent_id: str | None, run_id: str | None) -> None:
    try:
        from agent.notification_store import get_notification_store

        ok = bool(applied.get("ok"))
        get_notification_store().create(
            title=f"기기 권한 {'적용됨' if ok else '적용 실패'} — {details.get('label')} ({details.get('serial')})",
            body=(f"{details.get('package')}: {details.get('label')} {'허용됨 (adb로 확인)' if ok else applied.get('error')}"
                  + (f"\n단계: {' → '.join(applied.get('steps') or [])}" if applied.get("steps") else "")
                  + (f"\n화면: {applied.get('screenshot')}" if applied.get("screenshot") else "")),
            level="success" if ok else "error", run_id=run_id, agent_id=agent_id, reason="device_permission",
        )
    except Exception:
        logger.exception("device permission: result notification failed")


__all__ = ["KINDS", "OPERATION", "request_device_permission", "apply_device_permission", "on_approval_decided"]
