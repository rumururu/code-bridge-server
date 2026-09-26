"""A device permission is asked for on the phone and applied over adb after the decision."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import device_permissions, notification_store  # noqa: E402
from approvals import approval_store  # noqa: E402
from approvals.approval_service import decide_approval  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402
from routes import agents, approvals as approvals_routes  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class _Completed:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


class DevicePermissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "perm.db"
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        notification_store._notification_store = None
        database.init_db()
        self.addCleanup(self._restore)
        app = FastAPI()
        app.include_router(agents.router)
        app.include_router(approvals_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "k"
        self.client = TestClient(app)
        self.calls: list[list[str]] = []

    def _restore(self) -> None:
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        notification_store._notification_store = None
        database.DB_PATH = self._original

    def _fake_run(self, adb, serial, args, *, timeout=30):
        self.calls.append([serial, *args])
        if args[:3] == ["shell", "dumpsys", "deviceidle"] and len(args) == 4:
            return _Completed(stdout="system,com.mkideabox.devfeedbackhub,10123\n")
        return _Completed(stdout="")

    def test_a_request_becomes_a_pending_approval_and_a_notification(self):
        with patch.object(device_permissions, "_adb_path", return_value="/fake/adb"), \
             patch.object(device_permissions, "_run", side_effect=self._fake_run):
            r = self.client.post("/api/agent/devices/permissions/request", json={
                "serial": "R59N3035LQL", "package": "com.mkideabox.devfeedbackhub",
                "kind": "battery_optimization_exempt", "reason": "onboarding is stuck on it",
            })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "pending")
        approval = r.json()["approval"]
        self.assertEqual(approval["operation"], "device.control")
        self.assertEqual(approval["details"]["device_permission"], "battery_optimization_exempt")
        self.assertEqual(self.calls, [], "nothing runs before a decision")
        pending = self.client.get("/api/approvals/pending").json()["approvals"]
        self.assertEqual([p["id"] for p in pending], [approval["id"]])

    def test_approving_applies_the_permission_and_verifies_it(self):
        with patch.object(device_permissions, "_adb_path", return_value="/fake/adb"), \
             patch.object(device_permissions, "_run", side_effect=self._fake_run):
            approval = self.client.post("/api/agent/devices/permissions/request", json={
                "serial": "R59N3035LQL", "package": "com.mkideabox.devfeedbackhub", "kind": "battery_optimization_exempt",
            }).json()["approval"]
            outcome = decide_approval(approval["id"], decision="approve", channel="remote_client")
        self.assertEqual(outcome["device_permission"]["ok"], True)
        self.assertEqual(self.calls[0], ["R59N3035LQL", "shell", "dumpsys", "deviceidle", "whitelist", "+com.mkideabox.devfeedbackhub"])
        self.assertEqual(self.calls[1][:4], ["R59N3035LQL", "shell", "dumpsys", "deviceidle"])

    def test_denying_runs_nothing(self):
        with patch.object(device_permissions, "_adb_path", return_value="/fake/adb"), \
             patch.object(device_permissions, "_run", side_effect=self._fake_run):
            approval = self.client.post("/api/agent/devices/permissions/request", json={
                "serial": "R59N3035LQL", "package": "com.mkideabox.devfeedbackhub", "kind": "battery_optimization_exempt",
            }).json()["approval"]
            outcome = decide_approval(approval["id"], decision="deny", channel="remote_client")
        self.assertEqual(outcome["device_permission"], {"ok": False, "skipped": "denied"})
        self.assertEqual(self.calls, [])

    def test_an_unknown_kind_is_refused(self):
        r = self.client.post("/api/agent/devices/permissions/request", json={
            "serial": "x", "package": "y", "kind": "root_everything",
        })
        self.assertEqual(r.status_code, 400)

    def test_a_failed_verification_is_reported_not_claimed(self):
        def run_but_never_take(adb, serial, args, *, timeout=30):
            return _Completed(stdout="")
        with patch.object(device_permissions, "_adb_path", return_value="/fake/adb"), \
             patch.object(device_permissions, "_run", side_effect=run_but_never_take):
            applied = device_permissions.apply_device_permission({
                "device_permission": "battery_optimization_exempt", "serial": "s", "package": "p",
            })
        self.assertFalse(applied["ok"])
        self.assertIn("not verified", applied["error"])


class GoogleSignInMacroTest(unittest.TestCase):
    """The sign-in walk taps only what it recognises and proves success by the anchor."""

    LOGIN = ('<node text="이용약관에 동의합니다." checkable="true" checked="false" bounds="[10,10][100,40]"/>'
             '<node text="개인정보처리방침에 동의합니다." checkable="true" checked="false" bounds="[10,50][100,80]"/>'
             '<node text="Sign in with Google" bounds="[10,100][300,140]"/>')
    LOGIN_CHECKED = LOGIN.replace('checked="false"', 'checked="true"')
    PICKER = ('<node clickable="true" bounds="[0,150][400,300]"><node text="서만길" bounds="[0,0][10,10]"/>'
              '<node text="user@example.com" bounds="[10,200][300,240]"/></node>')
    HOME = '<node content-desc="autotask_home_app|thing" bounds="[0,0][10,10]"/>'

    def _walk(self, screens):
        it = iter(screens)
        state = {"current": next(it)}
        taps: list[tuple[int, int]] = []

        def fake_run(adb, serial, args, *, timeout=30):
            if args[:2] == ["shell", "input"]:
                taps.append((int(args[3]), int(args[4])))
                try:
                    state["current"] = next(it)
                except StopIteration:
                    pass
            if args[:2] == ["shell", "cat"]:
                return _Completed(stdout=state["current"])
            return _Completed(stdout="")

        with patch.object(device_permissions, "_adb_path", return_value="/fake/adb"), \
             patch.object(device_permissions, "_run", side_effect=fake_run), \
             patch.object(device_permissions._time, "sleep", lambda *_: None), \
             patch.object(device_permissions, "_screenshot", return_value="/tmp/x.png"):
            result = device_permissions.apply_device_permission({
                "device_permission": "app_google_sign_in", "serial": "M205N", "package": "com.x",
                "account_email": "user@example.com", "success_anchor": "autotask_home_app|",
            })
        return result, taps

    def test_consent_then_button_then_account_then_home(self):
        # two consent taps advance to the checked screen, button tap to the picker, chip tap to home
        result, taps = self._walk([self.LOGIN, self.LOGIN, self.LOGIN_CHECKED, self.PICKER, self.HOME])
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["steps"], ["checked 2 consent box(es)", "tapped 'Sign in with Google'", "picked account user@example.com"])
        # the account tap lands on the clickable row that encloses the email, not the bare text
        self.assertEqual(taps[-1], (200, 225))

    def test_a_bounce_back_to_the_sign_in_screen_after_picking_stops_the_walk(self):
        result, taps = self._walk([self.LOGIN_CHECKED, self.PICKER, self.LOGIN_CHECKED, self.PICKER, self.LOGIN_CHECKED])
        self.assertFalse(result["ok"])
        self.assertIn("app's own sign-in failed", result["error"])
        self.assertEqual(result["steps"], ["tapped 'Sign in with Google'", "picked account user@example.com"])
        self.assertEqual(len(taps), 2)

    def test_an_unknown_screen_stops_with_a_screenshot_and_no_typing(self):
        password_prompt = '<node text="비밀번호" bounds="[0,0][10,10]"/><node text="password" bounds="[0,0][10,10]"/>'
        result, taps = self._walk([self.LOGIN_CHECKED, password_prompt])
        self.assertFalse(result["ok"])
        self.assertIn("does not recognise", result["error"])
        self.assertEqual(result["screenshot"], "/tmp/x.png")
        self.assertEqual(len(taps), 1, "only the sign-in button was tapped")

    def test_already_signed_in_is_ok_without_a_tap(self):
        result, taps = self._walk([self.HOME])
        self.assertTrue(result["ok"])
        self.assertEqual(taps, [])

    def test_missing_account_or_anchor_is_refused(self):
        with patch.object(device_permissions, "_adb_path", return_value="/fake/adb"):
            result = device_permissions.apply_device_permission({"device_permission": "app_google_sign_in", "serial": "s", "package": "p"})
        self.assertFalse(result["ok"])
        self.assertIn("required", result["error"])
