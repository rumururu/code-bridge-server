"""A notify step's push half must never be able to fail the step.

`notification_store` already makes the message durable — it survives the
run and the inbox will show it whenever it is next opened, key or no key.
Push (agent/push_notifier.py) is strictly additive: it tries to also ring
the phone through FCM. Nothing about push being unconfigured, misconfigured,
or failing mid-send may turn an otherwise-successful notify step into a
failed one, because the thing the step exists to do — leave a message that
outlives the run — already happened by the time push is attempted.

This also pins the token bookkeeping the push path depends on: the same
token registered twice must not fan a single notify out twice, and a token
FCM reports as unregistered must stop being tried rather than costing a
doomed network call on every future notify.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store, push_notifier  # noqa: E402
from agent.notification_store import get_notification_store  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    _push_notification_best_effort,
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402
from pairing import pairing_service as pairing_service_module  # noqa: E402
from pairing.pairing_service import PairingService  # noqa: E402


class _OrchestratorTestBase(unittest.TestCase):
    """Shared DB + pairing-service isolation for the tests below."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = self.dir / "notify_push.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

        # Isolate the pairing singleton so registering a token here can never
        # touch a real operator's api_keys.json.
        self._original_pairing_service = pairing_service_module._pairing_service
        pairing_service_module._pairing_service = PairingService(config_dir=self.dir / "pairing")

        # Point push_notifier at a key that does not exist, so every test
        # starts from "push is off" unless it explicitly registers a token.
        self._env_patch = mock.patch.dict(
            os.environ,
            {push_notifier.SERVICE_ACCOUNT_ENV: str(self.dir / "no_such_key.json")},
        )
        self._env_patch.start()
        push_notifier.reset_for_tests()

        self.addCleanup(self._restore)

    def _restore(self):
        self._env_patch.stop()
        push_notifier.reset_for_tests()
        pairing_service_module._pairing_service = self._original_pairing_service
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _run_notify_workflow(self, notify_payload: dict) -> tuple[dict, list[dict]]:
        agent = self.store.create_agent(
            name="notifier bot",
            system_prompt="Leave a message.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "tell_user",
                    "type": "notify",
                    "name": "Tell the user",
                    "notify": notify_payload,
                }
            ],
        )
        task = self.store.create_task(
            title="Run notify workflow",
            assigned_agent_id=agent["id"],
            goal="Leave a message.",
        )
        result = prepare_task_orchestration(task["id"], provider_id="openai", auto_start=False)
        assert result is not None
        asyncio.run(execute_task_orchestration(result["execution"]))
        return task, self.store.list_task_steps(task["id"])


class NotifyStepWithoutAKeyTest(_OrchestratorTestBase):
    def test_the_notification_is_stored_and_the_step_completes(self):
        # No service-account key anywhere, no push tokens registered — this
        # is the state of most servers, and it must be indistinguishable from
        # success as far as the notify step and the inbox are concerned.
        self.assertFalse(push_notifier.is_configured())

        task, steps = self._run_notify_workflow(
            {"title": "Disk almost full", "body": "8% left", "level": "warning"}
        )

        self.assertEqual(steps[0]["status"], "completed")
        stored = get_notification_store().list_notifications()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["title"], "Disk almost full")


class NotifyStepPushFailureTest(_OrchestratorTestBase):
    def test_a_push_send_failure_does_not_fail_the_step(self):
        pairing = pairing_service_module.get_pairing_service()
        pairing._api_keys["client_1"] = {"device_name": "Phone", "api_key_sha256": "hash"}
        pairing.register_push_token("client_1", "fcm-token-1")

        with mock.patch(
            "agent.push_notifier.send_to_tokens",
            side_effect=RuntimeError("FCM is unreachable"),
        ):
            task, steps = self._run_notify_workflow(
                {"title": "Build failed", "level": "error"}
            )

        self.assertEqual(steps[0]["status"], "completed")
        stored = get_notification_store().list_notifications()
        self.assertEqual(len(stored), 1)


class PushNotificationBestEffortTest(_OrchestratorTestBase):
    """Direct unit coverage of the function `_execute_notify_workflow_step` calls."""

    def test_no_tokens_registered_is_a_silent_no_op(self):
        # Should not even try to import/initialize the Admin SDK.
        _push_notification_best_effort(
            notification={"id": "notif_1"}, title="t", body=None, level="info"
        )

    def test_a_dropped_token_is_removed_from_the_pairing_store(self):
        pairing = pairing_service_module.get_pairing_service()
        pairing._api_keys["client_1"] = {"device_name": "Phone", "api_key_sha256": "hash"}
        pairing.register_push_token("client_1", "dead-token")

        with mock.patch(
            "agent.push_notifier.send_to_tokens",
            return_value={"delivered": [], "dropped": ["dead-token"]},
        ):
            _push_notification_best_effort(
                notification={"id": "notif_1"}, title="t", body=None, level="info"
            )

        self.assertEqual(pairing.all_push_tokens(), [])

    def test_an_unexpected_error_never_escapes(self):
        pairing = pairing_service_module.get_pairing_service()
        pairing._api_keys["client_1"] = {"device_name": "Phone", "api_key_sha256": "hash"}
        pairing.register_push_token("client_1", "tok")

        with mock.patch(
            "agent.push_notifier.send_to_tokens", side_effect=RuntimeError("boom")
        ):
            try:
                _push_notification_best_effort(
                    notification={"id": "notif_1"}, title="t", body=None, level="info"
                )
            except RuntimeError:
                self.fail("a push failure must never escape into the caller")


class PushTokenRegistrationTest(unittest.TestCase):
    """`notification_id`/token bookkeeping the push route depends on."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.service = PairingService(config_dir=Path(self._tmp.name))
        self.service._api_keys["client_1"] = {
            "device_name": "Phone",
            "api_key_sha256": "hash1",
        }

    def test_registering_the_same_token_twice_keeps_one(self):
        self.service.register_push_token("client_1", "tok-a")
        self.service.register_push_token("client_1", "tok-a")
        self.assertEqual(self.service.all_push_tokens(), ["tok-a"])

    def test_an_unregistered_token_is_dropped(self):
        self.service.register_push_token("client_1", "tok-a")
        self.service.remove_push_token("tok-a")
        self.assertEqual(self.service.all_push_tokens(), [])

    def test_a_token_moves_with_a_reinstall_onto_a_new_client(self):
        self.service._api_keys["client_2"] = {
            "device_name": "Phone (reinstalled)",
            "api_key_sha256": "hash2",
        }
        self.service.register_push_token("client_1", "shared-token")
        self.service.register_push_token("client_2", "shared-token")

        # The old owner must not still have it, or one notify would fan out
        # to two client records for what is physically one device.
        self.assertEqual(self.service.all_push_tokens(), ["shared-token"])

    def test_registering_for_an_unknown_client_is_a_no_op(self):
        self.assertFalse(self.service.register_push_token("ghost", "tok"))
        self.assertEqual(self.service.all_push_tokens(), [])

    def test_find_client_id_by_api_key_resolves_the_owner(self):
        from pairing.pairing_service import _hash_api_key

        self.service._api_keys["client_3"] = {
            "device_name": "Phone 3",
            "api_key_sha256": _hash_api_key("real-key"),
        }
        self.assertEqual(self.service.find_client_id_by_api_key("real-key"), "client_3")
        self.assertIsNone(self.service.find_client_id_by_api_key("wrong-key"))


class PushNotifierConfigurationTest(unittest.TestCase):
    """No key on disk (or no firebase-admin) degrades to a no-op, never a crash."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ,
            {push_notifier.SERVICE_ACCOUNT_ENV: str(Path(self._tmp.name) / "missing.json")},
        )
        self._env_patch.start()
        push_notifier.reset_for_tests()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        self._env_patch.stop()
        push_notifier.reset_for_tests()

    def test_missing_key_reports_unconfigured(self):
        self.assertFalse(push_notifier.is_configured())

    def test_sending_with_no_key_returns_empty_and_does_not_raise(self):
        result = push_notifier.send_to_tokens(["tok"], title="hi")
        self.assertEqual(result, {"delivered": [], "dropped": []})

    def test_a_malformed_key_file_also_degrades_quietly(self):
        bad_path = Path(self._tmp.name) / "bad.json"
        bad_path.write_text("not json at all")
        with mock.patch.dict(os.environ, {push_notifier.SERVICE_ACCOUNT_ENV: str(bad_path)}):
            push_notifier.reset_for_tests()
            result = push_notifier.send_to_tokens(["tok"], title="hi")
        self.assertEqual(result, {"delivered": [], "dropped": []})

    def test_no_tokens_short_circuits_before_touching_the_sdk(self):
        self.assertEqual(
            push_notifier.send_to_tokens([], title="hi"), {"delivered": [], "dropped": []}
        )


if __name__ == "__main__":
    unittest.main()
