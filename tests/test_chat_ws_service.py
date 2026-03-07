import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import chat_ws_service
from remote_access_service import RemoteAccessLoginPayload


class ChatWsServiceTest(unittest.TestCase):
    def test_validate_chat_websocket_access_for_current_server_allows_missing_key_when_auth_disabled(self):
        fake_config = MagicMock()
        fake_config.api_key = ""
        fake_config.port = 8080
        fake_config.get_project.return_value = {"path": "/tmp/demo"}

        result = chat_ws_service.validate_chat_websocket_access_for_current_server(
            None,
            "demo",
            config=fake_config,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.project_path, "/tmp/demo")

    def test_validate_chat_websocket_access_for_current_server_requires_api_key_when_auth_enabled(self):
        fake_config = MagicMock()
        fake_config.api_key = "server-static-key"
        fake_config.get_project.return_value = {"path": "/tmp/demo"}

        result = chat_ws_service.validate_chat_websocket_access_for_current_server(
            None,
            "demo",
            config=fake_config,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.close_code, 4001)
        self.assertEqual(result.close_reason, "API key required")

    def test_validate_chat_websocket_access_for_current_server_project_missing(self):
        fake_pairing = MagicMock()
        fake_pairing.validate_api_key.return_value = True

        fake_config = MagicMock()
        fake_config.get_project.return_value = None

        result = chat_ws_service.validate_chat_websocket_access_for_current_server(
            "apikey",
            "missing",
            pairing_service=fake_pairing,
            config=fake_config,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Project missing not found")

    def test_resolve_chat_provider_selection_for_current_server_success(self):
        selection = SimpleNamespace(provider_name="Claude")

        result = chat_ws_service.resolve_chat_provider_selection_for_current_server(
            selection_resolver=lambda: selection
        )

        self.assertTrue(result.success)
        self.assertEqual(result.provider_name, "Claude")
        self.assertIs(result.selection, selection)

    def test_resolve_chat_provider_selection_for_current_server_invalid_provider_name(self):
        selection = SimpleNamespace(provider_name="")

        result = chat_ws_service.resolve_chat_provider_selection_for_current_server(
            selection_resolver=lambda: selection
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Invalid provider selection")


class ChatWsServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_chat_session_for_current_server_success(self):
        session_object = object()
        session_creator = AsyncMock(return_value=session_object)
        selection = SimpleNamespace(provider_name="Claude")

        result = await chat_ws_service.create_chat_session_for_current_server(
            "demo",
            "/tmp/demo",
            selection,
            session_creator=session_creator,
        )

        self.assertTrue(result.success)
        self.assertIs(result.session, session_object)
        session_creator.assert_awaited_once_with(
            project_name="demo",
            project_path="/tmp/demo",
            selection=selection,
        )

    async def test_process_firebase_auth_message_for_current_server_invalid_payload(self):
        result = await chat_ws_service.process_firebase_auth_message_for_current_server(
            message={"type": "firebase_auth"},
            local_port=8080,
        )

        self.assertEqual(result.payload.get("type"), "error")
        self.assertIn("id_token", result.payload.get("message", ""))

    async def test_process_firebase_auth_message_for_current_server_register_device_false(self):
        payload = RemoteAccessLoginPayload(
            id_token="token",
            refresh_token=None,
            auth_mode="refresh_token",
            register_device=False,
        )
        fake_auth = MagicMock()
        fake_auth.initialize = AsyncMock(return_value=True)
        fake_auth.authenticate_with_token = AsyncMock(return_value=True)

        register_mock = AsyncMock(return_value=True)
        result = await chat_ws_service.process_firebase_auth_message_for_current_server(
            message={"type": "firebase_auth"},
            local_port=8080,
            payload_parser=lambda _: (payload, None),
            firebase_auth=fake_auth,
            register_device_for_remote_access_fn=register_mock,
        )

        self.assertEqual(result.payload.get("type"), "status")
        self.assertIn("Firebase authenticated", result.payload.get("message", ""))
        register_mock.assert_not_awaited()

    async def test_process_disconnect_server_message_for_current_server_clears_auth(self):
        fake_auth = MagicMock()
        fake_auth.clear_auth = AsyncMock()

        result = await chat_ws_service.process_disconnect_server_message_for_current_server(
            firebase_auth=fake_auth
        )

        self.assertEqual(result.payload, {"type": "status", "message": "Server disconnected"})
        fake_auth.clear_auth.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
