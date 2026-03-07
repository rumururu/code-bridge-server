import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import preview_route_service


class _DummyRequest:
    def __init__(self, query_params=None):
        self.query_params = query_params or {}


class PreviewRouteServiceTest(unittest.TestCase):
    def test_create_preview_token_for_current_server_requires_running_port(self):
        fake_manager = MagicMock()
        fake_manager.get_server_port.return_value = None

        result = preview_route_service.create_preview_token_for_current_server(
            "demo",
            manager=fake_manager,
            preview_access=MagicMock(),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.payload.get("error"), "No running dev server for project demo")

    def test_create_preview_token_for_current_server_success(self):
        fake_manager = MagicMock()
        fake_manager.get_server_port.return_value = 5173

        fake_access = MagicMock()
        fake_access.generate_preview_token.return_value = "token123"
        fake_access.ttl_minutes = 15

        result = preview_route_service.create_preview_token_for_current_server(
            "demo",
            manager=fake_manager,
            preview_access=fake_access,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["token"], "token123")
        self.assertEqual(result.payload["project"], "demo")

    def test_authorize_project_preview_request_remote_requires_token(self):
        fake_access = MagicMock()
        fake_access.is_local_request.return_value = False
        fake_access.has_remote_session.return_value = False

        result = preview_route_service.authorize_project_preview_request_for_current_server(
            _DummyRequest(),
            "demo",
            preview_access=fake_access,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 403)
        self.assertEqual(result.payload.get("error"), "Preview token required for remote access")
        self.assertIn("hint", result.payload)

    def test_authorize_project_preview_request_binds_remote_session_when_token_valid(self):
        fake_access = MagicMock()
        fake_access.is_local_request.return_value = False
        fake_access.validate_preview_token.return_value = True

        request = _DummyRequest(query_params={"preview_token": "abc"})
        result = preview_route_service.authorize_project_preview_request_for_current_server(
            request,
            "demo",
            preview_access=fake_access,
        )

        self.assertTrue(result.success)
        fake_access.bind_remote_session.assert_called_once_with(request, "demo")

    def test_resolve_last_preview_project_target_for_current_server_requires_session(self):
        fake_access = MagicMock()
        fake_access.get_last_previewed_project.return_value = "demo"
        fake_access.is_local_request.return_value = False
        fake_access.has_remote_session.return_value = False

        result = preview_route_service.resolve_last_preview_project_target_for_current_server(
            _DummyRequest(),
            preview_access=fake_access,
            manager=MagicMock(),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 403)
        self.assertEqual(result.payload.get("error"), "Preview session not authorized")


class PreviewRouteServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_preview_request_for_current_server_delegates(self):
        fake_proxy = MagicMock()
        fake_proxy.proxy_request = AsyncMock(return_value="ok")

        result = await preview_route_service.proxy_preview_request_for_current_server(
            request=_DummyRequest(),
            target_port=5173,
            path="",
            proxy=fake_proxy,
        )

        self.assertEqual(result, "ok")
        fake_proxy.proxy_request.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
