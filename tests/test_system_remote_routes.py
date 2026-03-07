import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from remote_access_service import (
    RemoteAccessActionResult,
    RemoteAccessLoginResult,
    RemoteFirebaseStatus,
    RemoteMdnsStatus,
    RemoteNetworkStatus,
    RemoteTunnelStatus,
)
from routes.deps import verify_api_key
from routes.system_remote import router as system_remote_router


class SystemRemoteRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(system_remote_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_network_status_uses_service_payload(self):
        payload = RemoteNetworkStatus(
            mdns=RemoteMdnsStatus(
                available=False,
                enabled=False,
                registered=False,
                server_name="demo",
            ),
            tunnel=RemoteTunnelStatus(
                available=True,
                enabled=True,
                running=True,
                url="https://t",
            ),
            firebase=RemoteFirebaseStatus(
                available=True,
                enabled=True,
                authenticated=True,
                user_id="u",
                device_id="d",
            ),
        )

        with patch("routes.system_remote.build_remote_network_status_for_current_server", return_value=payload):
            response = self.client.get("/api/system/network-status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload.as_response_fields())

    def test_remote_access_login_missing_token_returns_400(self):
        response = self.client.post("/api/system/remote-access/login", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("id_token", response.json().get("error", ""))

    def test_remote_access_login_success_without_register(self):
        with patch(
            "routes.system_remote.login_for_remote_access_request_json_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessLoginResult(
                    success=True,
                    status_code=200,
                    user_id="u1",
                    device_id="d1",
                    device_name="dev",
                    auth_mode="refresh_token",
                )
            ),
        ) as mock_login:
            response = self.client.post(
                "/api/system/remote-access/login",
                json={"id_token": "token", "register_device": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("success"))
        self.assertEqual(response.json().get("user_id"), "u1")
        self.assertEqual(response.json().get("device_id"), "d1")
        mock_login.assert_awaited_once()

    def test_remote_access_login_maps_service_error_status(self):
        with patch(
            "routes.system_remote.login_for_remote_access_request_json_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessLoginResult(
                    success=False,
                    status_code=401,
                    error="Token verification failed",
                )
            ),
        ):
            response = self.client.post(
                "/api/system/remote-access/login",
                json={"id_token": "token"},
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json().get("error"), "Token verification failed")

    def test_remote_access_logout_maps_service_success(self):
        with patch(
            "routes.system_remote.logout_remote_access_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessActionResult(
                    success=True,
                    status_code=200,
                )
            ),
        ):
            response = self.client.post("/api/system/remote-access/logout")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True})

    def test_remote_access_disconnect_maps_service_message(self):
        with patch(
            "routes.system_remote.disconnect_remote_access_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessActionResult(
                    success=True,
                    status_code=200,
                    message="Disconnected from remote access",
                )
            ),
        ):
            response = self.client.post("/api/system/remote-access/disconnect")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("message"), "Disconnected from remote access")

    def test_remote_access_disconnect_maps_service_error(self):
        with patch(
            "routes.system_remote.disconnect_remote_access_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessActionResult(
                    success=False,
                    status_code=400,
                    error="Firebase is not available",
                )
            ),
        ):
            response = self.client.post("/api/system/remote-access/disconnect")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "Firebase is not available")

    def test_tunnel_start_maps_service_error(self):
        with patch(
            "routes.system_remote.start_tunnel_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessActionResult(
                    success=False,
                    status_code=500,
                    error="boom",
                )
            ),
        ):
            response = self.client.post("/api/system/tunnel/start")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json().get("error"), "boom")

    def test_tunnel_stop_maps_service_message(self):
        with patch(
            "routes.system_remote.stop_tunnel_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessActionResult(
                    success=True,
                    status_code=200,
                    message="No tunnel running",
                )
            ),
        ):
            response = self.client.post("/api/system/tunnel/stop")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("message"), "No tunnel running")

    def test_tunnel_start_maps_service_success_url(self):
        with patch(
            "routes.system_remote.start_tunnel_for_current_server",
            new=AsyncMock(
                return_value=RemoteAccessActionResult(
                    success=True,
                    status_code=200,
                    url="https://demo.tunnel",
                )
            ),
        ):
            response = self.client.post("/api/system/tunnel/start")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "url": "https://demo.tunnel"})
