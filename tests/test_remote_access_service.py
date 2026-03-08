import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import remote_access_service
from pairing import PairingVerifyTokenResult


class RemoteAccessServiceTest(unittest.IsolatedAsyncioTestCase):
    def test_parse_remote_access_login_payload_defaults(self):
        payload, error = remote_access_service.parse_remote_access_login_payload(
            {"id_token": "  token  "}
        )

        self.assertIsNone(error)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.id_token, "token")
        self.assertEqual(payload.auth_mode, "refresh_token")
        self.assertIsNone(payload.refresh_token)
        self.assertTrue(payload.register_device)

    def test_parse_remote_access_login_payload_missing_token(self):
        payload, error = remote_access_service.parse_remote_access_login_payload({})

        self.assertIsNone(payload)
        self.assertIn("id_token", error or "")

    def test_build_remote_network_status_with_defaults(self):
        config = SimpleNamespace(
            server_name="demo-server",
            remote_access_enabled=True,
            firebase_enabled=True,
        )
        fake_firebase_auth = MagicMock()
        fake_firebase_auth.get_status.return_value = {
            "authenticated": True,
            "user_id": "user-1",
            "server_id": "server-1",
        }

        with (
            patch.object(remote_access_service, "get_tunnel_service", return_value=None),
            patch.object(remote_access_service, "get_firebase_auth", return_value=fake_firebase_auth),
        ):
            status = remote_access_service.build_remote_network_status(config)

        self.assertEqual(status.mdns.server_name, "demo-server")
        self.assertTrue(status.firebase.authenticated)
        self.assertEqual(status.firebase.user_id, "user-1")
        self.assertFalse(status.tunnel.running)
        self.assertEqual(
            status.as_response_fields()["firebase"]["server_id"],
            "server-1",
        )

    def test_build_remote_network_status_for_current_server_uses_config(self):
        fake_config = SimpleNamespace(
            server_name="demo-server",
            remote_access_enabled=True,
            firebase_enabled=False,
        )
        expected_status = remote_access_service.RemoteNetworkStatus(
            mdns=remote_access_service.RemoteMdnsStatus(
                available=False,
                enabled=False,
                registered=False,
                server_name="demo-server",
            ),
            tunnel=remote_access_service.RemoteTunnelStatus(
                available=False,
                enabled=True,
                running=False,
                url=None,
            ),
            firebase=remote_access_service.RemoteFirebaseStatus(
                available=False,
                enabled=False,
                authenticated=False,
                user_id=None,
                server_id=None,
            ),
        )

        with (
            patch.object(remote_access_service, "get_config", return_value=fake_config),
            patch.object(
                remote_access_service,
                "build_remote_network_status",
                return_value=expected_status,
            ) as mock_build,
        ):
            status = remote_access_service.build_remote_network_status_for_current_server()

        self.assertEqual(status, expected_status)
        mock_build.assert_called_once_with(fake_config)

    async def test_start_tunnel_unavailable_returns_400(self):
        with patch.object(remote_access_service, "TUNNEL_AVAILABLE", False):
            result = await remote_access_service.start_tunnel_for_remote_access(local_port=8080)

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.error, "Tunnel service not available")
        self.assertEqual(result.as_response_fields(), {"error": "Tunnel service not available"})

    async def test_start_tunnel_success_returns_url(self):
        fake_tunnel_service = MagicMock()
        fake_tunnel_service.start = AsyncMock(return_value="https://demo.tunnel")

        with (
            patch.object(remote_access_service, "TUNNEL_AVAILABLE", True),
            patch.object(remote_access_service, "get_tunnel_service", return_value=fake_tunnel_service),
        ):
            result = await remote_access_service.start_tunnel_for_remote_access(local_port=8080)

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.url, "https://demo.tunnel")
        self.assertIsNone(result.error)
        self.assertEqual(
            result.as_response_fields(),
            {"success": True, "url": "https://demo.tunnel"},
        )

    async def test_start_tunnel_for_current_server_uses_config_port(self):
        with (
            patch.object(remote_access_service, "get_config", return_value=SimpleNamespace(port=9191)),
            patch.object(
                remote_access_service,
                "start_tunnel_for_remote_access",
                new=AsyncMock(
                    return_value=remote_access_service.RemoteAccessActionResult(
                        success=True,
                        status_code=200,
                        url="https://demo.tunnel",
                    )
                ),
            ) as mock_start,
        ):
            result = await remote_access_service.start_tunnel_for_current_server()

        self.assertTrue(result.success)
        self.assertEqual(result.url, "https://demo.tunnel")
        mock_start.assert_awaited_once_with(local_port=9191)

    async def test_stop_tunnel_without_service_returns_message(self):
        with patch.object(remote_access_service, "get_tunnel_service", return_value=None):
            result = await remote_access_service.stop_tunnel_for_remote_access()

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.message, "No tunnel running")

    async def test_register_pairing_remote_access_returns_empty_when_disabled(self):
        with patch.object(remote_access_service, "FIREBASE_AVAILABLE", False):
            result = await remote_access_service.register_pairing_remote_access(
                firebase_id_token="token",
                firebase_refresh_token=None,
                auth_mode="refresh_token",
                local_port=8080,
            )

        self.assertEqual(result.as_response_fields(), {})

    async def test_register_pairing_remote_access_returns_error_on_auth_failure(self):
        fake_firebase_auth = MagicMock()
        fake_firebase_auth.authenticate_with_token = AsyncMock(return_value=False)

        with (
            patch.object(remote_access_service, "FIREBASE_AVAILABLE", True),
            patch.object(remote_access_service, "get_firebase_auth", return_value=fake_firebase_auth),
        ):
            result = await remote_access_service.register_pairing_remote_access(
                firebase_id_token="token",
                firebase_refresh_token=None,
                auth_mode="refresh_token",
                local_port=8080,
            )

        self.assertEqual(
            result.as_response_fields(),
            {
                "firebase_registered": False,
                "firebase_error": "Token verification failed",
            },
        )

    async def test_login_for_remote_access_returns_400_when_firebase_unavailable(self):
        payload = remote_access_service.RemoteAccessLoginPayload(
            id_token="token",
            refresh_token=None,
            auth_mode="refresh_token",
            register_device=True,
        )

        result = await remote_access_service.login_for_remote_access(
            payload,
            firebase_enabled=True,
            local_port=8080,
            firebase_available=False,
            firebase_auth=MagicMock(),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.error, "Firebase is not available")

    async def test_login_for_remote_access_success_without_register_device(self):
        payload = remote_access_service.RemoteAccessLoginPayload(
            id_token="token",
            refresh_token="refresh",
            auth_mode="refresh_token",
            register_device=False,
        )
        fake_firebase_auth = MagicMock()
        fake_firebase_auth.authenticate_with_token = AsyncMock(return_value=True)
        fake_firebase_auth.get_status.return_value = {
            "user_id": "user-1",
            "server_id": "server-1",
            "server_name": "Pixel",
            "auth_mode": "refresh_token",
        }

        with patch.object(
            remote_access_service,
            "register_device_for_remote_access",
            new=AsyncMock(),
        ) as mock_register:
            result = await remote_access_service.login_for_remote_access(
                payload,
                firebase_enabled=True,
                local_port=8080,
                firebase_available=True,
                firebase_auth=fake_firebase_auth,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.user_id, "user-1")
        self.assertEqual(result.as_response_fields()["server_name"], "Pixel")
        mock_register.assert_not_awaited()

    async def test_login_for_remote_access_for_current_server_uses_config(self):
        payload = remote_access_service.RemoteAccessLoginPayload(
            id_token="token",
            refresh_token=None,
            auth_mode="refresh_token",
            register_device=True,
        )
        fake_firebase_auth = MagicMock()

        with (
            patch.object(
                remote_access_service,
                "get_config",
                return_value=SimpleNamespace(firebase_enabled=True, port=9090),
            ),
            patch.object(
                remote_access_service,
                "login_for_remote_access",
                new=AsyncMock(
                    return_value=remote_access_service.RemoteAccessLoginResult(
                        success=True,
                        status_code=200,
                        user_id="u1",
                    )
                ),
            ) as mock_login,
        ):
            result = await remote_access_service.login_for_remote_access_for_current_server(
                payload,
                firebase_available=True,
                firebase_auth=fake_firebase_auth,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.user_id, "u1")
        mock_login.assert_awaited_once_with(
            payload,
            firebase_enabled=True,
            local_port=9090,
            firebase_available=True,
            firebase_auth=fake_firebase_auth,
        )

    async def test_login_for_remote_access_body_for_current_server_returns_parse_error(self):
        result = await remote_access_service.login_for_remote_access_body_for_current_server({})

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertIn("id_token", result.error or "")

    async def test_login_for_remote_access_request_json_for_current_server_returns_400_on_invalid_json(self):
        async def bad_loader():
            raise ValueError("bad json")

        result = await remote_access_service.login_for_remote_access_request_json_for_current_server(bad_loader)

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.error, "Invalid request body. Expected JSON with 'id_token' field.")

    async def test_login_for_remote_access_request_json_for_current_server_delegates(self):
        async def loader():
            return {"id_token": "token"}

        with patch.object(
            remote_access_service,
            "login_for_remote_access_body_for_current_server",
            new=AsyncMock(
                return_value=remote_access_service.RemoteAccessLoginResult(
                    success=True,
                    status_code=200,
                    user_id="u1",
                )
            ),
        ) as mock_login:
            result = await remote_access_service.login_for_remote_access_request_json_for_current_server(loader)

        self.assertTrue(result.success)
        self.assertEqual(result.user_id, "u1")
        mock_login.assert_awaited_once_with({"id_token": "token"})

    async def test_verify_pairing_flow_returns_400_on_pairing_failure(self):
        fake_pairing = MagicMock()
        fake_pairing.verify_pair_token.return_value = PairingVerifyTokenResult(
            success=False,
            status_code=400,
            error="Pair token expired",
        )

        result = await remote_access_service.verify_pairing_flow(
            pairing_service=fake_pairing,
            pair_token="expired",
            client_id=None,
            device_name=None,
            firebase_id_token=None,
            firebase_refresh_token=None,
            auth_mode="refresh_token",
            local_port=8080,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.error, "Pair token expired")
        self.assertEqual(result.as_response_fields()["error"], "Pair token expired")

    async def test_verify_pairing_flow_merges_remote_registration_fields(self):
        fake_pairing = MagicMock()
        fake_pairing.verify_pair_token.return_value = PairingVerifyTokenResult(
            success=True,
            status_code=200,
            api_key="k1",
            server_id="server-1",
            client_id="c1",
        )

        with patch.object(
            remote_access_service,
            "register_pairing_remote_access",
            new=AsyncMock(
                return_value=remote_access_service.PairingRemoteAccessResult(
                    firebase_registered=True,
                )
            ),
        ) as mock_remote:
            result = await remote_access_service.verify_pairing_flow(
                pairing_service=fake_pairing,
                pair_token="valid",
                client_id="c1",
                device_name="phone",
                firebase_id_token="id-token",
                firebase_refresh_token="refresh-token",
                auth_mode="refresh_token",
                local_port=9090,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.api_key, "k1")
        self.assertTrue(result.firebase_registered)
        self.assertEqual(result.as_response_fields()["api_key"], "k1")
        self.assertTrue(result.as_response_fields()["firebase_registered"])
        mock_remote.assert_awaited_once_with(
            firebase_id_token="id-token",
            firebase_refresh_token="refresh-token",
            auth_mode="refresh_token",
            local_port=9090,
        )

    async def test_verify_pairing_flow_for_current_server_uses_config_port(self):
        fake_pairing = MagicMock()

        with (
            patch.object(remote_access_service, "get_config", return_value=SimpleNamespace(port=9191)),
            patch.object(
                remote_access_service,
                "verify_pairing_flow",
                new=AsyncMock(
                    return_value=remote_access_service.PairVerifyFlowResult(
                        success=True,
                        status_code=200,
                        api_key="k2",
                    )
                ),
            ) as mock_verify,
        ):
            result = await remote_access_service.verify_pairing_flow_for_current_server(
                pairing_service=fake_pairing,
                pair_token="valid",
                client_id="c2",
                device_name="Phone",
                firebase_id_token=None,
                firebase_refresh_token=None,
                auth_mode="refresh_token",
            )

        self.assertTrue(result.success)
        self.assertEqual(result.api_key, "k2")
        mock_verify.assert_awaited_once_with(
            pairing_service=fake_pairing,
            pair_token="valid",
            client_id="c2",
            device_name="Phone",
            firebase_id_token=None,
            firebase_refresh_token=None,
            auth_mode="refresh_token",
            local_port=9191,
        )

    async def test_verify_pair_token_for_current_server_uses_default_pairing_service(self):
        fake_pairing = MagicMock()

        with (
            patch.object(remote_access_service, "get_pairing_service", return_value=fake_pairing),
            patch.object(
                remote_access_service,
                "verify_pairing_flow_for_current_server",
                new=AsyncMock(
                    return_value=remote_access_service.PairVerifyFlowResult(
                        success=True,
                        status_code=200,
                        api_key="k3",
                    )
                ),
            ) as mock_verify,
        ):
            result = await remote_access_service.verify_pair_token_for_current_server(
                pair_token="valid",
                client_id="c3",
                device_name="Phone",
                firebase_id_token=None,
                firebase_refresh_token=None,
                auth_mode="refresh_token",
            )

        self.assertTrue(result.success)
        self.assertEqual(result.api_key, "k3")
        mock_verify.assert_awaited_once_with(
            pairing_service=fake_pairing,
            pair_token="valid",
            client_id="c3",
            device_name="Phone",
            firebase_id_token=None,
            firebase_refresh_token=None,
            auth_mode="refresh_token",
        )

    async def test_logout_remote_access_unavailable_returns_400(self):
        result = await remote_access_service.logout_remote_access(
            firebase_available=False,
            firebase_auth=MagicMock(),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.error, "Firebase is not available")

    async def test_logout_remote_access_calls_sign_out(self):
        fake_firebase_auth = MagicMock()
        fake_firebase_auth.sign_out = AsyncMock()

        result = await remote_access_service.logout_remote_access(
            firebase_available=True,
            firebase_auth=fake_firebase_auth,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        fake_firebase_auth.sign_out.assert_awaited_once()

    async def test_logout_remote_access_for_current_server_uses_optional_context(self):
        fake_firebase_auth = MagicMock()

        with patch.object(
            remote_access_service,
            "logout_remote_access",
            new=AsyncMock(
                return_value=remote_access_service.RemoteAccessActionResult(
                    success=True,
                    status_code=200,
                )
            ),
        ) as mock_logout:
            with patch.object(remote_access_service, "get_firebase_auth", return_value=fake_firebase_auth):
                result = await remote_access_service.logout_remote_access_for_current_server()

        self.assertTrue(result.success)
        mock_logout.assert_awaited_once_with(
            firebase_available=remote_access_service.FIREBASE_AVAILABLE,
            firebase_auth=fake_firebase_auth,
        )

    async def test_disconnect_remote_access_calls_clear_auth(self):
        fake_firebase_auth = MagicMock()
        fake_firebase_auth.clear_auth = AsyncMock()

        result = await remote_access_service.disconnect_remote_access(
            firebase_available=True,
            firebase_auth=fake_firebase_auth,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.message, "Disconnected from remote access")
        self.assertEqual(
            result.as_response_fields(),
            {"success": True, "message": "Disconnected from remote access"},
        )
        fake_firebase_auth.clear_auth.assert_awaited_once()

    async def test_disconnect_remote_access_for_current_server_uses_optional_context(self):
        fake_firebase_auth = MagicMock()

        with patch.object(
            remote_access_service,
            "disconnect_remote_access",
            new=AsyncMock(
                return_value=remote_access_service.RemoteAccessActionResult(
                    success=True,
                    status_code=200,
                    message="Disconnected from remote access",
                )
            ),
        ) as mock_disconnect:
            with patch.object(remote_access_service, "get_firebase_auth", return_value=fake_firebase_auth):
                result = await remote_access_service.disconnect_remote_access_for_current_server()

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Disconnected from remote access")
        mock_disconnect.assert_awaited_once_with(
            firebase_available=remote_access_service.FIREBASE_AVAILABLE,
            firebase_auth=fake_firebase_auth,
        )

    async def test_stop_tunnel_for_current_server_delegates(self):
        with patch.object(
            remote_access_service,
            "stop_tunnel_for_remote_access",
            new=AsyncMock(
                return_value=remote_access_service.RemoteAccessActionResult(
                    success=True,
                    status_code=200,
                    message="No tunnel running",
                )
            ),
        ) as mock_stop:
            result = await remote_access_service.stop_tunnel_for_current_server()

        self.assertTrue(result.success)
        self.assertEqual(result.message, "No tunnel running")
        mock_stop.assert_awaited_once_with()
