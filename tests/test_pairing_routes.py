import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from pairing.pairing import (
    PairTokenStatus,
    PairingQrResult,
    PairingRevokeResult,
    PairingStatus,
    SSOPairingResult,
)
from pairing.pairing_page_service import PairingPageRenderResult
from remote.remote_access_service import PairVerifyFlowResult
from routes.deps import require_localhost_only, verify_api_key
from routes.pairing import router as pairing_router


class PairingRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(pairing_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        app.dependency_overrides[require_localhost_only] = lambda: True
        self.app = app
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    # Sample tokens matching secrets.token_hex(16) — 32 lowercase hex chars.
    _VALID_PAIR_TOKEN = "0123456789abcdef0123456789abcdef"
    _OTHER_VALID_PAIR_TOKEN = "fedcba9876543210fedcba9876543210"

    def test_verify_pair_token_failure_returns_400(self):
        with patch(
            "routes.pairing.verify_pair_token_for_current_server",
            new=AsyncMock(
                return_value=PairVerifyFlowResult(
                    success=False,
                    status_code=400,
                    error="Pair token expired",
                )
            ),
        ):
            response = self.client.post(
                "/api/pair/verify",
                json={"pair_token": self._OTHER_VALID_PAIR_TOKEN},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "Pair token expired")

    def test_verify_pair_token_rejects_malformed_token(self):
        # Malformed payload must never reach the verification flow.
        with patch(
            "routes.pairing.verify_pair_token_for_current_server",
            new=AsyncMock(),
        ) as mock_verify_flow:
            response = self.client.post(
                "/api/pair/verify",
                json={"pair_token": "not-a-real-token"},
            )

        self.assertEqual(response.status_code, 400)
        mock_verify_flow.assert_not_called()

    def test_get_pair_qr_uses_pairing_data_response_builder(self):
        qr_result = PairingQrResult(
            success=True,
            status_code=200,
            qr_url="codebridge://pair/demo",
            payload={"pair_token": "t1"},
            local_url="http://127.0.0.1:8080",
            tunnel_url=None,
            expires_in_seconds=300,
        )

        with patch("routes.pairing.build_current_pairing_qr_result", return_value=qr_result):
            response = self.client.get("/api/pair/qr")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), qr_result.as_response_fields())

    def test_get_pair_page_returns_html_from_service(self):
        with patch(
            "routes.pairing.build_pairing_page_html_for_current_server",
            return_value=PairingPageRenderResult(success=True, status_code=200, content="<html>ok</html>"),
        ):
            response = self.client.get("/pair")

        self.assertEqual(response.status_code, 200)
        self.assertIn("ok", response.text)

    def test_get_pair_page_returns_service_error_status(self):
        with patch(
            "routes.pairing.build_pairing_page_html_for_current_server",
            return_value=PairingPageRenderResult(
                success=False,
                status_code=500,
                content="Failed to build pairing page",
            ),
        ):
            response = self.client.get("/pair")

        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to build pairing page", response.text)

    def test_verify_pair_token_success_without_firebase_skips_remote_registration(self):
        with patch(
            "routes.pairing.verify_pair_token_for_current_server",
            new=AsyncMock(
                return_value=PairVerifyFlowResult(
                    success=True,
                    status_code=200,
                    api_key="key-1",
                    client_id="client-1",
                )
            ),
        ) as mock_verify_flow:
            response = self.client.post(
                "/api/pair/verify",
                json={"pair_token": self._VALID_PAIR_TOKEN},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("success"))
        self.assertEqual(response.json().get("api_key"), "key-1")
        mock_verify_flow.assert_awaited_once_with(
            pair_token=self._VALID_PAIR_TOKEN,
            client_id=None,
            device_name=None,
            firebase_id_token=None,
            firebase_refresh_token=None,
            force_replace=False,
        )

    def test_verify_pair_token_firebase_registration_failure_returns_error(self):
        with patch(
            "routes.pairing.verify_pair_token_for_current_server",
            new=AsyncMock(
                return_value=PairVerifyFlowResult(
                    success=False,
                    status_code=502,
                    error="Firebase registration failed",
                )
            ),
        ) as mock_verify_flow:
            response = self.client.post(
                "/api/pair/verify",
                json={
                    "pair_token": self._VALID_PAIR_TOKEN,
                    "firebase_id_token": "id-token",
                    "firebase_refresh_token": "refresh-token",
                },
            )

        self.assertEqual(response.status_code, 502)
        body = response.json()
        self.assertEqual(body.get("error"), "Firebase registration failed")
        mock_verify_flow.assert_awaited_once_with(
            pair_token=self._VALID_PAIR_TOKEN,
            client_id=None,
            device_name=None,
            firebase_id_token="id-token",
            firebase_refresh_token="refresh-token",
            force_replace=False,
        )

    def test_verify_sso_pairing_registration_failure_returns_error(self):
        with patch(
            "routes.pairing.verify_sso_pairing_for_current_server",
            new=AsyncMock(
                return_value=SSOPairingResult(
                    success=False,
                    status_code=502,
                    error="Firebase registration failed",
                )
            ),
        ) as mock_sso:
            response = self.client.post(
                "/api/pair/sso",
                json={
                    "firebase_id_token": "id-token",
                    "firebase_refresh_token": "refresh-token",
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json().get("error"), "Firebase registration failed")
        mock_sso.assert_awaited_once_with(
            firebase_id_token="id-token",
            firebase_refresh_token="refresh-token",
            client_id=None,
            device_name=None,
            force_replace=False,
        )

    def test_get_pair_status_passes_through_service_payload(self):
        status = PairingStatus(
            server_id="server-1",
            active_clients=1,
            pending_tokens=1,
            clients=[],
        )

        with patch("routes.pairing.get_pairing_status_for_current_server", return_value=status):
            response = self.client.get("/api/pair/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "server_id": "server-1",
                "active_clients": 1,
                "pending_tokens": 1,
                "clients": [],
            },
        )

    def test_get_token_status_passes_through_service_payload(self):
        status = PairTokenStatus(
            exists=True,
            used=False,
            expired=False,
        )

        # Token must match the secrets.token_hex(16) shape — 32 hex chars.
        valid_token = "0123456789abcdef0123456789abcdef"
        with patch("routes.pairing.get_pair_token_status_for_current_server", return_value=status):
            response = self.client.get(f"/api/pair/token-status/{valid_token}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"exists": True, "used": False, "expired": False})

    def test_get_token_status_rejects_malformed_token(self):
        # The route should reject obviously-wrong tokens cheaply (400),
        # without invoking the pairing service.
        with patch(
            "routes.pairing.get_pair_token_status_for_current_server",
        ) as service:
            response = self.client.get("/api/pair/token-status/not-a-real-token")

        self.assertEqual(response.status_code, 400)
        service.assert_not_called()

    def test_revoke_paired_client_returns_404_when_missing(self):
        revoke_result = PairingRevokeResult(
            success=False,
            status_code=404,
            error="Client missing-client not found",
        )

        with patch("routes.pairing.revoke_paired_client_for_current_server", return_value=revoke_result):
            response = self.client.delete("/api/pair/clients/missing-client")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("error"), "Client missing-client not found")

    def test_revoke_paired_client_returns_success_payload(self):
        revoke_result = PairingRevokeResult(
            success=True,
            status_code=200,
            message="Client c1 revoked",
        )

        with patch("routes.pairing.revoke_paired_client_for_current_server", return_value=revoke_result):
            response = self.client.delete("/api/pair/clients/c1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "message": "Client c1 revoked"})

    def test_register_push_token_success(self):
        with patch(
            "routes.pairing.register_push_token_for_current_server", return_value=True
        ) as mock_register:
            response = self.client.post(
                "/api/pair/push-token",
                json={"token": "fcm-token-1", "platform": "android"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True})
        mock_register.assert_called_once_with(True, "fcm-token-1", platform="android")

    def test_register_push_token_defaults_platform_to_android(self):
        with patch(
            "routes.pairing.register_push_token_for_current_server", return_value=True
        ) as mock_register:
            response = self.client.post("/api/pair/push-token", json={"token": "fcm-token-1"})

        self.assertEqual(response.status_code, 200)
        mock_register.assert_called_once_with(True, "fcm-token-1", platform="android")

    def test_register_push_token_unknown_client_returns_404(self):
        with patch("routes.pairing.register_push_token_for_current_server", return_value=False):
            response = self.client.post(
                "/api/pair/push-token",
                json={"token": "fcm-token-1"},
            )

        self.assertEqual(response.status_code, 404)

    def test_register_push_token_requires_a_paired_key_not_ip_login(self):
        self.app.dependency_overrides[verify_api_key] = lambda: "__ip_login__"
        try:
            with patch(
                "routes.pairing.register_push_token_for_current_server"
            ) as mock_register:
                response = self.client.post(
                    "/api/pair/push-token",
                    json={"token": "fcm-token-1"},
                )
        finally:
            self.app.dependency_overrides[verify_api_key] = lambda: True

        self.assertEqual(response.status_code, 401)
        mock_register.assert_not_called()
