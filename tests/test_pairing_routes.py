import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from pairing import (
    PairTokenStatus,
    PairingQrResult,
    PairingRevokeResult,
    PairingStatus,
)
from pairing_page_service import PairingPageRenderResult
from remote_access_service import PairVerifyFlowResult
from routes.deps import verify_api_key
from routes.pairing import router as pairing_router


class PairingRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(pairing_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

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
            response = self.client.post("/api/pair/verify", json={"pair_token": "expired"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "Pair token expired")

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
            response = self.client.post("/api/pair/verify", json={"pair_token": "valid"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("success"))
        self.assertEqual(response.json().get("api_key"), "key-1")
        mock_verify_flow.assert_awaited_once_with(
            pair_token="valid",
            client_id=None,
            device_name=None,
            firebase_id_token=None,
            firebase_refresh_token=None,
            auth_mode="refresh_token",
        )

    def test_verify_pair_token_success_with_firebase_merges_remote_fields(self):
        with patch(
            "routes.pairing.verify_pair_token_for_current_server",
            new=AsyncMock(
                return_value=PairVerifyFlowResult(
                    success=True,
                    status_code=200,
                    api_key="key-2",
                    client_id="client-2",
                    firebase_registered=False,
                    firebase_error="Token verification failed",
                )
            ),
        ) as mock_verify_flow:
            response = self.client.post(
                "/api/pair/verify",
                json={
                    "pair_token": "valid",
                    "firebase_id_token": "id-token",
                    "firebase_refresh_token": "refresh-token",
                    "auth_mode": "refresh_token",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body.get("success"))
        self.assertFalse(body.get("firebase_registered"))
        self.assertEqual(body.get("firebase_error"), "Token verification failed")
        mock_verify_flow.assert_awaited_once_with(
            pair_token="valid",
            client_id=None,
            device_name=None,
            firebase_id_token="id-token",
            firebase_refresh_token="refresh-token",
            auth_mode="refresh_token",
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

        with patch("routes.pairing.get_pair_token_status_for_current_server", return_value=status):
            response = self.client.get("/api/pair/token-status/token-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"exists": True, "used": False, "expired": False})

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
