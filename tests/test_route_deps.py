import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from auth_service import ApiKeyValidationResult
from routes.deps import verify_api_key


def _make_mock_request(*, from_tunnel: bool = False) -> MagicMock:
    """Create a mock Request object."""
    request = MagicMock()
    if from_tunnel:
        request.headers.get = lambda h: "test-value" if h == "CF-Ray" else None
    else:
        request.headers.get = lambda h: None
    return request


class RouteDepsTest(unittest.IsolatedAsyncioTestCase):
    async def test_verify_api_key_returns_validated_key(self):
        request = _make_mock_request(from_tunnel=False)
        with patch(
            "routes.deps.validate_api_key_for_current_server",
            return_value=ApiKeyValidationResult(success=True, api_key="token-1"),
        ):
            validated = await verify_api_key(
                request=request, x_api_key="token-1", api_key=None
            )

        self.assertEqual(validated, "token-1")

    async def test_verify_api_key_raises_http_exception_on_failure(self):
        request = _make_mock_request(from_tunnel=False)
        with patch(
            "routes.deps.validate_api_key_for_current_server",
            return_value=ApiKeyValidationResult(success=False, error="API key required"),
        ):
            with self.assertRaises(HTTPException) as exc_ctx:
                await verify_api_key(request=request, x_api_key=None, api_key=None)

        self.assertEqual(exc_ctx.exception.status_code, 401)
        self.assertEqual(exc_ctx.exception.detail, "API key required")

    async def test_verify_api_key_blocks_tunnel_access_without_key(self):
        """Tunnel access without API key should be blocked."""
        request = _make_mock_request(from_tunnel=True)
        with patch(
            "routes.deps.validate_api_key_for_current_server",
            return_value=ApiKeyValidationResult(
                success=True, api_key="__ip_login__", is_ip_login=True
            ),
        ):
            with self.assertRaises(HTTPException) as exc_ctx:
                await verify_api_key(request=request, x_api_key=None, api_key=None)

        self.assertEqual(exc_ctx.exception.status_code, 401)
        self.assertIn("external access", exc_ctx.exception.detail)

    async def test_verify_api_key_blocks_tunnel_even_with_ip_login(self):
        """IP login mode should NOT allow tunnel access - tunnel always requires API key."""
        request = _make_mock_request(from_tunnel=True)
        with patch(
            "routes.deps.validate_api_key_for_current_server",
            return_value=ApiKeyValidationResult(
                success=True, api_key="__ip_login__", is_ip_login=True
            ),
        ):
            with self.assertRaises(HTTPException) as exc_ctx:
                await verify_api_key(request=request, x_api_key=None, api_key=None)

        self.assertEqual(exc_ctx.exception.status_code, 401)
        self.assertIn("external access", exc_ctx.exception.detail)

    async def test_verify_api_key_allows_local_with_ip_login(self):
        """IP login mode should allow LOCAL access without API key."""
        request = _make_mock_request(from_tunnel=False)
        with patch(
            "routes.deps.validate_api_key_for_current_server",
            return_value=ApiKeyValidationResult(
                success=True, api_key="__ip_login__", is_ip_login=True
            ),
        ):
            validated = await verify_api_key(
                request=request, x_api_key=None, api_key=None
            )

        self.assertEqual(validated, "__ip_login__")


if __name__ == "__main__":
    unittest.main()
