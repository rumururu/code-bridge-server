import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
from starlette.requests import Request

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from preview import PreviewProxy


def _make_request(
    method: str,
    *,
    path: str = "/",
    query: bytes = b"",
    body: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "query_string": query,
        "headers": headers
        or [
            (b"host", b"localhost:8080"),
            (b"x-test", b"value"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("localhost", 8080),
    }

    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=receive)


class PreviewProxyTest(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_request_forwards_request_and_strips_host_header(self):
        client = AsyncMock()
        client.request = AsyncMock(
            return_value=httpx.Response(
                200,
                content=b"ok",
                headers={
                    "content-type": "text/plain",
                    "content-length": "999",
                    "x-proxy": "1",
                },
            )
        )

        proxy = PreviewProxy(client=client)
        request = _make_request("POST", path="/preview/demo", query=b"q=1", body=b"payload")

        response = await proxy.proxy_request(request, 5173, "api/path")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b"ok")
        self.assertEqual(response.headers.get("x-proxy"), "1")
        self.assertEqual(response.headers.get("content-length"), "2")

        client.request.assert_awaited_once()
        args, kwargs = client.request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "http://localhost:5173/api/path?q=1")
        self.assertEqual(kwargs["content"], b"payload")
        self.assertNotIn("host", kwargs["headers"])

    async def test_proxy_request_returns_503_on_connect_error(self):
        client = AsyncMock()
        client.request = AsyncMock(side_effect=httpx.ConnectError("connect failed"))

        proxy = PreviewProxy(client=client)
        request = _make_request("GET")

        response = await proxy.proxy_request(request, 5173, "")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Dev server not running", response.body.decode())

    async def test_proxy_request_returns_500_on_unexpected_error(self):
        client = AsyncMock()
        client.request = AsyncMock(side_effect=RuntimeError("boom"))

        proxy = PreviewProxy(client=client)
        request = _make_request("GET")

        response = await proxy.proxy_request(request, 5173, "")

        self.assertEqual(response.status_code, 500)
        self.assertIn("Proxy error: boom", response.body.decode())


if __name__ == "__main__":
    unittest.main()
