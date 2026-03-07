"""Preview proxy for dev servers."""

import httpx
from fastapi import Request, Response

STRIP_RESPONSE_HEADERS = {"transfer-encoding", "content-encoding", "content-length"}


class PreviewProxy:
    """Reverse proxy for dev server previews."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    def _build_target_url(self, request: Request, target_port: int, path: str) -> str:
        target_url = f"http://localhost:{target_port}/{path}"
        if request.query_params:
            target_url = f"{target_url}?{request.query_params}"
        return target_url

    def _build_forward_headers(self, request: Request) -> dict[str, str]:
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("Host", None)
        return headers

    async def proxy_request(
        self,
        request: Request,
        target_port: int,
        path: str = "",
    ) -> Response:
        """Proxy request to target dev server."""
        target_url = self._build_target_url(request, target_port, path)
        headers = self._build_forward_headers(request)

        try:
            body = await request.body()
            resp = await self._client.request(
                request.method,
                target_url,
                headers=headers,
                content=body if body else None,
            )

            response_headers = dict(resp.headers)
            for header in STRIP_RESPONSE_HEADERS:
                response_headers.pop(header, None)
                response_headers.pop(header.title(), None)

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )

        except httpx.ConnectError:
            return Response(
                content="Dev server not running or not accessible",
                status_code=503,
                media_type="text/plain",
            )
        except Exception as e:
            return Response(
                content=f"Proxy error: {str(e)}",
                status_code=500,
                media_type="text/plain",
            )

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


# Global proxy instance
_preview_proxy: PreviewProxy | None = None


def get_preview_proxy() -> PreviewProxy:
    """Get global preview proxy instance."""
    global _preview_proxy
    if _preview_proxy is None:
        _preview_proxy = PreviewProxy()
    return _preview_proxy
